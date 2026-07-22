# Isolation du cache de conversation PAR MODÈLE (standard 2026-07-22, remplace le
# « cache souverain » save/restore) : la sonde du bench mesure si le cache survit à
# la pollution du slot (A -> B -> A) ; verdict dans model.toml (cache_isolation),
# exploité par serve/swap (--parallel monté à 2 pour CE modèle seulement).
from __future__ import annotations

import tomllib

from loom.config import _parse_model
from loom.runtime.hardware import HardwareProfile
from loom.runtime.server_args import resolve_parallel
from loom.runtime.swap import _model_cmd
from loom.setup.topology import TOPO_RAM, ServerProbe, isolation_needed


# ── Verdict pur ──────────────────────────────────────────────────────────────


def test_isolation_needed_bimodal():
    # Hybride : le retour re-préfille l'essentiel -> isolation nécessaire.
    assert isolation_needed(600, 590) is True
    # Classique : le prompt-cache natif recycle tout sauf le suffixe.
    assert isolation_needed(600, 5) is False
    # Frontière 50 % (mesure bimodale en pratique : ~0 % ou ~100 %).
    assert isolation_needed(100, 50) is True
    assert isolation_needed(100, 49) is False


def test_isolation_needed_mesure_illisible_sans_verdict():
    # 1er passage vide = sonde illisible : on n'impose JAMAIS un doublement de
    # KV sans preuve.
    assert isolation_needed(0, 0) is False
    assert isolation_needed(0, 500) is False


# ── Sonde A -> pollution -> A ────────────────────────────────────────────────


def test_probe_isolation_sequence(monkeypatch):
    from loom.setup import topology as topo

    monkeypatch.setattr(topo.time, "sleep", lambda s: None)
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=4,
        ngl=0,
        topology=TOPO_RAM,
        kill=lambda proc: None,
    )
    seen: list[tuple[str, bool]] = []

    def fake_completion(prompt, n_predict, cache_prompt=False):
        seen.append((prompt, cache_prompt))
        # 1er A : tout traité ; B : pollution ; retour A : 4 tokens (suffixe).
        n = {1: 600, 2: 610, 3: 4}[len(seen)]
        return {"timings": {"prompt_n": n}}

    monkeypatch.setattr(probe, "_start", lambda ctx: object())
    monkeypatch.setattr(probe, "_completion", fake_completion)
    first, back = probe.probe_isolation()
    assert (first, back) == (600, 4)
    # 3 appels, tous avec cache_prompt (sans lui la sonde ne mesure rien).
    assert [c for _, c in seen] == [True, True, True]
    # B ne partage AUCUN préfixe avec A, et le retour re-commence par A.
    a1, b, a2 = (p for p, _ in seen)
    assert not b.startswith(a1[:20])
    assert a2.startswith(a1)


def test_probe_n_parallel_traverse_les_flags(monkeypatch):
    # P2 (le conseilleur simule l'exécutant) : isolation jugée nécessaire ->
    # la calibration doit charger avec --parallel 2 et le KV réellement doublé.
    from loom.setup import topology as topo

    captured: dict = {}

    def fake_build(**kw):
        captured.update(kw)
        raise RuntimeError("stop après capture")

    monkeypatch.setattr(topo, "build_server_args", fake_build)
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=4,
        ngl=0,
        topology=TOPO_RAM,
        n_parallel=2,
    )
    try:
        probe._start(4096)
    except RuntimeError:
        pass
    assert captured["n_parallel"] == 2


# ── Verdict -> config -> ligne de commande ───────────────────────────────────


def test_resolve_parallel():
    assert resolve_parallel(1, False) == 1
    assert resolve_parallel(1, True) == 2
    assert resolve_parallel(2, True) == 2  # déjà isolé : pas de sur-enchère
    assert resolve_parallel(3, False) == 3  # global explicite respecté
    assert resolve_parallel(0, False) == 1  # borne basse


def test_model_toml_porte_cache_isolation():
    base = {"repo": "r/x", "filename": "x.gguf", "n_layers": 40, "size_mb": 100}
    m = _parse_model({**base, "cache_isolation": True}, "x")
    assert m.cache_isolation is True
    # Absent -> False (jamais sondé = pas d'isolation imposée).
    assert _parse_model(base, "x").cache_isolation is False


def test_swap_cmd_monte_parallel_pour_le_modele_isole():
    profile = HardwareProfile(False, None, 0, 8)
    base = {"repo": "r/x", "filename": "x.gguf", "n_layers": 40, "size_mb": 100}
    kw = dict(
        profile=profile,
        llama_bin="llama-server",
        models_dir="/models",
        context=8192,
    )
    isole = _model_cmd(_parse_model({**base, "cache_isolation": True}, "a"), **kw)
    assert "--parallel 2" in isole
    assert "-c 16384" in isole  # fenêtre PAR SLOT préservée : -c doublé
    sain = _model_cmd(_parse_model(base, "b"), **kw)
    assert "--parallel 1" in sain
    assert "-c 8192" in sain


# ── Écriture du verdict dans model.toml ──────────────────────────────────────


def test_set_model_cache_isolation_ecrit_et_remplace(tmp_path):
    from loom.setup.cli import _set_model_cache_isolation

    mdir = tmp_path / "m"
    mdir.mkdir()
    p = mdir / "model.toml"
    p.write_text('filename = "x.gguf"\ncontext = 8192\n', encoding="utf-8")
    gguf = mdir / "x.gguf"

    _set_model_cache_isolation(gguf, True, "retour 590/600 tokens retraités")
    d = tomllib.loads(p.read_text(encoding="utf-8"))
    assert d["cache_isolation"] is True
    assert "sondé par le bench" in p.read_text(encoding="utf-8")

    # Re-sonde avec verdict inverse : REMPLACE (ligne + stamp), pas de doublon.
    _set_model_cache_isolation(gguf, False, "retour 4/600 tokens retraités")
    txt = p.read_text(encoding="utf-8")
    d = tomllib.loads(txt)
    assert d["cache_isolation"] is False
    assert txt.count("cache_isolation =") == 1
    assert txt.count("sondé par le bench") == 1
    assert d["context"] == 8192  # le reste du fichier est intact


def test_set_model_cache_isolation_sans_toml_ne_leve_pas(tmp_path):
    from loom.setup.cli import _set_model_cache_isolation

    _set_model_cache_isolation(tmp_path / "absent.gguf", True, "d")  # no-op


# ── slot_kv : OFF par défaut (standard 2026-07-22) ───────────────────────────


def test_slot_kv_off_par_defaut_et_reactivable(tmp_path):
    from loom.config import load_config

    d = tmp_path / "models" / "local" / "text" / "m1"
    d.mkdir(parents=True)
    (d / "model.toml").write_text(
        'repo = "org/x-GGUF"\nfilename = "x.gguf"\nn_layers = 10\nsize_mb = 100\n',
        encoding="utf-8",
    )
    defaults = tmp_path / "defaults.toml"
    defaults.write_text(
        f"""
[server]
context = 8192
port = 8080
bin = "llama-server"

[chat]
history_path = "{(tmp_path / "var" / "conversation.json").as_posix()}"

[storage]
models_root = "{(tmp_path / "models").as_posix()}"
""",
        encoding="utf-8",
    )
    assert load_config(defaults).slot_kv is False
    # Réactivable explicitement (vieux llama-server sans prompt-cache RAM natif).
    local = tmp_path / "local.toml"
    local.write_text("[server]\nslot_kv = true\n", encoding="utf-8")
    assert load_config(defaults, local).slot_kv is True
