# tests/test_checkpoints_effectifs.py
"""Nombre EFFECTIF de checkpoints d'état récurrent par mesure (revue 2026-10-10).

32 est le défaut du serveur (`ctx_checkpoints`), donc un MAXIMUM : le nombre créé
pendant une mesure dépend du prompt, de `--checkpoint-min-step` et de la longueur de
la génération. L'estimation mémoire compte le plafond ; la mesure doit dire combien il
y en a eu vraiment. Le serveur le journalise (« created context checkpoint N of M
(… size = X MiB) ») : la sonde capture son journal (stdout + stderr, les builds
loggent sur l'un ou l'autre) dans un fichier temporaire, le lit après la mesure, et
chaque échantillon porte le compte — ou dit « non mesuré » quand le journal est muet.
"""

from __future__ import annotations

import os

from loom.runtime.hardware import HardwareProfile
from loom.setup import topology as topo
from loom.setup.placement import _agreger, checkpoints_text, validate_final
from loom.setup.topology import (
    TOPO_MOE_HYBRIDE,
    ProbeResult,
    ServerProbe,
    parse_checkpoints,
)

# Extrait RÉEL de var/logs/llama/ternary-bonsai-2-27b-q2_0.log (Bonsai 2, 2026-10).
JOURNAL = """0.03.892.408 I srv    load_model: context checkpoints enabled, max = 32, min spacing = 512
2.46.807.006 I slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 9009, pos_max = 9009, n_tokens = 9010, size = 149.626 MiB)
2.57.749.859 I slot create_check: id  0 | task 0 | created context checkpoint 2 of 32 (pos_min = 9513, pos_max = 9513, n_tokens = 9514, size = 149.626 MiB)
2.59.595.918 I slot create_check: id  0 | task 0 | created context checkpoint 3 of 32 (pos_min = 9521, pos_max = 9521, n_tokens = 9522, size = 149.626 MiB)
3.10.740.324 I slot create_check: id  0 | task 77 | superseding context checkpoint at n_tokens = 9522
3.10.780.588 I slot create_check: id  0 | task 77 | created context checkpoint 3 of 32 (pos_min = 9521, pos_max = 9521, n_tokens = 9522, size = 149.626 MiB)
"""


def test_parse_checkpoints_lit_le_nombre_effectif_pas_le_plafond():
    cp = parse_checkpoints(JOURNAL)
    # 4 lignes « created » (un remplacement), mais 3 checkpoints vivants sur 32 possibles.
    assert cp["effectifs"] == 3 and cp["plafond"] == 32 and cp["crees"] == 4
    assert cp["taille_mb"] == 149.6 and cp["min_spacing"] == 512
    assert "journal serveur" in cp["source"]
    assert None not in cp.values()  # finit dans local.toml : pas de null


def _ligne(slot, n, n_tokens):
    return (
        f"0.05 I slot create_check: id  {slot} | task 0 | created context checkpoint "
        f"{n} of 32 (pos_min = {n_tokens - 1}, pos_max = {n_tokens - 1}, n_tokens = "
        f"{n_tokens}, size = 149.626 MiB)"
    )


def test_parse_checkpoints_compte_par_slot_puis_additionne():
    """Vu en réel (Bonsai 2, -lv 4) : « N of M » est le compte du SLOT après création,
    et repart de 1 quand le slot est réinitialisé. Vivants = dernier N de chaque slot,
    additionnés ; le plafond M vaut par slot."""
    deux_slots = "\n".join(
        [_ligne(1, 1, 5), _ligne(1, 2, 517), _ligne(0, 1, 7090), _ligne(0, 2, 7602)]
    )
    cp = parse_checkpoints(deux_slots)
    assert cp["effectifs"] == 4 and cp["par_slot"] == {"0": 2, "1": 2}
    assert cp["max_par_slot"] == 2 and cp["plafond"] == 32 and cp["crees"] == 4
    # Même slot réinitialisé (prompt sans cache) : 1, 2 puis 1, 2 -> 2 vivants.
    meme_slot = "\n".join(
        [_ligne(1, 1, 5), _ligne(1, 2, 517), _ligne(1, 1, 7090), _ligne(1, 2, 7602)]
    )
    cp = parse_checkpoints(meme_slot)
    assert cp["effectifs"] == 2 and cp["par_slot"] == {"1": 2} and cp["crees"] == 4


def test_parse_checkpoints_actives_mais_aucun_cree():
    cp = parse_checkpoints(JOURNAL.splitlines()[0])
    assert cp["effectifs"] == 0 and cp["plafond"] == 32 and cp["crees"] == 0
    assert "aucun créé" in cp["source"]


def test_parse_checkpoints_sans_ligne_dit_non_mesure():
    cp = parse_checkpoints("0.01 I srv load_model: prompt cache is enabled\n")
    assert cp.get("effectifs") is None and "non mesuré" in cp["source"]
    assert "non mesuré" in parse_checkpoints("")["source"]


_UMA = HardwareProfile(
    True, "Radeon 860M", 46_350, 16, vram_total_mb=48_789, backend="Vulkan"
)


def _demarrage_simule(monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(topo.urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(topo.time, "sleep", lambda s: None)


def _sonde(popen):
    dispo = iter([60_000, 24_800, 60_000, 24_800])
    return ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=_UMA,
        ram_avail=lambda: next(dispo),
        popen=popen,
        kill=lambda p: None,
    )


class _Proc:
    pid = 4242


def test_la_sonde_capture_le_journal_du_serveur_et_compte_les_checkpoints(monkeypatch):
    _demarrage_simule(monkeypatch)
    chemins: list[str] = []

    def popen(args, **kw):
        # Les deux flux vont au MÊME fichier : llama.cpp logge sur stdout ou stderr
        # selon les builds (cf. hardware.py).
        assert kw["stderr"] is topo.subprocess.STDOUT
        fh = kw["stdout"]
        chemins.append(fh.name)
        fh.write(JOURNAL.encode("utf-8"))
        fh.flush()
        return _Proc()

    res = _sonde(popen).run(8192, None)
    assert res.mem_mb == 35_200  # la mesure mémoire est inchangée
    assert res.checkpoints["effectifs"] == 3 and res.checkpoints["plafond"] == 32
    assert chemins and not os.path.exists(chemins[0])  # journal temporaire nettoyé


def test_la_sonde_dit_non_mesure_quand_le_journal_est_muet(monkeypatch):
    _demarrage_simule(monkeypatch)
    res = _sonde(lambda *a, **k: _Proc()).run(8192, None)
    assert res.checkpoints.get("effectifs") is None
    assert "non mesuré" in res.checkpoints["source"]


def test_un_lancement_rate_ne_laisse_pas_de_journal_temporaire(monkeypatch):
    chemins: list[str] = []

    def popen(args, **kw):
        chemins.append(kw["stdout"].name)
        raise RuntimeError("stop avant le vrai lancement")

    import pytest

    with pytest.raises(RuntimeError):
        _sonde(popen)._start(8192)
    assert chemins and not os.path.exists(chemins[0])


ECHEC_GGUF = """0.00.094.131 I srv    load_model: loading model 'x.gguf'
0.00.094.422 E gguf_init_from_file: failed to open GGUF file 'x.gguf' (No such file or directory)
0.00.095.394 E srv  llama_server: exiting due to model loading error
"""


def test_serveur_mort_au_chargement_echec_immediat_avec_la_cause(monkeypatch):
    """Vu en réel (2026-10-10) : un serveur mort en 0,1 s (GGUF introuvable) faisait
    attendre la sonde jusqu'au timeout de /health (600 s), puis lever « health
    timeout » sans la cause. Le process sorti est détecté tout de suite, et l'erreur
    porte la fin du journal serveur."""
    import pytest

    attentes: list[float] = []
    monkeypatch.setattr(topo.time, "sleep", lambda s: attentes.append(s))

    def _refus(*a, **k):
        raise ConnectionRefusedError("refusé")

    monkeypatch.setattr(topo.urllib.request, "urlopen", _refus)
    chemins: list[str] = []

    class _Mort:
        pid = 4242
        returncode = 1

        def poll(self):
            return 1

    def popen(args, **kw):
        chemins.append(kw["stdout"].name)
        kw["stdout"].write(ECHEC_GGUF.encode("utf-8"))
        kw["stdout"].flush()
        return _Mort()

    with pytest.raises(RuntimeError) as exc:
        _sonde(popen)._start(8192)
    msg = str(exc.value)
    assert "code 1" in msg and "failed to open GGUF" in msg and "exiting" in msg
    assert len(attentes) <= 1  # pas d'attente du timeout
    assert chemins and not os.path.exists(chemins[0])


def test_timeout_de_health_porte_aussi_la_fin_du_journal(monkeypatch):
    import pytest

    horloge = iter(range(0, 10_000, 100))
    monkeypatch.setattr(topo.time, "monotonic", lambda: next(horloge))
    monkeypatch.setattr(topo.time, "sleep", lambda s: None)

    def _refus(*a, **k):
        raise ConnectionRefusedError("refusé")

    monkeypatch.setattr(topo.urllib.request, "urlopen", _refus)

    class _Vivant:
        pid = 4242

        def poll(self):
            return None

    def popen(args, **kw):
        kw["stdout"].write(b"0.01 I srv load_model: loading model 'x.gguf'\n")
        kw["stdout"].flush()
        return _Vivant()

    probe = _sonde(popen)
    probe.health_timeout_s = 300
    with pytest.raises(RuntimeError) as exc:
        probe._start(8192)
    assert "health timeout" in str(exc.value) and "loading model" in str(exc.value)


def test_la_sonde_lance_le_serveur_en_verbosite_4_comme_l_executant():
    """Vu en réel (Bonsai 2, 2026-10-10) : en verbosité 3 (défaut) llama-server
    n'écrit AUCUNE ligne de checkpoint ; en -lv 4 (celle de l'exécutant, [server]
    log_verbosity) il écrit « created context checkpoint … ». Sans ça, le compte
    effectif serait toujours « non mesuré »."""
    import pytest

    captured: dict = {}

    def popen(args, **kw):
        captured["args"] = [str(a) for a in args]
        raise RuntimeError("stop avant le vrai lancement")

    with pytest.raises(RuntimeError):
        _sonde(popen)._start(8192)
    args = captured["args"]
    assert args[args.index("-lv") + 1] == "4"


def test_journal_verrouille_par_le_serveur_qui_meurt_est_quand_meme_supprime(
    monkeypatch,
):
    """Vu en réel : juste après taskkill, le serveur tient encore le fichier et la
    suppression échoue (le journal restait dans %TEMP%). Elle est réessayée."""
    _demarrage_simule(monkeypatch)
    vrai_remove = os.remove
    essais: list[str] = []

    def remove_capricieux(path):
        essais.append(path)
        if len(essais) == 1:
            raise PermissionError(
                "[WinError 32] fichier utilisé par un autre processus"
            )
        vrai_remove(path)

    monkeypatch.setattr(topo.os, "remove", remove_capricieux)
    chemins: list[str] = []

    def popen(args, **kw):
        chemins.append(kw["stdout"].name)
        kw["stdout"].write(JOURNAL.encode("utf-8"))
        kw["stdout"].flush()
        return _Proc()

    res = _sonde(popen).run(8192, None)
    assert res.checkpoints["effectifs"] == 3
    assert len(essais) >= 2 and not os.path.exists(chemins[0])


def _cp(n):
    return {
        "effectifs": n,
        "plafond": 32,
        "crees": n,
        "taille_mb": 149.6,
        "min_spacing": 512,
        "source": "journal serveur",
    }


def test_agregat_porte_le_maximum_effectif_ou_dit_non_mesure():
    base = {"tg_ts": 10.0, "pp_ts": 200.0, "mem_mb": 1}
    m = _agreger([{**base, "checkpoints": _cp(2)}, {**base, "checkpoints": _cp(3)}])
    assert m["checkpoints_effectifs"] == 3 and m["checkpoints_plafond"] == 32
    assert m["checkpoint_mb"] == 149.6 and "journal serveur" in m["checkpoints_detail"]
    # Journal muet : rien d'inventé, et dit.
    m2 = _agreger([{**base, "checkpoints": {"source": "non mesuré : journal vide"}}])
    assert (
        "checkpoints_effectifs" not in m2 and "non mesuré" in m2["checkpoints_detail"]
    )
    # Sonde qui ne rapporte rien (faux de test) : aucune clé ajoutée.
    assert "checkpoints_detail" not in _agreger([base])


def test_reglage_final_porte_les_checkpoints_effectifs():
    class _P:
        ngl, cpu_moe, n_cpu_moe, n_parallel, ubatch, batch = (
            999,
            False,
            None,
            2,
            512,
            2048,
        )

        def run(self, ctx, depth):
            return ProbeResult(
                ctx=ctx, mem_mb=1, tg_ts=10.0, pp_ts=200.0, checkpoints=_cp(3)
            )

    fin = validate_final(_P(), ctx=65_536, depth=16_384, n_layers=40, reps=1)
    assert fin["checkpoints_effectifs"] == 3
    assert fin["echantillons"][0]["checkpoints"]["effectifs"] == 3
    texte = checkpoints_text(fin)
    assert "checkpoints effectifs 3" in texte and "plafond 32 par slot" in texte
    assert "~449 Mio" in texte  # 3 x 149,6
    assert "149.6" in checkpoints_text(fin)
    assert "non mesuré" in checkpoints_text({"checkpoints_detail": "non mesuré : vide"})
    assert checkpoints_text({"tg_ts": 1.0}) == ""
