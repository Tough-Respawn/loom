# Incident du 2026-10-14 (session 5bc7e8a4e5d3) : conversation sur Bonsai, /remove-model
# d'un AUTRE modèle -> llama.swap.yaml réécrit -> rechargement global de llama-swap ->
# Bonsai arrêté, cache perdu (10 258 tokens réutilisés avant, 0 après, 210 s avant le
# premier token). Règle : modifier le catalogue n'interrompt pas un modèle inchangé.
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from loom.agent.client import LoomClient
from loom.agent.session import SessionStore
from loom.runtime import serve, swap_sync
from loom.web.app import create_app

FAKE_MODEL = "fake-model"


class _FakeClient:
    """Le strict nécessaire de LoomClient vu par la régénération et le préalable."""

    def __init__(self, running=(), reachable=True):
        self.running = set(running)
        self.reachable = reachable
        self.cold_calls = 0
        self.running_calls = 0
        self.slot_counts = {}

    def running_local(self, timeout=5.0):
        self.running_calls += 1
        if not self.reachable:
            return False, ""
        body = {
            "running": [{"model": m, "state": "ready"} for m in sorted(self.running)]
        }
        return True, json.dumps(body)

    def mark_all_cold(self):
        self.cold_calls += 1

    def local_server_root(self):
        return "http://swap.test"

    def unload_local(self, model=None, timeout=30.0):
        return True


def _sse_texts(body: bytes) -> str:
    out = []
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if evt["type"] == "text":
                out.append(evt.get("text", ""))
    return "\n".join(out)


def _fake_runtime_config(monkeypatch, text_dir):
    """La config suit les dossiers de modèles présents sur le disque, comme load_config."""

    def cfg(*a, **k):
        ids = sorted(p.name for p in text_dir.iterdir() if p.is_dir())
        return SimpleNamespace(
            models=[SimpleNamespace(id=i, cache_isolation=False) for i in ids],
            server_bin="llama-server",
            n_parallel=1,
        )

    monkeypatch.setattr(serve, "load_config", cfg)
    monkeypatch.setattr("loom.config.load_config", cfg)
    monkeypatch.setattr(serve, "detect_hardware", lambda *a, **k: object())
    monkeypatch.setattr(
        serve,
        "_swap_config",
        lambda c, profile: {
            "models": {m.id: {"cmd": f"llama-server -m {m.id}.gguf"} for m in c.models}
        },
    )


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from loom.web.routes import misc

    monkeypatch.setattr(misc, "llama_update_loop", lambda S: None)
    root = tmp_path / "root"
    text_dir = root / "local" / "text"
    for mid in ("A", "B"):
        (text_dir / mid).mkdir(parents=True)
        (text_dir / mid / f"{mid}.gguf").write_bytes(b"gguf")
    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    defaults = tmp_path / "defaults.toml"
    local = tmp_path / "local.toml"
    defaults.write_text("", encoding="utf-8")
    local.write_text("", encoding="utf-8")
    y = tmp_path / "llama-swap.yaml"
    monkeypatch.setattr(serve, "SWAP_YAML", y)
    _fake_runtime_config(monkeypatch, text_dir)
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL],
    )
    specs = [{"id": m, "dir": str(text_dir / m), "size_mb": 1} for m in ("A", "B")]
    app = create_app(
        client=None,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_MODEL, "A", "B"],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        config_defaults_path=str(defaults),
        config_local_path=str(local),
        local_models=specs,
        models_dir=str(text_dir),
        models_roots=[str(root)],
    )
    web = app.test_client()
    assert web.post("/session/new", data={}).status_code == 200
    # Config appliquée par llama-swap au départ : A et B.
    assert serve.sync_swap_yaml("d", "l", y, running=set()) == "written"
    return SimpleNamespace(web=web, S=app.S, yaml=y, text_dir=text_dir)


def _remove(env, mid):
    r = env.web.post("/chat", data={"message": "/remove-model"})
    listing = _sse_texts(r.data)
    line = next(ln for ln in listing.splitlines() if f". {mid} " in ln)
    idx = line.strip().split(".")[0]
    env.web.post("/chat", data={"message": idx})
    r = env.web.post("/chat", data={"message": "oui"})
    assert "supprimé" in _sse_texts(r.data)


def test_supprimer_un_autre_modele_ne_recharge_pas_llama_swap(env):
    # A tourne (la conversation en cours) ; on supprime B.
    applied = env.yaml.read_text(encoding="utf-8")
    fake = _FakeClient(running={"A"})
    env.S.client = fake
    _remove(env, "B")
    assert not (env.text_dir / "B").exists()
    # Le yaml surveillé n'a pas bougé : llama-swap ne recharge rien, A garde son cache.
    assert env.yaml.read_text(encoding="utf-8") == applied
    pending = swap_sync.pending_path(env.yaml).read_text(encoding="utf-8")
    assert "A" in pending and "B.gguf" not in pending
    assert fake.cold_calls == 0  # aucun slot déclaré froid : rien n'a été rechargé


def test_supprimer_quand_rien_ne_tourne_ecrit_tout_de_suite(env):
    env.S.client = _FakeClient(running=())
    _remove(env, "B")
    assert "B.gguf" not in env.yaml.read_text(encoding="utf-8")
    assert not swap_sync.pending_path(env.yaml).exists()


def test_regen_qui_recharge_le_modele_actif_le_declare_froid(env, monkeypatch):
    # Changement voulu de l'entrée du modèle actif (paramètre serveur) : le yaml est
    # écrit, llama-swap recharge tout -> plus aucun slot ne peut être présumé chaud.
    from loom.web.routes.config import _regen_swap_yaml

    fake = _FakeClient(running={"A"})
    env.S.client = fake
    monkeypatch.setattr(
        serve,
        "_swap_config",
        lambda c, profile: {
            "models": {
                m.id: {"cmd": f"llama-server -m {m.id}.gguf -c 4096"} for m in c.models
            }
        },
    )
    assert _regen_swap_yaml(env.S) == "written"
    assert "-c 4096" in env.yaml.read_text(encoding="utf-8")
    assert fake.cold_calls == 1


# ---- préalable : appliquer l'attente à la prochaine transition -----------------------


def _with_pending(tmp_path, monkeypatch):
    y = tmp_path / "llama-swap.yaml"
    monkeypatch.setattr(serve, "SWAP_YAML", y)
    y.write_text("ancien", encoding="utf-8")
    swap_sync.pending_path(y).write_text("nouveau", encoding="utf-8")
    calls = []

    def fake_apply(path, root, **k):
        calls.append(root)
        os.replace(swap_sync.pending_path(path), path)
        return "applied"

    monkeypatch.setattr(swap_sync, "apply_pending", fake_apply)
    return y, calls


def test_preflight_applique_l_attente_avant_un_modele_qui_ne_tourne_pas(
    tmp_path, monkeypatch
):
    from loom.web.routes.config import _swap_preflight

    y, calls = _with_pending(tmp_path, monkeypatch)
    fake = _FakeClient(running={"A"})
    _swap_preflight(SimpleNamespace(client=fake), "C")
    assert calls == ["http://swap.test"]
    assert y.read_text(encoding="utf-8") == "nouveau"
    assert fake.cold_calls == 1  # le rechargement a arrêté A


def test_preflight_garde_l_attente_si_le_modele_tourne_deja(tmp_path, monkeypatch):
    from loom.web.routes.config import _swap_preflight

    y, calls = _with_pending(tmp_path, monkeypatch)
    fake = _FakeClient(running={"A"})
    _swap_preflight(SimpleNamespace(client=fake), "A")
    assert calls == []
    assert y.read_text(encoding="utf-8") == "ancien"
    assert swap_sync.pending_path(y).exists()
    assert fake.cold_calls == 0


def test_preflight_sans_attente_ne_coute_rien(tmp_path, monkeypatch):
    from loom.web.routes.config import _swap_preflight

    monkeypatch.setattr(serve, "SWAP_YAML", tmp_path / "llama-swap.yaml")
    fake = _FakeClient(running={"A"})
    _swap_preflight(SimpleNamespace(client=fake), "C")
    assert fake.running_calls == 0  # aucun appel réseau sur le chemin courant


def test_preflight_llama_swap_injoignable_pose_le_fichier(tmp_path, monkeypatch):
    from loom.web.routes.config import _swap_preflight

    y, calls = _with_pending(tmp_path, monkeypatch)
    fake = _FakeClient(reachable=False)
    _swap_preflight(SimpleNamespace(client=fake), "C")
    assert calls == [
        None
    ]  # rien à attendre : llama-swap relira le fichier au lancement
    assert fake.cold_calls == 0


# ---- branchement sur le client ---------------------------------------------------------


def test_le_client_appelle_le_prealable_avant_toute_requete_locale():
    c = LoomClient(
        base_url="http://127.0.0.1:9/v1", routes={"dist": {"base_url": "http://x/v1"}}
    )
    seen = []
    c.local_preflight = seen.append
    c._resolve("A")
    c._resolve("dist")  # distant : pas de llama-swap
    c._resolve(None)
    assert seen == ["A"]


def test_le_client_appelle_le_prealable_avant_une_action_de_slot(monkeypatch):
    import urllib.request

    c = LoomClient(base_url="http://127.0.0.1:9/v1")
    seen = []
    c.local_preflight = seen.append
    monkeypatch.setattr(c, "running_local", lambda timeout=5.0: (False, ""))

    def refuse(*a, **k):
        raise OSError("connexion refusée")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert c.restore_slot("A", "turnend.kv", force=True) is False
    assert seen == ["A"]


def test_un_prealable_en_echec_ne_bloque_jamais_la_requete():
    c = LoomClient(base_url="http://127.0.0.1:9/v1")

    def boom(model):
        raise RuntimeError("préalable cassé")

    c.local_preflight = boom
    oai, api_model, native = c._resolve("A")
    assert api_model == "A"


def test_l_app_branche_le_prealable_sur_le_client(tmp_path, monkeypatch):
    from loom.web.routes import misc

    monkeypatch.setattr(misc, "llama_update_loop", lambda S: None)
    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="p",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL],
    )
    client = LoomClient(base_url="http://127.0.0.1:9/v1")
    app = create_app(
        client=client,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_MODEL],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
    )
    assert callable(getattr(app.S.client, "local_preflight", None))


@pytest.mark.parametrize(
    "statut, dechargements",
    [("written", ["B"]), ("deferred", []), ("unchanged", []), (False, [])],
)
def test_reglage_d_un_modele_ne_decharge_que_si_le_yaml_est_ecrit(
    env, monkeypatch, statut, dechargements
):
    # « inchangé » déchargeait le modèle actif pour rien ; « différé » vise un modèle
    # qui ne tourne pas. Seule une écriture réelle justifie le rechargement.
    import threading

    from loom.web.routes import models as routes_models

    (env.text_dir / "B" / "model.toml").write_text(
        'filename = "B.gguf"\n', encoding="utf-8"
    )
    done = threading.Event()
    calls = []

    class _Client(_FakeClient):
        def unload_local(self, model=None, timeout=30.0):
            calls.append(model)
            done.set()
            return True

    env.S.client = _Client(running={"A"})
    monkeypatch.setattr(routes_models, "_regen_swap_yaml", lambda S: statut)
    r = env.web.post(
        "/models/local/set", json={"id": "B", "key": "context", "value": 8192}
    )
    assert r.status_code == 200
    done.wait(timeout=2 if dechargements else 0.3)
    assert calls == dechargements
    assert r.get_json()["applies"] == ("model-reload" if statut else "restart")
