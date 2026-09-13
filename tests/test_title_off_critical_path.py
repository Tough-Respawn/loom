"""Étape 3 de l'audit timings (2026-09-13) : le titre hors du chemin critique.

Avant : le titre local était inféré AVANT `done` (21 s de slot, annulé au timeout,
puis 400 « invalid temperature »), et le titre distant faisait attendre `done`
jusqu'à 8 s. Contrat : titre PROVISOIRE (début du message) posé et publié dès le
début du tour ; `done` ne dépend jamais du modèle de titrage ; l'inférence du
« vrai » titre tourne après coup, hors verrou, sur le slot annexe — jamais sur
un modèle local à un seul slot (elle évincerait le cache de la conversation).
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

from .fakes import FakeOAI, FakeRegistry, turn_text

MODEL = "fake-local"
REMOTE = "remote-x"


def _sse_events(body: bytes) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture()
def title_env(tmp_env, monkeypatch):
    """Factory : app complète sur un modèle local (ou distant) simulé, titre
    inféré par un faux `infer_title` LENT qui note son thread appelant."""

    def build(*, remote=False, slots=2, title="Titre modèle", delay=0.3):
        from loom.agent.client import LoomClient

        client = LoomClient("http://127.0.0.1:9/v1")
        fake = FakeOAI([turn_text("réponse.")])
        model = REMOTE if remote else MODEL
        if remote:
            client.add_remote_route(
                REMOTE,
                {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "f/x"},
            )
            client._routes[REMOTE]["client"] = fake
        else:
            client._client = fake
        client.slot_counts = {MODEL: slots}
        calls: list[tuple] = []

        def infer_title(m, message):
            calls.append((threading.current_thread().name, time.monotonic()))
            time.sleep(delay)
            return title

        monkeypatch.setattr(client, "infer_title", infer_title)
        monkeypatch.setattr(
            client,
            "running_local",
            lambda timeout=0.0: (True, json.dumps({"running": [{"model": MODEL}]})),
        )
        monkeypatch.setattr(client, "save_slot", lambda *a, **k: False)
        monkeypatch.setattr(client, "warm_context", lambda *a, **k: True)
        monkeypatch.setattr(client, "try_hot_resume", lambda *a, **k: False)
        store = SessionStore(
            tmp_env / "sessions",
            default_system_prompt="prompt de test",
            default_model=model,
            known_models=[model],
        )
        app = create_app(
            client=client,
            skills_dir=str(tmp_env / "skills"),
            session_store=store,
            models=[model],
            remote_model_ids=[REMOTE] if remote else [],
            keepwarm_enabled=False,
            workspace_dir=str(tmp_env / "workspace"),
            user_skills_dir=str(tmp_env / "skills_user"),
            plugins_dir=str(tmp_env / "plugins"),
            remote_store_path=str(tmp_env / "remote_models.json"),
            tool_factory=lambda tools, ws, conv: FakeRegistry(),
        )
        web = app.test_client()
        r = web.post("/session/new", data={})  # titre par défaut « Nouvelle session »
        assert r.status_code == 200
        return web, store, calls, r.get_json()["id"]

    return build


MESSAGE = "Explique-moi comment llama-server choisit son slot quand il y en a deux"


def _wait_title(store, sid, expected, timeout=3.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cur = store.load(sid).title
        if cur == expected:
            return cur
        time.sleep(0.05)
    return store.load(sid).title


def test_titre_provisoire_publie_avant_la_reponse(title_env):
    web, store, calls, sid = title_env()
    t0 = time.monotonic()
    r = web.post("/chat", data={"message": MESSAGE, "session_id": sid})
    elapsed = time.monotonic() - t0
    events = _sse_events(r.data)
    types = [e["type"] for e in events]
    assert types[-1] == "done", events[-3:]
    # Le titre provisoire = début du message, publié AVANT le premier texte.
    i_title = types.index("session_title")
    assert events[i_title]["title"] == MESSAGE[:48].strip()
    assert i_title < types.index("text")
    # Et `done` n'a pas attendu le faux modèle de titrage (0,3 s de délai simulé).
    assert elapsed < 0.25 or all(name != "MainThread" for name, _ in calls)


def test_le_vrai_titre_arrive_apres_coup_hors_flux(title_env):
    web, store, calls, sid = title_env(slots=2)
    r = web.post("/chat", data={"message": MESSAGE, "session_id": sid})
    assert _sse_events(r.data)[-1]["type"] == "done"
    assert _wait_title(store, sid, "Titre modèle") == "Titre modèle"
    # Inféré dans un thread dédié, jamais dans le thread de la requête.
    assert calls and all(name.startswith("loom-title") for name, _ in calls), calls


def test_pas_de_titre_modele_sur_un_local_a_un_seul_slot(title_env):
    # Un seul slot : l'inférence évincerait le cache de la conversation -> on garde
    # le titre provisoire, aucun appel modèle.
    web, store, calls, sid = title_env(slots=1)
    r = web.post("/chat", data={"message": MESSAGE, "session_id": sid})
    assert _sse_events(r.data)[-1]["type"] == "done"
    time.sleep(0.4)
    assert calls == []
    assert store.load(sid).title == MESSAGE[:48].strip()


def test_distant_done_n_attend_pas_le_titre(title_env):
    web, store, calls, sid = title_env(remote=True, delay=1.0)
    t0 = time.monotonic()
    r = web.post("/chat", data={"message": MESSAGE, "session_id": sid})
    elapsed = time.monotonic() - t0
    events = _sse_events(r.data)
    assert events[-1]["type"] == "done"
    assert elapsed < 0.8, elapsed  # l'ancien code attendait le titre jusqu'à 8 s
    assert _wait_title(store, sid, "Titre modèle") == "Titre modèle"


def test_un_titre_deja_pose_n_est_jamais_retouche(title_env):
    web, store, calls, _ = title_env()
    r = web.post("/session/new", data={"title": "Mon titre"})
    sid = r.get_json()["id"]
    r = web.post("/chat", data={"message": MESSAGE, "session_id": sid})
    events = _sse_events(r.data)
    assert events[-1]["type"] == "done"
    assert "session_title" not in [e["type"] for e in events]
    time.sleep(0.4)
    assert calls == [] and store.load(sid).title == "Mon titre"
