"""Revue étape 2 (2026-09-13) : après un déchargement (/machine/unload marque les
slots froids), le PREMIER /chat doit tenter la reprise à chaud AVANT d'appeler le
modèle — sinon ce parcours re-préfille tout, la reprise n'existant que dans
l'amorçage (boot, changement de modèle/session, fin de tour)."""

from __future__ import annotations

import json
import threading

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

from .fakes import FakeOAI, FakeRegistry, turn_text

MODEL = "fake-local"


def _sse_events(body: bytes) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture()
def local_chat(tmp_env, monkeypatch):
    """App complète sur un modèle LOCAL simulé : FakeOAI derrière le client local,
    serveur déclaré vivant, save/restore réseau neutralisés sauf la reprise espionnée."""
    from loom.agent.client import LoomClient

    client = LoomClient("http://127.0.0.1:9/v1")
    fake = FakeOAI([turn_text("réponse.")])
    client._client = fake
    calls: list[tuple] = []
    monkeypatch.setattr(
        client,
        "running_local",
        lambda timeout=0.0: (True, json.dumps({"running": [{"model": MODEL}]})),
    )
    monkeypatch.setattr(client, "save_slot", lambda *a, **k: False)
    # L'amorçage (session/new, fin de tour) tourne dans un thread « loom-prime » et
    # consommerait le faux script : neutralisé. On distingue SA reprise de celle
    # du /chat par le thread appelant.
    monkeypatch.setattr(client, "warm_context", lambda *a, **k: True)
    monkeypatch.setattr(
        client,
        "try_hot_resume",
        lambda model, sid: (
            calls.append(
                (
                    "hot_resume",
                    model,
                    sid,
                    len(fake.calls),
                    threading.current_thread().name,
                )
            )
            or False
        ),
    )
    store = SessionStore(
        tmp_env / "sessions",
        default_system_prompt="prompt de test",
        default_model=MODEL,
        known_models=[MODEL],
    )
    app = create_app(
        client=client,
        skills_dir=str(tmp_env / "skills"),
        session_store=store,
        models=[MODEL],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_env / "workspace"),
        user_skills_dir=str(tmp_env / "skills_user"),
        plugins_dir=str(tmp_env / "plugins"),
        remote_store_path=str(tmp_env / "remote_models.json"),
        tool_factory=lambda tools, ws, conv: FakeRegistry(),
    )
    web = app.test_client()
    r = web.post("/session/new", data={"title": "session testée"})
    assert r.status_code == 200
    return web, fake, calls, r.get_json()["id"]


def test_chat_tente_la_reprise_a_chaud_avant_le_modele(local_chat):
    web, fake, calls, sid = local_chat
    r = web.post("/chat", data={"message": "bonjour", "session_id": sid})
    assert r.status_code == 200
    events = _sse_events(r.data)
    assert events[-1]["type"] == "done", events[-3:]
    # La tentative du /chat lui-même (thread de la requête, pas le thread d'amorçage),
    # faite AVANT le premier appel modèle (aucun call FakeOAI enregistré à ce moment).
    mine = [c for c in calls if not c[4].startswith("loom-prime")]
    assert mine and mine[0][:4] == ("hot_resume", MODEL, sid, 0), calls
    assert len(fake.calls) == 1
