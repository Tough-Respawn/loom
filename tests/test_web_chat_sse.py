# Caractérisation du chemin /chat COMPLET (la couture create_app <-> stream_chat_tools) :
# vrai LoomClient branché sur un FakeOAI via une route distante (pas de bloc serveur
# local), vrai flux SSE, vraie persistance session/timeline. C'est LE test qui doit
# survivre aux refactors P2-1 et P2-3 ensemble.
from __future__ import annotations

import json

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

from .fakes import FakeOAI, FakeRegistry, turn_text, turn_tools

MODEL = "remote-x"


def _sse_events(body: bytes) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture()
def chat_env(tmp_env):
    """Factory : construit l'app complète autour d'un script FakeOAI donné."""

    def build(scripts, handlers=None):
        from loom.agent.client import LoomClient

        client = LoomClient("http://127.0.0.1:9/v1")
        client.add_remote_route(
            MODEL,
            {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "fake/x"},
        )
        fake = FakeOAI(scripts)
        client._routes[MODEL]["client"] = fake
        registry = FakeRegistry(handlers or {"list_dir": lambda a: "a.txt\nb.txt"})
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
            remote_model_ids=[MODEL],
            keepwarm_enabled=False,
            workspace_dir=str(tmp_env / "workspace"),
            user_skills_dir=str(tmp_env / "skills_user"),
            plugins_dir=str(tmp_env / "plugins"),
            remote_store_path=str(tmp_env / "remote_models.json"),
            tool_factory=lambda tools, ws, conv: registry,
        )
        web = app.test_client()
        # Titre EXPLICITE : sinon un thread de titrage part en course et consomme
        # le script FakeOAI (titrage immédiat des modèles distants, app.py ~1726).
        r = web.post("/session/new", data={"title": "session testée"})
        assert r.status_code == 200
        return web, fake, registry, r.get_json()["id"]

    return build


def test_chat_texte_simple_sse_et_persistance(chat_env, tmp_env):
    web, fake, _, sid = chat_env([turn_text("Bonjour humain.")])
    r = web.post("/chat", data={"message": "salut", "session_id": sid})
    assert r.status_code == 200
    assert r.content_type.startswith("text/event-stream")

    events = _sse_events(r.data)
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    texte = "".join(e.get("text", "") for e in events if e["type"] == "text")
    assert texte == "Bonjour humain."
    assert len(fake.calls) == 1

    # persistance : session.json porte l'échange complet
    saved = json.loads(
        (tmp_env / "sessions" / sid / "session.json").read_text(encoding="utf-8")
    )
    contents = json.dumps(saved, ensure_ascii=False)
    assert "salut" in contents and "Bonjour humain." in contents


def test_chat_tool_call_sse_et_timeline(chat_env, tmp_env):
    web, fake, registry, sid = chat_env(
        [
            turn_tools([("call_1", "list_dir", '{"path": "."}')]),
            turn_text("il y a deux fichiers."),
        ]
    )
    r = web.post("/chat", data={"message": "liste le dossier", "session_id": sid})
    assert r.status_code == 200

    events = _sse_events(r.data)
    types = [e["type"] for e in events]
    assert "tool_call" in types and "tool_result" in types
    assert types[-1] == "done"
    assert registry.calls == [("list_dir", {"path": "."})]

    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["name"] == "list_dir" and tr["ok"] is True

    # la timeline persistée ({event, data}) reprend le tour complet (sous-ensemble _TL)
    tl = web.get(f"/session/{sid}/timeline").get_json()["events"]
    seq = [e["event"] for e in tl]
    assert seq == ["user", "tool_call", "tool_result", "text"]
    assert tl[2]["data"]["name"] == "list_dir" and tl[2]["data"]["ok"] is True


def test_chat_verrou_relache_apres_le_tour(chat_env):
    # Après un tour terminé, la session accepte un nouveau /chat (pas de 202) :
    # le finally du générateur relâche bien le verrou de session.
    web, _, _, sid = chat_env([turn_text("un."), turn_text("deux.")])
    r1 = web.post("/chat", data={"message": "premier", "session_id": sid})
    assert r1.status_code == 200
    # la réponse est STREAMÉE : le verrou n'est relâché qu'une fois le flux consommé
    assert _sse_events(r1.data)[-1]["type"] == "done"
    r2 = web.post("/chat", data={"message": "second", "session_id": sid})
    assert r2.status_code == 200
    assert _sse_events(r2.data)[-1]["type"] == "done"


def test_resend_apres_stop_genere_sans_202(chat_env):
    # Bug STOP+reprise : après un STOP (cancel_event posé), le verrou de session reste
    # tenu le temps du teardown de la génération interrompue. Un message RENVOYÉ pendant
    # cette fenêtre NE DOIT PAS partir en file (202) — il attend la libération du verrou
    # puis GÉNÈRE. Sans le fix, il tombe en 202 et n'est jamais généré (message perdu).
    import threading
    import time

    from loom.web.routes.helpers import _lock_for

    web, _, _, sid = chat_env([turn_text("je reprends.")])
    S = web.application.S
    lock = _lock_for(S, sid)
    lock.acquire()  # génération en cours d'interruption : verrou encore tenu
    web.post("/cancel", data={"session_id": sid})  # STOP demandé sur CETTE session
    # Le teardown de la génération interrompue relâche le verrou incessamment :
    threading.Thread(
        target=lambda: (time.sleep(0.3), lock.release()), daemon=True
    ).start()
    r = web.post("/chat", data={"message": "reprends", "session_id": sid})
    assert r.status_code == 200, f"resend après STOP doit générer, reçu {r.status_code}"
    assert _sse_events(r.data)[-1]["type"] == "done"


def test_note_en_vol_reste_202_sans_stop(chat_env):
    # Garde-fou : SANS STOP en cours, un message envoyé pendant une génération active
    # reste une note en vol (202). Le fix ne doit pas casser cette sémantique.
    from loom.web.routes.helpers import _lock_for

    web, _, _, sid = chat_env([turn_text("x.")])
    S = web.application.S
    _lock_for(S, sid).acquire()  # génération active, AUCUN /cancel
    r = web.post("/chat", data={"message": "btw note", "session_id": sid})
    assert r.status_code == 202


def test_chat_erreur_api_flux_error_generique(chat_env):
    # Une exception NON-openai pendant la génération (ici : script épuisé) remonte
    # jusqu'au try de generate() qui la capture : dernier event SSE = "error" avec
    # message GÉNÉRIQUE (pas de fuite d'interne, cf. P3-4), et le flux se ferme sans
    # exception côté client.
    web, _, _, sid = chat_env([])
    r = web.post("/chat", data={"message": "salut", "session_id": sid})
    assert r.status_code == 200
    events = _sse_events(r.data)
    assert events[-1]["type"] == "error"
    assert events[-1]["message"] == "erreur interne"
