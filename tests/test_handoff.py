from __future__ import annotations

import json

import pytest

from loom.agent.session import SessionStore
from loom.agent.compaction import _inject_notes
from loom.web.app import create_app
from loom.web.routes.helpers import _lock_for

from .fakes import FakeOAI, FakeRegistry, turn_text


MODEL = "remote-x"


@pytest.fixture()
def chat_env(tmp_env):
    def build(scripts):
        from loom.agent.client import LoomClient

        client = LoomClient("http://127.0.0.1:9/v1")
        client.add_remote_route(
            MODEL,
            {
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "k",
                "model": "fake/x",
            },
        )
        fake = FakeOAI(scripts)
        client._routes[MODEL]["client"] = fake
        registry = FakeRegistry({"list_dir": lambda args: "a.txt"})
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
            tool_factory=lambda tools, workspace, conversation: registry,
        )
        web = app.test_client()
        target_sid = web.post(
            "/session/new", data={"title": "Session cible"}
        ).get_json()["id"]
        return web, fake, registry, target_sid

    return build


def _sse_events(body: bytes) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


def test_handoff_ajoute_a_la_cible_declenche_et_ne_touche_pas_la_source(
    chat_env, tmp_env
):
    web, fake, _, target_sid = chat_env([turn_text("Je vérifie ce travail.")])
    source = web.post("/session/new", data={"title": "Analyse profonde"}).get_json()
    source_sid = source["id"]

    # Un chemin existant cité dans la réponse transférée est du contexte : il ne doit
    # pas faire adopter ce dossier à la session cible.
    transferred = f"Implémente ce plan dans {tmp_env}." + (" détail" * 800)
    target_before = web.application.S.session_store.load(target_sid)
    source_before = web.application.S.session_store.load(source_sid)

    response = web.post(
        "/handoff",
        data={
            "message": transferred,
            "session_id": target_sid,
            "source_session_id": source_sid,
            "provenance": "[]",
            "handoff_id": "handoff:test-direct",
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.data)
    assert events[-1]["type"] == "done"
    text_event = next(e for e in events if e["type"] == "text")
    assert text_event["text"] == "Je vérifie ce travail."
    assert text_event["provenance"] == [
        {
            "session_id": source_sid,
            "title": "Analyse profonde",
            "model": "remote-x",
        }
    ]

    # Le modèle reçoit la provenance ET le contenu, tandis que la source reste intacte.
    model_messages = fake.calls[0]["messages"]
    prompt = next(m["content"] for m in model_messages if m["role"] == "user")
    assert "Message transferred from another Loom session" in prompt
    assert "remote-x" in prompt and "Analyse profonde" in prompt
    assert transferred in prompt
    assert web.application.S.session_store.load(source_sid).conversation.messages == (
        source_before.conversation.messages
    )

    target_after = web.application.S.session_store.load(target_sid)
    assert target_after.workspace == target_before.workspace
    assert target_after.conversation.messages[-1]["content"] == "Je vérifie ce travail."
    assert transferred in target_after.conversation.messages[-2]["content"]

    timeline = web.get(f"/session/{target_sid}/timeline").get_json()["events"]
    assert [e["event"] for e in timeline] == ["user", "text"]
    assert timeline[0]["data"] == {
        "content": transferred,
        "provenance": text_event["provenance"],
        "handoff_id": "handoff:test-direct",
    }
    assert timeline[1]["data"]["provenance"] == text_event["provenance"]


def test_handoff_chaine_la_provenance_sans_plafond_de_profondeur(chat_env):
    web, fake, _, target_sid = chat_env([turn_text("Correction prête.")])
    source_sid = web.post(
        "/session/new", data={"title": "Vérification Kimi"}
    ).get_json()["id"]
    previous = [
        {
            "session_id": f"old-{i}",
            "title": f"étape {i}",
            "model": f"modèle-{i}",
        }
        for i in range(25)
    ]

    response = web.post(
        "/handoff",
        data={
            "message": "Voici les défauts à corriger.",
            "session_id": target_sid,
            "source_session_id": source_sid,
            "provenance": json.dumps(previous),
            "handoff_id": "handoff:test-chain",
        },
    )
    events = _sse_events(response.data)
    chain = next(e for e in events if e["type"] == "text")["provenance"]
    assert len(chain) == 26
    assert chain[:25] == previous
    assert chain[-1]["title"] == "Vérification Kimi"
    prompt = next(
        m["content"] for m in fake.calls[0]["messages"] if m["role"] == "user"
    )
    assert "modèle-0" in prompt and "modèle-24" in prompt
    assert "Vérification Kimi" in prompt


def test_handoff_refuse_source_cible_inconnue_et_envoi_vers_soi(web):
    source_sid = web.post(
        "/session/new", data={"title": "Source existante"}
    ).get_json()["id"]
    base = {
        "message": "réponse",
        "provenance": "[]",
        "source_session_id": source_sid,
    }
    assert (
        web.post("/handoff", data={**base, "session_id": "deadbeefdead"}).status_code
        == 404
    )
    assert (
        web.post(
            "/handoff",
            data={
                **base,
                "source_session_id": "deadbeefdead",
                "session_id": source_sid,
            },
        ).status_code
        == 404
    )
    assert (
        web.post("/handoff", data={**base, "session_id": source_sid}).status_code == 400
    )


def test_handoff_cible_occupee_part_dans_la_file_structuree(web):
    source_sid = web.post(
        "/session/new", data={"title": "Deepseek analyse"}
    ).get_json()["id"]
    target_sid = web.post(
        "/session/new", data={"title": "Ornith développe"}
    ).get_json()["id"]
    target_before = web.application.S.session_store.load(target_sid)
    lock = _lock_for(web.application.S, target_sid)
    lock.acquire()
    try:
        response = web.post(
            "/handoff",
            data={
                "message": "Construis cette solution.",
                "session_id": target_sid,
                "source_session_id": source_sid,
                "provenance": "[]",
                "handoff_id": "handoff:test-queue",
                "queue_only": "1",
            },
        )
    finally:
        lock.release()

    assert response.status_code == 202
    queued = web.application.S.notes.drain(target_sid)
    assert len(queued) == 1
    assert queued[0]["display"] == "Construis cette solution."
    assert queued[0]["handoff_id"] == "handoff:test-queue"
    assert queued[0]["provenance"][-1]["session_id"] == source_sid
    assert "Message transferred" in queued[0]["text"]
    # Avant son prochain point d'arrêt, la conversation cible n'est pas mutée sous
    # la génération en cours ; la note structurée sera injectée par cette boucle.
    assert web.application.S.session_store.load(target_sid).conversation.messages == (
        target_before.conversation.messages
    )


def test_handoff_queue_only_repond_409_si_la_cible_est_deja_libre(web):
    source_sid = web.post("/session/new", data={"title": "Source"}).get_json()["id"]
    target_sid = web.post("/session/new", data={"title": "Cible"}).get_json()["id"]
    response = web.post(
        "/handoff",
        data={
            "message": "travail",
            "session_id": target_sid,
            "source_session_id": source_sid,
            "provenance": "[]",
            "queue_only": "1",
        },
    )
    assert response.status_code == 409
    assert web.application.S.notes.drain(target_sid) == []


def test_note_handoff_injecte_le_texte_et_reemet_les_metadonnees():
    conversation = []
    queued = {
        "text": "[Message transferred] contenu",
        "display": "contenu",
        "provenance": [{"model": "ornith"}],
        "handoff_id": "handoff:structured",
    }
    events = list(_inject_notes(lambda: [queued], conversation))

    assert conversation == [
        {
            "role": "user",
            "content": (
                "[User note received mid-turn — take it into account and continue "
                "the task] [Message transferred] contenu"
            ),
        }
    ]
    assert events[0][0] == "note"
    assert events[0][1]["display"] == "contenu"
    assert events[0][1]["provenance"] == [{"model": "ornith"}]
    assert events[0][1]["handoff_id"] == "handoff:structured"
