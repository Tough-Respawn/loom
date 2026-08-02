# Intégration complète de `/chat`: client simulé, SSE et persistance réelle.
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

    def build(scripts, handlers=None, monitor_hub=None):
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
            monitor_hub=monitor_hub,
        )
        web = app.test_client()
        # Un titre explicite empêche le thread de titrage de consommer le faux client.
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

    tl = web.get(f"/session/{sid}/timeline").get_json()["events"]
    seq = [e["event"] for e in tl]
    assert seq == ["user", "tool_call", "tool_result", "text"]
    assert tl[2]["data"]["name"] == "list_dir" and tl[2]["data"]["ok"] is True


def test_monitor_event_sse_timeline_and_structured_persistence(chat_env, tmp_env):
    class Hub:
        def __init__(self):
            self.sent = False

        def drain(self, sid):
            if self.sent:
                return []
            self.sent = True
            return [
                {
                    "id": "evt1",
                    "monitor_id": "mon1",
                    "description": "build",
                    "text": "tests verts",
                    "model_content": "donnée externe balisée",
                    "final": False,
                }
            ]

    web, fake, _, sid = chat_env([turn_text("Bien reçu.")], monitor_hub=Hub())
    r = web.post("/chat", data={"message": "surveille", "session_id": sid})
    assert r.status_code == 200
    events = _sse_events(r.data)
    monitor_event = next(e for e in events if e["type"] == "monitor_event")
    assert monitor_event["description"] == "build"
    assert monitor_event["text"] == "tests verts"

    saved = json.loads(
        (tmp_env / "sessions" / sid / "session.json").read_text(encoding="utf-8")
    )
    messages = saved["conversation"]["messages"]
    assistant = next(m for m in messages if m.get("tool_calls"))
    tool = next(m for m in messages if m["role"] == "tool")
    assert assistant["tool_calls"][0]["function"]["name"] == "monitor"
    assert tool["content"] == "donnée externe balisée"
    assert any(
        e["event"] == "monitor_event"
        for e in web.get(f"/session/{sid}/timeline").get_json()["events"]
    )
    assert any(m.get("tool_calls") for m in fake.calls[0]["messages"])


def test_chat_verrou_relache_apres_le_tour(chat_env):
    # Le générateur doit relâcher le verrou après consommation complète du flux.
    web, _, _, sid = chat_env([turn_text("un."), turn_text("deux.")])
    r1 = web.post("/chat", data={"message": "premier", "session_id": sid})
    assert r1.status_code == 200
    # Le verrou reste tenu tant que le flux n'est pas consommé.
    assert _sse_events(r1.data)[-1]["type"] == "done"
    r2 = web.post("/chat", data={"message": "second", "session_id": sid})
    assert r2.status_code == 200
    assert _sse_events(r2.data)[-1]["type"] == "done"


def test_resend_apres_stop_genere_sans_202(chat_env):
    # Après STOP, attendre le teardown plutôt que mettre le nouveau message en file orpheline.
    import threading
    import time

    from loom.web.routes.helpers import _lock_for

    web, _, _, sid = chat_env([turn_text("je reprends.")])
    S = web.application.S
    lock = _lock_for(S, sid)
    lock.acquire()  # génération en cours d'interruption : verrou encore tenu
    web.post("/cancel", data={"session_id": sid})  # STOP demandé sur CETTE session
    threading.Thread(
        target=lambda: (time.sleep(0.3), lock.release()), daemon=True
    ).start()
    r = web.post("/chat", data={"message": "reprends", "session_id": sid})
    assert r.status_code == 200, f"resend après STOP doit générer, reçu {r.status_code}"
    assert _sse_events(r.data)[-1]["type"] == "done"


def test_cancel_ferme_le_stream_distant_bloque(chat_env):
    # `/cancel` doit fermer un stream distant figé pour rendre la libération du verrou bornée.
    import threading

    import httpx

    from .fakes import _FakeStream

    class _BlockingStream:
        """Stream distant qui BLOQUE à l'itération jusqu'à close() ; close() débloque
        et fait lever httpx.ReadError (comme un stream SDK fermé sous le lecteur)."""

        def __init__(self):
            self.started = threading.Event()
            self._closed = threading.Event()

        def __iter__(self):
            self.started.set()
            self._closed.wait(timeout=5)  # bloque jusqu'à close()
            raise httpx.ReadError("stream fermé par /cancel")
            yield

        def close(self):
            self._closed.set()

    web, fake, _, sid = chat_env([turn_text("je repars après annulation.")])
    S = web.application.S
    S.interrupt_wait = 1.0  # borne l'attente du resend (sinon 15 s si le fix manque)

    bstream = _BlockingStream()
    holder = {"first": True}

    def create(**kwargs):
        fake.calls.append(kwargs)
        if holder["first"]:
            holder["first"] = False
            return bstream
        return _FakeStream(fake.scripts.pop(0))

    fake.chat.completions.create = create

    out: dict = {}

    def run_chat():
        r = web.post("/chat", data={"message": "bloque-toi", "session_id": sid})
        out["status"] = r.status_code

    t = threading.Thread(target=run_chat, daemon=True)
    t.start()
    assert bstream.started.wait(timeout=5), "le stream distant n'a jamais démarré"

    assert web.post("/cancel", data={"session_id": sid}).status_code == 204
    t.join(timeout=5)
    assert not t.is_alive(), "le /chat bloqué ne s'est pas terminé après /cancel"

    r2 = web.post("/chat", data={"message": "reprends", "session_id": sid})
    assert r2.status_code == 200, (
        f"verrou de session non relâché après /cancel (reçu {r2.status_code})"
    )
    assert _sse_events(r2.data)[-1]["type"] == "done"


def test_note_en_vol_reste_202_sans_stop(chat_env):
    # Sans STOP, un message concurrent doit rester une note en vol.
    from loom.web.routes.helpers import _lock_for

    web, _, _, sid = chat_env([turn_text("x.")])
    S = web.application.S
    _lock_for(S, sid).acquire()  # génération active, AUCUN /cancel
    r = web.post("/chat", data={"message": "btw note", "session_id": sid})
    assert r.status_code == 202


def test_chat_outils_dans_le_workspace_de_la_session_cible(tmp_env, monkeypatch):
    # Les outils doivent suivre la session cible même si le focus change pendant le flux.
    from loom.agent.client import LoomClient
    from loom.agent.session import SessionStore

    from .fakes import FakeOAI, FakeRegistry, turn_text

    captured = {}

    client = LoomClient("http://127.0.0.1:9/v1")
    client.add_remote_route(
        MODEL, {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "fake/x"}
    )
    client._routes[MODEL]["client"] = FakeOAI([turn_text("ok.")])
    registry = FakeRegistry({"list_dir": lambda a: "x"})
    store = SessionStore(
        tmp_env / "sessions",
        default_system_prompt="prompt de test",
        default_model=MODEL,
        known_models=[MODEL],
    )

    def tf(tools, ws, conv):
        captured["ws"] = ws
        return registry

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
        tool_factory=tf,
    )
    web = app.test_client()
    from loom.web.routes.helpers import _get_session

    sidA = web.post("/session/new", data={"title": "A"}).get_json()["id"]
    web.post("/session/workspace", data={"session_id": sidA, "workspace": "C:/wsA"})
    sessB = _get_session(
        app.S, web.post("/session/new", data={"title": "B"}).get_json()["id"]
    )
    sessB.workspace = "C:/wsB"
    monkeypatch.setattr("loom.web.routes.chat._session", lambda _S: sessB)

    r = web.post("/chat", data={"message": "salut", "session_id": sidA})
    assert r.status_code == 200
    assert _sse_events(r.data)[-1]["type"] == "done"
    assert captured["ws"] == "C:/wsA", (
        f"outils dans le mauvais workspace : {captured['ws']}"
    )


def test_chat_ne_ressuscite_pas_une_session_supprimee(chat_env):
    # Une session supprimée avant l'acquisition du verrou ne doit pas être recréée par save().
    import shutil

    web, _, _, sid = chat_env([turn_text("ok.")])
    S = web.application.S
    shutil.rmtree(S.session_store.session_dir(sid))  # supprimé sous les pieds de /chat
    r = web.post("/chat", data={"message": "reprends", "session_id": sid})
    assert r.status_code == 404
    assert not S.session_store.session_dir(sid).exists(), (
        "save() a ressuscité une session supprimée"
    )


def test_init_adopte_la_session_cible_pas_la_focus(tmp_env):
    # `/init` adopte le dossier dans la session cible, jamais dans le focus global.
    from types import SimpleNamespace

    from loom.agent.session import SessionStore
    from loom.web.routes.commands import _handle_init_command

    store = SessionStore(
        tmp_env / "sessions",
        default_system_prompt="p",
        default_model="m",
        known_models=["m"],
    )
    target = store.create()  # session CIBLE
    focus = store.create()  # session FOCUS, distincte
    S = SimpleNamespace(session_store=store, cur={"session": focus})
    projet = tmp_env / "projet"
    projet.mkdir()

    _handle_init_command(S, f"/init {projet}", target)

    want = str(projet.resolve())
    assert target.workspace == want, "la session cible n'a pas adopté le dossier"
    assert focus.workspace != want, "la session focus a été modifiée à tort"


def test_prime_slot_utilise_le_workspace_de_la_session_cible(app):
    # L'amorçage KV doit utiliser le workspace de la session ciblée.
    from types import SimpleNamespace

    from loom.web.routes.priming import _prime_slot

    S = app.S
    S.remote_model_ids = set()
    S.image_model_ids = set()
    S.video_model_ids = set()
    captured = {}

    def fake_warm(msgs, sp, **kw):
        captured["sp"] = sp
        return True

    S.client = SimpleNamespace(warm_context=fake_warm)
    S.tool_factory = lambda tools, ws, conv: None

    store = S.session_store
    target = store.create()
    target.workspace = "C:/wsA_prime"
    store.save(target)
    focus = store.create()
    focus.workspace = "C:/wsB_prime"
    store.save(focus)
    S.cur["session"] = focus  # focus DISTINCTE de la cible

    assert _prime_slot(S, target) is True
    assert "C:/wsA_prime" in captured["sp"], (
        "l'amorçage n'utilise pas le workspace de la cible"
    )
    assert "C:/wsB_prime" not in captured["sp"], (
        "l'amorçage a utilisé le workspace de la focus"
    )


def test_chat_id_explicite_inconnu_repond_404(web_sess):
    # Un identifiant explicite inconnu doit répondre 404, sans repli sur le focus.
    r = web_sess.post("/chat", data={"message": "x", "session_id": "deadbeefdead"})
    assert r.status_code == 404


def test_stop_avec_note_en_file_repart_de_la_note(tmp_env):
    # Une note suivie de STOP doit relancer depuis cette note, pas la laisser orpheline.
    from types import SimpleNamespace as NS

    from loom.agent.client import LoomClient
    from loom.agent.session import SessionStore
    from loom.web.routes.helpers import _cancel_for

    from .fakes import FakeRegistry, _FakeStream, chunk, usage_chunk

    holder = {}

    class MidCancelStream:
        def __iter__(self):
            yield chunk(content="je pars dans la ")
            holder["S"].notes.push(holder["sid"], "non, fais plutôt un bouton discret")
            _cancel_for(holder["S"], holder["sid"]).set()
            yield chunk(content="mauvaise direction", finish="stop")
            yield usage_chunk()

        def close(self):
            pass

    class ScriptedOAI:
        def __init__(self):
            self.n = 0
            self.chat = NS(completions=NS(create=self._create))

        def _create(self, **kw):
            self.n += 1
            if self.n == 1:
                return MidCancelStream()
            return _FakeStream(
                [
                    chunk(content="ok, je repars de ta note : bouton.", finish="stop"),
                    usage_chunk(),
                ]
            )

    client = LoomClient("http://127.0.0.1:9/v1")
    client.add_remote_route(
        MODEL, {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "fake/x"}
    )
    client._routes[MODEL]["client"] = ScriptedOAI()
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
        tool_factory=lambda t, w, c: FakeRegistry({"list_dir": lambda a: "x"}),
    )
    web = app.test_client()
    holder["S"] = app.S
    sid = web.post("/session/new", data={"title": "t"}).get_json()["id"]
    holder["sid"] = sid

    r = web.post("/chat", data={"message": "commence", "session_id": sid})
    assert r.status_code == 200
    texts = "".join(
        e.get("text", "") for e in _sse_events(r.data) if e["type"] == "text"
    )
    assert "je repars de ta note" in texts, (
        "le STOP n'a pas enchaîné sur la note en file"
    )
    assert app.S.notes.drain(sid) == [], "la note est restée orpheline dans la file"


def test_chat_erreur_api_flux_error_generique(chat_env):
    # Une erreur interne doit devenir un événement SSE générique sans fuite de détails.
    web, _, _, sid = chat_env([])
    r = web.post("/chat", data={"message": "salut", "session_id": sid})
    assert r.status_code == 200
    events = _sse_events(r.data)
    assert events[-1]["type"] == "error"
    assert events[-1]["message"] == "erreur interne"
