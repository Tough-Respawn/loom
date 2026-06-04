# tests/test_web.py
import pytest

from loom.conversation import Conversation
from loom.web.app import _build_user_content, _sse, create_app


class _FakeImage:
    def __init__(self, filename, mimetype, blob):
        self.filename = filename
        self.mimetype = mimetype
        self._blob = blob

    def read(self):
        return self._blob


def test_build_user_content_text_only():
    assert _build_user_content("salut", None) == "salut"


def test_build_user_content_multimodal_when_image():
    img = _FakeImage("shot.png", "image/png", b"\x89PNG")
    content = _build_user_content("vois ?", img)
    assert content[0] == {"type": "text", "text": "vois ?"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_user_content_rejects_non_image():
    with pytest.raises(ValueError):
        _build_user_content("x", _FakeImage("note.txt", "text/plain", b"hi"))


def test_build_user_content_rejects_too_large():
    big = _FakeImage("big.png", "image/png", b"\x00" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ValueError):
        _build_user_content("x", big)


def test_sse_preserves_accents_and_framing():
    out = _sse("text", text="réflexion")
    assert out.startswith("data: ") and out.endswith("\n\n")
    assert "réflexion" in out  # pas d'échappement \uXXXX


class FakeClient:
    def __init__(self, events):
        self._events = events
        self.last_system_prompt = None
        self.last_model = None
        self.last_thinking = None

    def stream_chat(
        self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
    ):
        # mémorise le dernier message pour les assertions image
        self.last_messages = messages
        self.last_system_prompt = system_prompt
        self.last_model = model
        self.last_thinking = thinking
        yield from self._events


def _make(tmp_path, events=(("content", "Hel"), ("content", "lo")), budget=100000):
    conv = Conversation(system_prompt="sys", model="gemma")
    history = tmp_path / "conv.json"
    skills_dir = tmp_path / "skills"
    (skills_dir / "dagster").mkdir(parents=True)
    (skills_dir / "dagster" / "SKILL.md").write_text(
        "---\nname: dagster\ndescription: archi\n---\nARCHI_DAGSTER_XYZ",
        encoding="utf-8",
    )
    fake = FakeClient(list(events))
    app = create_app(
        conv,
        fake,
        history,
        skills_dir,
        max_tokens=2048,
        context_budget=budget,
        keep_recent=3,
        models=["gemma", "qwen"],
        interrupt_wait=0.3,  # tests rapides : pas d'attente réelle de 15 s
    )
    app.config["_fake_client"] = fake
    return app, conv, history


def test_index_lists_models(tmp_path):
    app, _, _ = _make(tmp_path)
    body = app.test_client().get("/").get_data(as_text=True)
    assert "gemma" in body and "qwen" in body


def test_run_active_and_replay_without_run(tmp_path):
    # Contrat de reattach : sans run en cours, /run/active dit has=False et /run/replay
    # se termine immédiatement (un "done"), sans bloquer.
    app, _, _ = _make(tmp_path)
    c = app.test_client()
    st = c.get("/run/active").get_json()
    assert st["has"] is False and st["running"] is False
    assert "done" in c.get("/run/replay").get_data(as_text=True)


def test_post_model_updates_conversation(tmp_path):
    app, conv, _ = _make(tmp_path)
    resp = app.test_client().post("/model", data={"model": "qwen"})
    assert resp.status_code == 200
    assert conv.model == "qwen"


def test_chat_sends_conversation_model(tmp_path):
    app, conv, _ = _make(tmp_path)
    conv.set_model("qwen")
    app.test_client().post("/chat", data={"message": "salut"})
    assert app.config["_fake_client"].last_model == "qwen"


def test_chat_uses_tool_loop_and_emits_tool_events(tmp_path):
    from loom.conversation import Conversation
    from loom.tools import ToolRegistry, ToolSpec
    from loom.web.app import create_app

    conv = Conversation(system_prompt="sys", model="gemma")
    skills = tmp_path / "skills"
    skills.mkdir()

    class ToolClient:
        def stream_chat_tools(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            registry=None,
            thinking=True,
            permission=None,
            confirm=None,
        ):
            yield ("tool_call", {"id": "c1", "name": "read_file"})
            yield (
                "tool_result",
                {"name": "read_file", "ok": True, "preview": "CONTENU"},
            )
            yield ("content", "Le fichier contient X.")

    reg = ToolRegistry(
        [ToolSpec("read_file", "lit", {"type": "object"}, lambda a: "x")]
    )
    app = create_app(
        conv,
        ToolClient(),
        tmp_path / "c.json",
        skills,
        context_budget=100000,
        tool_registry=reg,
    )
    body = (
        app.test_client()
        .post("/chat", data={"message": "lis x"})
        .get_data(as_text=True)
    )
    assert '"type": "tool_call"' in body
    assert '"type": "tool_result"' in body
    assert "Le fichier contient X." in body
    assert conv.messages[-1] == {
        "role": "assistant",
        "content": "Le fichier contient X.",
    }


def test_tool_decision_sets_pending_event(tmp_path):
    import threading

    app, _, _ = _make(tmp_path)
    ev = threading.Event()
    app.config["_pending"]["abc"] = {"event": ev, "approved": False}
    resp = app.test_client().post("/tool_decision", data={"id": "abc", "approve": "1"})
    assert resp.status_code == 204
    assert app.config["_pending"]["abc"]["approved"] is True
    assert ev.is_set()


def test_tool_decision_unknown_id_is_noop(tmp_path):
    app, _, _ = _make(tmp_path)
    resp = app.test_client().post("/tool_decision", data={"id": "nope", "approve": "1"})
    assert resp.status_code == 204  # idempotent, ne crashe pas


def test_post_tools_updates_conversation(tmp_path):
    app, conv, _ = _make(tmp_path)
    resp = app.test_client().post("/tools", data={"tool": ["read_file", "run_shell"]})
    assert resp.status_code == 200
    assert conv.active_tools == ["read_file", "run_shell"]


def test_index_lists_available_tools(tmp_path):
    from loom.conversation import Conversation
    from loom.tools import AVAILABLE_TOOLS
    from loom.web.app import create_app

    conv = Conversation(system_prompt="sys", model="gemma")
    skills = tmp_path / "skills"
    skills.mkdir()
    app = create_app(
        conv,
        FakeClient([]),
        tmp_path / "c.json",
        skills,
        available_tools=AVAILABLE_TOOLS,
    )
    body = app.test_client().get("/").get_data(as_text=True)
    assert "read_file" in body and "run_shell" in body


def test_chat_builds_registry_from_active_tools(tmp_path):
    from loom.conversation import Conversation
    from loom.web.app import create_app

    conv = Conversation(system_prompt="sys", model="gemma")
    conv.set_tools(["read_file"])
    skills = tmp_path / "skills"
    skills.mkdir()
    seen = {}

    def factory(active):
        seen["active"] = list(active)
        from loom.tools import ToolRegistry, ToolSpec

        return ToolRegistry(
            [ToolSpec("read_file", "lit", {"type": "object"}, lambda a: "x")]
        )

    class ToolClient:
        def stream_chat_tools(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            registry=None,
            thinking=True,
            permission=None,
            confirm=None,
        ):
            yield ("content", "ok")

    app = create_app(
        conv,
        ToolClient(),
        tmp_path / "c.json",
        skills,
        context_budget=100000,
        tool_factory=factory,
    )
    app.test_client().post("/chat", data={"message": "lis"}).get_data()
    assert seen["active"] == ["read_file"]


def test_chat_passes_thinking_flag(tmp_path):
    app, conv, _ = _make(tmp_path)
    conv.set_thinking(False)
    app.test_client().post("/chat", data={"message": "salut"})
    assert app.config["_fake_client"].last_thinking is False


def test_post_thinking_toggles_conversation(tmp_path):
    app, conv, _ = _make(tmp_path)
    resp = app.test_client().post("/thinking", data={"thinking": "0"})
    assert resp.status_code == 200
    assert conv.thinking is False
    app.test_client().post("/thinking", data={"thinking": "1"})
    assert conv.thinking is True


def test_post_skills_updates_conversation(tmp_path):
    app, conv, _ = _make(tmp_path)
    resp = app.test_client().post("/skills", data={"skill": ["dagster"]})
    assert resp.status_code == 200
    assert conv.active_skills == ["dagster"]


def test_chat_injects_active_skill_into_system_prompt(tmp_path):
    app, conv, _ = _make(tmp_path)
    conv.set_skills(["dagster"])
    client = app.test_client()
    client.post("/chat", data={"message": "salut"})
    # FakeClient mémorise le system_prompt reçu
    assert "ARCHI_DAGSTER_XYZ" in app.config["_fake_client"].last_system_prompt


def test_index_lists_skills(tmp_path):
    app, _, _ = _make(tmp_path)
    body = app.test_client().get("/").get_data(as_text=True)
    assert "dagster" in body


def test_index_renders_history(tmp_path):
    app, conv, _ = _make(tmp_path)
    conv.add("user", "coucou-test")
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"coucou-test" in resp.data


def test_reset_clears_history(tmp_path):
    app, conv, history = _make(tmp_path)
    conv.add("user", "x")
    client = app.test_client()
    resp = client.post("/reset")
    assert resp.status_code == 200
    assert conv.messages == []
    assert history.exists()  # persisté


def test_chat_streams_reasoning_then_content(tmp_path):
    app, conv, _ = _make(
        tmp_path,
        events=[
            ("reasoning", "je pense"),
            ("content", "Bon"),
            ("content", "jour"),
        ],
    )
    resp = app.test_client().post("/chat", data={"message": "salut"})
    body = resp.get_data(as_text=True)
    assert '"type": "reasoning"' in body
    assert "je pense" in body
    assert '"type": "text"' in body
    assert "Bon" in body and "jour" in body
    assert '"type": "done"' in body
    # les tokens "Bon" + "jour" sont assemblés en "Bonjour" côté serveur
    assert conv.messages[1] == {"role": "assistant", "content": "Bonjour"}


def test_chat_with_image_builds_multimodal_content(tmp_path):
    import io

    app, conv, _ = _make(tmp_path)
    data = {
        "message": "que vois-tu ?",
        "image": (io.BytesIO(b"\x89PNG\r\n\x1a\nFAKE"), "shot.png"),
    }
    resp = app.test_client().post(
        "/chat", data=data, content_type="multipart/form-data"
    )
    assert resp.status_code == 200
    user_content = conv.messages[0]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_chat_rejects_empty_message(tmp_path):
    app, _, _ = _make(tmp_path)
    resp = app.test_client().post("/chat", data={"message": "   "})
    assert resp.status_code == 400


def test_chat_busy_returns_429(tmp_path):
    app, _, _ = _make(tmp_path)
    app.config["_chat_lock"].acquire()
    try:
        resp = app.test_client().post("/chat", data={"message": "x"})
        assert resp.status_code == 429
    finally:
        app.config["_chat_lock"].release()


def test_chat_empty_answer_placeholder(tmp_path):
    app, conv, _ = _make(
        tmp_path, events=[("reasoning", "je pense")]
    )  # aucun 'content'
    resp = app.test_client().post("/chat", data={"message": "x"})
    body = resp.get_data(as_text=True)
    assert "seulement réfléchi" in body
    assert conv.messages[1]["role"] == "assistant"
    assert conv.messages[1]["content"] != ""


def _drive_partial(app, n_chunks):
    """Lance /chat, consomme n_chunks events, puis ferme le flux (= interruption)."""
    with app.test_request_context("/chat", method="POST", data={"message": "x"}):
        resp = app.view_functions["chat"]()
        gen = resp.response
        for _ in range(n_chunks):
            next(gen)
        gen.close()  # GeneratorExit dans generate() = nouvelle soumission a aborté


def test_chat_persists_partial_on_interrupt(tmp_path):
    app, conv, _ = _make(
        tmp_path,
        events=[("content", "Hel"), ("content", "lo"), ("content", " monde")],
    )
    _drive_partial(app, 2)  # reçoit "Hel" + "lo"
    assert conv.messages[-1] == {"role": "assistant", "content": "Hello"}


def test_chat_releases_lock_on_interrupt(tmp_path):
    app, conv, _ = _make(
        tmp_path, events=[("content", "Hel"), ("content", "lo"), ("content", "!")]
    )
    _drive_partial(app, 1)
    # le verrou doit être libre malgré l'interruption
    assert app.config["_chat_lock"].acquire(blocking=False) is True
    app.config["_chat_lock"].release()


def test_chat_no_429_after_interrupt(tmp_path):
    app, conv, _ = _make(
        tmp_path, events=[("content", "Hel"), ("content", "lo"), ("content", "!")]
    )
    _drive_partial(app, 1)
    # une nouvelle requête passe (pas de 429), le verrou ayant été relâché
    resp = app.test_client().post("/chat", data={"message": "suite"})
    assert resp.status_code == 200


def test_chat_interrupt_before_any_token_persists_nothing(tmp_path):
    app, conv, _ = _make(tmp_path, events=[("reasoning", "je pense"), ("content", "x")])
    _drive_partial(app, 1)  # ne consomme que le reasoning, aucun content
    # rien de pertinent reçu : on ne pollue pas l'historique avec une bulle vide
    assert all(m["role"] != "assistant" for m in conv.messages)


def test_chat_stops_when_cancel_event_set_mid_stream(tmp_path):
    """Le flag cancel (posé par une nouvelle soumission) stoppe net la boucle."""
    from loom.conversation import Conversation
    from loom.web.app import create_app

    conv = Conversation(system_prompt="sys", model="gemma")
    skills = tmp_path / "skills"
    skills.mkdir()
    holder = {}

    class CancellingClient:
        def stream_chat(
            self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
        ):
            yield ("content", "Hel")
            yield ("content", "lo")
            holder["ev"].set()  # une nouvelle soumission demande l'annulation
            yield ("content", " monde")  # ne doit PAS être conservé

    app = create_app(
        conv, CancellingClient(), tmp_path / "c.json", skills, context_budget=100000
    )
    holder["ev"] = app.config["_cancel_event"]
    app.test_client().post("/chat", data={"message": "x"}).get_data()
    assert conv.messages[-1] == {"role": "assistant", "content": "Hello"}


def test_chat_clears_cancel_event_before_generating(tmp_path):
    """Un cancel résiduel ne doit pas tuer la requête suivante."""
    app, conv, _ = _make(tmp_path, events=[("content", "Salut")])
    app.config["_cancel_event"].set()  # résidu
    app.test_client().post("/chat", data={"message": "x"}).get_data()
    assert conv.messages[-1] == {"role": "assistant", "content": "Salut"}


# ---- multi-agent : route /run ----


def _make_multi(tmp_path, scripts=None):
    from loom.agents import Agent

    conv = Conversation(system_prompt="sys", model="gemma")
    skills = tmp_path / "skills"
    skills.mkdir()
    scripts = scripts or {
        "m1": [("reasoning", "r1"), ("content", "PLAN1")],
        "m2": [("content", "CODE2")],
    }

    class MultiClient:
        def stream_chat(
            self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
        ):
            yield from scripts.get(model, [("content", f"out-{model}")])

    agents = [
        Agent(id="a", role="plan", model="m1", system_prompt="planifie"),
        Agent(id="b", role="code", model="m2", system_prompt="code"),
    ]
    app = create_app(
        conv,
        MultiClient(),
        tmp_path / "c.json",
        skills,
        context_budget=100000,
        interrupt_wait=0.3,
        agents=agents,
        pipeline=["a", "b"],
    )
    return app


def test_run_returns_sse_with_agent_events(tmp_path):
    app = _make_multi(tmp_path)
    resp = app.test_client().post("/run", data={"task": "fais X", "mode": "pipeline"})
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert '"type": "agent_start"' in body
    assert '"type": "agent_done"' in body
    assert '"agent": "a"' in body
    assert '"agent": "b"' in body
    assert "PLAN1" in body
    assert "CODE2" in body


def test_run_uses_configured_max_revisions(tmp_path):
    from loom.agents import Agent

    conv = Conversation(system_prompt="sys", model="gemma")
    skills = tmp_path / "skills"
    skills.mkdir()

    class RevClient:
        def stream_chat(
            self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
        ):
            if model == "rev":
                yield ("content", "VERDICT: BLOQUANT il manque un fichier")
            else:
                yield ("content", f"out-{model}")

    agents = [
        Agent(id="p", role="plan", model="m1", system_prompt="p"),
        Agent(id="c", role="code", model="m2", system_prompt="c"),
        Agent(id="r", role="reviewer", model="rev", system_prompt="r"),
    ]
    app = create_app(
        conv,
        RevClient(),
        tmp_path / "c.json",
        skills,
        context_budget=100000,
        interrupt_wait=0.3,
        agents=agents,
        pipeline=["p", "c", "r"],
        max_revisions=3,
    )
    body = (
        app.test_client()
        .post("/run", data={"task": "x", "mode": "pipeline"})
        .get_data(as_text=True)
    )
    # le relecteur bloque toujours => 3 révisions (la valeur configurée), pas 1
    assert body.count('"type": "revision"') == 3


def test_run_targets_chosen_workspace(tmp_path):
    from loom.agents import Agent
    from loom.tools import ToolRegistry, ToolSpec

    captured = {}
    conv = Conversation(system_prompt="sys", model="gemma")
    skills = tmp_path / "skills"
    skills.mkdir()
    target = tmp_path / "tictactoo"

    class C:
        def stream_chat_tools(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            registry=None,
            thinking=True,
            permission=None,
            confirm=None,
        ):
            yield ("content", "ok")

        def stream_chat(self, *a, **k):
            yield ("content", "plan")

    def factory(active, workspace=None):
        captured["ws"] = workspace
        return ToolRegistry(
            [ToolSpec("write_file", "w", {"type": "object"}, lambda a: "ok")]
        )

    agents = [
        Agent(id="b", role="code", model="m2", system_prompt="c", tools=["write_file"])
    ]
    app = create_app(
        conv,
        C(),
        tmp_path / "c.json",
        skills,
        context_budget=100000,
        interrupt_wait=0.3,
        agents=agents,
        pipeline=["b"],
        tool_factory=factory,
        workspace_dir=str(tmp_path),
    )
    body = (
        app.test_client()
        .post("/run", data={"task": "x", "workspace": str(target), "mode": "pipeline"})
        .get_data(as_text=True)
    )
    assert '"type": "run_info"' in body and "tictactoo" in body
    assert captured["ws"] == str(target.resolve())  # outils liés au dossier cible
    assert target.exists()  # le dossier cible est créé


def test_run_default_workspace_emits_run_info(tmp_path):
    app = _make_multi(tmp_path)  # workspace_dir défaut = "."
    body = (
        app.test_client()
        .post("/run", data={"task": "x", "mode": "pipeline"})
        .get_data(as_text=True)
    )
    assert '"type": "run_info"' in body


def test_run_rejects_empty_task(tmp_path):
    app = _make_multi(tmp_path)
    resp = app.test_client().post("/run", data={"task": "   "})
    assert resp.status_code == 400


def test_run_busy_returns_429(tmp_path):
    app = _make_multi(tmp_path)
    app.config["_chat_lock"].acquire()
    try:
        resp = app.test_client().post("/run", data={"task": "x", "mode": "pipeline"})
        assert resp.status_code == 429
    finally:
        app.config["_chat_lock"].release()


def test_run_releases_lock_after_completion(tmp_path):
    app = _make_multi(tmp_path)
    app.test_client().post("/run", data={"task": "x", "mode": "pipeline"}).get_data()
    assert app.config["_chat_lock"].acquire(blocking=False) is True
    app.config["_chat_lock"].release()


class _ClassifyClient:
    """Client minimal exposant complete() pour /classify (et /pick-folder)."""

    def __init__(self, reply="BUILD"):
        self.reply = reply

    def complete(
        self,
        messages,
        system_prompt,
        max_tokens=8,
        model=None,
        thinking=False,
        temperature=None,
    ):
        return self.reply


def _make_classify(tmp_path, reply="BUILD"):
    conv = Conversation(system_prompt="sys", model="m")
    history = tmp_path / "conv.json"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)
    return create_app(
        conv,
        _ClassifyClient(reply),
        history,
        skills_dir,
        models=["m"],
        interrupt_wait=0.3,
    )


def test_classify_endpoint_returns_mode(tmp_path):
    app = _make_classify(tmp_path, reply="BUILD")
    resp = app.test_client().post("/classify", data={"message": "crée un jeu"})
    assert resp.status_code == 200
    assert resp.get_json()["mode"] == "build"


def test_classify_empty_message_is_chat(tmp_path):
    app = _make_classify(tmp_path, reply="BUILD")
    resp = app.test_client().post("/classify", data={"message": "  "})
    assert resp.get_json()["mode"] == "chat"


def test_pick_folder_returns_selected_path(tmp_path, monkeypatch):
    import loom.web.app as appmod

    class _Proc:
        returncode = 0
        stdout = "C:/Users/Amine/projet\n"
        stderr = ""

    monkeypatch.setattr(appmod.subprocess, "run", lambda *a, **k: _Proc())
    app = _make_classify(tmp_path)
    resp = app.test_client().post("/pick-folder")
    assert resp.get_json()["path"] == "C:/Users/Amine/projet"


def test_pick_folder_cancel_returns_empty(tmp_path, monkeypatch):
    import loom.web.app as appmod

    class _Proc:
        returncode = 0
        stdout = "\n"
        stderr = ""

    monkeypatch.setattr(appmod.subprocess, "run", lambda *a, **k: _Proc())
    app = _make_classify(tmp_path)
    resp = app.test_client().post("/pick-folder")
    assert resp.get_json()["path"] == ""


# ---- sessions first-class (session_store fourni) ----


def _make_session_app(tmp_path, client=None, **kw):
    from loom.session import SessionStore

    skills = tmp_path / "skills"
    skills.mkdir()
    store = SessionStore(tmp_path / "sessions", default_system_prompt="sys")
    conv = Conversation(system_prompt="sys", model="gemma")  # ignoré en mode session
    app = create_app(
        conv,
        client or FakeClient([("content", "ok")]),
        tmp_path / "c.json",
        skills,
        context_budget=100000,
        interrupt_wait=0.3,
        models=["gemma"],
        session_store=store,
        **kw,
    )
    return app, store


def test_session_chat_persists_to_active_session(tmp_path):
    app, store = _make_session_app(tmp_path, client=FakeClient([("content", "Salut")]))
    app.test_client().post("/chat", data={"message": "coucou"}).get_data()
    active = store.active()
    assert active is not None
    roles = [(m["role"], m["content"]) for m in active.conversation.messages]
    assert ("user", "coucou") in roles
    assert ("assistant", "Salut") in roles


def test_sessions_list_new_and_activate(tmp_path):
    app, store = _make_session_app(tmp_path)
    c = app.test_client()
    first = c.get("/sessions").get_json()
    assert first["active"]  # une session active existe d'office
    created = c.post(
        "/session/new", data={"workspace": "C:/proj", "title": "Calc"}
    ).get_json()
    listing = c.get("/sessions").get_json()
    assert listing["active"] == created["id"]  # la nouvelle est active
    assert any(
        s["id"] == created["id"] and s["title"] == "Calc" for s in listing["sessions"]
    )
    # bascule vers la première
    c.post("/session/activate", data={"id": first["active"]})
    assert c.get("/sessions").get_json()["active"] == first["active"]


def test_session_delete_removes_and_keeps_an_active(tmp_path):
    app, store = _make_session_app(tmp_path)
    c = app.test_client()
    created = c.post("/session/new", data={"title": "jetable"}).get_json()
    c.post("/session/delete", data={"id": created["id"]})
    listing = c.get("/sessions").get_json()
    assert all(s["id"] != created["id"] for s in listing["sessions"])
    assert listing["active"]  # il reste toujours une session active


def test_run_records_into_active_session(tmp_path):
    from loom.agents import Agent

    class MultiClient:
        def stream_chat(
            self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
        ):
            yield ("content", f"out-{model}")

    agents = [Agent(id="a", role="plan", model="m1", system_prompt="p")]
    app, store = _make_session_app(
        tmp_path, client=MultiClient(), agents=agents, pipeline=["a"]
    )
    app.test_client().post(
        "/run", data={"task": "fais X", "mode": "pipeline"}
    ).get_data()
    active = store.active()
    assert len(active.runs) == 1
    assert active.runs[0].task == "fais X"
    # une trace du run est aussi dans la conversation (le modèle reprend le fil)
    assert any("fais X" in str(m["content"]) for m in active.conversation.messages)
