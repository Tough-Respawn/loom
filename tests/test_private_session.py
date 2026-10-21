# Session privée (2026-10-21) : rien ne sort de la machine. Outils réseau retirés, shell
# et navigateur limités à la machine (proxy mort), prompt refigé quand on coche la case.
import http.server
import threading

import pytest

from loom.agent.conversation import Conversation
from loom.tools import PRIVATE_TOOL_MSG, ToolError, build_registry
from loom.tools.browser import make_check_page
from loom.tools.shell import _child_env, make_run_shell

ENABLED = ["web_search", "fetch_url", "watch_video", "run_shell", "read_file"]


def test_outils_reseau_retires_et_expliques(tmp_path):
    reg = build_registry(str(tmp_path), 10_000, ENABLED, private=True)
    for name in ("web_search", "fetch_url", "watch_video"):
        assert name not in reg
        assert reg.run(name, {}) == f"erreur: {PRIVATE_TOOL_MSG}"
    assert "run_shell" in reg and "read_file" in reg
    # Hors session privée, rien ne change.
    assert "web_search" in build_registry(str(tmp_path), 10_000, ENABLED)


def test_proxy_mort_seulement_en_session_privee():
    assert _child_env(offline=True)["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert "127.0.0.1" in _child_env(offline=True)["NO_PROXY"]
    assert _child_env().get("HTTPS_PROXY") != "http://127.0.0.1:9"


@pytest.fixture
def local_server():
    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def _fetch(url: str) -> str:
    return (
        f'python -c "import urllib.request as u; '
        f"print(u.urlopen('{url}', timeout=5).status)\""
    )


def test_shell_prive_atteint_la_machine_mais_pas_internet(tmp_path, local_server):
    shell = make_run_shell(str(tmp_path), offline=True)
    assert shell.run({"command": _fetch(local_server)}).strip() == "200"
    out = shell.run({"command": _fetch("https://example.com/")})
    assert out.startswith("erreur")


def test_check_page_prive_refuse_une_page_distante(tmp_path):
    page = make_check_page(str(tmp_path), offline=True)
    with pytest.raises(ToolError, match="session privée"):
        page.run({"url": "https://example.com"})
    with pytest.raises(ToolError, match="session privée"):
        page.run({"url": "example.com"})


def test_cocher_la_case_refige_le_prompt(monkeypatch):
    from types import SimpleNamespace as NS

    from loom.web.routes import system_prompt as sp

    renders = []
    monkeypatch.setattr(
        sp,
        "_render_system_prompt",
        lambda S, conv, workspace=None, scratch=None: (
            renders.append(conv.local_only) or f"privé={conv.local_only}",
            False,
        ),
    )
    monkeypatch.setattr(sp, "log_event", lambda *a, **k: None)
    S = NS(remote_model_ids=set(), remote_weak_ids=set())
    conv = Conversation(system_prompt="base", model="bonsai")
    assert sp._build_system_prompt(S, conv, workspace="C:/a")[0] == "privé=False"
    conv.set_local_only(True)
    assert sp._build_system_prompt(S, conv, workspace="C:/a")[0] == "privé=True"
    assert renders == [False, True]


# --- Couche Loom : modèles distants, appels automatiques, mémoire, historique ---------

LOCAL, REMOTE = "loc-model", "rem-model"


def test_client_refuse_toute_route_distante_en_session_privee():
    from loom.agent.egress import PRIVATE, PrivateEgressError

    from .fakes import make_client, turn_text

    client, fake = make_client([turn_text("ok")], remote=True)
    tok = PRIVATE.set(True)
    try:
        with pytest.raises(PrivateEgressError):
            list(
                client.stream_chat(
                    [{"role": "user", "content": "x"}], "", 5, model="remote-x"
                )
            )
    finally:
        PRIVATE.reset(tok)
    assert fake.calls == []  # rien n'est parti
    list(
        client.stream_chat([{"role": "user", "content": "x"}], "", 5, model="remote-x")
    )
    assert len(fake.calls) == 1  # hors session privée, la route marche


@pytest.fixture()
def priv_env(tmp_env, monkeypatch):
    """App complète : un modèle LOCAL simulé, un DISTANT simulé configuré PAR DÉFAUT."""
    from types import SimpleNamespace as NS

    from loom.agent.client import LoomClient
    from loom.agent.session import SessionStore
    from loom.web.app import create_app
    from loom.web.routes import chat as chat_mod

    from .fakes import FakeOAI, FakeRegistry, turn_text

    client = LoomClient("http://127.0.0.1:9/v1")
    local_fake = FakeOAI([turn_text("réponse locale") for _ in range(8)])
    client._client = local_fake
    # Serveur local « déjà démarré » : sinon le tour l'attend jusqu'à 600 s.
    client.running_local = lambda timeout=5.0: (True, "")
    client.add_remote_route(
        REMOTE, {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "r/x"}
    )
    remote_fake = FakeOAI([])  # tout appel serait enregistré (puis lèverait)
    client._routes[REMOTE]["client"] = remote_fake
    store = SessionStore(
        tmp_env / "sessions",
        default_system_prompt="prompt de test",
        default_model=REMOTE,
        known_models=[LOCAL, REMOTE],
    )
    app = create_app(
        client=client,
        skills_dir=str(tmp_env / "skills"),
        session_store=store,
        models=[LOCAL, REMOTE],
        remote_model_ids=[REMOTE],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_env / "workspace"),
        user_skills_dir=str(tmp_env / "skills_user"),
        plugins_dir=str(tmp_env / "plugins"),
        remote_store_path=str(tmp_env / "remote_models.json"),
        tool_factory=lambda tools, ws, conv: FakeRegistry({}),
    )
    # Tâches différées : on capture ce qu'elles reçoivent et le contexte privé.
    seen = []

    def fake_maintenance(S, sess, msgs, actions, answer, model, do_reflect, **kw):
        from loom.agent.egress import PRIVATE

        seen.append(NS(private=PRIVATE.get(), do_reflect=do_reflect, model=model))

    monkeypatch.setattr(chat_mod, "_post_turn_maintenance", fake_maintenance)
    S = app.S
    S.settings["reflect_enabled"] = True
    S.settings["reflect_min_actions"] = 0
    S.reflect_stores = object()
    return NS(
        web=app.test_client(), S=S, local=local_fake, remote=remote_fake, seen=seen
    )


def _new_session(web, model):
    r = web.post("/session/new", data={"model": model, "title": "t"})
    assert r.status_code == 200
    return r.get_json()["id"]


def _wait_threads():
    for t in threading.enumerate():
        if t.name in ("loom-post-turn", "loom-title"):
            t.join(timeout=10)


def test_session_privee_ne_touche_jamais_le_distant_par_defaut(priv_env):
    web = priv_env.web
    sid = _new_session(web, LOCAL)
    assert web.post("/local_only", data={"local_only": "1"}).status_code == 200
    r = web.post("/chat", data={"message": "salut", "session_id": sid})
    assert r.status_code == 200 and b"r\xc3\xa9ponse locale" in r.data
    _wait_threads()
    assert priv_env.remote.calls == []  # le distant par défaut n'a jamais été appelé
    assert priv_env.local.calls  # le tour est bien passé par le local
    # Tâche différée : contexte privé conservé, réflexion (mémoire partagée) coupée.
    assert priv_env.seen[-1].private is True and priv_env.seen[-1].do_reflect is False


def test_session_normale_garde_la_reflexion(priv_env):
    web = priv_env.web
    sid = _new_session(web, LOCAL)
    r = web.post("/chat", data={"message": "salut", "session_id": sid})
    assert r.status_code == 200 and r.data  # lire le flux déroule le tour
    _wait_threads()
    assert priv_env.seen[-1].private is False and priv_env.seen[-1].do_reflect is True


def test_modele_distant_refuse_et_historique_prive_protege(priv_env):
    web, S = priv_env.web, priv_env.S
    sid = _new_session(web, REMOTE)
    # Activer « privé » avec un modèle distant : refusé.
    assert web.post("/local_only", data={"local_only": "1"}).status_code == 409
    web.post("/model", data={"model": LOCAL})
    assert web.post("/local_only", data={"local_only": "1"}).status_code == 200
    # Choisir un distant dans une session privée : refusé, modèle inchangé, toast.
    r = web.post("/model", data={"model": REMOTE})
    assert "loomToast" in r.headers.get("HX-Trigger", "")
    from loom.web.routes.helpers import _session

    assert _session(S).conversation.model == LOCAL  # le modèle n'a pas changé
    r = web.post("/chat", data={"message": "secret", "session_id": sid})
    assert r.status_code == 200 and b"r\xc3\xa9ponse locale" in r.data
    _wait_threads()
    # Décocher « privé » une fois l'historique écrit : refusé, la session reste privée.
    assert web.post("/local_only", data={"local_only": "0"}).status_code == 409
    assert _session(S).conversation.local_only is True
    assert priv_env.remote.calls == []


def test_tour_refuse_si_le_modele_prive_est_distant(priv_env):
    web, S = priv_env.web, priv_env.S
    sid = _new_session(web, LOCAL)
    web.post("/local_only", data={"local_only": "1"})
    # État incohérent forcé (session ancienne, appel direct) : le tour est refusé.
    from loom.web.routes.helpers import _session

    _session(S).conversation.model = REMOTE
    r = web.post("/chat", data={"message": "secret", "session_id": sid})
    assert r.status_code == 409
    assert priv_env.remote.calls == []


class _Mem:
    """Mémoire partagée simulée : ce qui y est écrit est relu par toutes les sessions."""

    def __init__(self):
        self.items = []

    def remember(self, text, **kw):
        self.items.append(text)

    def recall(self, query, k=5):
        from types import SimpleNamespace as NS

        return [NS(text=t, source="") for t in self.items if query in t][:k]


def test_un_souvenir_prive_ne_ressort_pas_dans_une_session_distante(tmp_path):
    from types import SimpleNamespace as NS

    from loom.tools import PRIVATE_MEMORY_MSG

    mem = _Mem()
    summaries = []
    memory = NS(
        provider=mem,
        paths={},
        summarize=lambda q, hits: summaries.append(q) or "synthèse",
        threshold=1,
    )
    priv = build_registry(
        str(tmp_path), 10_000, ["remember", "recall"], memory=memory, private=True
    )
    out = priv.run("remember", {"text": "code secret 1234"})
    assert out == f"erreur: {PRIVATE_MEMORY_MSG}" and mem.items == []
    # Côté session distante, rien de la session privée n'est retrouvable.
    pub = build_registry(str(tmp_path), 10_000, ["recall"], memory=memory)
    assert "1234" not in pub.run("recall", {"query": "secret"})
    # Lecture permise en privé, sans synthèse par un modèle (configuré à part).
    mem.items.append("souvenir ancien partagé")
    assert "souvenir ancien" in priv.run("recall", {"query": "souvenir"})
    assert summaries == []


def test_image_sans_vision_desactivee_explicitement(tmp_path):
    """Modèle local sans vision : read_image refuse en le disant, sans aucun repli vers
    un autre modèle (pas de descripteur)."""
    from loom.tools.read import VISION_UNAVAILABLE

    reg = build_registry(
        str(tmp_path), 10_000, ["read_image"], active_is_vision=False, private=True
    )
    assert reg.run("read_image", {"path": "x.png"}) == f"erreur: {VISION_UNAVAILABLE}"
