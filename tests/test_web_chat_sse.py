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


def test_cancel_ferme_le_stream_distant_bloque(chat_env):
    # (3) hung-remote : sur un modèle distant bloqué au moment du STOP, cancel_event
    # n'est lu qu'ENTRE deux chunks. Si l'itération du stream ne rend jamais la main
    # (modèle distant lent/figé), le finally qui relâche le verrou de session n'est
    # jamais atteint -> session morte (tout /chat suivant -> 202 pour toujours). /cancel
    # doit FERMER le stream actif : close() lève httpx.ReadError, la boucle la classe en
    # api_error, le finally s'exécute et le verrou est libéré de façon BORNÉE.
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

    # STOP : doit fermer le stream bloqué et libérer le verrou de façon bornée.
    assert web.post("/cancel", data={"session_id": sid}).status_code == 204
    t.join(timeout=5)
    assert not t.is_alive(), "le /chat bloqué ne s'est pas terminé après /cancel"

    # Verrou relâché : un nouveau /chat GÉNÈRE (200), pas coincé en 202 pour toujours.
    r2 = web.post("/chat", data={"message": "reprends", "session_id": sid})
    assert r2.status_code == 200, (
        f"verrou de session non relâché après /cancel (reçu {r2.status_code})"
    )
    assert _sse_events(r2.data)[-1]["type"] == "done"


def test_note_en_vol_reste_202_sans_stop(chat_env):
    # Garde-fou : SANS STOP en cours, un message envoyé pendant une génération active
    # reste une note en vol (202). Le fix ne doit pas casser cette sémantique.
    from loom.web.routes.helpers import _lock_for

    web, _, _, sid = chat_env([turn_text("x.")])
    S = web.application.S
    _lock_for(S, sid).acquire()  # génération active, AUCUN /cancel
    r = web.post("/chat", data={"message": "btw note", "session_id": sid})
    assert r.status_code == 202


def test_chat_outils_dans_le_workspace_de_la_session_cible(tmp_env, monkeypatch):
    # Anti-mélange d'onglets : /chat cible `sess` (session_id). Même si la session
    # FOCUS change (autre onglet activé pendant la génération -> _session(S) renvoie une
    # autre session), les OUTILS doivent tourner dans le workspace de la session CIBLE,
    # jamais celui de la focus. Sans le fix, chat.py:650 lit _session(S).workspace.
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
    # une AUTRE session B, avec un workspace différent, que _session renverra (focus)
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
    # Fenêtre de résurrection concurrente : /chat capture l'objet session AVANT
    # d'acquérir son verrou. Un /session/delete peut supprimer entre-temps. On simule
    # l'état résultant (dossier supprimé, objet encore en cache) : /chat, une fois le
    # verrou acquis, doit détecter la disparition et ABANDONNER, sans que save() ne
    # recrée la session.
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
    # /init <dir> doit adopter le dossier dans la session PASSÉE (cible capturée par
    # /chat), pas dans S.cur (focus) : sinon un onglet activé pendant la génération
    # détournerait l'adoption. Test unitaire du contrat de _handle_init_command.
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
    # L'amorçage KV (_prime_slot) construit le prompt avec le workspace de la session
    # CIBLE passée, pas celui de S.cur (focus) : sinon on amorce le cache avec le
    # loom.md/dossier d'un autre onglet.
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
    # Un session_id EXPLICITE mais inconnu ne doit pas retomber en silence sur la
    # session focus (l'onglet enverrait son message dans une autre session) : 404,
    # comme /fork et /compact.
    r = web_sess.post("/chat", data={"message": "x", "session_id": "deadbeefdead"})
    assert r.status_code == 404


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
