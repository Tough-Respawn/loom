"""Étape 4 de l'audit timings (2026-09-13) : le warm est PRÉEMPTIBLE.

Avant : le warm de fin de tour (ou le keep-warm, ou reflect) tenait le verrou
local pendant tout son prefill ; un message arrivé pendant ce temps attendait
jusqu'à sa fin (54 s vécus). Contrat : le warm publie son flux dans un porte-flux
partagé ; un message qui trouve le verrou tenu par un amorçage FERME ce flux
(llama-server annule la tâche à la déconnexion et garde le cache déjà calculé),
le warm rend la main aussitôt, le tour paie lui-même le contexte restant.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace as NS

import pytest

from loom.agent.client import LoomClient
from loom.agent.session import SessionStore
from loom.web.app import create_app
from loom.web.routes.helpers import _abort_warm

from .fakes import FakeOAI, FakeRegistry, turn_text

MODEL = "fake-local"


class _BlockingStream:
    """Flux qui bloque (comme un prefill silencieux) jusqu'à close()."""

    def __init__(self):
        self.started = threading.Event()
        self.released = threading.Event()
        self.closed = False

    def __iter__(self):
        self.started.set()
        self.released.wait(timeout=5)
        if self.closed:
            raise RuntimeError("stream closed")
        return iter(())

    def close(self):
        self.closed = True
        self.released.set()


# ---- client -------------------------------------------------------------------


def test_warm_context_publie_son_flux_et_signale_l_abandon():
    client = LoomClient("http://127.0.0.1:9/v1")
    stream = _BlockingStream()
    client._client = NS(chat=NS(completions=NS(create=lambda **kw: stream)))
    holder: dict = {}
    result: list = []
    t = threading.Thread(
        target=lambda: result.append(
            client.warm_context(
                [{"role": "user", "content": "."}], "sys", stream_holder=holder
            )
        )
    )
    t.start()
    assert stream.started.wait(timeout=2)
    assert holder["stream"] is stream  # visible de l'extérieur, donc interruptible
    holder["abort"] = True
    holder["stream"].close()
    t.join(timeout=2)
    assert result == [False]  # abandonné, pas « amorcé »
    # Le flux est retiré, mais le signal RESTE : il vaut pour toute la maintenance
    # (titre, ping…) jusqu'à la libération du verrou par son détenteur.
    assert "stream" not in holder and holder.get("abort") is True


def test_warm_context_saute_si_l_abandon_est_deja_demande():
    client = LoomClient("http://127.0.0.1:9/v1")
    created: list = []
    client._client = NS(chat=NS(completions=NS(create=lambda **kw: created.append(kw))))
    holder = {"abort": True}
    assert (
        client.warm_context(
            [{"role": "user", "content": "."}], "sys", stream_holder=holder
        )
        is False
    )
    assert created == [] and holder == {"abort": True}  # signal conservé


def test_abort_warm_ferme_le_flux_courant():
    stream = _BlockingStream()
    S = NS(warm_holder={"stream": stream})
    assert _abort_warm(S) is True
    assert stream.closed and S.warm_holder["abort"] is True
    assert _abort_warm(NS()) is False  # pas de porte-flux : rien à faire


def test_reflect_transmet_le_porte_flux():
    from loom.agent.reflect import reflect

    seen: dict = {}

    def stream_chat(messages, system_prompt, max_tokens=0, **kw):
        seen.update(kw)
        yield ("content", "pas de json")

    client = NS(stream_chat=stream_chat)
    holder: dict = {}
    reflect(
        [{"role": "user", "content": "x"}],
        ["a"],
        "r",
        client=client,
        model="m",
        provider=None,
        paths={},
        learned_dir="",
        stream_holder=holder,
    )
    assert seen["stream_holder"] is holder


# ---- parcours web complet -------------------------------------------------------


def _sse_events(body: bytes) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture()
def warm_env(tmp_env, monkeypatch):
    client = LoomClient("http://127.0.0.1:9/v1")
    fake = FakeOAI([turn_text("un."), turn_text("deux.")])
    client._client = fake
    client.slot_counts = {MODEL: 2}
    warm = {"stream": None, "started": threading.Event()}

    def warm_context(*a, stream_holder=None, **k):
        # Simule un prefill d'amorçage LONG qui honore le porte-flux.
        if stream_holder is not None and stream_holder.get("abort"):
            stream_holder["abort"] = False
            return False
        s = _BlockingStream()
        warm["stream"] = s
        if stream_holder is not None:
            stream_holder["stream"] = s
        warm["started"].set()
        try:
            for _ in s:
                pass
        except RuntimeError:
            return False
        finally:
            if stream_holder is not None:
                stream_holder.pop("stream", None)
                stream_holder.pop("abort", None)
        return True

    monkeypatch.setattr(client, "warm_context", warm_context)
    monkeypatch.setattr(
        client,
        "running_local",
        lambda timeout=0.0: (True, json.dumps({"running": [{"model": MODEL}]})),
    )
    monkeypatch.setattr(client, "save_slot", lambda *a, **k: False)
    monkeypatch.setattr(client, "try_hot_resume", lambda *a, **k: False)
    monkeypatch.setattr(client, "infer_title", lambda *a, **k: "")
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
    r = web.post("/session/new", data={"title": "t"})
    assert r.status_code == 200
    return web, warm, r.get_json()["id"]


def test_un_message_interrompt_le_warm_de_fin_de_tour(warm_env):
    web, warm, sid = warm_env
    # L'amorçage de session/new (thread loom-prime) peut avoir démarré : on l'attend
    # et on le laisse là — c'est justement lui (ou celui de fin de tour) que le
    # message doit interrompre.
    r = web.post("/chat", data={"message": "premier", "session_id": sid})
    assert _sse_events(r.data)[-1]["type"] == "done"
    assert warm["started"].wait(timeout=3), "aucun warm n'a démarré après le tour"
    t0 = time.monotonic()
    r = web.post("/chat", data={"message": "second", "session_id": sid})
    elapsed = time.monotonic() - t0
    events = _sse_events(r.data)
    assert events[-1]["type"] == "done", events[-3:]
    assert elapsed < 2.0, elapsed  # sans préemption : 5 s (le faux warm) ou 54 s (réel)
    assert warm["stream"].closed  # le flux du warm a bien été fermé par le message
    assert any(e["type"] == "notice" and "interrompu" in e["text"] for e in events), [
        e for e in events if e["type"] == "notice"
    ]


# ---- revue 2 (2026-09-13) : signal conservé jusqu'à la libération du verrou ----------


class _RaceStream(_BlockingStream):
    """Flux dont l'itération ne doit JAMAIS commencer (annulation pendant l'ouverture)."""

    def __iter__(self):
        raise AssertionError("le flux ne doit pas être lu après une annulation")


def test_warm_context_revérifie_l_annulation_des_l_ouverture_du_flux():
    client = LoomClient("http://127.0.0.1:9/v1")
    holder: dict = {}
    stream = _RaceStream()

    def create(**kw):
        holder["abort"] = True  # le message arrive PENDANT l'ouverture HTTP
        return stream

    client._client = NS(chat=NS(completions=NS(create=create)))
    assert (
        client.warm_context(
            [{"role": "user", "content": "."}], "sys", stream_holder=holder
        )
        is False
    )
    assert stream.closed and "stream" not in holder and holder["abort"] is True


def test_infer_title_revérifie_l_annulation_des_l_ouverture_du_flux():
    holder: dict = {}
    stream = _RaceStream()

    def create(**kw):
        holder["abort"] = True
        return stream

    oai = NS(chat=NS(completions=NS(create=create)))
    oai.with_options = lambda **kw: oai
    fake_self = NS(
        _resolve=lambda m: (oai, "m", True),
        is_remote=lambda m: False,
        annex_slot=lambda m: 1,
    )
    assert LoomClient.infer_title(fake_self, "orn", "x", stream_holder=holder) == ""
    assert stream.closed and holder["abort"] is True


def test_keepwarm_annule_ne_ping_pas_et_libere_le_signal(monkeypatch):
    import threading as _th

    import loom.web.routes.maintenance as m

    pings: list = []
    client = NS(stream_chat=lambda *a, **k: pings.append(k) or iter(()))
    sess = NS(conversation=NS(model=MODEL), id="s")
    S = NS(
        settings={"keepwarm_interval": 1.0, "keepwarm_enabled": True},
        last_activity=[time.time() - 100],
        local_gen_lock=_th.Lock(),
        local_busy={"reason": ""},
        cur={"session": sess},
        remote_model_ids=set(),
        client=client,
        warm_holder={},
    )

    def prime_aborted(S_, sess_):
        S_.warm_holder["abort"] = True  # un message a coupé le warm
        return False

    monkeypatch.setattr(m, "_prime_slot", prime_aborted)
    m._keepwarm_tick(S)
    assert pings == []  # pas de ping de repli pendant qu'un message attend
    assert S.warm_holder.get("abort") is False  # signal remis à zéro à la libération
    assert S.local_gen_lock.acquire(blocking=False)  # verrou libéré


@pytest.fixture()
def maint_env(tmp_env, monkeypatch):
    """Session SANS titre (titre demandé) + warm bloquant : le message qui interrompt
    le warm ne doit pas voir la maintenance enchaîner sur le titre."""
    client = LoomClient("http://127.0.0.1:9/v1")
    fake = FakeOAI([turn_text("un."), turn_text("deux.")])
    client._client = fake
    client.slot_counts = {MODEL: 2}
    seen = {"stream": None, "started": threading.Event(), "titles": [], "holder": None}

    def warm_context(*a, stream_holder=None, **k):
        seen["holder"] = stream_holder
        if stream_holder is not None and stream_holder.get("abort"):
            return False
        s = _BlockingStream()
        seen["stream"] = s
        stream_holder["stream"] = s
        seen["started"].set()
        try:
            for _ in s:
                pass
        except RuntimeError:
            return False
        finally:
            stream_holder.pop("stream", None)
        return True

    def infer_title(m, message, stream_holder=None):
        seen["titles"].append(message)
        return "Titre modèle"

    monkeypatch.setattr(client, "warm_context", warm_context)
    monkeypatch.setattr(client, "infer_title", infer_title)
    monkeypatch.setattr(
        client,
        "running_local",
        lambda timeout=0.0: (True, json.dumps({"running": [{"model": MODEL}]})),
    )
    monkeypatch.setattr(client, "save_slot", lambda *a, **k: False)
    monkeypatch.setattr(client, "try_hot_resume", lambda *a, **k: False)
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
    r = web.post("/session/new", data={})  # « Nouvelle session » : titre demandé
    assert r.status_code == 200
    return web, seen, store, r.get_json()["id"]


def test_warm_interrompu_aucun_titre_ne_suit_sous_le_verrou(maint_env):
    web, seen, store, sid = maint_env
    r = web.post("/chat", data={"message": "premier", "session_id": sid})
    assert _sse_events(r.data)[-1]["type"] == "done"
    assert seen["started"].wait(timeout=3)
    t0 = time.monotonic()
    r = web.post("/chat", data={"message": "second", "session_id": sid})
    elapsed = time.monotonic() - t0
    assert _sse_events(r.data)[-1]["type"] == "done"
    assert elapsed < 2.0, elapsed
    assert seen["stream"].closed
    time.sleep(0.3)
    assert seen["titles"] == []  # le titre n'a PAS été tenté après l'interruption
    assert store.load(sid).title == "premier"  # provisoire conservé
    assert seen["holder"].get("abort") is False  # signal remis à zéro à la libération
