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
    assert "stream" not in holder and not holder.get("abort")  # porte-flux nettoyé


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
    assert created == [] and holder == {"abort": False}


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
