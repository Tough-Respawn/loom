# Amorçage du cache KV au chargement (« prefill au load ») : dès qu'un modèle LOCAL
# est sélectionné ou qu'une session est créée/activée, le préfixe exact du prochain
# tour (system prompt + schémas d'outils) est pré-préfillé en FOND — le premier
# message ne paie plus que son delta. Invariant durable : ces déclencheurs passent
# par warm_context avec le VRAI préfixe (jamais le ping « poubelle » qui écrasait
# le slot KV). Route-level : on observe warm_context sur un client espion.
from __future__ import annotations

import time

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

MODEL = "fake-local"


class PrimeSpy:
    """Faux client : enregistre les warm_context ; serveur local « joignable »."""

    def __init__(self):
        self.calls = []

    def warm_context(
        self, messages, system_prompt, model=None, registry=None, thinking=True
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "system_prompt": system_prompt,
                "model": model,
            }
        )
        return True

    def running_local(self, timeout=5.0):
        return True, "{}"

    def unload_local(self):
        return True


def _wait_calls(spy, n, timeout=3.0):
    """Les amorces tournent dans un thread daemon : on attend (borné) leur trace."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(spy.calls) >= n:
            return True
        time.sleep(0.02)
    return len(spy.calls) >= n


@pytest.fixture()
def spy():
    return PrimeSpy()


@pytest.fixture()
def web_prime(tmp_env, spy):
    store = SessionStore(
        tmp_env / "sessions",
        default_system_prompt="prompt de test",
        default_model=MODEL,
        known_models=[MODEL],
    )
    app = create_app(
        client=spy,
        skills_dir=str(tmp_env / "skills"),
        session_store=store,
        models=[MODEL],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_env / "workspace"),
        user_skills_dir=str(tmp_env / "skills_user"),
        plugins_dir=str(tmp_env / "plugins"),
        remote_store_path=str(tmp_env / "remote_models.json"),
        tool_factory=lambda tools, ws, conv: None,
    )
    return app.test_client()


def test_session_new_amorce_le_prefill(web_prime, spy):
    assert web_prime.post("/session/new", data={}).status_code == 200
    assert _wait_calls(spy, 1), "aucune amorce après /session/new"
    call = spy.calls[0]
    assert call["model"] == MODEL
    # Fil vide : préfixe statique + placeholder user minimal (certains templates
    # refusent une requête sans message user), jamais le vrai contenu d'un tour.
    assert call["messages"] == [{"role": "user", "content": "."}]
    assert call["system_prompt"]  # system prompt complet, jamais vide (≠ ping)


def test_session_activate_amorce_le_prefill(web_prime, spy):
    a = web_prime.post("/session/new", data={}).get_json()
    assert _wait_calls(spy, 1)
    n = len(spy.calls)
    assert web_prime.post("/session/activate", data={"id": a["id"]}).status_code == 200
    assert _wait_calls(spy, n + 1), "aucune amorce après /session/activate"


def test_model_update_amorce_le_prefill(web_prime, spy):
    web_prime.post("/session/new", data={})
    assert _wait_calls(spy, 1)
    n = len(spy.calls)
    assert web_prime.post("/model", data={"model": MODEL}).status_code == 200
    assert _wait_calls(spy, n + 1), "aucune amorce après sélection du modèle"
    assert spy.calls[-1]["model"] == MODEL
