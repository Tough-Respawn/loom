"""Classement des erreurs API : un llama-server mort au lancement n'est PAS un overflow."""

import httpx
from openai import BadRequestError, InternalServerError

from loom.agent.errors import _classify_api_error

_REQ = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")


def _err(cls, code, body):
    return cls(
        f"Error code: {code} - {body}",
        response=httpx.Response(code, request=_REQ),
        body=body,
    )


def test_llama_swap_upstream_mort_est_backend_down():
    body = {
        "error": "unspecific error: upstream command exited prematurely",
        "src": "llama-swap",
    }
    assert _classify_api_error(_err(InternalServerError, 500, body)) == "backend_down"


def test_autre_500_reste_overflow():
    assert _classify_api_error(_err(InternalServerError, 500, {"error": "boom"})) == (
        "overflow"
    )


def test_depassement_de_contexte_inchange():
    body = {"error": "request (40000 tokens) exceeds the available context size"}
    assert _classify_api_error(_err(BadRequestError, 400, body)) == "context_overflow"
