# Stub OpenAI-compatible pour le banc E2E Playwright : streame des réponses
# scriptées, AUCUN modèle réel. Deux comportements :
#   - "lentement" dans le dernier message user => flux LENT (~60 s) pour tester
#     les interactions pendant une génération (blocage images, note en vol) ;
#   - sinon réponse rapide ; si une note en vol figure dans le contexte reçu,
#     la réponse contient "(note en vol bien visible dans le contexte)" —
#     preuve OBSERVABLE que l'injection a atteint l'appel modèle.
# Lancement : uv run python tests/e2e/stub_openai.py  (port 18081)
from __future__ import annotations

import json
import time

from flask import Flask, Response, request

app = Flask(__name__)


def _chunk(delta=None, finish=None):
    return (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "stub",
                "choices": [
                    {"index": 0, "delta": delta or {}, "finish_reason": finish}
                ],
            }
        )
        + "\n\n"
    )


def _usage_chunk():
    return (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "stub",
                "choices": [],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 21,
                    "total_tokens": 63,
                },
            }
        )
        + "\n\n"
    )


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "stub", "object": "model"}]}


@app.post("/v1/chat/completions")
def completions():
    body = request.get_json(force=True)
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    last = ""
    if users:
        c = users[-1].get("content", "")
        last = c if isinstance(c, str) else json.dumps(c)
    slow = "lentement" in last
    notes = [m for m in body.get("messages", []) if "note received mid-turn" in str(m)]

    if slow:
        words = ("je réfléchis posément, morceau par morceau, " * 8).split(" ")
        delay = 1.2
    else:
        reponse = "Réponse du stub : bien reçu."
        if notes:
            reponse += " (note en vol bien visible dans le contexte)"
        words = reponse.split(" ")
        delay = 0.05

    def gen():
        for w in words:
            yield _chunk({"content": w + " "})
            time.sleep(delay)
        yield _chunk(finish="stop")
        yield _usage_chunk()
        yield "data: [DONE]\n\n"

    if not body.get("stream"):
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": " ".join(words)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 21, "total_tokens": 63},
        }
    return Response(gen(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=18081, threaded=True)
