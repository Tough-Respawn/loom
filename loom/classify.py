# loom/classify.py
"""Routage d'intention : décide si une demande relève du BUILD (créer/modifier des
fichiers de code) ou du CHAT (question/discussion). Un seul appel court ; défaut sûr = chat."""

from __future__ import annotations

from loom.prompts import CLASSIFY_SYSTEM


def classify_intent(client, message: str, *, model: str | None) -> str:
    """Renvoie 'build' ou 'chat'. Défaut 'chat' (sûr) si la réponse n'est pas 'build'."""
    raw = client.complete(
        [{"role": "user", "content": message}],
        CLASSIFY_SYSTEM,
        max_tokens=4,
        model=model,
        thinking=False,
        temperature=0.0,
    )
    return "build" if "build" in (raw or "").strip().lower() else "chat"
