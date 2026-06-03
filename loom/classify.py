# loom/classify.py
"""Routage d'intention : décide si une demande relève du BUILD (créer/modifier des
fichiers de code) ou du CHAT (question/discussion). Un seul appel court ; défaut sûr = chat."""

from __future__ import annotations

_CLASSIFY_SYS = (
    "Tu es un routeur d'intention. Tu lis la demande et tu réponds UN SEUL mot : "
    "BUILD si elle consiste à CRÉER ou MODIFIER des fichiers de code dans un projet "
    "(faire une app/un jeu/un script, corriger ou refactorer du code) ; sinon CHAT "
    "(question, explication, discussion). Réponds uniquement BUILD ou CHAT."
)


def classify_intent(client, message: str, *, model: str | None) -> str:
    """Renvoie 'build' ou 'chat'. Défaut 'chat' (sûr) si la réponse n'est pas 'build'."""
    raw = client.complete(
        [{"role": "user", "content": message}],
        _CLASSIFY_SYS,
        max_tokens=4,
        model=model,
        thinking=False,
        temperature=0.0,
    )
    return "build" if "build" in (raw or "").strip().lower() else "chat"
