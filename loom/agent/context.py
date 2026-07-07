# loom/agent/context.py
"""Gestion de la fenêtre de contexte : estimation de tokens + résumé automatique."""

from __future__ import annotations

from loom.utils import estimate_tokens


def effective_context_budget(
    configured_budget: int, context: int, max_tokens: int, margin: int = 512
) -> int:
    """Budget de prompt SÛR : garantit `prompt + max_tokens + marge <= contexte`.

    Empêche la troncature (le bug d'origine : budget 3000 + max_tokens 2048 > contexte 4096).
    On ne laisse jamais le prompt dépasser ce qui laisse la place à la réponse complète.
    """
    safe = context - max_tokens - margin
    return max(256, min(configured_budget, safe))


def _content_tokens(content) -> int:
    if isinstance(content, str):
        return estimate_tokens(content)
    total = 0
    for part in content:
        if part.get("type") == "text":
            total += estimate_tokens(part.get("text", ""))
        elif part.get("type") == "image_url":
            total += 1000  # coût visuel approximatif d'une image
    return total


def conversation_tokens(system_prompt: str, messages: list[dict]) -> int:
    total = estimate_tokens(system_prompt)
    for m in messages:
        total += _content_tokens(m["content"])
    return total


def needs_summary(system_prompt: str, messages: list[dict], budget: int) -> bool:
    return conversation_tokens(system_prompt, messages) > budget


def summarize(conversation, client, budget: int, keep_recent: int) -> bool:
    """Résumé PRÉ-TOUR (proactif) : si l'historique dépasse `budget`, remplace les vieux
    messages par un résumé dense. Renvoie True si un résumé a bien eu lieu.

    Délègue à la PRIMITIVE UNIQUE `client.summarize_slice` (partagée avec l'étage de la
    boucle d'outils et le bouton manuel) : anglais télégraphique, littéraux préservés,
    strip du <think>, et FAIL-SOFT (modèle injoignable / API en erreur -> '' au lieu de
    lever -> ici on renvoie False, plus jamais de 500 sur /chat comme le WinError 10061)."""
    msgs = conversation.messages
    if not needs_summary(conversation.system_prompt, msgs, budget):
        return False
    if len(msgs) <= keep_recent:
        return False
    old, recent = msgs[:-keep_recent], msgs[-keep_recent:]
    model = getattr(conversation, "model", None)
    summary = client.summarize_slice(old, model=model)
    if not summary:
        return False
    conversation.messages = [
        {"role": "user", "content": f"[Conversation summary so far: {summary}]"},
        *recent,
    ]
    return True
