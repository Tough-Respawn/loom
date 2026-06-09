# loom/agent/context.py
"""Gestion de la fenêtre de contexte : estimation de tokens + résumé automatique."""

from __future__ import annotations

SUMMARY_SYSTEM = "Tu résumes des conversations de façon concise et fidèle, en français."
SUMMARY_INSTRUCTION = (
    "Résume la conversation suivante en quelques phrases, en gardant les faits, décisions "
    "et informations importantes. Voici la conversation :\n\n"
)


def estimate_tokens(text: str) -> int:
    """Estimation grossière : ~1 token pour 4 caractères (min 1)."""
    return max(1, len(text) // 4)


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


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            text = c
        else:
            text = " ".join(p.get("text", "[image]") for p in c)
        lines.append(f"{m['role']}: {text}")
    return "\n".join(lines)


def summarize(conversation, client, budget: int, keep_recent: int) -> bool:
    """Si au-dessus du budget, remplace les vieux messages par un résumé. Renvoie True si résumé."""
    msgs = conversation.messages
    if not needs_summary(conversation.system_prompt, msgs, budget):
        return False
    if len(msgs) <= keep_recent:
        return False
    old, recent = msgs[:-keep_recent], msgs[-keep_recent:]
    prompt = SUMMARY_INSTRUCTION + _render(old)
    summary = "".join(
        text
        for kind, text in client.stream_chat(
            [{"role": "user", "content": prompt}], SUMMARY_SYSTEM
        )
        if kind == "content"
    )
    conversation.messages = [
        {
            "role": "user",
            "content": f"[Résumé de la conversation précédente : {summary}]",
        },
        *recent,
    ]
    return True
