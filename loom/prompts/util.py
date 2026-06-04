# loom/prompts/util.py
"""Petits utilitaires partagés par les builders de prompts (bornage du contexte)."""

from __future__ import annotations


def clip(text: str, cap: int) -> str:
    """Tronque `text` à `cap` caractères avec un marqueur explicite (borne le prompt).

    Le bornage est vital en fan-out : le pool KV (-c) est PARTAGÉ entre les requêtes
    concurrentes, un prompt non borné sature le contexte et tronque la génération."""
    if text is None:
        return ""
    return text if len(text) <= cap else text[:cap] + "\n…[tronqué]"
