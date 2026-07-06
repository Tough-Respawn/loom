"""Protocols de typage pour les outils (duck-typing documenté).

Ces Protocol décrivent les interfaces minimales attendues par les outils,
sans imposer d'héritage. Utilisés uniquement pour l'annotation statique.
"""

from __future__ import annotations

from typing import Any, Protocol


class ChatClient(Protocol):
    """Client de chat minimal (LoomClient ou mock)."""

    def chat(self, *args: Any, **kwargs: Any) -> Any: ...


class MemoryProvider(Protocol):
    """Provider mémoire minimal (LocalMemory ou équivalent)."""

    def remember(self, *args: Any, **kwargs: Any) -> Any: ...

    def recall(self, *args: Any, **kwargs: Any) -> Any: ...


class Conversation(Protocol):
    """Conversation minimale (liste de messages)."""

    messages: list[dict]
