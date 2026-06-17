"""Mémoire à deux étages de Loom. Interface `MemoryProvider` + sélection par config.

v1 ne livre QUE le provider `local` (SQLite FTS5, offline, stdlib). Les providers
externes (mem0/supermemory/redis) sont des points d'extension NON implémentés : on
reste 100% offline. La sélection les importe paresseusement et lève une erreur claire
si demandés — jamais de crash silencieux au boot, jamais de réseau par défaut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Snippet:
    """Un épisode retrouvé par recall (longue traîne cherchable)."""

    text: str
    kind: str = "episodic"
    source: str = ""
    score: float = 0.0


@runtime_checkable
class MemoryProvider(Protocol):
    def remember(
        self, text: str, *, kind: str = "episodic", source: str = ""
    ) -> None: ...

    def recall(self, query: str, *, k: int = 5) -> list[Snippet]: ...


def get_provider(name: str, *, db_path: str) -> MemoryProvider:
    """Renvoie le provider mémoire. v1 : seul `local` est implémenté.

    Les providers externes sont importés paresseusement et lèvent une erreur explicite
    (jamais activés par défaut, jamais de réseau). Cf. design §2/§5.3/§12.
    """
    if name == "local":
        from loom.memory.local import LocalMemory

        return LocalMemory(db_path)
    if name in ("mem0", "supermemory", "redis"):
        raise NotImplementedError(
            f"provider mémoire '{name}' non implémenté en v1 (Loom reste offline). "
            "Seul 'local' est disponible. Cf. design §2/§12."
        )
    raise ValueError(f"provider mémoire inconnu : '{name}' (attendu : 'local').")
