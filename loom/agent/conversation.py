# loom/agent/conversation.py
"""Mémoire de conversation : historique des messages + persistance JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Conversation:
    system_prompt: str
    messages: list[dict] = field(default_factory=list)
    active_tools: list[str] = field(default_factory=list)
    model: str = ""
    thinking: bool = True
    # Plan de tâches de manage_todos : par conversation (donc par session) et persisté
    # ici -> survit au redémarrage, ne déborde plus d'une session à l'autre.
    todos: list[dict] = field(default_factory=list)
    # Notes de write_note/read_note : mémoire DURABLE qui échappe à la microcompaction
    # (laquelle purge les résultats d'outils). Le modèle y consigne ses trouvailles et
    # relit sa note plutôt que de re-lire un fichier entier. Par session, persisté ici.
    notes: list[str] = field(default_factory=list)
    # Objectif de complétion (commande /goal) : condition vérifiable qui maintient l'agent au
    # travail jusqu'à ce qu'un évaluateur la juge atteinte (façon /goal de Claude Code). Vide =
    # pas d'objectif. Effacé quand atteint. Par session, persisté ici.
    goal: str = ""

    def add(self, role: str, content: str | list) -> None:
        self.messages.append({"role": role, "content": content})

    def reset(self) -> None:
        self.messages = []
        self.todos = []  # nouvelle conversation = plan vierge
        self.notes = []  # ...et notes vierges
        self.goal = ""  # ...et objectif effacé

    def set_goal(self, goal: str) -> None:
        self.goal = (goal or "").strip()

    def set_tools(self, names: list[str]) -> None:
        self.active_tools = list(names)

    def set_model(self, model: str) -> None:
        self.model = model

    def set_thinking(self, thinking: bool) -> None:
        self.thinking = bool(thinking)

    def to_messages(self) -> list[dict]:
        return list(self.messages)

    def to_dict(self) -> dict:
        """État sérialisable (réutilisé par Session pour s'inclure sans dupliquer)."""
        return {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "active_tools": self.active_tools,
            "model": self.model,
            "thinking": self.thinking,
            "todos": self.todos,
            "notes": self.notes,
            "goal": self.goal,
        }

    @classmethod
    def from_dict(cls, data: dict, default_system_prompt: str) -> "Conversation":
        """Reconstruit depuis un dict tolérant aux anciens formats (clés absentes)."""
        return cls(
            system_prompt=data.get("system_prompt", default_system_prompt),
            messages=list(data.get("messages", [])),
            active_tools=list(data.get("active_tools", [])),
            model=data.get("model", ""),
            thinking=bool(data.get("thinking", True)),
            todos=list(data.get("todos", [])),
            notes=list(data.get("notes", [])),
            goal=data.get("goal", ""),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path, default_system_prompt: str) -> "Conversation":
        path = Path(path)
        if not path.exists():
            return cls(system_prompt=default_system_prompt)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data, default_system_prompt)
        except (json.JSONDecodeError, OSError):
            return cls(system_prompt=default_system_prompt)
