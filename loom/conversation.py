# loom/conversation.py
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
    active_skills: list[str] = field(default_factory=list)
    active_tools: list[str] = field(default_factory=list)
    model: str = ""
    thinking: bool = True

    def add(self, role: str, content: str | list) -> None:
        self.messages.append({"role": role, "content": content})

    def reset(self) -> None:
        self.messages = []

    def set_skills(self, names: list[str]) -> None:
        self.active_skills = list(names)

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
            "active_skills": self.active_skills,
            "active_tools": self.active_tools,
            "model": self.model,
            "thinking": self.thinking,
        }

    @classmethod
    def from_dict(cls, data: dict, default_system_prompt: str) -> "Conversation":
        """Reconstruit depuis un dict tolérant aux anciens formats (clés absentes)."""
        return cls(
            system_prompt=data.get("system_prompt", default_system_prompt),
            messages=list(data.get("messages", [])),
            active_skills=list(data.get("active_skills", [])),
            active_tools=list(data.get("active_tools", [])),
            model=data.get("model", ""),
            thinking=bool(data.get("thinking", True)),
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
