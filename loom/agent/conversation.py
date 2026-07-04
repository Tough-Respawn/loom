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
    # Skills DÉSACTIVÉS pour cette session : retirés du catalogue et de use_skill. Vide = tous
    # actifs (défaut, rétro-compat). Par session -> on module la surface de skills par projet.
    disabled_skills: list[str] = field(default_factory=list)
    # Overrides de skill AU NIVEAU SESSION : nom -> texte SKILL.md édité. N'écrit PAS le fichier
    # sur disque (ça, c'est « enregistrer pour toutes les sessions »). Le catalogue et use_skill
    # utilisent ce texte à la place du fichier pour cette session uniquement.
    skill_overrides: dict[str, str] = field(default_factory=dict)
    # Compteurs de consommation RÉELS, cumulés sur toute la session (persistés). Une API sans
    # état refacture TOUT le contexte en INPUT à chaque appel d'outil : `tokens_in` explose vs
    # l'output visible. On somme donc input/output/coût sur TOUS les appels (pas par tour).
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0  # part de tokens_in servie par le cache de préfixe
    cost_usd: float = 0.0
    api_calls: int = 0

    def add_usage(
        self,
        prompt: int,
        completion: int,
        cached: int,
        price_in: float,
        price_out: float,
        price_cached: float = 0.0,
    ) -> None:
        """Cumule un appel API dans les compteurs de session + le coût ($ / M tokens).

        `cached` = part de `prompt` servie par le cache (facturée `price_cached` au lieu de
        `price_in`) -> le coût distingue input frais et input caché ; price_cached=0 = pas de
        remise (borne haute). Sert à MESURER si le prompt caching mord."""
        prompt = int(prompt or 0)
        completion = int(completion or 0)
        cached = min(int(cached or 0), prompt)  # jamais > input total
        self.tokens_in += prompt
        self.tokens_out += completion
        self.tokens_cached += cached
        self.api_calls += 1
        pc = price_cached if price_cached > 0 else price_in
        fresh = prompt - cached
        self.cost_usd += (
            fresh / 1e6 * price_in + cached / 1e6 * pc + completion / 1e6 * price_out
        )

    def usage_totals(self) -> dict:
        pct = round(self.tokens_cached / self.tokens_in * 100) if self.tokens_in else 0
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "cache_pct": pct,
            "cost_usd": round(self.cost_usd, 4),
            "api_calls": self.api_calls,
        }

    def add(self, role: str, content: str | list) -> None:
        self.messages.append({"role": role, "content": content})

    def reset(self) -> None:
        self.messages = []
        self.todos = []  # nouvelle conversation = plan vierge
        self.notes = []  # ...et notes vierges
        self.goal = ""  # ...et objectif effacé
        # Compteur de consommation remis à zéro : le fil repart, le cumul aussi.
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_cached = 0
        self.cost_usd = 0.0
        self.api_calls = 0

    def set_goal(self, goal: str) -> None:
        self.goal = (goal or "").strip()

    def set_disabled_skills(self, names: list[str]) -> None:
        self.disabled_skills = list(names)

    def set_skill_override(self, name: str, body: str | None) -> None:
        """Pose (ou retire si body=None) l'override de session d'un skill."""
        if body is None:
            self.skill_overrides.pop(name, None)
        else:
            self.skill_overrides[name] = body

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
            "disabled_skills": self.disabled_skills,
            "skill_overrides": self.skill_overrides,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "cost_usd": self.cost_usd,
            "api_calls": self.api_calls,
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
            disabled_skills=list(data.get("disabled_skills", [])),
            skill_overrides=dict(data.get("skill_overrides", {})),
            tokens_in=int(data.get("tokens_in", 0) or 0),
            tokens_out=int(data.get("tokens_out", 0) or 0),
            tokens_cached=int(data.get("tokens_cached", 0) or 0),
            cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
            api_calls=int(data.get("api_calls", 0) or 0),
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
