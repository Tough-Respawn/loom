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
    # Schémas différés déjà chargés (persistés) : pas re-demandés après redémarrage.
    deferred_loaded: list[str] = field(default_factory=list)
    model: str = ""
    thinking: bool = True
    # Une session privée interdit tout routage distant ; seul l'utilisateur la déclare.
    local_only: bool = False
    # Le plan persiste par session et ne doit jamais déborder dans une autre.
    todos: list[dict] = field(default_factory=list)
    # Ces notes survivent à la microcompaction des résultats d'outils.
    notes: list[str] = field(default_factory=list)
    # L'objectif persiste jusqu'à ce que son évaluateur le juge atteint.
    goal: str = ""
    # Une liste vide conserve tous les skills actifs pour cette session.
    disabled_skills: list[str] = field(default_factory=list)
    # Un override de session masque le fichier sans le modifier sur disque.
    skill_overrides: dict[str, str] = field(default_factory=dict)
    # Cumuler tous les appels : une API sans état refacture le contexte à chaque tour d'outil.
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0  # part de tokens_in servie par le cache de préfixe
    cost_usd: float = 0.0
    api_calls: int = 0
    # Contrairement aux totaux, ce champ mesure seulement l'occupation du dernier appel.
    context_tokens: int = 0
    # Persister le wizard pour qu'il survive à un rafraîchissement de page.
    wizard: dict | None = None

    def add_usage(
        self,
        prompt: int,
        completion: int,
        cached: int,
        price_in: float,
        price_out: float,
        price_cached: float = 0.0,
        set_context: bool = True,
    ) -> None:
        """Cumule un appel API dans les compteurs de session + le coût ($ / M tokens).

        `cached` = part de `prompt` servie par le cache (facturée `price_cached` au lieu de
        `price_in`) -> le coût distingue input frais et input caché ; price_cached=0 = pas de
        remise (borne haute). Sert à MESURER si le prompt caching mord.

        `set_context=False` pour la conso d'un SOUS-AGENT (dispatch_agent) : ses tokens comptent
        dans les totaux (coût/N×), mais son prompt N'EST PAS le contexte du fil principal ->
        on ne touche pas la jauge de remplissage."""
        prompt = int(prompt or 0)
        completion = int(completion or 0)
        cached = min(int(cached or 0), prompt)  # jamais > input total
        self.tokens_in += prompt
        self.tokens_out += completion
        self.tokens_cached += cached
        self.api_calls += 1
        # Seul le dernier appel du fil principal définit l'occupation courante.
        if set_context:
            self.context_tokens = prompt
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
            "context_tokens": self.context_tokens,
        }

    def add(self, role: str, content: str | list) -> None:
        self.messages.append({"role": role, "content": content})

    def reset(self) -> None:
        self.messages = []
        self.todos = []  # nouvelle conversation = plan vierge
        self.notes = []  # ...et notes vierges
        self.goal = ""  # ...et objectif effacé
        self.wizard = None  # wizard abandonné avec le fil
        self.deferred_loaded = []
        # Compteur de consommation remis à zéro : le fil repart, le cumul aussi.
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_cached = 0
        self.cost_usd = 0.0
        self.api_calls = 0
        self.context_tokens = 0

    def set_goal(self, goal: str) -> None:
        self.goal = (goal or "").strip()

    def set_wizard(self, state: dict | None) -> None:
        self.wizard = state

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

    def set_local_only(self, local_only: bool) -> None:
        self.local_only = bool(local_only)

    def to_messages(self) -> list[dict]:
        return list(self.messages)

    def to_dict(self) -> dict:
        """État sérialisable (réutilisé par Session pour s'inclure sans dupliquer)."""
        return {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "active_tools": self.active_tools,
            "deferred_loaded": self.deferred_loaded,
            "model": self.model,
            "thinking": self.thinking,
            "local_only": self.local_only,
            "todos": self.todos,
            "notes": self.notes,
            "goal": self.goal,
            "wizard": self.wizard,
            "disabled_skills": self.disabled_skills,
            "skill_overrides": self.skill_overrides,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "cost_usd": self.cost_usd,
            "api_calls": self.api_calls,
            "context_tokens": self.context_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict, default_system_prompt: str) -> "Conversation":
        """Reconstruit depuis un dict tolérant aux anciens formats (clés absentes)."""
        return cls(
            system_prompt=data.get("system_prompt", default_system_prompt),
            messages=list(data.get("messages", [])),
            active_tools=list(data.get("active_tools", [])),
            deferred_loaded=list(data.get("deferred_loaded", [])),
            model=data.get("model", ""),
            thinking=bool(data.get("thinking", True)),
            local_only=bool(data.get("local_only", False)),
            todos=list(data.get("todos", [])),
            notes=list(data.get("notes", [])),
            goal=data.get("goal", ""),
            wizard=data.get("wizard"),
            disabled_skills=list(data.get("disabled_skills", [])),
            skill_overrides=dict(data.get("skill_overrides", {})),
            tokens_in=int(data.get("tokens_in", 0) or 0),
            tokens_out=int(data.get("tokens_out", 0) or 0),
            tokens_cached=int(data.get("tokens_cached", 0) or 0),
            cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
            api_calls=int(data.get("api_calls", 0) or 0),
            context_tokens=int(data.get("context_tokens", 0) or 0),
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
