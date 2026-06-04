# loom/agents.py
"""Agents multi-étapes : modèle de données + fonctions pures du pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loom.skills import compose_system_prompt, load_skill


@dataclass
class Agent:
    """Un agent du pipeline : rôle, modèle et prompt système propres."""

    id: str
    role: str
    model: str
    system_prompt: str
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    max_tokens: int | None = None
    thinking: bool = True


@dataclass
class RunStep:
    """Résultat d'un agent : réflexion + contenu + fichiers écrits."""

    agent_id: str
    role: str
    model: str
    reasoning: str
    content: str
    written: list[str] = field(default_factory=list)


@dataclass
class AgentRun:
    """Exécution complète du pipeline pour une tâche."""

    task: str
    steps: list[RunStep] = field(default_factory=list)
    stories: list = field(default_factory=list)  # user stories (planif profonde)


def resolve_agents(configs: list[Agent], pipeline: list[str]) -> list[Agent]:
    """Sélectionne les agents dans l'ordre du pipeline (ids inconnus ignorés)."""
    by_id = {a.id: a for a in configs}
    return [by_id[aid] for aid in pipeline if aid in by_id]


def build_step_messages(task: str, prior_steps: list[RunStep]) -> list[dict]:
    """Construit les messages : tâche en tête, puis le content de chaque étape.

    Le reasoning des étapes précédentes n'est jamais propagé (contenu seul).
    """
    messages: list[dict] = [{"role": "user", "content": task}]
    for step in prior_steps:
        messages.append(
            {"role": "user", "content": f"Étape {step.role}: {step.content}"}
        )
    return messages


def compose_agent_system_prompt(agent: Agent, skills_dir: str) -> str:
    """Compose le prompt système de l'agent en injectant ses skills actifs."""
    active = [s for s in (load_skill(skills_dir, name) for name in agent.skills) if s]
    return compose_system_prompt(agent.system_prompt, active)


def is_blocking(content: str) -> bool:
    """Vrai si la revue n'a PAS explicitement validé le travail (verdict relecteur).

    Le relecteur doit terminer par 'VERDICT: OK' ou 'VERDICT: BLOQUANT'. On reste
    tolérant : 'non bloquant' / 'pas bloquant' priment sur 'bloquant'. En l'absence
    d'un 'VERDICT: OK' explicite, la revue est jugée non concluante → bloquante,
    ce qui force une passe de correction (bornée par max_revisions).
    """
    low = content.lower()
    if "non bloquant" in low or "non-bloquant" in low or "pas bloquant" in low:
        return False
    if "bloquant" in low:
        return True
    return re.search(r"verdict\s*:?\s*ok", low) is None


def is_reviewer(role: str) -> bool:
    """Vrai si le rôle désigne un relecteur (seul à armer la boucle review→fix).

    Évite qu'un pipeline sans relecteur (ex. plan→code) déclenche une révision
    juste parce que la dernière étape ne porte pas de 'VERDICT: OK'.
    """
    low = role.lower()
    return "rev" in low or "relect" in low
