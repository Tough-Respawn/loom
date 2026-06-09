# loom/tools/skills.py
"""Outil use_skill : charge à la demande le corps d'un skill du catalogue (déclenché par le
modèle d'après les descriptions annoncées dans le prompt système)."""

from __future__ import annotations

from collections.abc import Callable

from loom.skills import Skill, load_skill_body
from loom.tools.base import ToolError, ToolSpec


def make_use_skill(skills_provider: Callable[[], list[Skill]]) -> ToolSpec:
    def run(args: dict) -> str:
        name = (args.get("name") or "").strip()
        if not name:
            raise ToolError("argument 'name' manquant (nom du skill à charger)")
        skills = skills_provider()
        body = load_skill_body(skills, name)
        if body is None:
            valid = ", ".join(s.name for s in skills) or "(aucun)"
            raise ToolError(f"skill inconnu '{name}'. Skills valides : {valid}")
        return body

    return ToolSpec(
        name="use_skill",
        description=(
            "Charge les instructions d'un skill listé dans « Skills disponibles » du prompt "
            "système. Appelle-le DÈS qu'un skill correspond à la demande, puis suis son "
            "contenu. Argument : name (le nom exact du catalogue, ex. 'plugin:skill')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Nom exact du skill au catalogue.",
                }
            },
            "required": ["name"],
        },
        run=run,
    )
