# loom/tools/skills.py
"""Outil use_skill : charge à la demande le corps d'un skill du catalogue (déclenché par le
modèle d'après les descriptions annoncées dans le prompt système)."""

from __future__ import annotations

from collections.abc import Callable

from loom.extend.skills import Skill, load_skill_body
from loom.tools.base import ToolError, ToolSpec


def make_use_skill(skills_provider: Callable[[], list[Skill]]) -> ToolSpec:
    def run(args: dict) -> str:
        name = (args.get("name") or "").strip()
        if not name:
            raise ToolError("argument 'name' manquant (nom du skill à charger)")
        skills = skills_provider()
        body = load_skill_body(skills, name)
        if body is None:
            # Le modèle appelle souvent le nom COURT ('brainstorming') au lieu du nom
            # namespacé du catalogue ('superpowers:brainstorming'), ou varie la casse.
            # On résout par égalité insensible à la casse, puis par suffixe UNIQUE, avant
            # d'échouer (analogue de la leçon `^` : accepter l'écriture équivalente).
            low = name.lower()
            exact_ci = [s for s in skills if s.name.lower() == low]
            by_suffix = [s for s in skills if s.name.lower().split(":")[-1] == low]
            match = exact_ci or by_suffix
            if len(match) == 1:
                body = load_skill_body(skills, match[0].name)
            if body is None:
                # Confusion outil/skill (vu en session : use_skill('dispatch_agent')) :
                # rediriger vers l'appel direct au lieu d'un simple « skill inconnu ».
                from loom.tools.base import AVAILABLE_TOOLS

                if low in (t["name"] for t in AVAILABLE_TOOLS):
                    raise ToolError(
                        f"'{name}' est un OUTIL, pas un skill : appelle-le "
                        "directement (comme n'importe quel autre outil)."
                    )
                valid = ", ".join(s.name for s in skills) or "(aucun)"
                hint = (
                    " (plusieurs skills matchent ce nom court, précise le namespace)"
                    if len(match) > 1
                    else ""
                )
                raise ToolError(
                    f"skill inconnu '{name}'{hint}. Skills valides : {valid}"
                )
        return body

    return ToolSpec(
        name="use_skill",
        description=(
            "Loads the instructions of a skill listed in 'Available skills' of the system "
            "prompt. Call it AS SOON AS a skill matches the request, then follow its "
            "content. Argument: name (the exact catalog name, e.g. 'plugin:skill')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact skill name in the catalog.",
                }
            },
            "required": ["name"],
        },
        run=run,
    )
