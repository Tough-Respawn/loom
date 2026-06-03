# loom/skills.py
"""Skills : fichiers markdown de connaissance injectables dans le contexte (format Claude Code)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str


def _parse_skill_md(text: str, fallback_name: str) -> tuple[str, str, str]:
    """Parse frontmatter (name/description) + corps. Renvoie (name, description, body)."""
    name, description, body = fallback_name, "", text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            for line in front.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key, val = key.strip().lower(), val.strip()
                    if key == "name" and val:
                        name = val
                    elif key == "description":
                        description = val
    return name, description, body


def list_skills(skills_dir: str | Path) -> list[Skill]:
    """Scanne <skills_dir>/<nom>/SKILL.md et renvoie les skills trouvés (triés par nom de dossier)."""
    skills_dir = Path(skills_dir)
    out: list[Skill] = []
    if not skills_dir.exists():
        return out
    for sub in sorted(skills_dir.iterdir()):
        md = sub / "SKILL.md"
        if sub.is_dir() and md.exists():
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            name, desc, body = _parse_skill_md(text, sub.name)
            out.append(Skill(name=name, description=desc, body=body))
    return out


def load_skill(skills_dir: str | Path, name: str) -> Skill | None:
    for skill in list_skills(skills_dir):
        if skill.name == name:
            return skill
    return None


def compose_system_prompt(base: str, active: list[Skill]) -> str:
    """Concatène le prompt de base et le corps de chaque skill actif."""
    parts = [base]
    for skill in active:
        parts.append(f"# Skill : {skill.name}\n{skill.body}")
    return "\n\n".join(parts)
