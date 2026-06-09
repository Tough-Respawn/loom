# loom/skills.py
"""Skills : modules de connaissance markdown DÉCLENCHÉS PAR LE MODÈLE (façon Claude Code).

On annonce au modèle un CATALOGUE (nom : description) ; quand un skill est pertinent il
appelle l'outil use_skill(nom) qui renvoie le corps. Plus d'activation manuelle. Les skills
viennent du dossier local (loom/skills, non namespacés) ET des plugins installés (namespacés
`plugin:nom`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str
    base_dir: str = ""  # dossier du SKILL.md (pour résoudre references/)


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


def _load_skill_file(md: Path, namespace: str | None) -> Skill | None:
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return None
    name, desc, body = _parse_skill_md(text, md.parent.name)
    if namespace:
        name = f"{namespace}:{name}"
    return Skill(name=name, description=desc, body=body, base_dir=str(md.parent))


def _scan_dir(skills_dir: str | Path, namespace: str | None) -> list[Skill]:
    skills_dir = Path(skills_dir)
    out: list[Skill] = []
    if not skills_dir.exists():
        return out
    for sub in sorted(skills_dir.iterdir()):
        md = sub / "SKILL.md"
        if sub.is_dir() and md.exists():
            sk = _load_skill_file(md, namespace)
            if sk:
                out.append(sk)
    return out


def collect_skills(
    local_dir: str | Path, plugins_root_path: str | Path | None = None
) -> list[Skill]:
    """Agrège les skills locaux (non namespacés) + ceux des plugins installés
    (namespacés `plugin:nom`)."""
    skills = _scan_dir(local_dir, namespace=None)
    if plugins_root_path is not None:
        from loom.plugins import discover_plugins

        for plugin in discover_plugins(plugins_root_path):
            for md in plugin.skills:
                sk = _load_skill_file(md, namespace=plugin.name)
                if sk:
                    skills.append(sk)
    return skills


def render_catalog(skills: list[Skill]) -> str:
    """Bloc injecté au prompt système : la liste nom : description + comment déclencher."""
    if not skills:
        return ""
    lines = [
        "# Skills disponibles",
        "Quand l'un de ces skills correspond à la demande, APPELLE l'outil "
        "`use_skill(name)` pour charger ses instructions, puis suis-les. Ne devine pas "
        "leur contenu.",
    ]
    for s in skills:
        desc = " ".join(s.description.split())
        if len(desc) > 220:
            desc = desc[:217] + "…"
        lines.append(f"- {s.name} : {desc}")
    return "\n".join(lines)


def load_skill_body(skills: list[Skill], name: str) -> str | None:
    """Corps d'un skill par nom (préfixé du dossier de base pour lire references/)."""
    for s in skills:
        if s.name == name:
            head = (
                f"Base directory for this skill: {s.base_dir}\n\n" if s.base_dir else ""
            )
            return f"{head}{s.body}"
    return None
