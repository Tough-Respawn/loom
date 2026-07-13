# loom/extend/skills.py
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
    # Champs optionnels des skills AUTO-APPRIS (namespace `learned`). Absents des skills
    # officiels/plugins -> défauts, rétro-compat totale.
    learned: bool = False
    uses: int = 0
    created_at: str = ""
    updated_at: str = ""
    # FAMILLE du skill (badge UI + droits) : "loom" = défaut du package (non supprimable
    # depuis l'UI), "user" = ajouté par l'utilisateur, "learned" = auto-appris,
    # "plugin" = installé via le store. Posée par collect_skills selon la source.
    origin: str = "loom"


def _parse_skill_md(text: str, fallback_name: str) -> tuple[str, str, str, dict]:
    """Parse frontmatter + corps. Renvoie (name, description, body, meta) où meta porte les
    champs optionnels des skills appris (learned/uses/created_at/updated_at) avec défauts."""
    name, description, body = fallback_name, "", text
    meta = {"learned": False, "uses": 0, "created_at": "", "updated_at": ""}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            body = text[end + 4 :].lstrip("\r\n")
            for line in front.splitlines():
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key, val = key.strip().lower(), val.strip()
                if key == "name" and val:
                    name = val
                elif key == "description":
                    description = val
                elif key == "learned":
                    meta["learned"] = val.lower() in ("true", "1", "yes", "oui")
                elif key == "uses":
                    meta["uses"] = int(val) if val.isdigit() else 0
                elif key in ("created_at", "updated_at"):
                    meta[key] = val
    return name, description, body, meta


def _load_skill_file(md: Path, namespace: str | None) -> Skill | None:
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return None
    name, desc, body, meta = _parse_skill_md(text, md.parent.name)
    if namespace:
        name = f"{namespace}:{name}"
    # base_dir ABSOLU : le workspace d'une session peut être n'importe où, un chemin
    # relatif rendrait les references/ introuvables depuis read_file.
    return Skill(
        name=name,
        description=desc,
        body=body,
        base_dir=str(md.parent.resolve()),
        **meta,
    )


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
    local_dir: str | Path,
    plugins_root_path: str | Path | None = None,
    learned_dir: str | Path | None = None,
    user_dir: str | Path | None = None,
) -> list[Skill]:
    """Agrège les skills du package (non namespacés, origin=loom) + AJOUTÉS PAR
    L'UTILISATEUR (non namespacés, origin=user, prioritaires en cas de nom en double) +
    AUTO-APPRIS (namespace `learned`) + ceux des plugins installés (namespacés
    `plugin:nom`)."""
    skills = _scan_dir(local_dir, namespace=None)
    for s in skills:
        s.origin = "loom"
    if user_dir is not None:
        seen = {s.name for s in skills}
        for s in _scan_dir(user_dir, namespace=None):
            s.origin = "user"
            # Un skill user au même nom qu'un skill package REMPLACE ce dernier
            # (l'utilisateur a la main), sans doublon au catalogue.
            if s.name in seen:
                skills = [k for k in skills if k.name != s.name]
            skills.append(s)
    if learned_dir is not None:
        for s in _scan_dir(learned_dir, namespace="learned"):
            s.origin = "learned"
            skills.append(s)
    if plugins_root_path is not None:
        from loom.extend.plugins import discover_plugins

        for plugin in discover_plugins(plugins_root_path):
            for md in plugin.skills:
                sk = _load_skill_file(md, namespace=plugin.name)
                if sk:
                    sk.origin = "plugin"
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
        mark = " ⟳" if getattr(s, "learned", False) else ""
        lines.append(f"- {s.name}{mark} : {desc}")
    return "\n".join(lines)


def effective_skills(
    skills: list[Skill],
    *,
    overrides: dict[str, str] | None = None,
    disabled: list[str] | None = None,
) -> list[Skill]:
    """Skills EFFECTIFS pour une session : applique les overrides (texte SKILL.md édité,
    re-parsé) et RETIRE les skills désactivés. Sert au catalogue ET à use_skill, pour que
    le modèle ne voie/charge que ce qui est actif, avec le contenu de session s'il existe.

    L'override ne peut PAS changer l'identité (name) du skill : on garde le nom d'origine
    quoi que dise le frontmatter édité (sinon on casserait le lien catalogue <-> use_skill)."""
    overrides = overrides or {}
    skip = set(disabled or [])
    out: list[Skill] = []
    for s in skills:
        if s.name in skip:
            continue
        raw = overrides.get(s.name)
        if raw is None:
            out.append(s)
            continue
        _, desc, body, meta = _parse_skill_md(raw, s.name)
        out.append(
            Skill(name=s.name, description=desc, body=body, base_dir=s.base_dir, **meta)
        )
    return out


def skill_source_path(skill: Skill) -> Path | None:
    """Chemin du fichier SKILL.md d'un skill (pour lire/écrire son texte brut)."""
    if not skill.base_dir:
        return None
    return Path(skill.base_dir) / "SKILL.md"


def read_skill_source(skill: Skill) -> str:
    """Texte brut du SKILL.md sur disque (frontmatter + corps), pour l'éditeur.

    Repli : si le fichier est illisible, reconstitue un document minimal depuis les
    champs connus (ne perd pas la main sur l'édition)."""
    p = skill_source_path(skill)
    if p is not None:
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            pass
    name = skill.name.split(":")[-1]
    return f"---\nname: {name}\ndescription: {skill.description}\n---\n\n{skill.body}"


def write_skill_source(skill: Skill, text: str) -> Path:
    """Écrit le texte brut dans le SKILL.md sur disque (« enregistrer pour toutes les
    sessions »). Renvoie le chemin écrit. Lève OSError si non écrivable."""
    p = skill_source_path(skill)
    if p is None:
        raise OSError(f"skill '{skill.name}' sans fichier source (base_dir absent)")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def load_skill_body(skills: list[Skill], name: str) -> str | None:
    """Corps d'un skill par nom (préfixé du dossier de base pour lire references/)."""
    for s in skills:
        if s.name == name:
            head = (
                f"Base directory for this skill: {s.base_dir}\n\n" if s.base_dir else ""
            )
            return f"{head}{s.body}"
    return None
