# loom/planning.py
"""Planification PROFONDE : au-delà du plan 1-shot, on (1) auto-critique le design pour
combler ses trous, puis (2) on le découpe en petites USER STORIES portant des critères
d'acceptation OBSERVABLES, et (3) on EXTERNALISE le tout en .md sous `<workspace>/.loom/`.

Pourquoi externaliser : le dev et le vérificateur s'appuient sur ces fichiers pour
continuer sans porter tout le contexte en mémoire (cf. harness Anthropic : le système de
fichiers est la mémoire de travail). Les critères d'acceptation alimentent ensuite le
vérificateur orienté-intention."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loom.parallel import FileSpec
from loom.prompts import (
    CRITIQUE_SYSTEM,
    DECOMPOSE_SYSTEM,
    critique_prompt,
    decompose_prompt,
)


@dataclass
class UserStory:
    """Une US exécutable : petite, ordonnée, avec des critères OBSERVABLES."""

    id: str
    title: str
    detail: str
    acceptance: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "detail": self.detail,
            "acceptance": list(self.acceptance),
            "files": list(self.files),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserStory":
        return cls(
            id=str(d.get("id", "")),
            title=str(d.get("title", "")),
            detail=str(d.get("detail", "")),
            acceptance=[str(a) for a in d.get("acceptance", [])],
            files=[str(f) for f in d.get("files", [])],
        )


def critique_design(client, design: str, task: str, *, model: str | None) -> str:
    """Passe d'AUTO-CRITIQUE : le modèle attaque son propre plan et liste les trous, qu'on
    annexe au design. Robuste : critique vide -> design inchangé (jamais de régression)."""
    raw = client.complete(
        [{"role": "user", "content": critique_prompt(design, task)}],
        CRITIQUE_SYSTEM,
        max_tokens=1024,
        model=model,
        thinking=False,
    )
    refinements = (raw or "").strip()
    if not refinements:
        return design
    return f"{design}\n\n## Critique et raffinements\n{refinements}"


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("aucun objet JSON")
    return json.loads(text[start : end + 1])


def decompose_into_stories(
    client, design: str, specs: list[FileSpec], task: str, *, model: str | None
) -> list[UserStory]:
    """Découpe le plan en US (JSON). Robuste : si le JSON casse ou est vide, on dégrade
    en UNE US par fichier (utile plutôt que rien, cohérent avec la philosophie KISS)."""
    files = [s.path for s in specs]
    raw = client.complete(
        [{"role": "user", "content": decompose_prompt(design, task, files)}],
        DECOMPOSE_SYSTEM,
        max_tokens=2048,
        model=model,
        thinking=False,
    )
    try:
        data = _extract_json(raw)
        items = data.get("stories", []) if isinstance(data, dict) else []
        stories = [
            UserStory.from_dict(it)
            for it in items
            if isinstance(it, dict) and it.get("id")
        ]
        if stories:
            return stories
    except (ValueError, json.JSONDecodeError, TypeError):
        pass
    # Fallback déterministe : une US par fichier planifié.
    return [
        UserStory(
            id=f"US-{i:02d}",
            title=spec.path,
            detail=spec.role or f"Produire {spec.path}",
            acceptance=[],
            files=[spec.path],
        )
        for i, spec in enumerate(specs, start=1)
    ]


def story_md(story: UserStory) -> str:
    """Rendu .md d'une US (critères d'acceptation en cases à cocher)."""
    accept = "\n".join(f"- [ ] {a}" for a in story.acceptance) or "- [ ] (à définir)"
    files = ", ".join(story.files) or "(non précisé)"
    return (
        f"# {story.id} — {story.title}\n\n"
        f"{story.detail}\n\n"
        f"**Fichiers :** {files}\n\n"
        f"## Critères d'acceptation\n{accept}\n"
    )


def stories_for_file(stories: list[UserStory], path: str) -> str:
    """Texte compact des US touchant `path` (id, titre, détail, critères), pour le prompt
    de génération de ce fichier. Vide si aucune US ne le concerne."""
    lines: list[str] = []
    for s in stories:
        if path in s.files:
            crit = " ; ".join(s.acceptance)
            crit = f" Acceptation : {crit}." if crit else ""
            lines.append(f"- {s.id} {s.title} : {s.detail}.{crit}")
    return "\n".join(lines)


def acceptance_text(stories: list[UserStory]) -> str:
    """Concatène les critères d'acceptation de toutes les US (vide si aucune n'en a).
    Sert d'ancrage au vérificateur orienté-intention."""
    lines = [
        f"{s.id} {s.title} : " + " ; ".join(s.acceptance)
        for s in stories
        if s.acceptance
    ]
    return "\n".join(lines)


def plan_md(design: str, stories: list[UserStory]) -> str:
    """Rendu .md du PLAN global (design + index des US)."""
    index = "\n".join(f"- {s.id} — {s.title}" for s in stories)
    return f"# Plan\n\n{design}\n\n## User stories\n{index}\n"


def write_plan_artifacts(
    workspace: str, design: str, stories: list[UserStory], *, write=None
) -> str:
    """Écrit le PLAN et les US en .md sous `.loom/` (chemins RELATIFS). Renvoie le dossier.

    `write(relpath, content)` (optionnel) : le run réel passe son writer borné au dossier
    cible (les tests un faux writer -> pas de pollution disque). Par défaut, écriture FS
    directe sous `<workspace>/.loom/`. Libère le contexte : dev et vérificateur relisent
    ces fichiers au lieu de tout porter en mémoire."""
    if write is None:

        def write(rel, content):  # writer FS par défaut, rooté sur le workspace
            p = Path(workspace) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return str(p)

    write(".loom/PLAN.md", plan_md(design, stories))
    for story in stories:
        write(f".loom/us/{story.id}.md", story_md(story))
    return str(Path(workspace) / ".loom")
