# loom/explore.py
"""Exploration déterministe : récolte la « ground truth » brownfield.

Quand la tâche utilisateur cite des fichiers existants, on les lit (de façon
bornée) pour que la phase PLAN dispose d'un vrai contexte. AUCUN appel LLM,
AUCUNE boucle d'outils ici : uniquement des lectures de système de fichiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from loom.context import estimate_tokens
from loom.tools.base import _resolve_in_root

_PATH_RE = re.compile(r"[\w\-./]+\.\w{1,6}")
_PROJECT_EXTS = (".html", ".css", ".js", ".mjs", ".json")


def list_project_files(workspace: str, *, max_files: int = 25) -> list[str]:
    """Liste (bornée) les fichiers web déjà présents sous `workspace`, en chemins relatifs
    POSIX. Sert au plan brownfield (réutiliser EXACTEMENT l'existant, ne pas réinventer
    l'archi). Ignore les dossiers cachés (dont `.loom`). Borne aussi le PARCOURS pour ne
    jamais s'étouffer sur un arbre géant."""
    root = Path(workspace)
    if not root.is_dir():
        return []
    out: list[str] = []
    visited = 0
    for p in sorted(root.rglob("*")):
        visited += 1
        if visited > 2000 or len(out) >= max_files:
            break
        if not p.is_file() or p.suffix.lower() not in _PROJECT_EXTS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return out


@dataclass
class ExploreResult:
    summary: str  # ground-truth concaténée, bornée (vide si greenfield)
    files: list[str]  # chemins relatifs réellement lus


def _candidate_paths(task: str) -> list[str]:
    """Chemins cités dans `task`, dédupliqués en gardant l'ordre d'apparition."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _PATH_RE.findall(task):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


def explore(
    task: str,
    workspace: str,
    *,
    context: int = 8192,
    max_files: int = 3,
    max_bytes: int = 12_000,
    budget_ratio: float = 0.6,
) -> ExploreResult:
    """Lit les fichiers cités dans `task` qui existent sous `workspace`, bornés."""
    root = Path(workspace)
    budget = budget_ratio * context
    blocks: list[str] = []
    files: list[str] = []

    for rel in _candidate_paths(task):
        if len(files) >= max_files:
            break
        try:
            target = _resolve_in_root(root, rel)
        except Exception:  # noqa: BLE001 - ToolError/autre : on n'échoue jamais sur un chemin
            continue
        if not target.is_file():
            continue
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - lecture impossible = on ignore
            continue
        if len(content) > max_bytes:
            content = content[:max_bytes] + "\n…[tronqué]"
        block = f"----- {rel} -----\n{content}\n"
        if estimate_tokens("".join(blocks) + block) > budget:
            break
        blocks.append(block)
        files.append(rel)

    return ExploreResult(summary="".join(blocks), files=files)
