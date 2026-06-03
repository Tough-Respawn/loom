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
