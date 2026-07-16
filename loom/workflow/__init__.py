# loom/workflow/__init__.py
"""Workflows : un SCRIPT Python écrit par le modèle orchestre N sous-agents.

`from loom.workflow import run_workflow, parse_meta, WorkflowError` est le point
d'entrée ; l'implémentation vit dans `runtime`.
"""

from __future__ import annotations

from loom.workflow.runtime import (
    MAX_AGENTS,
    MAX_ITEMS,
    WorkflowError,
    parse_meta,
    run_workflow,
)

__all__ = [
    "MAX_AGENTS",
    "MAX_ITEMS",
    "WorkflowError",
    "parse_meta",
    "run_workflow",
]
