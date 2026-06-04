# loom/tools/base.py
"""Cœur des outils : erreur métier, spec, registre, résolution de chemin bornée.

Un outil = un `ToolSpec` (nom, description, schéma JSON des arguments, fonction
`run`). Le `ToolRegistry` expose les schémas au format OpenAI `tools=[...]` et
exécute un appel par nom en transformant toute erreur en message exploitable par
le modèle (jamais d'exception qui casserait la boucle de streaming).

C'est le socle commun : read/verify (read.py), write/edit (fs.py), run_shell
(shell.py), web_search (web.py) s'enregistrent dessus sans toucher au transport.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ToolError(Exception):
    """Erreur métier d'un outil, message en français montrable au modèle."""


# Univers des outils proposés dans l'UI (activables par conversation). `danger`
# marque ceux qui modifient le système (gardés par le mode permission).
AVAILABLE_TOOLS = [
    {"name": "find_files", "label": "find_files", "danger": False},
    {"name": "search_text", "label": "search_text", "danger": False},
    {"name": "list_dir", "label": "list_dir", "danger": False},
    {"name": "read_file", "label": "read_file", "danger": False},
    {"name": "read_document", "label": "read_document", "danger": False},
    {"name": "web_search", "label": "web_search", "danger": False},
    {"name": "fetch_url", "label": "fetch_url", "danger": False},
    {"name": "write_file", "label": "write_file", "danger": True},
    {"name": "edit_file", "label": "edit_file", "danger": True},
    {"name": "run_shell", "label": "run_shell", "danger": True},
]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema des arguments
    run: Callable[[dict], str]

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Collection d'outils : expose les schémas et exécute par nom (sans lever)."""

    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {s.name: s for s in specs}

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def openai_tools(self) -> list[dict]:
        return [s.to_openai() for s in self._specs.values()]

    def run(self, name: str, args: dict) -> str:
        spec = self._specs.get(name)
        if spec is None:
            return f"erreur: outil inconnu '{name}'"
        try:
            return spec.run(args)
        except ToolError as exc:
            return f"erreur: {exc}"
        except Exception as exc:  # noqa: BLE001 - on ne casse jamais la boucle
            return f"erreur inattendue: {exc}"


def _resolve_in_root(root: Path, rel: str) -> Path:
    """Résout `rel` sous `root` en refusant toute évasion (path traversal)."""
    root = root.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ToolError(f"chemin hors du périmètre autorisé : {rel}")
    return target
