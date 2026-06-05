# loom/tools/base.py
"""Cœur des outils : erreur métier, spec, registre, résolution de chemin bornée.

Un outil = un `ToolSpec` (nom, description, schéma JSON des arguments, fonction
`run`). Le `ToolRegistry` expose les schémas au format OpenAI `tools=[...]` et
exécute un appel par nom en transformant toute erreur en message exploitable par
le modèle (jamais d'exception qui casserait la boucle de streaming).

C'est le socle commun : read/document/image (read.py), localisation (search.py),
write/edit (fs.py), run_shell (shell.py), web (web.py), todos (todo.py) et
dispatch_agent (agent.py) s'enregistrent dessus sans toucher au transport.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
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
    {"name": "read_image", "label": "read_image", "danger": False},
    {"name": "web_search", "label": "web_search", "danger": False},
    {"name": "fetch_url", "label": "fetch_url", "danger": False},
    {"name": "dispatch_agent", "label": "dispatch_agent", "danger": False},
    {"name": "manage_todos", "label": "manage_todos", "danger": False},
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
    # Outil STREAMANT (optionnel) : au lieu de rendre une str d'un bloc, il yield les
    # events de sa propre activité (mêmes tuples que stream_chat_tools). La boucle les
    # relaie à l'UI EN DIRECT et reconstruit le résultat final. Sert à `dispatch_agent`
    # pour qu'on VOIE ce que fait le sous-agent. `run` reste le repli (1 bloc).
    run_stream: Callable[[dict], Iterator[tuple[str, object]]] | None = None

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

    def is_streaming(self, name: str) -> bool:
        """Vrai si l'outil expose une exécution STREAMANTE (run_stream)."""
        spec = self._specs.get(name)
        return bool(spec and spec.run_stream)

    def run_stream(self, name: str, args: dict) -> Iterator[tuple[str, object]]:
        """Exécute un outil streamant en relayant ses events ; ne lève jamais (toute
        erreur devient un event ('content', 'erreur: …') que la boucle traite comme
        un résultat d'outil en échec)."""
        spec = self._specs.get(name)
        if spec is None or spec.run_stream is None:
            yield ("content", f"erreur: outil non streamant '{name}'")
            return
        try:
            yield from spec.run_stream(args)
        except ToolError as exc:
            yield ("content", f"erreur: {exc}")
        except Exception as exc:  # noqa: BLE001 - on ne casse jamais la boucle
            yield ("content", f"erreur inattendue: {exc}")


def _resolve_in_root(root: Path, rel: str) -> Path:
    """Résout un chemin d'outil. PLUS DE CONFINEMENT (Loom agit sur tout le système,
    comme un agent généraliste) :
    - chemin ABSOLU (ex. `C:/Users/.../x`, `/home/.../x`) -> utilisé tel quel ;
    - chemin RELATIF -> résolu sous `root`, qui n'est qu'un DOSSIER DE TRAVAIL par défaut.

    Le garde-fou n'est plus le périmètre mais la deny-list dure de loom.permissions
    (rm -rf, format, …), incontournable même ici.
    """
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    return (Path(root).resolve() / p).resolve()
