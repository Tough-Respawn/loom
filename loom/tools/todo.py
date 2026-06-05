# loom/tools/todo.py
"""Outil manage_todos : mémoire de travail EXTERNE pour la boucle tool-use.

Un petit modèle perd le fil d'une tâche multi-étapes au bout de quelques appels
d'outils. `manage_todos` lui donne un bloc-notes persistant : il y pose son plan
(une liste de tâches statut par statut) et le réécrit à mesure qu'il avance. Le
rendu renvoyé À CHAQUE appel lui rappelle où il en est -> il ne repart pas de
zéro ni n'oublie une étape.

Sémantique de REMPLACEMENT total (comme l'outil TodoWrite de Claude Code) : le
modèle renvoie la liste COMPLÈTE à jour, on ne fusionne pas. L'état vit dans la
`Conversation` de la session active (`conversation.todos`) : par session, persisté
dans session.json -> survit au redémarrage et ne déborde pas d'une session à l'autre.
"""

from __future__ import annotations

from loom.tools.base import ToolError, ToolSpec

# Marqueurs ASCII (pas d'emoji) : à faire / en cours / fait.
_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}
_MAX_TODOS = 30


def _render(items: list[dict]) -> str:
    """Rend la liste en checklist lisible, avec un compteur d'avancement."""
    if not items:
        return "(aucune tâche)"
    done = sum(1 for i in items if i["status"] == "done")
    lines = [f"Plan ({done}/{len(items)} fait) :"]
    lines += [f"{_MARK[i['status']]} {i['content']}" for i in items]
    return "\n".join(lines)


def make_manage_todos(conversation) -> ToolSpec:
    """Outil manage_todos : écrit/met à jour le plan de la conversation active.

    `conversation` : tout objet portant une liste `.todos` (la Conversation de la
    session). On mute `conversation.todos` en place ; la persistance (session.json)
    est faite par la sauvegarde de fin de tour, comme pour les messages.
    """

    def run(args: dict) -> str:
        todos = args.get("todos")
        if not isinstance(todos, list):
            raise ToolError("argument 'todos' : une liste de tâches est attendue")
        if len(todos) > _MAX_TODOS:
            raise ToolError(
                f"trop de tâches ({len(todos)}) : garde un plan court (<= {_MAX_TODOS})"
            )
        clean: list[dict] = []
        for raw in todos:
            if not isinstance(raw, dict):
                raise ToolError("chaque tâche doit être un objet {content, status}")
            content = (raw.get("content") or "").strip()
            if not content:
                raise ToolError("chaque tâche doit avoir un 'content' non vide")
            status = (raw.get("status") or "pending").strip()
            if status not in _MARK:
                raise ToolError(
                    f"statut invalide '{status}' : pending | in_progress | done"
                )
            clean.append({"content": content, "status": status})
        conversation.todos = clean
        return _render(clean)

    return ToolSpec(
        name="manage_todos",
        description=(
            "Tient à jour ta liste de tâches pour une demande en plusieurs étapes : "
            "pose ton plan, puis réémets la liste COMPLÈTE à chaque progrès en changeant "
            "les statuts. Sert de mémoire externe pour ne pas perdre le fil. Chaque tâche "
            "= {content, status} avec status parmi pending, in_progress, done. Marque "
            "in_progress AVANT d'attaquer une étape, done une fois vérifiée."
        ),
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Liste COMPLÈTE des tâches, à jour.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Intitulé de la tâche.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "done"],
                                "description": "État de la tâche.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        run=run,
    )
