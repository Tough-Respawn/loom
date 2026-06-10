# loom/tools/note.py
"""Outils write_note / read_note : mémoire DURABLE de la boucle tool-use.

Les résultats d'outils (lectures de fichiers, recherches) peuvent être PURGÉS du
contexte par la microcompaction quand la fenêtre se remplit -> le modèle perd ce qu'il
avait lu et le re-lit en boucle (thrash observé en session). Une note échappe à cette
purge : le modèle y consigne ses trouvailles (chemins, valeurs clés, décisions) et relit
sa note (petite) au lieu de re-lire un fichier entier.

Persistées dans `conversation.notes` (session.json), comme les todos -> par session,
survit au redémarrage, ne déborde pas d'une session à l'autre. Append par défaut ;
`replace=true` repart d'une note vierge (quand l'ancienne est périmée).
"""

from __future__ import annotations

from loom.tools.base import ToolError, ToolSpec

_MAX_NOTES = 50  # garde-fou : au-delà, la mémoire enfle au lieu d'aider
_MAX_NOTE_CHARS = 4000  # une note = une synthèse, pas un dump de fichier


def _render(notes: list[str]) -> str:
    if not notes:
        return "(aucune note)"
    lines = [f"Notes ({len(notes)}) :"]
    lines += [f"{i + 1}. {n}" for i, n in enumerate(notes)]
    return "\n".join(lines)


def make_write_note(conversation) -> ToolSpec:
    """Outil write_note : ajoute (ou remplace) une note dans la mémoire de la session.

    `conversation` : objet portant une liste `.notes` (la Conversation de la session).
    On mute en place ; la persistance (session.json) est faite par la sauvegarde de fin
    de tour, comme pour les messages et les todos.
    """

    def run(args: dict) -> str:
        note = (args.get("note") or "").strip()
        if not note:
            raise ToolError("argument 'note' : un texte non vide est attendu")
        if len(note) > _MAX_NOTE_CHARS:
            raise ToolError(
                f"note trop longue ({len(note)} car., max {_MAX_NOTE_CHARS}) : "
                "résume l'essentiel (chemins, valeurs, décisions), pas le contenu brut"
            )
        replace = bool(args.get("replace", False))
        if replace:
            conversation.notes = [note]
        else:
            if len(conversation.notes) >= _MAX_NOTES:
                raise ToolError(
                    f"trop de notes ({len(conversation.notes)}, max {_MAX_NOTES}) : "
                    "relis-les (read_note) et réécris une synthèse avec replace=true"
                )
            conversation.notes.append(note)
        return _render(conversation.notes)

    return ToolSpec(
        name="write_note",
        description=(
            "Consigne une note DURABLE dans ta mémoire de session. À utiliser pour ce que "
            "tu dois retenir au-delà de quelques tours : chemins de fichiers, valeurs "
            "clés, décisions, état d'avancement. Les résultats d'outils (lectures, "
            "recherches) peuvent être effacés du contexte quand il se remplit — une note, "
            "elle, persiste : relis-la (read_note) au lieu de re-lire un fichier entier. "
            "Append par défaut ; replace=true repart d'une note propre quand l'ancienne "
            "est périmée. Note une SYNTHÈSE, pas un copier-coller de fichier."
        ),
        parameters={
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Le contenu de la note (synthèse concise).",
                },
                "replace": {
                    "type": "boolean",
                    "description": (
                        "Si true, efface les notes existantes et repart de celle-ci. "
                        "Défaut false (ajoute à la suite)."
                    ),
                },
            },
            "required": ["note"],
        },
        run=run,
    )


def make_read_note(conversation) -> ToolSpec:
    """Outil read_note : relit toutes les notes de la session (mémoire durable)."""

    def run(args: dict) -> str:
        return _render(conversation.notes)

    return ToolSpec(
        name="read_note",
        description=(
            "Relit tes notes de session (mémoire durable écrite avec write_note). "
            "Réflexe à avoir quand tu reprends le fil ou que tu doutes d'avoir oublié "
            "quelque chose : tes notes survivent à la purge du contexte, pas les "
            "résultats d'outils. Moins coûteux que re-lire un fichier."
        ),
        parameters={"type": "object", "properties": {}},
        run=run,
    )
