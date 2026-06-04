# loom/tools/fs.py
"""Outils d'écriture : write_file (création/écrasement) et edit_file (remplace).

Bornés au workspace via `_resolve_in_root` (anti path-traversal). Écriture
ATOMIQUE (fichier .tmp + os.replace, comme `Conversation.save`) pour ne jamais
laisser de fichier partiel. Encodage utf-8, `newline=''` afin de préserver le
contenu byte-exact (pas de traduction \\n -> \\r\\n sous Windows).
"""

from __future__ import annotations

import os
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root


def _atomic_write(path: Path, content: str) -> None:
    """Écrit `content` en utf-8 de façon atomique (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(content)
    os.replace(tmp, path)


def make_write_file(workspace_dir: str, max_bytes: int) -> ToolSpec:
    """Outil write_file borné au workspace, taille plafonnée, écriture atomique."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        content = args.get("content")
        if content is None:
            raise ToolError("argument 'content' manquant")
        if len(content.encode("utf-8")) > max_bytes:
            raise ToolError(f"contenu trop volumineux (> {max_bytes} octets)")
        path = _resolve_in_root(root, rel)
        _atomic_write(path, content)
        return f"écrit : {rel} ({len(content)} caractères)"

    return ToolSpec(
        name="write_file",
        description=(
            "Crée ou écrase un fichier du workspace avec le contenu fourni. "
            "Utilise un chemin relatif au workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Chemin du fichier, relatif au workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "Contenu complet à écrire dans le fichier.",
                },
            },
            "required": ["path", "content"],
        },
        run=run,
    )


def _occurrence_lines(text: str, sub: str) -> list[int]:
    """Numéros de ligne (1-based) où `sub` apparaît — pour désambiguïser."""
    lines: list[int] = []
    idx = text.find(sub)
    while idx != -1:
        lines.append(text.count("\n", 0, idx) + 1)
        idx = text.find(sub, idx + len(sub))
    return lines


def make_edit_file(workspace_dir: str) -> ToolSpec:
    """Outil edit_file : remplace old_string par new_string (1 occurrence, ou toutes
    avec replace_all). Erreurs EXPLOITABLES par le modèle : n° de ligne des occurrences
    sur ambiguïté, indice CRLF sur 'introuvable' (cf. gap P2.2 docs/harness-strategy.md)."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        old_string = args.get("old_string")
        new_string = args.get("new_string")
        if old_string is None or new_string is None:
            raise ToolError("arguments 'old_string' et 'new_string' requis")
        if old_string == "":
            raise ToolError("old_string vide")
        replace_all = bool(args.get("replace_all", False))
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if path.is_dir():
            raise ToolError(f"'{rel}' est un répertoire, pas un fichier")
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"fichier binaire non éditable : {rel}") from exc
        count = text.count(old_string)
        if count == 0:
            crlf = old_string.replace("\n", "\r\n")
            hint = (
                " (le fichier utilise des fins de ligne CRLF — vérifie les retours"
                " à la ligne d'old_string)"
                if crlf != old_string and crlf in text
                else ""
            )
            raise ToolError(f"old_string introuvable dans {rel}{hint}")
        if count > 1 and not replace_all:
            locs = ", ".join(str(n) for n in _occurrence_lines(text, old_string)[:12])
            raise ToolError(
                f"old_string ambigu : {count} occurrences (lignes {locs}) dans {rel}. "
                "Ajoute du contexte pour rendre old_string unique, OU passe replace_all=true."
            )
        if replace_all:
            _atomic_write(path, text.replace(old_string, new_string))
            return f"modifié : {rel} ({count} occurrence(s))"
        _atomic_write(path, text.replace(old_string, new_string, 1))
        return f"modifié : {rel}"

    return ToolSpec(
        name="edit_file",
        description=(
            "Remplace old_string par new_string dans un fichier du workspace. Par défaut "
            "old_string doit être UNIQUE (sinon l'erreur liste les lignes des occurrences) ; "
            "passe replace_all=true pour remplacer TOUTES les occurrences identiques."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Chemin du fichier, relatif au workspace.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Texte exact à remplacer (unique, sauf si replace_all).",
                },
                "new_string": {
                    "type": "string",
                    "description": "Texte de remplacement.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Remplacer TOUTES les occurrences identiques (défaut: false).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        run=run,
    )
