# loom/tools/read.py
"""Outil de lecture : read_file (lecture bornée au workspace). READ-only."""

from __future__ import annotations

from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root


def make_read_file(
    workspace_dir: str, extensions: list[str], max_bytes: int
) -> ToolSpec:
    """Outil read_file borné au workspace, extensions autorisées, taille plafonnée."""
    root = Path(workspace_dir)
    allowed = {e.lower() for e in extensions}

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if path.is_dir():
            raise ToolError(f"'{rel}' est un répertoire, pas un fichier")
        if allowed and path.suffix.lower() not in allowed:
            raise ToolError(f"extension non autorisée : {path.suffix or '(aucune)'}")
        data = path.read_bytes()
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"fichier binaire non lisible : {rel}") from exc
        if truncated:
            text += f"\n...[tronqué à {max_bytes} octets]"
        return text

    return ToolSpec(
        name="read_file",
        description=(
            "Lit le contenu d'un fichier texte du workspace et le renvoie. "
            "Utilise un chemin relatif au workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Chemin du fichier, relatif au workspace.",
                }
            },
            "required": ["path"],
        },
        run=run,
    )
