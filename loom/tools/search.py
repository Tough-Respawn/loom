# loom/tools/search.py
"""Outils de LOCALISATION : find_files (glob), search_text (grep), list_dir (ls).

Le réflexe n°1 d'un agent : un petit modèle ne connaît pas les chemins, il CHERCHE
au lieu de deviner. Tous bornés au workspace (anti-traversal via _resolve_in_root),
ignorant les dossiers lourds, et PLAFONNÉS (pas d'étouffement sur un arbre géant)."""

from __future__ import annotations

import re
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root

# Dossiers jamais parcourus : ils noient le résultat et étouffent le scan.
_SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".next",
        ".cache",
        ".idea",
        ".vscode",
        ".tox",
        "site-packages",
    }
)
# Extensions considérées « texte » pour le grep par défaut (sans filtre glob).
_TEXT_EXT = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".json",
        ".html",
        ".htm",
        ".css",
        ".js",
        ".mjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".yaml",
        ".yml",
        ".sh",
        ".cfg",
        ".ini",
        ".xml",
        ".csv",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".php",
    }
)


def _skipped(rel: Path) -> bool:
    """Vrai si un chemin relatif traverse un dossier ignoré."""
    return any(part in _SKIP_DIRS for part in rel.parts)


def make_find_files(workspace_dir: str, *, max_results: int = 200) -> ToolSpec:
    """Outil find_files : liste les fichiers du workspace matchant un motif glob."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        pattern = (args.get("pattern") or "").strip()
        if not pattern:
            raise ToolError("argument 'pattern' manquant (ex: '**/*.py', 'src/*.js')")
        if ".." in pattern:
            raise ToolError("motif invalide : pas de '..' (reste dans le workspace)")
        base = root.resolve()
        hits: list[str] = []
        for p in base.glob(pattern):
            if not p.is_file():
                continue
            rel = p.relative_to(base)
            if _skipped(rel):
                continue
            hits.append(rel.as_posix())
            if len(hits) >= max_results:
                break
        if not hits:
            return f"aucun fichier ne correspond à : {pattern}"
        hits.sort()
        return "\n".join(hits)

    return ToolSpec(
        name="find_files",
        description=(
            "Trouve les fichiers du workspace par MOTIF glob (ex: '**/*.py', "
            "'**/config.*'). Sert à LOCALISER un fichier ou voir la structure quand "
            "tu ne connais pas le chemin exact. Renvoie des chemins relatifs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Motif glob relatif au workspace (ex: '**/*.md').",
                }
            },
            "required": ["pattern"],
        },
        run=run,
    )


def make_search_text(
    workspace_dir: str,
    *,
    max_matches: int = 80,
    max_file_bytes: int = 1_000_000,
    max_files_scanned: int = 3000,
) -> ToolSpec:
    """Outil search_text : grep regex sur le contenu des fichiers du workspace."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        pattern = args.get("pattern") or ""
        if not pattern:
            raise ToolError("argument 'pattern' manquant")
        globf = (args.get("glob") or "").strip()
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"expression régulière invalide : {exc}") from exc
        base = root.resolve()
        files = base.glob(globf) if globf else base.rglob("*")
        out: list[str] = []
        scanned = 0
        for p in files:
            if len(out) >= max_matches or scanned >= max_files_scanned:
                break
            if not p.is_file():
                continue
            rel = p.relative_to(base)
            if _skipped(rel):
                continue
            if not globf and p.suffix.lower() not in _TEXT_EXT:
                continue
            try:
                if p.stat().st_size > max_file_bytes:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned += 1
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    out.append(f"{rel.as_posix()}:{i}: {line.strip()[:200]}")
                    if len(out) >= max_matches:
                        break
        if not out:
            return f"aucune correspondance pour : {pattern}"
        return "\n".join(out)

    return ToolSpec(
        name="search_text",
        description=(
            "Cherche une EXPRESSION RÉGULIÈRE dans le contenu des fichiers du "
            "workspace et renvoie les correspondances (fichier:ligne: texte). Sert à "
            "trouver OÙ est défini/utilisé un symbole, une chaîne, une fonction. "
            "Filtre optionnel 'glob' (ex: '**/*.py'). Pour lire un fichier entier, "
            "utilise plutôt read_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Expression régulière à chercher (syntaxe Python re).",
                },
                "glob": {
                    "type": "string",
                    "description": "Filtre glob optionnel des fichiers (ex: '**/*.js').",
                },
            },
            "required": ["pattern"],
        },
        run=run,
    )


def make_list_dir(workspace_dir: str, *, max_entries: int = 300) -> ToolSpec:
    """Outil list_dir : contenu d'un dossier du workspace (dossiers puis fichiers)."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or ".").strip() or "."
        target = _resolve_in_root(root, rel)
        if not target.exists():
            raise ToolError(f"introuvable : {rel}")
        if not target.is_dir():
            raise ToolError(f"'{rel}' n'est pas un dossier (utilise read_file)")
        entries: list[str] = []
        items = sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        for p in items:
            if p.name in _SKIP_DIRS:
                continue
            entries.append(p.name + ("/" if p.is_dir() else ""))
            if len(entries) >= max_entries:
                break
        if not entries:
            return f"(dossier vide : {rel})"
        return "\n".join(entries)

    return ToolSpec(
        name="list_dir",
        description=(
            "Liste le contenu d'un dossier du workspace (sous-dossiers suffixés '/'). "
            "Sert à explorer un dossier inconnu. Pour une recherche large par motif, "
            "find_files est mieux."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Dossier relatif au workspace (défaut : '.').",
                }
            },
            "required": [],
        },
        run=run,
    )
