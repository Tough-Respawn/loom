# loom/tools/search.py
"""Outils de LOCALISATION : find_files (glob), search_text (grep), list_dir (ls).

Le réflexe n°1 d'un agent : un petit modèle ne connaît pas les chemins, il CHERCHE
au lieu de deviner. Acceptent les chemins ABSOLUS (agir partout) comme RELATIFS au
dossier de travail ; ignorent les dossiers lourds ; PLAFONNÉS (pas d'étouffement)."""

from __future__ import annotations

import re
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root

# Dossiers jamais parcourus : ils noient le résultat et étouffent le scan.
# .claude : worktrees d'agents (chacun une copie COMPLÈTE du projet) + config interne
# -> sans ça, find_files/search_text renvoient N duplicatas par fichier.
_SKIP_DIRS = frozenset(
    {
        ".claude",
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
    """Vrai si un chemin traverse un dossier ignoré."""
    return any(part in _SKIP_DIRS for part in rel.parts)


def _glob_base_and_pattern(root: Path, pattern: str) -> tuple[Path, str]:
    """Sépare un motif glob en (dossier de base concret, motif relatif).

    Motif RELATIF -> (root, motif). Motif ABSOLU (ex. `C:/a/b/**/*.py`) -> on prend le
    plus long préfixe SANS joker comme base, et le reste comme motif (pathlib ne sait
    pas globber un motif absolu directement). Sert à chercher hors du dossier de travail.
    """
    p = Path(pattern)
    if not p.is_absolute():
        return root.resolve(), pattern
    magic = ("*", "?", "[")
    base_parts: list[str] = []
    rest: list[str] = []
    for part in p.parts:
        if rest or any(m in part for m in magic):
            rest.append(part)
        else:
            base_parts.append(part)
    base = Path(*base_parts) if base_parts else p
    if not rest:  # chemin concret sans joker -> matche ce seul nom
        return base.parent, base.name
    return base, "/".join(rest)


def _display(p: Path, base: Path) -> str:
    """Chemin relatif à `base` s'il est dedans, sinon chemin absolu (posix)."""
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return p.as_posix()


def make_find_files(workspace_dir: str, *, max_results: int = 200) -> ToolSpec:
    """Outil find_files : liste les fichiers du workspace matchant un motif glob."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        pattern = (args.get("pattern") or "").strip()
        if not pattern:
            raise ToolError("argument 'pattern' manquant (ex: '**/*.py', 'src/*.js')")
        base = root.resolve()
        gbase, gpat = _glob_base_and_pattern(base, pattern)
        hits: list[str] = []
        try:
            it = gbase.glob(gpat)
        except (ValueError, NotImplementedError) as exc:
            raise ToolError(f"motif invalide : {pattern} ({exc})") from exc
        for p in it:
            if not p.is_file():
                continue
            shown = _display(p, base)
            if _skipped(Path(shown)):
                continue
            hits.append(shown)
            if len(hits) >= max_results:
                break
        if not hits:
            return f"aucun fichier ne correspond à : {pattern}"
        hits.sort()
        return "\n".join(hits)

    return ToolSpec(
        name="find_files",
        description=(
            "Trouve des fichiers par MOTIF glob. Relatif au dossier de travail "
            "(ex: '**/*.py', '**/config.*') OU absolu pour chercher ailleurs "
            "(ex: 'C:/Users/moi/Desktop/projet/**/*.md'). Sert à LOCALISER un fichier "
            "ou voir la structure. Renvoie les chemins (relatifs si dans le dossier de "
            "travail, absolus sinon)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Motif glob relatif au dossier de travail (ex: '**/*.md') "
                        "ou ABSOLU pour chercher ailleurs (ex: 'C:/Users/moi/docs/**/*.pdf')."
                    ),
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
        if globf:
            gbase, gpat = _glob_base_and_pattern(base, globf)
            files = gbase.glob(gpat)
        else:
            files = base.rglob("*")
        out: list[str] = []
        scanned = 0
        for p in files:
            if len(out) >= max_matches or scanned >= max_files_scanned:
                break
            if not p.is_file():
                continue
            shown = _display(p, base)
            if _skipped(Path(shown)):
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
                    out.append(f"{shown}:{i}: {line.strip()[:200]}")
                    if len(out) >= max_matches:
                        break
        if not out:
            return f"aucune correspondance pour : {pattern}"
        return "\n".join(out)

    return ToolSpec(
        name="search_text",
        description=(
            "Cherche une EXPRESSION RÉGULIÈRE dans le contenu des fichiers et renvoie "
            "les correspondances (fichier:ligne: texte). Sert à trouver OÙ est "
            "défini/utilisé un symbole, une chaîne, une fonction. Filtre optionnel "
            "'glob' relatif au dossier de travail (ex: '**/*.py') ou absolu pour "
            "chercher ailleurs. Pour lire un fichier entier, utilise plutôt read_file."
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
            "Liste le contenu d'un dossier (sous-dossiers suffixés '/'). Chemin relatif "
            "au dossier de travail OU absolu (ex: 'C:/Users/moi/Desktop/projet'). Sert à "
            "explorer un dossier. Pour une recherche large par motif, find_files est mieux."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Dossier à lister : relatif au dossier de travail (défaut '.') "
                        "ou chemin absolu (ex: 'C:/Users/moi/Desktop/projet')."
                    ),
                }
            },
            "required": [],
        },
        run=run,
    )
