"""Outils de LOCALISATION : find_files (glob), search_text (grep), list_dir (ls).

Le réflexe n°1 d'un agent : un petit modèle ne connaît pas les chemins, il CHERCHE
au lieu de deviner. Acceptent les chemins ABSOLUS (agir partout) comme RELATIFS au
dossier de travail ; ignorent les dossiers lourds ; PLAFONNÉS (pas d'étouffement)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root


@lru_cache(maxsize=1)
def _rg_path() -> str | None:
    """Chemin de l'exécutable ripgrep s'il est installé, sinon None (repli Python).

    Résolu une seule fois : `rg` est plus rapide et respecte .gitignore. On tombe
    proprement sur le scanner Python pur si `rg` est absent (Loom reste autonome).

    PATH d'abord ; en repli l'emplacement d'install winget (Windows), car winget ne met
    pas toujours son shim sur le PATH du process courant -> Loom trouve quand même `rg`.
    Glob indépendant de la version installée."""
    found = shutil.which("rg")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        pkgs = Path(local) / "Microsoft" / "WinGet" / "Packages"
        try:
            for exe in sorted(pkgs.glob("*ripgrep*/**/rg.exe"), reverse=True):
                return str(exe)
        except OSError:
            pass
    return None


# Ignorer les dossiers générés et worktrees qui dupliquent ou noient les résultats.
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
        # Un motif sans dossier est récursif, comme avec ripgrep ou fd.
        if "/" not in pattern and "\\" not in pattern and "**" not in pattern:
            pattern = "**/" + pattern
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
            "Finds files by glob pattern — to LOCATE a file or see the structure. "
            "Returns paths ready for read_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern relative to the working directory (e.g. '**/*.md') "
                        "or ABSOLUTE to search elsewhere (e.g. 'C:/Users/moi/docs/**/*.pdf')."
                    ),
                }
            },
            "required": ["pattern"],
        },
        run=run,
    )


_RG_LINE = re.compile(r"^(.*?):(\d+):(.*)$")


def _rg_search(
    rg: str,
    base: Path,
    search_dir: Path,
    pattern: str,
    gpat: str,
    max_matches: int,
    max_file_bytes: int,
) -> str | None:
    """Recherche via ripgrep. Renvoie le texte de résultat, ou None si `rg` échoue
    (motif incompatible avec la regex Rust, erreur d'exécution) -> repli Python.

    On lance `rg` avec cwd=search_dir et chemin `.` : les chemins sortis sont RELATIFS
    (pas de `C:` initial qui casserait le parsing `fichier:ligne:texte` sous Windows).
    On les réabsolutise pour l'affichage relatif-à-base, comme le repli Python."""
    cmd = [
        rg,
        "--line-number",
        "--no-heading",
        "--color",
        "never",
        "--max-filesize",
        str(max_file_bytes),
        "-e",
        pattern,
    ]
    if gpat:
        cmd += ["-g", gpat]
    else:
        for d in _SKIP_DIRS:
            cmd += ["-g", f"!{d}"]
    cmd.append(".")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(search_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # Pour ripgrep, 1 signifie « aucune correspondance », pas une erreur.
    if proc.returncode not in (0, 1):
        return None
    out: list[str] = []
    for raw in proc.stdout.splitlines():
        m = _RG_LINE.match(raw)
        if not m:
            continue
        rel, ln, content = m.group(1), m.group(2), m.group(3)
        shown = _display((search_dir / rel).resolve(), base)
        if _skipped(Path(shown)):
            continue
        out.append(f"{shown}:{ln}: {content.strip()[:200]}")
        if len(out) >= max_matches:
            break
    if not out:
        return f"aucune correspondance pour : {pattern}"
    return "\n".join(out)


def _py_search(
    base: Path,
    pattern: str,
    globf: str,
    max_matches: int,
    max_file_bytes: int,
    max_files_scanned: int,
) -> str:
    """Scanner regex pur Python (repli quand ripgrep est absent/incompatible)."""
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"expression régulière invalide : {exc}") from exc
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


def make_search_text(
    workspace_dir: str,
    *,
    max_matches: int = 80,
    max_file_bytes: int = 1_000_000,
    max_files_scanned: int = 3000,
) -> ToolSpec:
    """Outil search_text : grep regex sur le contenu des fichiers du workspace.

    Utilise ripgrep (`rg`) s'il est installé — rapide, respecte .gitignore — sinon un
    scanner Python pur (Loom reste autonome sans dépendance externe)."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        pattern = args.get("pattern") or ""
        if not pattern:
            raise ToolError("argument 'pattern' manquant")
        globf = (args.get("glob") or "").strip()
        # Valider tôt pour produire une erreur plus claire que celle du processus.
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"expression régulière invalide : {exc}") from exc
        base = root.resolve()
        if globf:
            search_dir, gpat = _glob_base_and_pattern(base, globf)
        else:
            search_dir, gpat = base, ""
        rg = _rg_path()
        if rg:
            res = _rg_search(
                rg, base, Path(search_dir), pattern, gpat, max_matches, max_file_bytes
            )
            if res is not None:
                return res
        return _py_search(
            base, pattern, globf, max_matches, max_file_bytes, max_files_scanned
        )

    return ToolSpec(
        name="search_text",
        description=(
            "Searches for a REGULAR EXPRESSION in file contents and returns the "
            "matches (file:line: text). Use it to find WHERE a symbol, a string, or a "
            "function is defined/used. Optional 'glob' filter relative to the working "
            "directory (e.g. '**/*.py') or absolute to search elsewhere. To read an "
            "entire file, use read_file instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for (Python re syntax).",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob filter on files (e.g. '**/*.js').",
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
        # Renvoyer des chemins copiables, relatifs au dossier demandé.
        prefix = (
            "" if rel in (".", "./", "") else rel.replace("\\", "/").rstrip("/") + "/"
        )
        entries: list[str] = []
        items = sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        for p in items:
            if p.name in _SKIP_DIRS:
                continue
            entries.append(prefix + p.name + ("/" if p.is_dir() else ""))
            if len(entries) >= max_entries:
                break
        if not entries:
            return f"(dossier vide : {rel})"
        return "\n".join(entries)

    return ToolSpec(
        name="list_dir",
        description=(
            "Lists a directory (sub-directories suffixed with '/'). Returns "
            "READY-TO-USE paths: copy them as-is into read_file/edit_file. For a "
            "broad pattern search, find_files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory: relative to the working directory (default '.') "
                        "or absolute."
                    ),
                }
            },
            "required": [],
        },
        run=run,
    )
