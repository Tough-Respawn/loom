# loom/tools/format.py
"""Outil format_code : reformate proprement un fichier selon son langage.

Le 4B produit souvent du code mal indenté / non conforme (PEP8 en Python, alignement en
JS/HTML/CSS). Plutôt que de lui faire deviner la bonne commande `ruff`/`prettier` et ses
unix-ismes via run_shell, cet outil DÉDIÉ encapsule la bonne invocation cross-OS :
- Python (.py/.pyi)  -> ruff (via `uvx ruff`) : `check --fix` (corrige les soucis PEP8
  SÛRS : imports inutiles, tri, etc.) puis `format` (style/indentation).
- Web (.js/.ts/.jsx/.tsx/.json/.html/.css/.scss/.md/.yaml/.vue) -> prettier (via `npx`).

Reformate le fichier EN PLACE et renvoie ce qui a changé + les problèmes restants (lint
non corrigeable automatiquement, erreur de syntaxe repérée par prettier) — un retour
actionnable que le modèle corrige. Si le formateur n'est pas installé : message
ACTIONNABLE (comme check_page pour playwright), jamais une exception opaque.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root

EXT_RUFF = frozenset({".py", ".pyi"})
EXT_PRETTIER = frozenset(
    {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".json",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".less",
        ".md",
        ".markdown",
        ".yaml",
        ".yml",
        ".vue",
    }
)

_RUFF_HINT = (
    "ruff introuvable. Il est normalement livré avec Loom (`uv sync`) ; sinon "
    "`uv tool install ruff` (ou assure-toi que `ruff`/`uvx` est sur le PATH)."
)
_PRETTIER_HINT = (
    "prettier introuvable (hors-ligne). Installe-le une fois : `npm install -g prettier` "
    "(node requis), puis relance format_code."
)


def _run(argv: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def _clean(*parts: str) -> str:
    """Concatène stdout+stderr en supprimant le bruit vide."""
    return "\n".join(p.strip() for p in parts if p and p.strip())


def make_format_code(workspace_dir: str) -> ToolSpec:
    """Outil format_code borné au workspace pour les chemins relatifs (absolus acceptés)."""
    root = Path(workspace_dir)

    def _format_python(path: Path) -> str:
        # ruff est une dépendance de Loom (présent dans le venv) : on l'utilise EN PRIORITÉ
        # (hors-ligne, version pinnée). Repli sur `uvx ruff` si le binaire n'est pas sur le PATH.
        ruff = shutil.which("ruff")
        if ruff:
            base = [ruff]
        else:
            uvx = shutil.which("uvx")
            if not uvx:
                raise ToolError(_RUFF_HINT)
            base = [uvx, "ruff"]
        try:
            # 1) corrige les soucis PEP8 SÛRS (imports inutiles, tri…), 2) style.
            fix = _run([*base, "check", "--fix", str(path)], root, 60)
            fmt = _run([*base, "format", str(path)], root, 60)
        except subprocess.TimeoutExpired as exc:
            raise ToolError("ruff : délai dépassé (>60s)") from exc
        fix_out = _clean(fix.stdout, fix.stderr)
        fmt_out = _clean(fmt.stdout, fmt.stderr)
        lines = [f"format_code (ruff) : {path.name}"]
        if fmt_out:
            lines.append(fmt_out)
        # ruff check renvoie non-zéro s'il reste des soucis NON auto-corrigeables :
        # info utile (le modèle doit les corriger), pas un échec de l'outil.
        if fix.returncode != 0 and fix_out:
            lines.append("lint restant (à corriger) :")
            lines.append(fix_out)
        elif fix_out:
            lines.append(fix_out)
        return "\n".join(lines)

    def _format_web(path: Path) -> str:
        npx = shutil.which("npx")
        if not npx:
            raise ToolError(_PRETTIER_HINT)
        try:
            # --no-install : on REFUSE le download (Loom est offline) ; si prettier n'est
            # pas déjà présent, npx échoue et on renvoie le hint d'installation.
            proc = _run(
                [npx, "--no-install", "prettier", "--write", str(path)], root, 90
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError("prettier : délai dépassé (>90s)") from exc
        out = _clean(proc.stdout, proc.stderr)
        low = out.lower()
        if proc.returncode != 0:
            if (
                "could not determine executable" in low
                or "missing packages" in low
                or "not found" in low
                or "npx canceled" in low
            ):
                raise ToolError(_PRETTIER_HINT)
            # prettier signale aussi les ERREURS DE SYNTAXE (ligne/colonne) : retour
            # actionnable -> convention "erreur:" pour que la boucle marque l'échec.
            return f"erreur: prettier a échoué sur {path.name}.\n{out}"
        return _clean(f"format_code (prettier) : {path.name}", out)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant (chemin du fichier à formater)")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if not path.is_file():
            raise ToolError(
                f"'{rel}' n'est pas un fichier (format_code vise un fichier)"
            )
        ext = path.suffix.lower()
        if ext in EXT_RUFF:
            return _format_python(path)
        if ext in EXT_PRETTIER:
            return _format_web(path)
        raise ToolError(
            f"extension '{ext or '(aucune)'}' non gérée par format_code. "
            "Python (.py) via ruff ; web (.js/.ts/.jsx/.tsx/.json/.html/.css/.scss/"
            ".md/.yaml/.vue) via prettier."
        )

    return ToolSpec(
        name="format_code",
        description=(
            "Reformate proprement un fichier de code EN PLACE selon son langage : Python "
            "(.py) avec ruff (corrige les soucis PEP8 sûrs + indentation/style), web "
            "(.js/.ts/.jsx/.tsx/.json/.html/.css/.scss/.md/.yaml/.vue) avec prettier. "
            "Appelle-le APRÈS avoir écrit ou édité un fichier de code pour qu'il soit "
            "proprement indenté et conforme, au lieu de soigner l'alignement à la main. "
            "Renvoie ce qui a changé et les éventuels problèmes restants à corriger."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Chemin du fichier à reformater (relatif au dossier de travail "
                        "ou absolu)."
                    ),
                }
            },
            "required": ["path"],
        },
        run=run,
    )
