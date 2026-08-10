"""Outils d'intelligence de code — PYTHON UNIQUEMENT (ST-06, périmètre validé
par le banc A/B ST-05 : fix_prove 1/3 -> 3/3, detect_py -62 % de tokens ; le
volet JS/TS a été REFUSÉ, ne pas l'ajouter ici).

- code_outline(path)      : symboles et plages d'un .py via ast, SANS les corps.
- code_diagnostics(path)  : diagnostics structurés de ruff (déjà installé),
  normalisés (fichier, ligne, colonne, sévérité, code, message), triés, bornés.

Contrat produit (story ST-06) :
- LECTURE SEULE (classés READ_TOOLS dans loom.permissions — le refus silencieux
  d'un outil hors catégories a été VÉCU en ST-05, 17/17 appels perdus) ;
- DIFFÉRÉS (always_deferred) : jamais dans le préfixe, chargés via tool_search —
  le prompt et les outils par défaut ne bougent pas tant qu'ils ne sont pas
  chargés ;
- aucun téléchargement, aucun démon, pas de LSP/DAP/code action : ruff est un
  binaire local lancé en one-shot ; absent ou en crash -> ToolError actionnable,
  la session continue.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root

_MAX_DIAGS = 50
_MAX_OUTLINE = 200


def _outline_py(text: str, rel: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ToolError(
            f"{rel} : erreur de syntaxe ligne {exc.lineno} — l'outline exige un "
            "fichier parsable ; corrige d'abord (ou lis le fichier directement)."
        ) from None
    lines: list[str] = []

    def _sig(node) -> str:
        try:
            return ast.unparse(node.args)
        except Exception:  # noqa: BLE001 - la signature est un confort
            return "…"

    def _walk(nodes, indent: str) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                lines.append(
                    f"{indent}class {node.name}  [{node.lineno}-{node.end_lineno}]"
                )
                _walk(node.body, indent + "  ")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                lines.append(
                    f"{indent}{kind} {node.name}({_sig(node)})  "
                    f"[{node.lineno}-{node.end_lineno}]"
                )

    _walk(tree.body, "")
    return lines


def make_code_outline(workspace_dir: str) -> ToolSpec:
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        p = _resolve_in_root(root, rel)
        if not p.is_file():
            raise ToolError(f"fichier introuvable : {rel}")
        if p.suffix != ".py":
            raise ToolError(
                f"extension non couverte : {p.suffix or '(aucune)'} — code_outline "
                "ne couvre que Python (.py). Pour un autre langage : read_file."
            )
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = _outline_py(text, rel)
        if not lines:
            return f"{rel} : aucun symbole de niveau module."
        extra = len(lines) - _MAX_OUTLINE
        shown = "\n".join(lines[:_MAX_OUTLINE])
        note = f"\n… (+{extra} symboles)" if extra > 0 else ""
        return f"{rel} ({len(text.splitlines())} lignes) — symboles :\n{shown}{note}"

    return ToolSpec(
        name="code_outline",
        description=(
            "Structural view of a PYTHON file: classes, functions and methods "
            "with their exact line ranges, WITHOUT the bodies. Much cheaper than "
            "read_file on a large file — locate the symbol here, then read only "
            "the range you need. Read-only. Python (.py) only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Python file (relative to workspace, or absolute).",
                }
            },
            "required": ["path"],
        },
        run=run,
        deferred=True,
        always_deferred=True,
    )


def _sev(code: str) -> str:
    # E9 = erreurs de syntaxe/parse ; F821/822/823 = noms indéfinis -> error.
    if code.startswith("E9") or code in ("F821", "F822", "F823"):
        return "error"
    return "warning"


def make_code_diagnostics(
    workspace_dir: str, ruff_cmd=None, which=shutil.which
) -> ToolSpec:
    """`ruff_cmd` : argv injectable (tests hermétiques à faux serveur). Par
    défaut : le ruff DÉJÀ installé (which) — jamais téléchargé, jamais en démon."""
    root = Path(workspace_dir)
    if ruff_cmd is None:
        ruff = which("ruff")
        ruff_cmd = [ruff] if ruff else None

    def run(args: dict) -> str:
        rel = (args.get("path") or ".").strip() or "."
        severity = (args.get("severity") or "").strip().lower()
        p = _resolve_in_root(root, rel)
        if not p.exists():
            raise ToolError(f"chemin introuvable : {rel}")
        if p.is_file() and p.suffix != ".py":
            raise ToolError(
                f"extension non couverte : {p.suffix or '(aucune)'} — "
                "code_diagnostics ne couvre que Python (.py)."
            )
        if p.is_dir() and not any(p.rglob("*.py")):
            raise ToolError(f"aucun fichier Python sous : {rel}")
        if not ruff_cmd:
            raise ToolError(
                "aucun serveur de diagnostics Python disponible (ruff introuvable) "
                "— installe ruff, ou vérifie autrement (run_shell python -m "
                "py_compile, exécution réelle)."
            )
        try:
            proc = subprocess.run(
                [
                    *ruff_cmd,
                    "check",
                    "--no-fix",
                    "--isolated",
                    "--select",
                    "E9,F",
                    "--output-format",
                    "json",
                    str(p),
                ],
                capture_output=True,
                timeout=20,
                encoding="utf-8",
                errors="replace",
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise ToolError(
                f"le serveur de diagnostics Python n'a pas répondu : {exc}"
            ) from None
        if proc.returncode not in (0, 1):
            head = (proc.stderr or proc.stdout or "").strip().split("\n")[0][:160]
            raise ToolError(f"le serveur de diagnostics Python a échoué : {head}")
        try:
            raw = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            raise ToolError(
                "sortie du serveur de diagnostics Python illisible (JSON attendu)."
            ) from None
        diags = [
            {
                "file": d.get("filename", ""),
                "line": (d.get("location") or {}).get("row", 0),
                "col": (d.get("location") or {}).get("column", 0),
                "severity": _sev(str(d.get("code") or "")),
                "code": str(d.get("code") or ""),
                "message": str(d.get("message") or ""),
            }
            for d in raw
        ]
        if severity in ("error", "warning"):
            diags = [d for d in diags if d["severity"] == severity]
        diags.sort(key=lambda d: (d["file"], d["line"], d["col"]))
        if not diags:
            return f"aucun diagnostic sous {rel} : propre."
        extra = len(diags) - _MAX_DIAGS
        lines = [
            f"{d['file']}:{d['line']}:{d['col']} [{d['severity']}"
            + (f" {d['code']}" if d["code"] else "")
            + f"] {d['message']}"
            for d in diags[:_MAX_DIAGS]
        ]
        note = f"\n… (+{extra} diagnostics)" if extra > 0 else ""
        return f"{len(diags)} diagnostic(s) sous {rel} :\n" + "\n".join(lines) + note

    return ToolSpec(
        name="code_diagnostics",
        description=(
            "Structured PYTHON diagnostics (file, line, column, severity, code, "
            "message) for a file or directory, from the locally installed ruff "
            "(syntax errors, undefined names, import problems). Sorted, bounded, "
            "optional severity filter. Read-only. Use it to FIND errors and to "
            "PROVE they are gone after a fix. Python (.py) only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Python file or directory (default: workspace).",
                },
                "severity": {
                    "type": "string",
                    "enum": ["error", "warning"],
                    "description": "Optional filter.",
                },
            },
            "required": ["path"],
        },
        run=run,
        deferred=True,
        always_deferred=True,
    )
