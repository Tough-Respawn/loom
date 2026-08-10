"""ST-05 — PROTOTYPES d'intelligence de code, HORS PRODUCTION (banc A/B seulement).

Deux capacités expérimentales, jamais enregistrées dans le registre par défaut :

- code_outline(path)          : symboles et plages, SANS corps complet —
  Python via ast (plages exactes, signatures), JS/TS via motifs (ligne de début).
- code_diagnostics(path)      : diagnostics STRUCTURÉS d'un « serveur » déjà
  installé sur la machine (ruff pour Python, oxlint pour JS) — normalisés
  (fichier, ligne, colonne, sévérité, code, message), triés, bornés.

Contraintes de la story respectées ici :
- AUCUN téléchargement, AUCUN démarrage silencieux : uniquement des binaires
  déjà présents (shutil.which), lancés en one-shot, jamais en démon ;
- absence ou crash du serveur -> erreur ACTIONNABLE (fail-soft), jamais de boucle ;
- commandes INJECTABLES (py_cmd/js_cmd) pour les tests hermétiques à faux serveur.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root

_MAX_DIAGS = 50
_MAX_OUTLINE = 200

_JS_EXTS = {".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"}


# --------------------------------- outline -----------------------------------


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


_JS_SYMBOL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?"
    r"(?:(?:async\s+)?function\s+(?P<fn>\w+)"
    r"|class\s+(?P<cls>\w+)"
    r"|(?:const|let|var)\s+(?P<var>\w+)\s*=\s*(?:async\s*)?(?:\(|function\b|\w+\s*=>))"
)


def _outline_js(text: str, rel: str) -> list[str]:
    # Prototype : motifs de déclaration, LIGNE DE DÉBUT seulement (pas de plage
    # de fin sans vrai parseur — assumé et documenté pour le banc).
    out: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = _JS_SYMBOL.match(line)
        if not m:
            continue
        name = m.group("fn") or m.group("cls") or m.group("var")
        kind = "class" if m.group("cls") else "function"
        out.append(f"{kind} {name}  [ligne {i}]")
    return out


def make_code_outline(workspace_dir: str) -> ToolSpec:
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        p = _resolve_in_root(root, rel)
        if not p.is_file():
            raise ToolError(f"fichier introuvable : {rel}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix == ".py":
            lines = _outline_py(text, rel)
        elif p.suffix in _JS_EXTS:
            lines = _outline_js(text, rel)
        else:
            raise ToolError(
                f"extension non couverte : {p.suffix} (Python et JS/TS seulement)"
            )
        if not lines:
            return f"{rel} : aucun symbole de niveau module détecté."
        extra = len(lines) - _MAX_OUTLINE
        shown = "\n".join(lines[:_MAX_OUTLINE])
        note = f"\n… (+{extra} symboles)" if extra > 0 else ""
        total = len(text.splitlines())
        return f"{rel} ({total} lignes) — symboles :\n{shown}{note}"

    return ToolSpec(
        name="code_outline",
        description=(
            "Structural view of a source file: symbols (classes, functions, "
            "methods) with their line ranges, WITHOUT the bodies. Much cheaper "
            "than read_file for large files — use it to locate a symbol, then "
            "read only the range you need. Python and JS/TS."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Source file (relative to workspace, or absolute).",
                }
            },
            "required": ["path"],
        },
        run=run,
    )


# ------------------------------- diagnostics ----------------------------------


def _sev(code: str) -> str:
    # E9 = erreurs de syntaxe/parse ; F821/F823 = noms indéfinis -> error.
    if code.startswith("E9") or code in ("F821", "F823", "F822"):
        return "error"
    return "warning"


def _run_cmd(argv: list[str], timeout: int = 20):
    return subprocess.run(
        argv, capture_output=True, timeout=timeout, encoding="utf-8", errors="replace"
    )


def _py_diags(target: Path, py_cmd) -> list[dict]:
    if not py_cmd:
        raise ToolError(
            "aucun serveur de diagnostics Python disponible (ruff introuvable) — "
            "installe ruff ou lis/execute le fichier pour vérifier autrement."
        )
    proc = _run_cmd(
        [
            *py_cmd,
            "check",
            "--no-fix",
            "--isolated",
            "--select",
            "E9,F",
            "--output-format",
            "json",
            str(target),
        ]
    )
    if proc.returncode not in (0, 1):
        head = (proc.stderr or proc.stdout or "").strip().split("\n")[0][:160]
        raise ToolError(f"le serveur de diagnostics Python a échoué : {head}")
    try:
        raw = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        raise ToolError(
            "sortie du serveur de diagnostics Python illisible (JSON attendu)."
        ) from None
    return [
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


_UNIX_DIAG = re.compile(r"^(?P<f>.+?):(?P<l>\d+):(?P<c>\d+):\s*(?P<m>.+)$")


def _js_diags(target: Path, js_cmd) -> list[dict]:
    if not js_cmd:
        raise ToolError(
            "aucun serveur de diagnostics JS disponible (oxlint introuvable) — "
            "installe oxlint ou vérifie le fichier autrement (check_page, exécution)."
        )
    proc = _run_cmd([*js_cmd, "-f", "unix", "-D", "correctness", str(target)])
    if proc.returncode not in (0, 1):
        head = (proc.stderr or proc.stdout or "").strip().split("\n")[0][:160]
        raise ToolError(f"le serveur de diagnostics JS a échoué : {head}")
    out: list[dict] = []
    for line in (proc.stdout or "").splitlines():
        m = _UNIX_DIAG.match(line.strip())
        if not m:
            continue
        msg = m.group("m")
        code = ""
        if "[" in msg and msg.endswith("]"):
            msg, _, tail = msg.rpartition("[")
            code = tail.rstrip("]").strip()
        out.append(
            {
                "file": m.group("f"),
                "line": int(m.group("l")),
                "col": int(m.group("c")),
                "severity": "error" if "error" in msg.lower() else "warning",
                "code": code,
                "message": msg.strip().rstrip(":").strip(),
            }
        )
    return out


def make_code_diagnostics(
    workspace_dir: str, py_cmd=None, js_cmd=None, which=shutil.which
) -> ToolSpec:
    """`py_cmd`/`js_cmd` : argv de base injectables (tests hermétiques à faux
    serveur). Par défaut : ruff/oxlint DÉJÀ installés (which), jamais téléchargés."""
    root = Path(workspace_dir)
    if py_cmd is None:
        ruff = which("ruff")
        py_cmd = [ruff] if ruff else None
    if js_cmd is None:
        ox = which("oxlint")
        js_cmd = [ox] if ox else None

    def run(args: dict) -> str:
        rel = (args.get("path") or ".").strip() or "."
        severity = (args.get("severity") or "").strip().lower()
        p = _resolve_in_root(root, rel)
        if not p.exists():
            raise ToolError(f"chemin introuvable : {rel}")
        targets_py = (
            [p]
            if p.is_file() and p.suffix == ".py"
            else list(p.rglob("*.py"))
            if p.is_dir()
            else []
        )
        targets_js = (
            [p]
            if p.is_file() and p.suffix in _JS_EXTS
            else [f for ext in _JS_EXTS for f in p.rglob(f"*{ext}")]
            if p.is_dir()
            else []
        )
        if not targets_py and not targets_js:
            raise ToolError(
                f"aucun fichier Python ou JS/TS sous : {rel} (extensions couvertes : "
                ".py, " + ", ".join(sorted(_JS_EXTS)) + ")"
            )
        diags: list[dict] = []
        if targets_py:
            diags += _py_diags(p, py_cmd)
        if targets_js:
            for f in targets_js if p.is_dir() else [p]:
                diags += _js_diags(f, js_cmd)
        if severity in ("error", "warning"):
            diags = [d for d in diags if d["severity"] == severity]
        diags.sort(key=lambda d: (d["file"], d["line"], d["col"]))
        if not diags:
            return f"aucun diagnostic sous {rel} : propre."
        extra = len(diags) - _MAX_DIAGS
        shown = diags[:_MAX_DIAGS]
        lines = [
            f"{d['file']}:{d['line']}:{d['col']} [{d['severity']}"
            + (f" {d['code']}" if d["code"] else "")
            + f"] {d['message']}"
            for d in shown
        ]
        note = f"\n… (+{extra} diagnostics)" if extra > 0 else ""
        return f"{len(diags)} diagnostic(s) sous {rel} :\n" + "\n".join(lines) + note

    return ToolSpec(
        name="code_diagnostics",
        description=(
            "Structured diagnostics (file, line, column, severity, code, message) "
            "for a file or directory, from locally installed analyzers (ruff for "
            "Python, oxlint for JS/TS). Sorted and bounded; optional severity "
            "filter. Use it to FIND type/import/name errors and to PROVE they are "
            "gone after a fix."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory (default: workspace root).",
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
    )
