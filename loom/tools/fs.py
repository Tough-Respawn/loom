"""Outils d'écriture/édition. Trois outils DÉLIBÉRÉMENT distincts (cf. ADR 0003 : édition par numéro de ligne retirée) :
chacun neutralise une contrainte précise d'un petit modèle sur un contexte étroit.

- write_file   : créer / réécrire un petit fichier (baseline).
- append_file  : contourner l'OVERFLOW — un gros fichier ne tient pas dans une
                 réponse (plafond max_tokens) sans être tronqué ; on écrit par morceaux.
- edit_file    : édition chirurgicale par texte exact (old_string copié du fichier) — l'éditeur des blocs existants.

Ne PAS consolider par goût du minimalisme : moins d'outils = chaque outil restant
plus dur à piloter, ce qu'un 4B gère le moins bien. On ajoute un outil quand il
retire un mode d'échec qu'aucun autre ne couvre proprement.

Chemin absolu (écrire n'importe où) ou relatif au dossier de travail. Écriture
ATOMIQUE (fichier .tmp + os.replace, comme `Conversation.save`) pour ne jamais
laisser de fichier partiel. Encodage utf-8, `newline=''` afin de préserver le
contenu byte-exact (pas de traduction \\n -> \\r\\n sous Windows).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from loom.permissions import is_protected_write_path
from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.read import _decode_text_enc

# Après écriture, un lint non mutant remonte seulement les erreurs certaines.
# Il reste best-effort et isolé de la configuration du projet cible.
_LINT_PY_EXTS = frozenset({".py", ".pyi"})
_LINT_JS_EXTS = frozenset({".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"})
_LINT_MAX_LINES = 8  # budget contexte : plafond dur sur ce qu'on injecte

# Écarter les résumés pour ne garder que les diagnostics actionnables.
_LINT_NOISE = re.compile(r"^(Found \d+|\[\*\]|\d+ problems?\b)")


def _lint_diags(argv: list[str]) -> list[str]:
    """Lignes de diagnostic d'un linter, ou [] (sain, absent, timeout, rc inattendu).
    Contrat commun ruff/oxlint : rc 0 = sain, rc 1 = diagnostics, autre = silence."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 1:
        return []
    return [
        ln
        for ln in (proc.stdout or "").splitlines()
        if ln.strip() and not _LINT_NOISE.match(ln)
    ]


def _lint_auto_hint(path: Path) -> str:
    """Diagnostics plafonnés pour `path`, ou '' (extension non couverte, fichier
    sain, linter indisponible)."""
    ext = path.suffix.lower()
    if ext in _LINT_PY_EXTS:
        tool = shutil.which("ruff")
        if not tool:
            return ""
        label = "ruff"
        argv = [
            tool,
            "check",
            "--no-fix",
            "--isolated",
            "--select",
            "E9,F63,F7,F82",
            "--output-format",
            "concise",
            str(path),
        ]
    elif ext in _LINT_JS_EXTS:
        tool = shutil.which("oxlint")
        if not tool:
            return ""
        label = "oxlint"
        # Les variables inutilisées sont du bruit pendant une écriture par morceaux.
        argv = [
            tool,
            "-f",
            "unix",
            "-D",
            "correctness",
            "-A",
            "no-unused-vars",
            str(path),
        ]
    else:
        return ""
    lines = _lint_diags(argv)
    if not lines:
        return ""
    extra = len(lines) - _LINT_MAX_LINES
    shown = "\n".join(lines[:_LINT_MAX_LINES])
    more = f"\n… (+{extra} autres)" if extra > 0 else ""
    return f"\n{label} (auto) — erreurs à corriger :\n{shown}{more}"


def _guard_write_path(path: Path) -> None:
    """Barrière de sécurité DURE (indépendante de l'UI, comme la deny-list de run_shell) :
    refuse l'écriture sur un chemin système ou un dossier de secrets, MÊME en mode 'allow'.
    Le modèle reçoit une erreur d'outil exploitable, pas une bulle de confirmation."""
    if is_protected_write_path(str(path), []):
        raise ToolError(
            "chemin protégé par la politique de sécurité (dossier système ou secrets) "
            "— écriture refusée. Vise un autre emplacement."
        )


def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Écrit `content` de façon atomique (tmp + os.replace). `encoding` par défaut utf-8 ;
    edit_file le passe à l'encodage d'origine du fichier (utf-16/cp1252) pour ré-écrire
    à l'identique et ne pas casser un fichier PowerShell (UTF-16)."""
    _guard_write_path(path)  # choke point : write_file/edit/replace/insert passent ici
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding=encoding, newline="") as fh:
        fh.write(content)
    os.replace(tmp, path)


def make_write_file(
    workspace_dir: str, max_bytes: int, max_tokens: int = 8192
) -> ToolSpec:
    """Outil write_file borné au workspace, taille plafonnée, écriture atomique."""
    root = Path(workspace_dir)
    # Dériver la taille du budget évite qu'un appel JSON soit tronqué avant exécution.
    one_shot_cap = int(max_tokens * 1.75)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        content = args.get("content")
        if content is None:
            raise ToolError("argument 'content' manquant")
        if len(content.encode("utf-8")) > max_bytes:
            raise ToolError(f"contenu trop volumineux (> {max_bytes} octets)")
        if len(content) > one_shot_cap:
            raise ToolError(
                f"fichier trop gros pour un seul write_file ({len(content)} car. > "
                f"{one_shot_cap}) : l'appel serait probablement tronqué par la limite de "
                "tokens. Écris le SQUELETTE ici (imports + la 1re fonction/composant COMPLET), "
                "puis ajoute chaque unité suivante via append_file — une fonction / un composant "
                "ENTIER par appel, jamais coupé au milieu."
            )
        path = _resolve_in_root(root, rel)
        _atomic_write(path, content)
        return f"écrit : {rel} ({len(content)} caractères)" + _lint_auto_hint(path)

    return ToolSpec(
        name="write_file",
        description=(
            "Creates or overwrites a file with the provided content (new file, or "
            "complete rewrite of a SMALL one). LARGE file (>~150 lines — the output "
            "would truncate): write the skeleton here, then append_file one COMPLETE "
            "logical unit per call (whole function/component), never cut mid-unit."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": ("Relative to the working directory, or absolute."),
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write.",
                },
            },
            "required": ["path", "content"],
        },
        run=run,
    )


def make_append_file(
    workspace_dir: str, max_bytes: int, max_tokens: int = 8192
) -> ToolSpec:
    """Outil append_file : AJOUTE du contenu à la fin d'un fichier (le crée si absent).

    Clé du « chunking » : un gros fichier dont le contenu entier ne tient pas dans la
    limite de tokens d'UNE réponse (sinon l'appel d'outil est tronqué -> JSON cassé) est
    écrit en PLUSIEURS petits appels. Pas d'écriture atomique (mode append) : on accumule.
    """
    root = Path(workspace_dir)
    one_shot_cap = int(max_tokens * 1.75)  # un morceau = une unité logique, pas un dump

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        content = args.get("content")
        if content is None:
            raise ToolError("argument 'content' manquant")
        if len(content.encode("utf-8")) > max_bytes:
            raise ToolError(
                f"morceau trop volumineux (> {max_bytes} octets) — découpe en plus petit"
            )
        if len(content) > one_shot_cap:
            raise ToolError(
                f"morceau trop gros ({len(content)} car. > {one_shot_cap}) : append UNE unité "
                "logique complète à la fois (une fonction / un composant), pas plus."
            )
        path = _resolve_in_root(root, rel)
        _guard_write_path(path)  # append n'utilise pas _atomic_write : garde explicite
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="") as fh:
            fh.write(content)
        # Chaque morceau doit laisser un fichier syntaxiquement valide.
        return f"ajouté : {rel} (+{len(content)} caractères)" + _lint_auto_hint(path)

    return ToolSpec(
        name="append_file",
        description=(
            "Appends content to the END of a file (creates it if missing). The way to "
            "build a LARGE file without hitting the output limit: write_file the "
            "start, then append in small chunks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": ("Relative to the working directory, or absolute."),
                },
                "content": {
                    "type": "string",
                    "description": "Chunk to append.",
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
    sur ambiguïté, indice CRLF sur 'introuvable' (erreurs exploitables par un petit modèle)."""
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
        # Réécrire avec l'encodage détecté garde éditables les fichiers UTF-16 et cp1252.
        raw, _enc = _decode_text_enc(path.read_bytes())
        if raw is None:
            raise ToolError(
                f"fichier binaire non éditable : {rel} (aucun encodage texte détecté)"
            )
        # Chercher en LF puis restaurer le style original rend l'édition indépendante de l'OS.
        is_crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        old_string = old_string.replace("\r\n", "\n")
        new_string = new_string.replace("\r\n", "\n")
        count = text.count(old_string)
        if count == 0:
            # Signaler les numéros décoratifs copiés depuis `read_file`.
            prefix_hint = ""
            if re.search(r"(?m)^\s*\d+→", old_string):
                prefix_hint = (
                    " ATTENTION : ton old_string contient le préfixe « N→ » de "
                    "l'affichage read_file — RETIRE-le (numéros de ligne et flèche ne "
                    "sont PAS dans le fichier)."
                )
            raise ToolError(
                f"old_string introuvable dans {rel} — copie l'extrait EXACT du fichier "
                "(indentation et espaces au caractère près). Pour AJOUTER du code à la FIN, "
                f"utilise append_file, pas edit_file.{prefix_hint}"
            )
        if count > 1 and not replace_all:
            locs = ", ".join(str(n) for n in _occurrence_lines(text, old_string)[:12])
            raise ToolError(
                f"old_string ambigu : {count} occurrences (lignes {locs}) dans {rel}. "
                "Ajoute du contexte pour rendre old_string unique, OU passe replace_all=true."
            )
        result = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        if is_crlf:
            result = result.replace("\n", "\r\n")
        _atomic_write(path, result, encoding=_enc)
        msg = (
            f"modifié : {rel} ({count} occurrence(s))"
            if replace_all
            else f"modifié : {rel}"
        )
        # Relire seulement les changements assez grands pour justifier ce coût.
        if "\n" in old_string or "\n" in new_string or len(new_string) > 200:
            line = _occurrence_lines(text, old_string)[0]
            msg += (
                f" — édition multi-ligne : relis la zone modifiée (read_file, "
                f"start_line {line}) pour VÉRIFIER avant d'affirmer que c'est bon."
            )
        return msg + _lint_auto_hint(path)

    return ToolSpec(
        name="edit_file",
        description=(
            "Edits an existing file by replacing old_string with new_string. Read the "
            "file first and copy the EXACT snippet (whitespace character-for-character) "
            "into old_string; it must be UNIQUE (on ambiguity the error lists the "
            "lines — add context, or replace_all=true). Large rewrite -> write_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": ("Relative to the working directory, or absolute."),
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace (unique, unless replace_all).",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace ALL identical occurrences (default: false).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        run=run,
    )
