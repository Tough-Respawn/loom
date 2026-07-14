# loom/tools/fs.py
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

# ---- Diagnostics auto après écriture Python (« palier 1 LSP ») -------------------
# Le modèle n'appelle format_code que s'il y pense (et le 4B confabule l'exécution) :
# on déplace le check de l'espace prompt vers l'espace harnais. Après CHAQUE écriture
# d'un .py, un `ruff check` NON-MUTANT (jamais --fix : ne pas invalider les lignes que
# le modèle vient de viser) colle les erreurs au résultat d'outil. Jamais bloquant :
# ruff absent / timeout / rc inattendu -> silence, l'écriture reste acquise.
_LINT_EXTS = frozenset({".py", ".pyi"})
# E9/F63/F7/F82 = syntaxe, comparaisons cassées, return/yield hors fonction, noms non
# définis : haute confiance uniquement. PAS F401/F841 (imports/variables inutilisés) :
# sur un squelette en cours de chunking ils pousseraient le modèle à « corriger » des
# imports dont le chunk suivant a besoin.
_LINT_SELECT = "E9,F63,F7,F82"
_LINT_MAX_LINES = 8  # budget contexte : plafond dur sur ce qu'on injecte


def _ruff_auto_hint(path: Path) -> str:
    """Diagnostics ruff plafonnés pour `path`, ou '' (pas un .py, fichier sain, ruff
    indisponible). `--isolated` : le workspace = de vrais projets, leur config ruff ne
    doit pas changer le comportement du harnais."""
    if path.suffix.lower() not in _LINT_EXTS:
        return ""
    ruff = shutil.which("ruff")
    if not ruff:
        return ""
    try:
        proc = subprocess.run(
            [
                ruff,
                "check",
                "--no-fix",
                "--isolated",
                "--select",
                _LINT_SELECT,
                "--output-format",
                "concise",
                str(path),
            ],
            capture_output=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    # ruff : 0 = sain, 1 = diagnostics ; tout autre code (config cassée, IO) = silence.
    if proc.returncode != 1:
        return ""
    lines = [
        ln
        for ln in (proc.stdout or "").splitlines()
        if ln.strip() and not ln.startswith(("Found ", "[*]"))
    ]
    if not lines:
        return ""
    extra = len(lines) - _LINT_MAX_LINES
    shown = "\n".join(lines[:_LINT_MAX_LINES])
    more = f"\n… (+{extra} autres)" if extra > 0 else ""
    return f"\nruff (auto) — erreurs à corriger :\n{shown}{more}"


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
    # Plafond « un seul write » DÉRIVÉ du budget de sortie (max_tokens), pas codé en dur : un
    # contenu plus gros risque d'être tronqué par la limite de tokens AVANT d'arriver ici (JSON
    # d'appel cassé). Au-delà -> on impose le découpage par UNITÉ LOGIQUE. ~1.75 car./token (la
    # moitié du budget réservée au raisonnement).
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
        return f"écrit : {rel} ({len(content)} caractères)" + _ruff_auto_hint(path)

    return ToolSpec(
        name="write_file",
        description=(
            "Creates or overwrites a file with the provided content. Path relative to the "
            "working directory OR absolute (e.g. 'C:/Users/me/Desktop/out.txt'). New "
            "file, or COMPLETE rewrite of a SMALL file. LARGE file (>~150 lines, which "
            "would exceed the token limit and be truncated): write the skeleton "
            "(imports + 1st unit) here, then complete it with append_file, one COMPLETE "
            "LOGICAL UNIT per call (whole function/component), never cut in the middle."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path: relative to the working directory or absolute."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file.",
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
        # Une erreur de syntaxe ICI = chunk coupé au milieu d'une unité logique
        # (la doctrine du chunking veut que le fichier parse après chaque append).
        return f"ajouté : {rel} (+{len(content)} caractères)" + _ruff_auto_hint(path)

    return ToolSpec(
        name="append_file",
        description=(
            "APPENDS content to the END of a file (creates it if it does not exist). Used to "
            "write a LARGE file WITHOUT exceeding the token limit: write_file for the "
            "beginning, then append_file several times for the rest, in SMALL chunks. "
            "Path relative to the working directory or absolute."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path: relative to the working directory or absolute."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Chunk of content to append to the end of the file.",
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
        # Décodage MULTI-ENCODAGE (comme read_file) : un fichier UTF-16 (défaut
        # PowerShell) ou cp1252 se lit et s'édite, on mémorise l'encodage pour ré-écrire
        # à l'identique. Avant, un `decode("utf-8")` sec déclarait « binaire » un fichier
        # que read_file venait pourtant de lire — impasse pour le modèle.
        raw, _enc = _decode_text_enc(path.read_bytes())
        if raw is None:
            raise ToolError(
                f"fichier binaire non éditable : {rel} (aucun encodage texte détecté)"
            )
        # Matching AGNOSTIQUE aux fins de ligne. read_file montre du LF (splitlines) -> le
        # modèle copie un old_string en LF, alors que le fichier sur disque est souvent CRLF
        # (Windows). AVANT, `text.count(old_string)` échouait sur tout extrait multi-ligne
        # d'un fichier CRLF. On normalise TOUT en LF pour chercher/remplacer, puis on
        # ré-applique le style du fichier (CRLF) à l'écriture -> l'édition marche quel que
        # soit le style, et le fichier garde ses fins de ligne d'origine.
        is_crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        old_string = old_string.replace("\r\n", "\n")
        new_string = new_string.replace("\r\n", "\n")
        count = text.count(old_string)
        if count == 0:
            # Piège fréquent : le modèle copie l'affichage de read_file, qui préfixe chaque
            # ligne d'un « N→ » (numéro de ligne + flèche) ABSENT du fichier réel.
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
        # Read-back CONDITIONNEL (édits à risque seulement : multi-ligne ou gros bloc).
        # Systématique, il gonflerait tours/tokens sur chaque petit édit ; ciblé, il
        # coupe le cycle « edit -> affirme -> faux » sans re-lire tout le fichier.
        if "\n" in old_string or "\n" in new_string or len(new_string) > 200:
            line = _occurrence_lines(text, old_string)[0]
            msg += (
                f" — édition multi-ligne : relis la zone modifiée (read_file, "
                f"start_line {line}) pour VÉRIFIER avant d'affirmer que c'est bon."
            )
        return msg + _ruff_auto_hint(path)

    return ToolSpec(
        name="edit_file",
        description=(
            "Edits an existing file by replacing old_string with new_string (path "
            "relative to the working directory or absolute). YOUR surgical editor: read the "
            "file (read_file), copy the EXACT snippet to change into old_string "
            "(indentation and whitespace character-for-character), replacement in new_string. "
            "old_string must be UNIQUE (otherwise the error lists the lines — add "
            "context, or replace_all=true for all occurrences). To rewrite "
            "a large portion, use write_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path: relative to the working directory or absolute."
                    ),
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
