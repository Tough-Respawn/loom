# loom/tools/fs.py
"""Outils d'écriture/édition. Cinq outils DÉLIBÉRÉMENT distincts (cf. ADR 0002) :
chacun neutralise une contrainte précise d'un petit modèle sur un contexte étroit.

- write_file   : créer / réécrire un petit fichier (baseline).
- append_file  : contourner l'OVERFLOW — un gros fichier ne tient pas dans une
                 réponse (plafond max_tokens) sans être tronqué ; on écrit par morceaux.
- replace_lines: contourner la RECOPIE EXACTE — un 4B ne recopie pas un bloc au
                 caractère près ; on adresse par NUMÉROS de ligne (read_file). Outil
                 d'édition principal pour un bloc au milieu.
- insert_lines : ajouter au MILIEU sans rien remplacer (adressage par ligne).
- edit_file    : petit remplacement UNIQUE par string exacte (un nom, un token).

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
from pathlib import Path

from loom.permissions import is_protected_write_path
from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.indent import (
    indent_error,
    is_python,
    py_compiles,
    snap_indent,
    target_indent,
)


def _guard_write_path(path: Path) -> None:
    """Barrière de sécurité DURE (indépendante de l'UI, comme la deny-list de run_shell) :
    refuse l'écriture sur un chemin système ou un dossier de secrets, MÊME en mode 'allow'.
    Le modèle reçoit une erreur d'outil exploitable, pas une bulle de confirmation."""
    if is_protected_write_path(str(path), []):
        raise ToolError(
            "chemin protégé par la politique de sécurité (dossier système ou secrets) "
            "— écriture refusée. Vise un autre emplacement."
        )


def _atomic_write(path: Path, content: str) -> None:
    """Écrit `content` en utf-8 de façon atomique (tmp + os.replace)."""
    _guard_write_path(path)  # choke point : write_file/edit/replace/insert passent ici
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
            "Crée ou écrase un fichier avec le contenu fourni. Chemin relatif au "
            "dossier de travail OU absolu (ex: 'C:/Users/moi/Desktop/out.txt'). Pour un "
            "GROS fichier (qui dépasserait la limite de tokens d'une réponse) : écris le "
            "DÉBUT ici, puis complète par petits morceaux avec append_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Chemin du fichier : relatif au dossier de travail ou absolu."
                    ),
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


def make_append_file(workspace_dir: str, max_bytes: int) -> ToolSpec:
    """Outil append_file : AJOUTE du contenu à la fin d'un fichier (le crée si absent).

    Clé du « chunking » : un gros fichier dont le contenu entier ne tient pas dans la
    limite de tokens d'UNE réponse (sinon l'appel d'outil est tronqué -> JSON cassé) est
    écrit en PLUSIEURS petits appels. Pas d'écriture atomique (mode append) : on accumule.
    """
    root = Path(workspace_dir)

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
        path = _resolve_in_root(root, rel)
        _guard_write_path(path)  # append n'utilise pas _atomic_write : garde explicite
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="") as fh:
            fh.write(content)
        return f"ajouté : {rel} (+{len(content)} caractères)"

    return ToolSpec(
        name="append_file",
        description=(
            "AJOUTE du contenu à la FIN d'un fichier (le crée s'il n'existe pas). Sert à "
            "écrire un GROS fichier SANS dépasser la limite de tokens : write_file pour le "
            "début, puis append_file plusieurs fois pour la suite, par PETITS morceaux. "
            "Chemin relatif au dossier de travail ou absolu."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Chemin du fichier : relatif au dossier de travail ou absolu."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Morceau de contenu à ajouter à la fin du fichier.",
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
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"fichier binaire non éditable : {rel}") from exc
        count = text.count(old_string)
        if count == 0:
            crlf = old_string.replace("\n", "\r\n")
            if crlf != old_string and crlf in text:
                hint = " — le fichier est en CRLF, mets des \\r\\n dans old_string"
            else:
                hint = (
                    " — old_string doit être COPIÉ TEL QUEL du fichier (indentation et "
                    "espaces EXACTS). Si tu veux AJOUTER du code à la FIN (compléter un "
                    "fichier), n'utilise PAS edit_file : utilise append_file."
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
            "Remplace old_string par new_string dans un fichier (chemin relatif au "
            "dossier de travail ou absolu). Réserve-le aux PETITS remplacements UNIQUES "
            "recopiables au caractère près (un nom, un token). Si tu connais les NUMÉROS "
            "de ligne (tu viens de lire le fichier) ou si le bloc est long/indenté, "
            "utilise replace_lines (pas de recopie exacte à risque). Par défaut "
            "old_string doit être UNIQUE (sinon l'erreur liste les lignes) ; "
            "replace_all=true remplace TOUTES les occurrences identiques."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Chemin du fichier : relatif au dossier de travail ou absolu."
                    ),
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


def _lines_and_nl(path: Path, rel: str) -> tuple[list[str], str]:
    """(lignes avec fins préservées, style de fin de ligne). Lève si binaire. Le découpage
    matche la numérotation de read_file (1-based) : len == nb de lignes affichées."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"fichier binaire non éditable : {rel}") from exc
    nl = "\r\n" if "\r\n" in text else "\n"
    return text.splitlines(keepends=True), nl


def _new_block(content: str, nl: str) -> str:
    """Normalise un bloc de remplacement aux fins de ligne du fichier, terminé par nl."""
    if content == "":
        return ""
    body = nl.join(content.replace("\r\n", "\n").split("\n"))
    return body if body.endswith(nl) else body + nl


def _render_context(text: str, lo: int, hi: int, pad: int = 4, note: str = "") -> str:
    """Rend les lignes [lo-pad .. hi+pad] de `text` avec leurs NUMÉROS (style read_file).
    `note` : en-tête personnalisé (sinon : message « état à jour, réutilise ces numéros »)."""
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return ""
    a = max(1, lo - pad)
    b = min(n, hi + pad)
    width = max(2, len(str(b)))
    body = "\n".join(f"{i:>{width}}→{lines[i - 1]}" for i in range(a, b + 1))
    head = note or (
        f"État À JOUR autour de l'édition (lignes {a}-{b} sur {n}, numéros corrects — "
        f"réutilise-les directement, ne refais pas de read_file) :"
    )
    return f"\n{head}\n{body}"


def _context_after_edit(path: Path, lo: int, hi: int, pad: int = 4) -> str:
    """Relit le fichier APRÈS écriture et rend les lignes [lo-pad .. hi+pad] re-numérotées
    (anti-thrash : le modèle enchaîne l'édition suivante sans refaire de read_file)."""
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return _render_context(text, lo, hi, pad)


def make_replace_lines(workspace_dir: str) -> ToolSpec:
    """Outil replace_lines : remplace les lignes [start..end] (1-based, incluses) par
    `content`. Édition robuste pour petit modèle : il référence les NUMÉROS de ligne (vus
    via read_file) au lieu de recopier le texte exact (raté sur l'indentation), et n'écrit
    que le bloc modifié (pas tout le fichier -> pas d'overflow)."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        try:
            start = int(args.get("start_line"))
            end = int(args.get("end_line"))
        except (TypeError, ValueError):
            raise ToolError(
                "'start_line' et 'end_line' doivent être des entiers (1-based)"
            ) from None
        content = args.get("content")
        if content is None:
            raise ToolError("argument 'content' manquant ('' = supprimer les lignes)")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if path.is_dir():
            raise ToolError(f"'{rel}' est un répertoire, pas un fichier")
        lines, nl = _lines_and_nl(path, rel)
        n = len(lines)
        if start < 1 or start > end:
            raise ToolError(
                f"plage invalide : start_line={start}, end_line={end} (1 ≤ start ≤ end)"
            )
        if end > n:
            raise ToolError(f"end_line={end} hors fichier : {rel} a {n} lignes")
        before_text = "".join(lines)
        suffix = path.suffix
        snapped = False
        if content != "":
            target = target_indent(lines, start - 1, suffix)
            new_content = snap_indent(content, target)
            snapped = new_content != content.replace("\r\n", "\n").replace("\r", "\n")
            content = new_content
        block = _new_block(content, nl)
        new_text = "".join(lines[: start - 1]) + block + "".join(lines[end:])
        # Validation DIFFÉRENTIELLE (Python) : on n'écrit pas si l'édition introduit une
        # erreur d'indentation dans un fichier qui compilait. On ne bloque QUE ce cas (les
        # états intermédiaires non-compilables d'une construction restent permis).
        if is_python(suffix) and py_compiles(before_text):
            err = indent_error(new_text)
            if err:
                return (
                    f"erreur: ton bloc casse l'indentation ({err}) — {rel} n'a PAS été "
                    "modifié. Réémets le bloc avec la bonne indentation (mêmes niveaux "
                    "que le code autour)."
                    + _render_context(
                        before_text,
                        start,
                        end,
                        note=f"État actuel (INCHANGÉ) de {rel} autour de la zone visée :",
                    )
                )
        _atomic_write(path, new_text)
        added = 0 if content == "" else content.count("\n") + 1
        head = f"remplacé : {rel} lignes {start}-{end} ({end - start + 1} → {added} lignes)"
        if snapped:
            head += " (bloc ré-indenté pour coller au contexte)"
        # Bornes de la zone éditée dans le NOUVEAU fichier (start .. start-1+added).
        new_hi = start if added == 0 else start - 1 + added
        tail = _context_after_edit(path, start, new_hi)
        if is_python(suffix) and not py_compiles(new_text):
            tail += "\nnote: le fichier ne compile pas encore — poursuis tes edits."
        return head + tail

    return ToolSpec(
        name="replace_lines",
        description=(
            "Remplace une PLAGE de lignes [start_line..end_line] (numéros vus dans "
            "read_file, 1-based, bornes INCLUSES) par `content`. LE bon outil pour "
            "corriger/remplacer un bloc au MILIEU d'un fichier : pas besoin de recopier "
            "l'ancien texte (juste ses numéros), et tu n'écris que le nouveau bloc (pas "
            "tout le fichier). content='' supprime les lignes. N'inclus PAS les préfixes "
            "'N→' de read_file dans le contenu."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Chemin du fichier (relatif au dossier de travail ou absolu).",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1re ligne à remplacer (1-based, incluse).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Dernière ligne à remplacer (incluse).",
                },
                "content": {
                    "type": "string",
                    "description": "Nouveau contenu des lignes. '' pour les supprimer.",
                },
            },
            "required": ["path", "start_line", "end_line", "content"],
        },
        run=run,
    )


def make_insert_lines(workspace_dir: str) -> ToolSpec:
    """Outil insert_lines : insère `content` APRÈS la ligne `after_line` (0 = au début).
    Pour AJOUTER du code au milieu sans rien remplacer."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        try:
            after = int(args.get("after_line"))
        except (TypeError, ValueError):
            raise ToolError(
                "'after_line' doit être un entier (0 = au tout début)"
            ) from None
        content = args.get("content")
        if not content:
            raise ToolError("argument 'content' manquant")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if path.is_dir():
            raise ToolError(f"'{rel}' est un répertoire, pas un fichier")
        lines, nl = _lines_and_nl(path, rel)
        n = len(lines)
        if after < 0 or after > n:
            raise ToolError(
                f"after_line={after} hors fichier : {rel} a {n} lignes (0..{n})"
            )
        before_text = "".join(lines)
        suffix = path.suffix
        # Ancre = la ligne qui SUIVRA le bloc inséré (lines[after], 0-based), sinon la
        # précédente : target_indent gère le repli + le cas « après un ':' ».
        target = target_indent(lines, after, suffix)
        new_content = snap_indent(content, target)
        snapped = new_content != content.replace("\r\n", "\n").replace("\r", "\n")
        content = new_content
        head = lines[:after]
        # si la dernière ligne gardée n'a pas de fin de ligne, l'ajouter (sinon collage)
        if head and not head[-1].endswith(("\n", "\r")):
            head = head[:-1] + [head[-1] + nl]
        new_text = "".join(head) + _new_block(content, nl) + "".join(lines[after:])
        if is_python(suffix) and py_compiles(before_text):
            err = indent_error(new_text)
            if err:
                return (
                    f"erreur: ton insertion casse l'indentation ({err}) — {rel} n'a PAS "
                    "été modifié. Réémets avec la bonne indentation."
                    + _render_context(
                        before_text,
                        after,
                        after + 1,
                        note=f"État actuel (INCHANGÉ) de {rel} au point d'insertion :",
                    )
                )
        _atomic_write(path, new_text)
        k = content.count("\n") + 1
        msg = f"inséré : {rel} après ligne {after} (+{k} lignes)"
        if snapped:
            msg += " (bloc ré-indenté pour coller au contexte)"
        # La zone insérée occupe les lignes [after+1 .. after+k] dans le nouveau fichier.
        tail = _context_after_edit(path, after + 1, after + k)
        if is_python(suffix) and not py_compiles(new_text):
            tail += "\nnote: le fichier ne compile pas encore — poursuis tes edits."
        return msg + tail

    return ToolSpec(
        name="insert_lines",
        description=(
            "Insère `content` APRÈS la ligne `after_line` (numéros vus dans read_file ; "
            "0 = tout au début). Pour AJOUTER du code au MILIEU d'un fichier sans rien "
            "remplacer. Pour ajouter à la toute fin, append_file est plus simple."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Chemin du fichier (relatif au dossier de travail ou absolu).",
                },
                "after_line": {
                    "type": "integer",
                    "description": "Insère après cette ligne (0 = au tout début).",
                },
                "content": {
                    "type": "string",
                    "description": "Contenu à insérer (lignes).",
                },
            },
            "required": ["path", "after_line", "content"],
        },
        run=run,
    )
