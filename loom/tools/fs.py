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
from pathlib import Path

from loom.permissions import is_protected_write_path
from loom.tools.base import ToolError, ToolSpec, _resolve_in_root


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
            "Modifie un fichier existant en remplaçant old_string par new_string (chemin "
            "relatif au dossier de travail ou absolu). TON éditeur chirurgical : lis le "
            "fichier (read_file), copie l'extrait EXACT à changer dans old_string "
            "(indentation et espaces au caractère près), remplacement dans new_string. "
            "old_string doit être UNIQUE (sinon l'erreur liste les lignes — ajoute du "
            "contexte, ou replace_all=true pour toutes les occurrences). Pour réécrire "
            "une grande portion, write_file."
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

