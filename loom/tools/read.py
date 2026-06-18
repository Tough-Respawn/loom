# loom/tools/read.py
"""Outils de lecture : read_file (texte) et read_document (PDF/Excel/Word -> texte).
READ-only. Acceptent un chemin absolu (n'importe où) ou relatif au dossier de travail."""

from __future__ import annotations

import base64
from pathlib import Path

from loom.agent.inline_image import wrap_image
from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.trust import untrusted

# Images servies au modèle multimodal (mmproj). MIME par extension.
_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _decode_text(data: bytes) -> str | None:
    """Décode des octets en texte en gérant les encodages courants sous Windows.
    Renvoie None si ça ressemble vraiment à du binaire (échec de tout décodage)."""
    if data[:3] == b"\xef\xbb\xbf":  # UTF-8 avec BOM
        return data[3:].decode("utf-8", errors="replace")
    if data[:2] == b"\xff\xfe":  # UTF-16 LE (BOM) — défaut PowerShell
        return data[2:].decode("utf-16-le", errors="replace")
    if data[:2] == b"\xfe\xff":  # UTF-16 BE (BOM)
        return data[2:].decode("utf-16-be", errors="replace")
    try:
        return data.decode("utf-8")  # cas le plus courant
    except UnicodeDecodeError:
        pass
    # UTF-16 sans BOM : beaucoup d'octets nuls (1 octet sur 2 pour de l'ASCII).
    if data and data.count(b"\x00") > len(data) // 4:
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
    try:
        return data.decode("cp1252")  # Windows-1252 (legacy), strict : binaire -> lève
    except UnicodeDecodeError:
        return None


def make_read_file(workspace_dir: str, max_bytes: int) -> ToolSpec:
    """Outil read_file : lit tout fichier TEXTE (taille plafonnée, fenêtrable).

    Plus d'allowlist d'extensions (Loom est généraliste : .env/.log/Dockerfile/CSV…
    sont légitimes) ; le garde anti-binaire de `_decode_text` rejette ce qui n'est pas
    du texte. Symétrique de write_file, qui n'a jamais restreint l'extension."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if path.is_dir():
            raise ToolError(f"'{rel}' est un répertoire, pas un fichier")
        data = path.read_bytes()
        text = _decode_text(data)
        if text is None:
            raise ToolError(f"fichier binaire non lisible : {rel}")
        # Fenêtre de lecture (1-based, optionnelle) : lire un GROS fichier par TRANCHES
        # plutôt que tout d'un coup -> on ne dépasse pas le contexte. Sans start_line on
        # part du début ; on s'arrête à line_count lignes OU au cap de caractères.
        try:
            start = int(args.get("start_line") or 1)
        except (TypeError, ValueError):
            raise ToolError("start_line doit être un entier (1-based)") from None
        if start < 1:
            raise ToolError("start_line doit être >= 1 (1-based)")
        lc = args.get("line_count")
        try:
            count = None if lc in (None, "") else int(lc)
        except (TypeError, ValueError):
            raise ToolError("line_count doit être un entier") from None

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return f"[fichier vide : {rel}]"
        if start > total:
            raise ToolError(f"start_line={start} hors fichier : {rel} a {total} lignes")
        end_limit = total if count is None else min(total, start - 1 + count)
        # Sélection bornée par line_count ET par le cap de caractères (garde-fou contexte).
        selected: list[str] = []
        chars = 0
        i = start - 1
        while i < end_limit and chars <= max_bytes:
            selected.append(lines[i])
            chars += len(lines[i]) + 1
            i += 1
        last = i  # index exclusif -> dernière ligne affichée = i (1-based)
        # NUMÉROS DE LIGNE (format `  12→contenu`), ABSOLUS : le modèle les référence pour
        # se repérer/naviguer. Pour ÉDITER : copie l'extrait EXACT (sans le préfixe
        # `N→`) dans edit_file(old_string).
        width = max(2, len(str(last)))
        numbered = "\n".join(
            f"{n:>{width}}→{line}" for n, line in enumerate(selected, start)
        )
        # Marqueur de fin EXPLICITE : sans lui, le modèle qui voit du code s'arrêter net
        # croit que SA LECTURE a été coupée et relit en boucle. Si tronqué, on indique la
        # COMMANDE pour lire la suite (mécanisme réel, pas un vœu pieux).
        if last >= total:
            footer = (
                f"\n[FIN DU FICHIER — lignes {start}–{total} sur {total}. Pour éditer : "
                "edit_file (copie l'extrait exact à changer). Si le code s'arrête net, "
                "c'est LE FICHIER qui est incomplet (pas ta lecture) : complète-le "
                "(append_file pour la fin) — ne relis pas en boucle.]"
            )
        else:
            footer = (
                f"\n[affiché lignes {start}–{last} sur {total} — le FICHIER est plus long. "
                f"Lis la suite avec read_file(path, start_line={last + 1}), ou cible une "
                "zone précise avec search_text puis read_file(start_line=…).]"
            )
        return numbered + footer

    return ToolSpec(
        name="read_file",
        description=(
            "Lit un fichier texte (avec numéros de ligne) et le renvoie. Chemin relatif "
            "au dossier de travail OU absolu (ex: 'C:/Users/moi/Desktop/notes.txt'). "
            "GROS fichier : lis par TRANCHES avec start_line (et éventuellement "
            "line_count) — le pied de réponse t'indique où continuer. Pour viser une "
            "zone précise, fais d'abord search_text puis read_file(start_line=…)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Chemin du fichier : relatif au dossier de travail ou absolu "
                        "(ex: 'C:/Users/moi/notes.txt')."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": (
                        "Première ligne à lire (1-based, défaut 1). Pour lire la suite "
                        "d'un gros fichier déjà entamé."
                    ),
                },
                "line_count": {
                    "type": "integer",
                    "description": (
                        "Nombre de lignes à lire depuis start_line (défaut : jusqu'à la "
                        "fin ou la limite de taille)."
                    ),
                },
            },
            "required": ["path"],
        },
        run=run,
    )


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    out: list[str] = []
    try:
        for ws in wb.worksheets:
            out.append(f"# Feuille : {ws.title}")
            for row in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                if any(cells):
                    out.append("\t".join(cells))
    finally:
        wb.close()
    return "\n".join(out)


def _read_docx(path: Path) -> str:
    import docx

    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


_DOC_READERS = {
    ".pdf": _read_pdf,
    ".xlsx": _read_xlsx,
    ".xlsm": _read_xlsx,
    ".docx": _read_docx,
}


def make_read_document(workspace_dir: str, max_chars: int = 20000) -> ToolSpec:
    """Outil read_document : extrait le TEXTE d'un PDF / Excel / Word du workspace.

    read_file rendrait du binaire illisible sur ces formats. Lazy-import des libs
    (pypdf/openpyxl/python-docx) : message clair si une lib manque, jamais de crash."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        ext = path.suffix.lower()
        reader = _DOC_READERS.get(ext)
        if reader is None:
            raise ToolError(
                f"format non géré ({ext or 'aucune extension'}) : read_document lit "
                ".pdf/.xlsx/.docx ; pour du texte brut, utilise read_file"
            )
        try:
            text = reader(path).strip()
        except ImportError as exc:
            raise ToolError(f"bibliothèque manquante pour {ext} : {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - document corrompu/protégé
            raise ToolError(f"lecture impossible de {rel} : {exc}") from exc
        if not text:
            return f"(document sans texte extractible : {rel})"
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...[tronqué à {max_chars} caractères]"
        return untrusted(text, f"document {rel}")

    return ToolSpec(
        name="read_document",
        description=(
            "Extrait et renvoie le TEXTE d'un document : PDF (.pdf), Excel (.xlsx) ou "
            "Word (.docx). Chemin relatif au dossier de travail OU absolu. Utilise-le "
            "pour lire/résumer une facture, un tableur, un rapport. Pour un fichier "
            "TEXTE (.txt/.md/.py...), utilise read_file à la place."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Chemin du document (.pdf/.xlsx/.docx) : relatif au dossier de "
                        "travail ou absolu (ex: 'C:/Users/moi/Desktop/facture.pdf')."
                    ),
                }
            },
            "required": ["path"],
        },
        run=run,
    )


def make_read_image(workspace_dir: str, max_bytes: int = 10 * 1024 * 1024) -> ToolSpec:
    """Outil read_image : fait VOIR une image du workspace au modèle multimodal.

    Le serveur sert déjà la vision (mmproj). L'image ne peut pas transiter par un
    message `tool` (texte seul) : l'outil renvoie une chaîne sentinelle (cf.
    loom.inline_image) que la boucle tool-use convertit en message `user` multimodal.
    """
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if path.is_dir():
            raise ToolError(f"'{rel}' est un répertoire, pas une image")
        mime = _IMAGE_MIME.get(path.suffix.lower())
        if mime is None:
            raise ToolError(
                f"format image non géré ({path.suffix or 'aucune extension'}) : "
                "png/jpg/jpeg/gif/webp/bmp"
            )
        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ToolError(
                f"image trop volumineuse ({len(data)} octets > {max_bytes})"
            )
        b64 = base64.b64encode(data).decode("ascii")
        return wrap_image(f"data:{mime};base64,{b64}", rel)

    return ToolSpec(
        name="read_image",
        description=(
            "Charge une IMAGE SUR DISQUE (png/jpg/jpeg/gif/webp/bmp) depuis son CHEMIN "
            "et te la fait VOIR : capture, photo, schéma, diagramme. UNIQUEMENT pour un "
            "fichier déjà présent sur disque dont tu connais le chemin (relatif au dossier "
            "de travail ou absolu). Une image COLLÉE/jointe au chat, tu la vois DÉJÀ : "
            "n'appelle PAS read_image pour elle et ne devine aucun chemin. Sert à décrire, "
            "lire un texte, comparer un rendu. PDF/Excel/Word -> read_document ; texte -> "
            "read_file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Chemin de l'image : relatif au dossier de travail ou absolu "
                        "(ex: 'C:/Users/moi/Desktop/capture.png')."
                    ),
                }
            },
            "required": ["path"],
        },
        run=run,
    )
