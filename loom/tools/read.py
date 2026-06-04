# loom/tools/read.py
"""Outils de lecture : read_file (texte) et read_document (PDF/Excel/Word -> texte).
READ-only, bornés au workspace."""

from __future__ import annotations

from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.trust import untrusted


def make_read_file(
    workspace_dir: str, extensions: list[str], max_bytes: int
) -> ToolSpec:
    """Outil read_file borné au workspace, extensions autorisées, taille plafonnée."""
    root = Path(workspace_dir)
    allowed = {e.lower() for e in extensions}

    def run(args: dict) -> str:
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant")
        path = _resolve_in_root(root, rel)
        if not path.exists():
            raise ToolError(f"fichier introuvable : {rel}")
        if path.is_dir():
            raise ToolError(f"'{rel}' est un répertoire, pas un fichier")
        if allowed and path.suffix.lower() not in allowed:
            raise ToolError(f"extension non autorisée : {path.suffix or '(aucune)'}")
        data = path.read_bytes()
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"fichier binaire non lisible : {rel}") from exc
        if truncated:
            text += f"\n...[tronqué à {max_bytes} octets]"
        return text

    return ToolSpec(
        name="read_file",
        description=(
            "Lit le contenu d'un fichier texte du workspace et le renvoie. "
            "Utilise un chemin relatif au workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Chemin du fichier, relatif au workspace.",
                }
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
            "Extrait et renvoie le TEXTE d'un document du workspace : PDF (.pdf), "
            "Excel (.xlsx) ou Word (.docx). Utilise-le pour lire/résumer une facture, "
            "un tableur, un rapport. Pour un fichier TEXTE (.txt/.md/.py...), utilise "
            "read_file à la place."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Chemin du document (.pdf/.xlsx/.docx), relatif au workspace.",
                }
            },
            "required": ["path"],
        },
        run=run,
    )
