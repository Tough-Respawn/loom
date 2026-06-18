# loom/tools/indent.py
"""Aides à l'indentation pour les éditions par numéro de ligne (replace_lines). Fonctions PURES (aucune I/O).

But : une édition par plage de lignes ne doit jamais transformer un fichier Python qui
compilait en fichier cassé pour cause d'indentation, et l'erreur courante « bloc collé
au mauvais indent de base » (typiquement colonne 0) est corrigée en recollant le bloc à
l'indentation de son contexte. Les autres langages (séparateurs ; / {}) ne sont pas
validés : l'indentation y est cosmétique.
"""

from __future__ import annotations

import re
import textwrap

_LEADING_WS = re.compile(r"^[ \t]*")
_PY_SUFFIXES = frozenset({".py", ".pyi", ".pyw"})


def is_python(suffix: str) -> bool:
    """Vrai si l'extension désigne un fichier Python (indentation = syntaxe)."""
    return suffix.lower() in _PY_SUFFIXES


def indent_of(line: str) -> str:
    """Whitespace de tête (espaces/tabs) d'une ligne ('' si aucune)."""
    return _LEADING_WS.match(line).group(0)


def indent_unit(lines: list[str]) -> str:
    """Unité d'indentation déduite du fichier : un tab si le fichier en utilise, sinon
    le plus PETIT niveau d'indentation en espaces rencontré ; défaut 4 espaces."""
    widths: list[int] = []
    uses_tab = False
    for line in lines:
        if not line.strip():
            continue
        ws = indent_of(line)
        if "\t" in ws:
            uses_tab = True
        elif ws:
            widths.append(len(ws))
    if uses_tab:
        return "\t"
    if widths:
        return " " * min(widths)
    return "    "


def snap_indent(content: str, target: str) -> str:
    """Recolle l'indentation d'un bloc sur `target`, en PRÉSERVANT l'indentation relative
    interne. Idempotent si le bloc est déjà bien indenté. Opère en '\\n' (l'appelant
    réapplique la fin de ligne du fichier)."""
    body = content.replace("\r\n", "\n").replace("\r", "\n")
    dedented = textwrap.dedent(body)
    out = [(target + line if line.strip() else "") for line in dedented.split("\n")]
    return "\n".join(out)


def target_indent(lines: list[str], anchor_idx: int, suffix: str) -> str:
    """Indentation (string réelle, tab-aware) à appliquer à un bloc placé à `anchor_idx`
    (index 0-based dans `lines` de la ligne qui occupera la place du bloc : pour replace,
    la 1re ligne remplacée ; pour insert, la ligne qui SUIT l'insertion)."""
    n = len(lines)
    cur = lines[anchor_idx] if 0 <= anchor_idx < n else ""
    prev = ""
    i = anchor_idx - 1
    while i >= 0:
        if lines[i].strip():
            prev = lines[i]
            break
        i -= 1
    if is_python(suffix) and prev.strip().endswith(":"):
        return indent_of(prev) + indent_unit(lines)
    if cur.strip():
        return indent_of(cur)
    if prev.strip():
        return indent_of(prev)
    return ""


def py_compiles(text: str) -> bool:
    """Vrai si `text` compile comme module Python (aucune SyntaxError)."""
    try:
        compile(text, "<edit>", "exec")
        return True
    except SyntaxError:
        return False


def indent_error(text: str) -> str | None:
    """Message court si `text` échoue à compiler avec une erreur D'INDENTATION
    (IndentationError/TabError) ; None sinon (y compris pour une SyntaxError non liée à
    l'indentation)."""
    try:
        compile(text, "<edit>", "exec")
        return None
    except (IndentationError, TabError) as exc:
        return f"{type(exc).__name__} ligne {exc.lineno}: {exc.msg}"
    except SyntaxError:
        return None
