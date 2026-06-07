# replace_lines / insert_lines indent-safe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make line-addressed edits (`replace_lines`, `insert_lines`) auto-correct a block's base indentation and never turn a Python file that compiled into one broken by indentation.

**Architecture:** Pure helpers in a new `loom/tools/indent.py` (snap-to-context indentation + Python compile checks). `replace_lines`/`insert_lines` in `fs.py` snap the model's block to the surrounding indent, then, for Python files only, run a *differential* check: if the edit introduces an `IndentationError`/`TabError` into a file that previously compiled, the write is refused (no change) with an actionable message; otherwise it proceeds (incremental construction is never blocked).

**Tech Stack:** Python 3, stdlib only (`textwrap`, `re`, `compile`). Verification by inline smoke scripts (`uv run python -c …`) — the project keeps **no pytest suite** (rule: Loom pas de tests).

---

## File Structure

- **Create** `loom/tools/indent.py` — pure, I/O-free indentation helpers + Python compile checks. Single responsibility, smoke-testable in isolation.
- **Modify** `loom/tools/fs.py` — extract a `_render_context(text, lo, hi)` helper from `_context_after_edit`, then wire snap + differential validation into `make_replace_lines` and `make_insert_lines`.

Conventions to follow (existing in `fs.py`): atomic writes via `_atomic_write`, line endings preserved via `_lines_and_nl`/`_new_block`, actionable French `ToolError`/`"erreur: …"` messages, the `_context_after_edit` re-numbered tail after edits.

---

## Task 1: Pure indentation helpers (`loom/tools/indent.py`)

**Files:**
- Create: `loom/tools/indent.py`

- [ ] **Step 1: Write the failing smoke**

Run this — it MUST fail (module does not exist yet):

```bash
uv run python -c "
from loom.tools.indent import is_python, indent_of, indent_unit, snap_indent, target_indent, py_compiles, indent_error
print('import OK')
"
```
Expected: `ModuleNotFoundError: No module named 'loom.tools.indent'`

- [ ] **Step 2: Create the module**

Create `loom/tools/indent.py` with exactly:

```python
# loom/tools/indent.py
"""Aides à l'indentation pour les éditions par numéro de ligne (replace_lines /
insert_lines). Fonctions PURES (aucune I/O).

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
```

- [ ] **Step 3: Run the import smoke (now passes)**

```bash
uv run python -c "
from loom.tools.indent import is_python, indent_of, indent_unit, snap_indent, target_indent, py_compiles, indent_error
print('import OK')
"
```
Expected: `import OK`

- [ ] **Step 4: Run the behaviour smoke**

```bash
uv run python -c "
from loom.tools.indent import snap_indent, target_indent, indent_error, py_compiles, indent_unit, is_python
# snap: a col-0 block snaps to a 8-space context, relative structure kept
blk = 'if x:\n        y = 1'
print('snap1:', repr(snap_indent(blk, '        ')))
assert snap_indent(blk, '        ') == '        if x:\n                y = 1'
# idempotent
assert snap_indent('        if x:\n                y = 1', '        ') == '        if x:\n                y = 1'
# target after a ':' opener -> +1 unit (4 spaces here)
lines = ['def f():\n', '    pass\n']
print('target:', repr(target_indent(lines, 1, '.py')))
assert target_indent(lines, 1, '.py') == '    '
# indent_unit detects 4 spaces
assert indent_unit(['a\n', '    b\n', '        c\n']) == '    '
# indent_error: a dedented body is flagged; a clean text is not
assert indent_error('def f():\npass\n') is not None
assert indent_error('def f():\n    pass\n') is None
# a NON-indentation syntax error returns None (not our guarantee)
assert indent_error('def f(:\n    pass\n') is None
assert is_python('.PY') and not is_python('.js')
print('behaviour OK')
"
```
Expected: ends with `behaviour OK` (and `snap1: '        if x:\n                y = 1'`).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check loom/tools/indent.py && uv run ruff format loom/tools/indent.py
git add loom/tools/indent.py
git commit -m "feat(edition): helpers indent.py (snap + checks compile Python)"
```

---

## Task 2: Extract `_render_context(text, lo, hi)` in `fs.py`

Reason: the rollback path needs to render the re-numbered context from an **in-memory** text (the unchanged BEFORE state), not by re-reading the file. Extract the rendering from `_context_after_edit`.

**Files:**
- Modify: `loom/tools/fs.py:272-293` (`_context_after_edit`)

- [ ] **Step 1: Replace `_context_after_edit` with a renderer + thin wrapper**

Replace the whole existing `_context_after_edit` function (currently fs.py:272-293) with:

```python
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
```

- [ ] **Step 2: Smoke — existing behaviour preserved**

```bash
uv run python -c "
from pathlib import Path
import tempfile, os
from loom.tools.fs import _render_context, _context_after_edit
t = 'a\nb\nc\nd\ne\n'
out = _render_context(t, 2, 3)
print(out)
assert '2→b' in out and '3→c' in out
p = Path(tempfile.mkdtemp())/'f.txt'; p.write_text(t, encoding='utf-8')
assert '2→b' in _context_after_edit(p, 2, 3)
print('render OK')
"
```
Expected: ends with `render OK`.

- [ ] **Step 3: Commit**

```bash
git add loom/tools/fs.py
git commit -m "refactor(edition): extrait _render_context (rendu contexte depuis texte)"
```

---

## Task 3: Wire snap + differential validation into `make_replace_lines`

**Files:**
- Modify: `loom/tools/fs.py` (imports near top; body of `make_replace_lines.run`, currently fs.py:303-336)

- [ ] **Step 1: Add the import**

At the top of `fs.py`, just after the existing line `from loom.tools.base import ToolError, ToolSpec, _resolve_in_root`, add:

```python
from loom.tools.indent import (
    indent_error,
    is_python,
    py_compiles,
    snap_indent,
    target_indent,
)
```

- [ ] **Step 2: Replace the tail of `make_replace_lines.run`**

In `make_replace_lines.run`, replace this existing block (currently fs.py:330-336):

```python
        block = _new_block(content, nl)
        _atomic_write(path, "".join(lines[: start - 1]) + block + "".join(lines[end:]))
        added = 0 if content == "" else content.count("\n") + 1
        head = f"remplacé : {rel} lignes {start}-{end} ({end - start + 1} → {added} lignes)"
        # Bornes de la zone éditée dans le NOUVEAU fichier (start .. start-1+added).
        new_hi = start if added == 0 else start - 1 + added
        return head + _context_after_edit(path, start, new_hi)
```

with:

```python
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
        # erreur d'indentation dans un fichier qui compilait. On ne bloque QUE ce cas
        # (les états intermédiaires non-compilables d'une construction restent permis).
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
        new_hi = start if added == 0 else start - 1 + added
        tail = _context_after_edit(path, start, new_hi)
        if is_python(suffix) and not py_compiles(new_text):
            tail += "\nnote: le fichier ne compile pas encore — poursuis tes edits."
        return head + tail
```

- [ ] **Step 3: Smoke — the four spec cases**

```bash
uv run python -c "
import tempfile
from pathlib import Path
from loom.tools.fs import make_replace_lines
d = Path(tempfile.mkdtemp())
rl = make_replace_lines(str(d)).run

# Case 1: col-0 block snapped to method indent, file compiles
f = d/'a.py'
f.write_text('class C:\n    def m(self):\n        self.x = 1\n        self.y = 2\n', encoding='utf-8')
r = rl({'path':'a.py','start_line':3,'end_line':3,'content':'if True:\n    self.x = 1'})
print('C1:', r.splitlines()[0])
import ast; ast.parse(f.read_text(encoding='utf-8')); 
assert 'ré-indenté' in r
print('C1 compiles OK')

# Case 2: edit that breaks indentation of a VALID file -> rollback (file unchanged).
# The content keeps a RELATIVE over-indent (2nd line deeper, no opener) that survives
# snap (snap preserves relative structure), so the result is a real IndentationError.
g = d/'b.py'
orig = 'def f():\n    a = 1\n    b = 2\n'
g.write_text(orig, encoding='utf-8')
r = rl({'path':'b.py','start_line':2,'end_line':2,'content':'a = 1\n  b = 2'})
print('C2:', r.splitlines()[0])
assert g.read_text(encoding='utf-8') == orig, 'file must be unchanged on indent regression'
assert 'erreur' in r and 'INCHANGÉ' in r
print('C2 rollback OK')

# Case 3: file ALREADY broken -> edit allowed + warning
h = d/'c.py'
h.write_text('def f(:\n    pass\n', encoding='utf-8')  # already invalid
r = rl({'path':'c.py','start_line':2,'end_line':2,'content':'    return 1'})
print('C3:', r.splitlines()[0])
assert 'remplacé' in r
print('C3 not blocked OK')

# Case 4: non-Python file -> snap applied, no validation
j = d/'d.js'
j.write_text('function f() {\n  let a = 1;\n}\n', encoding='utf-8')
r = rl({'path':'d.js','start_line':2,'end_line':2,'content':'let a = 2;'})
print('C4:', r.splitlines()[0])
assert 'remplacé' in r
print('C4 js OK')
print('ALL REPLACE SMOKES OK')
"
```
Expected: ends with `ALL REPLACE SMOKES OK` (C1 compiles, C2 rollback leaves file unchanged, C3 allowed, C4 js written).

- [ ] **Step 4: Lint + commit**

```bash
uv run ruff check loom/tools/fs.py && uv run ruff format loom/tools/fs.py
git add loom/tools/fs.py
git commit -m "feat(edition): replace_lines indent-safe (snap + rollback indent Python)"
```

---

## Task 4: Wire the same into `make_insert_lines`

**Files:**
- Modify: `loom/tools/fs.py` (body of `make_insert_lines.run`, currently fs.py:397-413)

- [ ] **Step 1: Replace the tail of `make_insert_lines.run`**

In `make_insert_lines.run`, replace this existing block (currently fs.py:403-413):

```python
        head = lines[:after]
        # si la dernière ligne gardée n'a pas de fin de ligne, l'ajouter (sinon collage)
        if head and not head[-1].endswith(("\n", "\r")):
            head = head[:-1] + [head[-1] + nl]
        _atomic_write(
            path, "".join(head) + _new_block(content, nl) + "".join(lines[after:])
        )
        k = content.count("\n") + 1
        msg = f"inséré : {rel} après ligne {after} (+{k} lignes)"
        # La zone insérée occupe les lignes [after+1 .. after+k] dans le nouveau fichier.
        return msg + _context_after_edit(path, after + 1, after + k)
```

with:

```python
        before_text = "".join(lines)
        suffix = path.suffix
        # Ancre = la ligne qui SUIVRA le bloc inséré (lines[after], 0-based), sinon la
        # précédente : target_indent gère le repli + le cas « après un ':' ».
        target = target_indent(lines, after, suffix)
        new_content = snap_indent(content, target)
        snapped = new_content != content.replace("\r\n", "\n").replace("\r", "\n")
        content = new_content
        head_lines = lines[:after]
        if head_lines and not head_lines[-1].endswith(("\n", "\r")):
            head_lines = head_lines[:-1] + [head_lines[-1] + nl]
        new_text = "".join(head_lines) + _new_block(content, nl) + "".join(lines[after:])
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
        tail = _context_after_edit(path, after + 1, after + k)
        if is_python(suffix) and not py_compiles(new_text):
            tail += "\nnote: le fichier ne compile pas encore — poursuis tes edits."
        return msg + tail
```

- [ ] **Step 2: Smoke — insert snap + rollback**

```bash
uv run python -c "
import tempfile
from pathlib import Path
from loom.tools.fs import make_insert_lines
d = Path(tempfile.mkdtemp())
il = make_insert_lines(str(d)).run

# Insert a col-0 body right after a 'def f():' opener -> snapped to body indent, compiles
f = d/'a.py'
f.write_text('def f():\n    return 0\n', encoding='utf-8')
r = il({'path':'a.py','after_line':1,'content':'x = 1\ny = 2'})
print('I1:', r.splitlines()[0])
import ast; ast.parse(f.read_text(encoding='utf-8'))
assert 'ré-indenté' in r
print('I1 compiles OK')

# Insert that breaks indentation of a valid file -> rollback.
# A dangling opener ('if x:') with no indented body following -> IndentationError.
g = d/'b.py'
orig = 'a = 1\nb = 2\n'
g.write_text(orig, encoding='utf-8')
r = il({'path':'b.py','after_line':1,'content':'if x:'})
print('I2:', r.splitlines()[0])
assert g.read_text(encoding='utf-8') == orig
assert 'erreur' in r
print('I2 rollback OK')
print('ALL INSERT SMOKES OK')
"
```
Expected: ends with `ALL INSERT SMOKES OK`.

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check loom/tools/fs.py && uv run ruff format loom/tools/fs.py
git add loom/tools/fs.py
git commit -m "feat(edition): insert_lines indent-safe (meme snap + validation)"
```

---

## Task 5: Live verification (manual)

**Files:** none (runtime check).

- [ ] **Step 1: Regression smoke of all five edit tools**

```bash
uv run python -c "
from loom.tools import build_registry
reg = build_registry('.', 40000, ['write_file','append_file','edit_file','replace_lines','insert_lines'])
print('registry tools:', sorted(t['function']['name'] for t in reg.openai_tools()))
"
```
Expected: lists the five edit tools (no import error).

- [ ] **Step 2: Live run (with the user)**

Drive Loom (Gemma) to build/fix a small Python file via `replace_lines` and confirm in `sessions/<id>/debug.log` that snapped edits compile and that a genuine indent-break returns the rollback message instead of corrupting the file. (User launches the stack; this step is observational — no code change.)

- [ ] **Step 3: Final note**

No further commit. The feature ships as Tasks 1–4.

---

## Self-Review

- **Spec coverage:** §4.1 snap_indent → Task 1. §4.2 indent_unit/target_indent → Task 1. §4.3 differential validation (rollback only on indent regression of a compiling file; warn otherwise; non-Python skipped) → Tasks 3 & 4. §4.4 transparency (re-indent note + `_context_after_edit`) → Tasks 2–4. §5 integration into both tools → Tasks 3 & 4. §7 verification by smokes → each task's smoke + Task 5. All covered.
- **Placeholders:** none — every step has full code/commands.
- **Type consistency:** helper names (`is_python`, `indent_of`, `indent_unit`, `snap_indent`, `target_indent`, `py_compiles`, `indent_error`, `_render_context`) used identically across Tasks 1–4. `target_indent(lines, anchor_idx, suffix)` signature matches every call (`start-1` for replace, `after` for insert).
- **Note on Case 2 smoke:** uses a guaranteed indent-breaker (`def f():` indented under nothing at line 1) so the assertion is deterministic regardless of snap behaviour.
