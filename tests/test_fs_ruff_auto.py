# Palier 1 « LSP » (2026-07-15) : après CHAQUE écriture d'un fichier code,
# write_file/append_file/edit_file collent au résultat d'outil des diagnostics
# NON-MUTANTS plafonnés (.py -> ruff, .js/.ts -> oxlint ; jamais de --fix, jamais
# bloquant). Ces tests figent le contrat : présence du hint sur code cassé, silence
# sur code sain / extension non couverte / style seul, fichier jamais modifié.
from __future__ import annotations

import shutil

import pytest

from loom.tools.fs import (
    _lint_auto_hint,
    make_append_file,
    make_edit_file,
    make_write_file,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="ruff absent du PATH (dépendance Loom)"
)

needs_oxlint = pytest.mark.skipif(
    shutil.which("oxlint") is None, reason="oxlint absent du PATH (binaire optionnel)"
)

BROKEN = "def f(:\n    return 1\n"  # erreur de syntaxe franche (E999)
CLEAN = "def f():\n    return 1\n"
UNDEFINED = "def f():\n    return manquant\n"  # F821 : nom non défini


def test_write_file_py_casse_remonte_ruff(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.py", "content": BROKEN})
    assert out.startswith("écrit : a.py")
    assert "ruff (auto)" in out
    # NON-MUTANT : le fichier reste byte-identique à ce que le modèle a écrit.
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == BROKEN


def test_write_file_py_sain_reste_silencieux(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.py", "content": CLEAN})
    assert "ruff" not in out


def test_write_file_extension_non_couverte_jamais_lintee(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.txt", "content": "def f(:\nfunction g( {"})
    assert "(auto)" not in out


def test_write_file_nom_non_defini_F821(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.py", "content": UNDEFINED})
    assert "F821" in out


def test_style_seul_reste_silencieux(tmp_path):
    # Import inutilisé (F401) = style/chunking en cours : EXCLU de la sélection.
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run(
        {"path": "a.py", "content": "import os\n\n\ndef f():\n    return 1\n"}
    )
    assert "ruff" not in out


def test_append_file_chunk_coupe_remonte_ruff(tmp_path):
    tool = make_append_file(str(tmp_path), max_bytes=1_000_000)
    tool.run({"path": "a.py", "content": CLEAN})
    out = tool.run({"path": "a.py", "content": "def g(:\n"})
    assert "ruff (auto)" in out


def test_edit_file_qui_casse_remonte_ruff(tmp_path):
    (tmp_path / "a.py").write_text(CLEAN, encoding="utf-8")
    tool = make_edit_file(str(tmp_path))
    out = tool.run({"path": "a.py", "old_string": "def f():", "new_string": "def f(:"})
    assert "ruff (auto)" in out


def test_hint_plafonne_a_8_lignes(tmp_path):
    # 12 noms non définis -> 8 lignes affichées + compteur du reste.
    body = "\n".join(f"x{i} = manquant{i}" for i in range(12)) + "\n"
    path = tmp_path / "a.py"
    path.write_text(body, encoding="utf-8")
    hint = _lint_auto_hint(path)
    diags = [ln for ln in hint.splitlines() if "F821" in ln]
    assert len(diags) == 8
    assert "(+4 autres)" in hint


# ---------- oxlint (.js/.ts) : mêmes contrats que ruff ----------

JS_BROKEN = "function f( {\n  return 1\n}\n"  # erreur de syntaxe franche
JS_CLEAN = "export function h(n) {\n  return n + 1\n}\n"
# valid-typeof (catégorie correctness) : bug quasi certain, doit remonter.
JS_TYPEOF = "export const t = typeof window === 'undefned'\n"
# Fonction pas encore appelée (no-unused-vars) : bruit de chunking, doit se taire.
JS_UNUSED = "function future(n) {\n  return n\n}\n"


@needs_oxlint
def test_write_file_js_casse_remonte_oxlint(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.js", "content": JS_BROKEN})
    assert "oxlint (auto)" in out
    # NON-MUTANT : le fichier reste byte-identique à ce que le modèle a écrit.
    assert (tmp_path / "a.js").read_text(encoding="utf-8") == JS_BROKEN


@needs_oxlint
def test_write_file_js_sain_reste_silencieux(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.js", "content": JS_CLEAN})
    assert "oxlint" not in out


@needs_oxlint
def test_write_file_js_correctness_remonte(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.js", "content": JS_TYPEOF})
    assert "valid-typeof" in out


@needs_oxlint
def test_write_file_js_unused_seul_reste_silencieux(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.js", "content": JS_UNUSED})
    assert "oxlint" not in out


@needs_oxlint
def test_edit_file_ts_qui_casse_remonte_oxlint(tmp_path):
    (tmp_path / "a.ts").write_text(JS_CLEAN, encoding="utf-8")
    tool = make_edit_file(str(tmp_path))
    out = tool.run(
        {"path": "a.ts", "old_string": "function h(n)", "new_string": "function h(n"}
    )
    assert "oxlint (auto)" in out
