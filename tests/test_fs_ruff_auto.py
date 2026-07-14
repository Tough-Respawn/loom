# Palier 1 « LSP » (2026-07-15) : après CHAQUE écriture d'un .py, write_file/append_file/
# edit_file collent au résultat d'outil les diagnostics `ruff check` NON-MUTANTS (syntaxe,
# noms non définis — jamais de --fix, jamais bloquant, plafonné à 8 lignes). Ces tests
# figent le contrat : présence du hint sur code cassé, silence sur code sain et non-.py,
# fichier jamais modifié par le check.
from __future__ import annotations

import shutil

import pytest

from loom.tools.fs import (
    _ruff_auto_hint,
    make_append_file,
    make_edit_file,
    make_write_file,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="ruff absent du PATH (dépendance Loom)"
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


def test_write_file_non_py_jamais_linte(tmp_path):
    tool = make_write_file(str(tmp_path), max_bytes=1_000_000)
    out = tool.run({"path": "a.js", "content": "function f( {"})
    assert "ruff" not in out


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
    hint = _ruff_auto_hint(path)
    diags = [ln for ln in hint.splitlines() if "F821" in ln]
    assert len(diags) == 8
    assert "(+4 autres)" in hint
