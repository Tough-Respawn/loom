# tests/test_code_proto_st05.py
"""ST-05 : tests HERMÉTIQUES des prototypes code_outline / code_diagnostics.

Aucun vrai linter requis : les commandes sont INJECTÉES (faux serveur = script
Python local qui imprime une sortie canonique), l'absence et le crash du serveur
sont simulés. Les prototypes vivent dans evals/ (hors production)."""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from evals.code_tools import make_code_diagnostics, make_code_outline
from loom.tools.base import ToolError

# --------------------------------- outline -----------------------------------


def test_outline_python_symboles_et_plages_sans_corps(tmp_path):
    (tmp_path / "m.py").write_text(
        textwrap.dedent(
            """
            class Caisse:
                def total(self, items):
                    s = 0
                    for it in items:
                        s += it
                    return s

                def vider(self):
                    pass


            def facture(caisse, remise=0.0):
                return caisse.total([]) - remise
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    out = make_code_outline(str(tmp_path)).run({"path": "m.py"})
    assert "class Caisse  [1-9]" in out
    assert "def total(self, items)  [2-6]" in out
    assert "def facture(caisse, remise=0.0)  [12-13]" in out
    assert "s += it" not in out  # jamais les corps


def test_outline_python_syntaxe_cassee_erreur_actionnable(tmp_path):
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    with pytest.raises(ToolError, match="ligne 1"):
        make_code_outline(str(tmp_path)).run({"path": "bad.py"})


def test_outline_js_symboles_avec_lignes(tmp_path):
    (tmp_path / "app.js").write_text(
        "export function fmtDate(d) {\n  return d;\n}\n"
        "const helper = (x) => x * 2;\n"
        "class Store {\n}\n",
        encoding="utf-8",
    )
    out = make_code_outline(str(tmp_path)).run({"path": "app.js"})
    assert "function fmtDate  [ligne 1]" in out
    assert "function helper  [ligne 4]" in out
    assert "class Store  [ligne 5]" in out


# ------------------------------- diagnostics ----------------------------------


def _fake_server(tmp_path, name: str, body: str) -> list[str]:
    """Un faux « serveur » : script Python qui ignore ses arguments et imprime
    une sortie canonique. Rend l'argv de base à injecter."""
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return [sys.executable, str(p)]


def _ruff_json(diags) -> str:
    return json.dumps(
        [
            {
                "code": c,
                "filename": f,
                "location": {"row": ln, "column": col},
                "message": m,
            }
            for f, ln, col, c, m in diags
        ]
    )


def test_diagnostics_python_normalises_tries_et_severites(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payload = _ruff_json(
        [
            ("a.py", 9, 1, "F401", "`os` imported but unused"),
            ("a.py", 2, 5, "F821", "Undefined name `total`"),
        ]
    )
    cmd = _fake_server(
        tmp_path, "fake_ruff.py", f"import sys\nsys.stdout.write({payload!r})\n"
    )
    spec = make_code_diagnostics(str(tmp_path), py_cmd=cmd)
    out = spec.run({"path": "a.py"})
    lines = out.splitlines()
    assert lines[0].startswith("2 diagnostic(s)")
    # Trié par (fichier, ligne) : la ligne 2 avant la ligne 9.
    assert "a.py:2:5 [error F821] Undefined name `total`" in lines[1]
    assert "a.py:9:1 [warning F401]" in lines[2]
    # Filtre de sévérité.
    only_err = spec.run({"path": "a.py", "severity": "error"})
    assert "F401" not in only_err and "F821" in only_err


def test_diagnostics_bornes_a_50(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    payload = _ruff_json(
        [("a.py", i, 1, "F821", f"Undefined name `n{i}`") for i in range(1, 61)]
    )
    cmd = _fake_server(
        tmp_path, "fake_ruff.py", f"import sys\nsys.stdout.write({payload!r})\n"
    )
    out = make_code_diagnostics(str(tmp_path), py_cmd=cmd).run({"path": "a.py"})
    assert "60 diagnostic(s)" in out
    assert "… (+10 diagnostics)" in out


def test_diagnostics_fichier_propre(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cmd = _fake_server(tmp_path, "fake_ruff.py", "import sys\nsys.stdout.write('[]')\n")
    out = make_code_diagnostics(str(tmp_path), py_cmd=cmd).run({"path": "a.py"})
    assert "propre" in out


def test_diagnostics_sans_serveur_erreur_actionnable(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.js").write_text("var x = 1;\n", encoding="utf-8")
    spec = make_code_diagnostics(str(tmp_path), which=lambda name: None)
    with pytest.raises(ToolError, match="ruff introuvable"):
        spec.run({"path": "a.py"})
    with pytest.raises(ToolError, match="oxlint introuvable"):
        spec.run({"path": "b.js"})


def test_diagnostics_serveur_qui_crashe_fail_soft(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cmd = _fake_server(
        tmp_path,
        "fake_ruff.py",
        "import sys\nsys.stderr.write('panic: boom')\nsys.exit(3)\n",
    )
    with pytest.raises(ToolError, match="a échoué"):
        make_code_diagnostics(str(tmp_path), py_cmd=cmd).run({"path": "a.py"})


def test_diagnostics_js_format_unix_parse(tmp_path):
    (tmp_path / "app.js").write_text("var x = 1;\n", encoding="utf-8")
    body = (
        "import sys\n"
        "sys.stdout.write('app.js:3:7: error: no-undef: fmtDate is not defined "
        "[no-undef]\\n')\n"
    )
    cmd = _fake_server(tmp_path, "fake_ox.py", body)
    out = make_code_diagnostics(str(tmp_path), js_cmd=cmd).run({"path": "app.js"})
    assert "app.js:3:7 [error no-undef]" in out
    assert "fmtDate is not defined" in out
