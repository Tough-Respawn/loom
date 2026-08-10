# tests/test_code_tools.py
"""ST-06 : outils d'intelligence de code de PRODUCTION (Python uniquement).

Hermétique : ruff est INJECTÉ (faux serveur = script local) ou simulé absent.
Couvre aussi les contrats produits : lecture seule (permissions explicites),
différé (jamais dans le préfixe tant que non chargé via tool_search), chemins
Windows (antislash, dossier accentué), fail-soft sans casser la session.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from loom.agent.conversation import Conversation
from loom.tools import build_registry
from loom.tools.base import ToolError
from loom.tools.code import make_code_diagnostics, make_code_outline

# --------------------------------- outline -----------------------------------


def test_outline_python_plages_exactes_sans_corps(tmp_path):
    (tmp_path / "m.py").write_text(
        textwrap.dedent(
            """
            class Caisse:
                def total(self, items):
                    s = 0
                    return s


            def facture(caisse, remise=0.0):
                return caisse.total([]) - remise
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    out = make_code_outline(str(tmp_path)).run({"path": "m.py"})
    assert "class Caisse  [1-4]" in out
    assert "def total(self, items)  [2-4]" in out
    assert "def facture(caisse, remise=0.0)  [7-8]" in out
    assert "s = 0" not in out  # jamais les corps


def test_outline_chemin_windows_absolu_et_accents(tmp_path):
    d = tmp_path / "propriété"
    d.mkdir()
    (d / "mod.py").write_text("def héla():\n    pass\n", encoding="utf-8")
    # Chemin ABSOLU avec antislashs Windows, dossier accentué.
    abs_win = str(d / "mod.py").replace("/", "\\")
    out = make_code_outline(str(tmp_path)).run({"path": abs_win})
    assert "def héla()  [1-2]" in out


def test_outline_syntaxe_cassee_et_non_python(tmp_path):
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    with pytest.raises(ToolError, match="ligne 1"):
        make_code_outline(str(tmp_path)).run({"path": "bad.py"})
    (tmp_path / "app.js").write_text("var x = 1;\n", encoding="utf-8")
    with pytest.raises(ToolError, match="ne couvre que Python"):
        make_code_outline(str(tmp_path)).run({"path": "app.js"})


# ------------------------------- diagnostics ----------------------------------


def _fake_ruff(tmp_path, diags) -> list[str]:
    payload = json.dumps(
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
    p = tmp_path / "fake_ruff.py"
    p.write_text(f"import sys\nsys.stdout.write({payload!r})\n", encoding="utf-8")
    return [sys.executable, str(p)]


def test_diagnostics_normalises_tries_bornes_filtres(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cmd = _fake_ruff(
        tmp_path,
        [("a.py", 9, 1, "F401", "unused"), ("a.py", 2, 5, "F821", "Undefined `t`")],
    )
    spec = make_code_diagnostics(str(tmp_path), ruff_cmd=cmd)
    out = spec.run({"path": "a.py"})
    lines = out.splitlines()
    assert lines[0].startswith("2 diagnostic(s)")
    assert "a.py:2:5 [error F821]" in lines[1]  # trié, sévérité mappée
    assert "a.py:9:1 [warning F401]" in lines[2]
    assert "F401" not in spec.run({"path": "a.py", "severity": "error"})


def test_diagnostics_borne_50(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cmd = _fake_ruff(tmp_path, [("a.py", i, 1, "F821", f"n{i}") for i in range(1, 61)])
    out = make_code_diagnostics(str(tmp_path), ruff_cmd=cmd).run({"path": "a.py"})
    assert "… (+10 diagnostics)" in out


def test_diagnostics_sans_ruff_erreur_actionnable_sans_crash(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    spec = make_code_diagnostics(str(tmp_path), which=lambda name: None)
    with pytest.raises(ToolError, match="ruff introuvable"):
        spec.run({"path": "a.py"})


def test_diagnostics_ruff_qui_crashe_fail_soft(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nsys.stderr.write('boom')\nsys.exit(3)\n", "utf-8")
    spec = make_code_diagnostics(str(tmp_path), ruff_cmd=[sys.executable, str(crash)])
    with pytest.raises(ToolError, match="a échoué"):
        spec.run({"path": "a.py"})


def test_diagnostics_refuse_hors_python(tmp_path):
    (tmp_path / "b.js").write_text("var x;\n", encoding="utf-8")
    spec = make_code_diagnostics(str(tmp_path), ruff_cmd=[sys.executable])
    with pytest.raises(ToolError, match="ne couvre que Python"):
        spec.run({"path": "b.js"})


# ------------------------- permissions : lecture seule -------------------------


def test_permissions_explicites_jamais_de_refus_silencieux():
    """La leçon ST-05 (17/17 appels refusés) : les deux outils sont classés
    READ_TOOLS -> allow dans TOUS les modes, jamais « ask » sans UI."""
    from loom.permissions import PermissionConfig, evaluate

    for mode in ("allow", "ask", "allowlist", "deny_all"):
        for name in ("code_outline", "code_diagnostics"):
            d = evaluate(name, {"path": "x.py"}, PermissionConfig(mode=mode))
            assert d.action == "allow", f"{name} en mode {mode} -> {d.action}"


# ----------------------- différé : préfixe inchangé ---------------------------


def _registry(tmp_path, enabled):
    conv = Conversation("p")
    conv.runtime_session_id = "code-test"
    return build_registry(str(tmp_path), 10_000, enabled, conversation=conv)


def test_prefixe_inchange_tant_que_non_charges(tmp_path):
    """Contrat ST-06 : les outils code n'apparaissent JAMAIS dans le préfixe
    (always_deferred). Ils sont listés au catalogue de tool_search et deviennent
    appelables après chargement explicite — et pas avant."""
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    reg = _registry(tmp_path, ["read_file", "code_outline", "code_diagnostics"])
    prefix = [t["function"]["name"] for t in reg.openai_tools()]
    assert "code_outline" not in prefix and "code_diagnostics" not in prefix
    assert "read_file" in prefix  # les outils par défaut, eux, ne bougent pas
    assert "tool_search" in prefix  # le mécanisme de découverte, comme pour MCP
    catalogue = next(
        t["function"]["description"]
        for t in reg.openai_tools()
        if t["function"]["name"] == "tool_search"
    )
    assert "code_outline" in catalogue and "code_diagnostics" in catalogue

    # Avant chargement : appel refusé avec l'erreur actionnable standard.
    out = reg.run("code_outline", {"path": "a.py"})
    assert out.startswith("erreur") and "tool_search" in out
    # Après chargement : appelable, et il fonctionne.
    reg.run("tool_search", {"names": ["code_outline"]})
    out2 = reg.run("code_outline", {"path": "a.py"})
    assert "def f()  [1-2]" in out2


def test_sans_outils_code_rien_ne_change(tmp_path):
    """Une config qui n'active pas les outils code garde un registre identique :
    pas de tool_search induit, pas de schéma parasite."""
    reg = _registry(tmp_path, ["read_file"])
    names = [t["function"]["name"] for t in reg.openai_tools()]
    assert names == ["read_file"]
