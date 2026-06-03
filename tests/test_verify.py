# tests/test_verify.py
import shutil

import pytest

from loom.verify import VerifyReport, format_report, verify_path


def _w(path, content):
    path.write_bytes(content.encode("utf-8"))  # write_bytes : pas de \r\n sous Windows


def test_python_syntax_error_detected(tmp_path):
    _w(tmp_path / "bad.py", "def f(:\n    pass\n")
    report = verify_path(str(tmp_path / "bad.py"))
    assert report.ok is False
    assert report.defects[0].kind == "syntax"
    assert "bad.py" in report.defects[0].location


def test_python_valid_ok(tmp_path):
    _w(tmp_path / "good.py", "def f():\n    return 1\n")
    assert verify_path(str(tmp_path / "good.py")).ok is True


def test_json_invalid_detected(tmp_path):
    _w(tmp_path / "data.json", '{"a": 1,}')
    report = verify_path(str(tmp_path / "data.json"))
    assert report.ok is False
    assert report.defects[0].kind == "json"


def test_verify_dir_aggregates(tmp_path):
    _w(tmp_path / "ok.py", "x = 1\n")
    _w(tmp_path / "bad.py", "x = (\n")
    _w(tmp_path / "page.html", "<html></html>")  # ignoré (non vérifiable)
    report = verify_path(str(tmp_path))
    assert report.ok is False
    assert any("bad.py" in d.location for d in report.defects)


def test_format_report_ok():
    assert "VERIFY OK" in format_report(VerifyReport(ok=True))


def test_format_report_lists_defects(tmp_path):
    _w(tmp_path / "bad.py", "def (:\n")
    out = format_report(verify_path(str(tmp_path / "bad.py")))
    assert "défaut" in out.lower() and "bad.py" in out


@pytest.mark.skipif(not shutil.which("node"), reason="node absent")
def test_js_syntax_error_detected_with_node(tmp_path):
    _w(tmp_path / "game.js", "function f() {\n  return 1;\n}}\n")  # accolade en trop
    report = verify_path(str(tmp_path / "game.js"))
    assert report.ok is False
    assert report.defects[0].kind == "syntax"


@pytest.mark.skipif(not shutil.which("node"), reason="node absent")
def test_js_valid_ok_with_node(tmp_path):
    _w(tmp_path / "ok.js", "const a = 1;\nfunction g() { return a; }\n")
    assert verify_path(str(tmp_path / "ok.js")).ok is True


# --- verify_files : liste explicite, borné (hard-gate P0.4) ------------


def test_verify_files_explicit_list_aggregates(tmp_path):
    from loom.verify import verify_files

    _w(tmp_path / "ok.py", "x = 1\n")
    _w(tmp_path / "bad.py", "x = (\n")
    r = verify_files([str(tmp_path / "ok.py"), str(tmp_path / "bad.py")])
    assert r.ok is False
    assert any("bad.py" in d.location for d in r.defects)


def test_verify_files_all_clean_ok(tmp_path):
    from loom.verify import verify_files

    _w(tmp_path / "a.py", "x = 1\n")
    _w(tmp_path / "b.json", '{"a": 1}')
    assert verify_files([str(tmp_path / "a.py"), str(tmp_path / "b.json")]).ok is True


def test_verify_files_ignores_unverifiable(tmp_path):
    from loom.verify import verify_files

    _w(tmp_path / "style.css", "body{color:red}")  # non vérifiable -> ignoré
    _w(tmp_path / "a.py", "x = 1\n")
    assert (
        verify_files([str(tmp_path / "style.css"), str(tmp_path / "a.py")]).ok is True
    )


# --- rejet déterministe des ES modules (non chargeables en file://) ------


def test_verify_files_rejects_es_modules(tmp_path):
    from loom.verify import verify_files

    _w(tmp_path / "index.html", '<script type="module" src="m.js"></script>')
    _w(tmp_path / "m.js", "export function go(){ return 1; }\n")
    r = verify_files([str(tmp_path / "index.html"), str(tmp_path / "m.js")])
    assert r.ok is False
    # un défaut 'modules' par fichier fautif : type=module ET export
    assert sum(d.kind == "modules" for d in r.defects) == 2


def test_verify_files_accepts_classic_scripts(tmp_path):
    from loom.verify import verify_files

    _w(tmp_path / "a.js", "function go(){ return 1; }\nwindow.go = go;\n")
    r = verify_files([str(tmp_path / "a.js")])
    assert all(d.kind != "modules" for d in r.defects)


# --- verify_path : garde-fous anti-blocage (skip dirs lourds + plafond) -


def test_verify_path_skips_heavy_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    _w(tmp_path / "node_modules" / "bad.py", "def (:\n")  # ne doit PAS être scanné
    _w(tmp_path / "good.py", "x = 1\n")
    assert verify_path(str(tmp_path)).ok is True


def test_verify_path_caps_oversized_dir(tmp_path, monkeypatch):
    import loom.verify as v

    monkeypatch.setattr(v, "_MAX_DIR_SCAN", 3)
    for i in range(6):
        _w(tmp_path / f"f{i}.py", "x = 1\n")
    r = v.verify_path(str(tmp_path))
    assert r.ok is False
    assert "trop vaste" in r.defects[0].evidence


# --- l'outil verify (registre) ----------------------------------------


def test_verify_tool_reports_defect(tmp_path):
    from loom.tools import make_verify

    _w(tmp_path / "bad.py", "def (:\n")
    out = make_verify(str(tmp_path)).run({"path": "bad.py"})
    assert "VERIFY" in out and "bad.py" in out


def test_verify_tool_ok_on_clean_dir(tmp_path):
    from loom.tools import make_verify

    _w(tmp_path / "ok.py", "x = 1\n")
    assert "VERIFY OK" in make_verify(str(tmp_path)).run({"path": "."})


def test_build_registry_includes_verify(tmp_path):
    from loom.tools import build_registry

    reg = build_registry(str(tmp_path), [".py"], 1000, enabled=["verify"])
    assert "verify" in reg


def test_verify_is_read_tool_allowed():
    from loom.permissions import PermissionConfig, evaluate

    # mode deny_all : verify reste autorisé car READ-only (parse, n'exécute pas)
    d = evaluate("verify", {"path": "."}, PermissionConfig(mode="deny_all"))
    assert d.action == "allow"


# --- Vérificateur RUNTIME web (jsdom) ----------------------------------


def _runtime_verifier_available():
    import subprocess

    if not shutil.which("node"):
        return False
    try:
        return (
            subprocess.run(
                ["node", "-e", "require('jsdom')"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001
        return False


_RUNTIME_OK = _runtime_verifier_available()


@pytest.mark.skipif(not _RUNTIME_OK, reason="node/jsdom absent")
def test_web_runtime_error_detected(tmp_path):
    _w(tmp_path / "index.html", '<div id="board"></div><script src="app.js"></script>')
    _w(
        tmp_path / "app.js",
        "document.addEventListener('DOMContentLoaded',()=>{ window.nope.boom(); });",
    )
    r = verify_path(str(tmp_path))
    assert r.ok is False
    assert any(d.kind in ("runtime", "render") for d in r.defects)


@pytest.mark.skipif(not _RUNTIME_OK, reason="node/jsdom absent")
def test_web_renders_cells_is_ok(tmp_path):
    _w(tmp_path / "index.html", '<div id="board"></div><script src="app.js"></script>')
    # plateau rendu ET interactif : un clic pose une marque (sinon le gate d'interaction
    # le refuse, à raison — le rendu seul ne prouve pas la jouabilité).
    _w(
        tmp_path / "app.js",
        "document.addEventListener('DOMContentLoaded',()=>{const b=document.getElementById('board');"
        "for(let i=0;i<9;i++){const c=document.createElement('div');c.className='cell';"
        "c.addEventListener('click',()=>{if(!c.textContent)c.textContent='X';});b.appendChild(c);}});",
    )
    r = verify_path(str(tmp_path))
    assert r.ok is True


@pytest.mark.skipif(not _RUNTIME_OK, reason="node/jsdom absent")
def test_web_empty_board_is_render_defect(tmp_path):
    _w(tmp_path / "index.html", '<div id="board"></div><script src="app.js"></script>')
    _w(tmp_path / "app.js", "const x = 1;")  # syntaxe OK mais ne rend rien
    r = verify_path(str(tmp_path))
    assert r.ok is False
    assert any(d.kind == "render" for d in r.defects)
