# tests/test_tools_search.py
import pytest

from loom.tools.base import ToolError
from loom.tools.search import make_find_files, make_list_dir, make_search_text


def _seed(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def play():\n    return 1\n", encoding="utf-8"
    )
    (tmp_path / "src" / "util.js").write_text(
        "function play(){ return 2 }\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Projet\nplay ici\n", encoding="utf-8")
    nm = tmp_path / "node_modules" / "lib"
    nm.mkdir(parents=True)
    (nm / "x.js").write_text("play in node_modules\n", encoding="utf-8")


def test_find_files_by_glob(tmp_path):
    _seed(tmp_path)
    out = make_find_files(str(tmp_path)).run({"pattern": "**/*.py"})
    assert out == "src/app.py"


def test_find_files_ignores_heavy_dirs(tmp_path):
    _seed(tmp_path)
    out = make_find_files(str(tmp_path)).run({"pattern": "**/*.js"})
    assert "src/util.js" in out
    assert "node_modules" not in out  # dossier lourd ignoré


def test_find_files_no_match(tmp_path):
    _seed(tmp_path)
    assert "aucun fichier" in make_find_files(str(tmp_path)).run({"pattern": "**/*.rs"})


def test_find_files_rejects_traversal(tmp_path):
    with pytest.raises(ToolError):
        make_find_files(str(tmp_path)).run({"pattern": "../*.py"})


def test_search_text_finds_symbol_with_locations(tmp_path):
    _seed(tmp_path)
    out = make_search_text(str(tmp_path)).run({"pattern": r"def play"})
    assert "src/app.py:1:" in out


def test_search_text_glob_filter_and_skips_node_modules(tmp_path):
    _seed(tmp_path)
    out = make_search_text(str(tmp_path)).run({"pattern": "play", "glob": "**/*.js"})
    assert "src/util.js" in out
    assert "node_modules" not in out


def test_search_text_invalid_regex_raises(tmp_path):
    with pytest.raises(ToolError):
        make_search_text(str(tmp_path)).run({"pattern": "([unclosed"})


def test_search_text_no_match(tmp_path):
    _seed(tmp_path)
    assert "aucune correspondance" in make_search_text(str(tmp_path)).run(
        {"pattern": "zzzz"}
    )


def test_list_dir_lists_dirs_then_files(tmp_path):
    _seed(tmp_path)
    out = make_list_dir(str(tmp_path)).run({"path": "."})
    lines = out.splitlines()
    assert "src/" in lines and "README.md" in lines
    assert lines.index("src/") < lines.index("README.md")  # dossiers d'abord


def test_list_dir_on_file_raises(tmp_path):
    _seed(tmp_path)
    with pytest.raises(ToolError):
        make_list_dir(str(tmp_path)).run({"path": "README.md"})
