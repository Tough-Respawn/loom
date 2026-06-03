# tests/test_tools.py
import pytest

from loom.tools import ToolError, build_registry, make_read_file


def _reg(tmp_path, exts=(".py", ".md", ".txt"), max_bytes=1000):
    return build_registry(
        workspace_dir=str(tmp_path),
        extensions=list(exts),
        max_bytes=max_bytes,
        enabled=["read_file"],
    )


def test_read_file_returns_content(tmp_path):
    # write_bytes : pas de traduction \n -> \r\n (Windows), on lit l'exact contenu
    (tmp_path / "note.md").write_bytes("# Titre\ncontenu".encode("utf-8"))
    reg = _reg(tmp_path)
    assert reg.run("read_file", {"path": "note.md"}) == "# Titre\ncontenu"


def test_read_file_in_subdir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("x = 1", encoding="utf-8")
    reg = _reg(tmp_path)
    assert reg.run("read_file", {"path": "sub/a.py"}) == "x = 1"


def test_read_file_rejects_path_traversal(tmp_path):
    (tmp_path.parent / "secret.txt").write_text("SECRET", encoding="utf-8")
    reg = _reg(tmp_path)
    out = reg.run("read_file", {"path": "../secret.txt"})
    assert "périmètre" in out.lower() or "hors" in out.lower()


def test_read_file_rejects_unknown_extension(tmp_path):
    (tmp_path / "data.bin").write_text("x", encoding="utf-8")
    reg = _reg(tmp_path, exts=(".md",))
    assert "extension" in reg.run("read_file", {"path": "data.bin"}).lower()


def test_read_file_missing(tmp_path):
    reg = _reg(tmp_path)
    assert "introuvable" in reg.run("read_file", {"path": "nope.md"}).lower()


def test_read_file_directory(tmp_path):
    (tmp_path / "dir").mkdir()
    reg = _reg(tmp_path)
    assert "répertoire" in reg.run("read_file", {"path": "dir"}).lower()


def test_read_file_binary_rejected(tmp_path):
    (tmp_path / "img.md").write_bytes(b"\xff\xfe\x00\x01\x02")
    reg = _reg(tmp_path)
    assert "binaire" in reg.run("read_file", {"path": "img.md"}).lower()


def test_read_file_truncated_when_too_big(tmp_path):
    (tmp_path / "big.txt").write_text("a" * 5000, encoding="utf-8")
    reg = _reg(tmp_path, max_bytes=100)
    out = reg.run("read_file", {"path": "big.txt"})
    assert "tronqué" in out.lower()
    assert len(out) < 5000


def test_read_file_missing_path_arg(tmp_path):
    reg = _reg(tmp_path)
    assert "path" in reg.run("read_file", {}).lower()


def test_registry_openai_tools_schema(tmp_path):
    reg = _reg(tmp_path)
    tools = reg.openai_tools()
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "read_file"
    assert "path" in tools[0]["function"]["parameters"]["properties"]


def test_registry_unknown_tool(tmp_path):
    reg = _reg(tmp_path)
    assert "inconnu" in reg.run("inexistant", {}).lower()


def test_registry_len_and_empty():
    reg = build_registry(".", [".md"], 100, enabled=[])
    assert len(reg) == 0


def test_make_read_file_raises_tool_error_direct(tmp_path):
    spec = make_read_file(str(tmp_path), [".md"], 100)
    with pytest.raises(ToolError):
        spec.run({"path": "absent.md"})


def test_registry_catches_tool_error_into_message(tmp_path):
    reg = _reg(tmp_path)
    # ne doit jamais lever : renvoie un message d'erreur exploitable par le modèle
    out = reg.run("read_file", {"path": "../../etc/passwd"})
    assert isinstance(out, str) and out.startswith("erreur")
