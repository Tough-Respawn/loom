# tests/test_tools_fs.py
import pytest

from loom.tools import ToolError, build_registry
from loom.tools.fs import make_edit_file, make_write_file


def _wreg(tmp_path, max_bytes=10000):
    return build_registry(
        workspace_dir=str(tmp_path),
        extensions=[".py", ".md", ".txt"],
        max_bytes=max_bytes,
        enabled=["write_file", "edit_file"],
    )


# --- write_file ---------------------------------------------------------

def test_write_file_no_integrity_check_for_plain_text(tmp_path):
    reg = _wreg(tmp_path)
    out = reg.run("write_file", {"path": "note.txt", "content": "(((pas du code"})
    assert "écrit" in out.lower() and "erreur" not in out.lower()


def test_write_file_round_trip(tmp_path):
    reg = _wreg(tmp_path)
    out = reg.run("write_file", {"path": "a.txt", "content": "salut\nmonde"})
    assert "erreur" not in out.lower()
    assert (tmp_path / "a.txt").read_bytes() == "salut\nmonde".encode("utf-8")


def test_write_file_creates_parent_dirs(tmp_path):
    reg = _wreg(tmp_path)
    reg.run("write_file", {"path": "sub/deep/b.txt", "content": "x"})
    assert (tmp_path / "sub" / "deep" / "b.txt").read_bytes() == b"x"


def test_write_file_overwrite(tmp_path):
    (tmp_path / "c.txt").write_bytes(b"ancien")
    reg = _wreg(tmp_path)
    reg.run("write_file", {"path": "c.txt", "content": "nouveau"})
    assert (tmp_path / "c.txt").read_bytes() == b"nouveau"


def test_write_file_accents_emoji_byte_exact(tmp_path):
    reg = _wreg(tmp_path)
    content = "éàü 🚀 fin"
    reg.run("write_file", {"path": "u.txt", "content": content})
    assert (tmp_path / "u.txt").read_bytes() == content.encode("utf-8")


def test_write_file_no_crlf_translation(tmp_path):
    reg = _wreg(tmp_path)
    reg.run("write_file", {"path": "nl.txt", "content": "a\nb\nc"})
    assert (tmp_path / "nl.txt").read_bytes() == b"a\nb\nc"


def test_write_file_rejects_outside_workspace(tmp_path):
    reg = _wreg(tmp_path)
    out = reg.run("write_file", {"path": "../evade.txt", "content": "x"})
    assert "erreur" in out.lower()
    assert not (tmp_path.parent / "evade.txt").exists()


def test_write_file_no_tmp_residue(tmp_path):
    reg = _wreg(tmp_path)
    reg.run("write_file", {"path": "d.txt", "content": "z"})
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_file_missing_path(tmp_path):
    reg = _wreg(tmp_path)
    assert "path" in reg.run("write_file", {"content": "x"}).lower()


def test_write_file_too_big(tmp_path):
    reg = _wreg(tmp_path, max_bytes=10)
    out = reg.run("write_file", {"path": "big.txt", "content": "a" * 100})
    assert "erreur" in out.lower()
    assert not (tmp_path / "big.txt").exists()


# --- edit_file ----------------------------------------------------------


def test_edit_file_nominal(tmp_path):
    (tmp_path / "e.txt").write_bytes(b"avant X apres")
    reg = _wreg(tmp_path)
    out = reg.run("edit_file", {"path": "e.txt", "old_string": "X", "new_string": "Y"})
    assert "erreur" not in out.lower()
    assert (tmp_path / "e.txt").read_bytes() == b"avant Y apres"


def test_edit_file_preserves_rest_byte_exact(tmp_path):
    (tmp_path / "f.txt").write_bytes("ligne1\nFOO\nligne3".encode("utf-8"))
    reg = _wreg(tmp_path)
    reg.run("edit_file", {"path": "f.txt", "old_string": "FOO", "new_string": "BAR"})
    assert (tmp_path / "f.txt").read_bytes() == "ligne1\nBAR\nligne3".encode("utf-8")


def test_edit_file_missing_file(tmp_path):
    reg = _wreg(tmp_path)
    out = reg.run(
        "edit_file", {"path": "nope.txt", "old_string": "a", "new_string": "b"}
    )
    assert "erreur" in out.lower()


def test_edit_file_old_string_absent(tmp_path):
    (tmp_path / "g.txt").write_bytes(b"contenu")
    reg = _wreg(tmp_path)
    out = reg.run(
        "edit_file", {"path": "g.txt", "old_string": "absent", "new_string": "x"}
    )
    assert "introuvable" in out.lower()


def test_edit_file_ambiguous(tmp_path):
    (tmp_path / "h.txt").write_bytes(b"x x x")
    reg = _wreg(tmp_path)
    out = reg.run("edit_file", {"path": "h.txt", "old_string": "x", "new_string": "y"})
    assert "ambigu" in out.lower()
    # fichier inchangé
    assert (tmp_path / "h.txt").read_bytes() == b"x x x"


def test_edit_file_ambiguous_lists_lines_and_suggests_replace_all(tmp_path):
    (tmp_path / "h2.txt").write_bytes(b"foo\nbar\nfoo\n")  # 'foo' lignes 1 et 3
    reg = _wreg(tmp_path)
    out = reg.run(
        "edit_file", {"path": "h2.txt", "old_string": "foo", "new_string": "z"}
    )
    assert "ambigu" in out.lower()
    assert "1" in out and "3" in out  # n° de ligne des occurrences
    assert "replace_all" in out


def test_edit_file_replace_all(tmp_path):
    (tmp_path / "h3.txt").write_bytes(b"foo\nbar\nfoo\n")
    reg = _wreg(tmp_path)
    out = reg.run(
        "edit_file",
        {"path": "h3.txt", "old_string": "foo", "new_string": "z", "replace_all": True},
    )
    assert "erreur" not in out.lower()
    assert (tmp_path / "h3.txt").read_bytes() == b"z\nbar\nz\n"


def test_edit_file_crlf_hint(tmp_path):
    (tmp_path / "h4.txt").write_bytes(b"alpha\r\nbeta\r\n")  # fins de ligne CRLF
    reg = _wreg(tmp_path)
    out = reg.run(
        "edit_file",
        {"path": "h4.txt", "old_string": "alpha\nbeta", "new_string": "x"},
    )
    assert "introuvable" in out.lower() and "crlf" in out.lower()


def test_edit_file_empty_old_string(tmp_path):
    (tmp_path / "i.txt").write_bytes(b"contenu")
    reg = _wreg(tmp_path)
    out = reg.run("edit_file", {"path": "i.txt", "old_string": "", "new_string": "x"})
    assert "vide" in out.lower()


def test_edit_file_rejects_outside_workspace(tmp_path):
    reg = _wreg(tmp_path)
    out = reg.run(
        "edit_file", {"path": "../x.txt", "old_string": "a", "new_string": "b"}
    )
    assert "erreur" in out.lower()


def test_edit_file_no_tmp_residue(tmp_path):
    (tmp_path / "j.txt").write_bytes(b"aZb")
    reg = _wreg(tmp_path)
    reg.run("edit_file", {"path": "j.txt", "old_string": "Z", "new_string": "Q"})
    assert list(tmp_path.glob("*.tmp")) == []


def test_edit_file_accents_round_trip(tmp_path):
    (tmp_path / "k.txt").write_bytes("café ☕ ici".encode("utf-8"))
    reg = _wreg(tmp_path)
    reg.run("edit_file", {"path": "k.txt", "old_string": "☕", "new_string": "🍵"})
    assert (tmp_path / "k.txt").read_bytes() == "café 🍵 ici".encode("utf-8")


# --- direct ToolError / schémas ----------------------------------------


def test_make_edit_file_raises_tool_error_direct(tmp_path):
    spec = make_edit_file(str(tmp_path))
    with pytest.raises(ToolError):
        spec.run({"path": "absent.txt", "old_string": "a", "new_string": "b"})


def test_make_write_file_schema(tmp_path):
    spec = make_write_file(str(tmp_path), 1000)
    schema = spec.to_openai()
    props = schema["function"]["parameters"]["properties"]
    assert "path" in props and "content" in props


def test_make_edit_file_schema(tmp_path):
    spec = make_edit_file(str(tmp_path))
    props = spec.to_openai()["function"]["parameters"]["properties"]
    assert {"path", "old_string", "new_string"} <= set(props)


def test_build_registry_registers_fs_tools(tmp_path):
    reg = _wreg(tmp_path)
    assert "write_file" in reg
    assert "edit_file" in reg
    assert len(reg) == 2
