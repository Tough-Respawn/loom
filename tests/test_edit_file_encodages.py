# edit_file réécrit le fichier dans son encodage d'origine : il ne doit ni corrompre
# des octets, ni changer l'ordre des octets, ni toucher aux lignes non éditées.
import pytest

from loom.tools.base import ToolError
from loom.tools.fs import make_edit_file


def _edit(tmp_path, name, old, new):
    return make_edit_file(str(tmp_path)).run(
        {"path": name, "old_string": old, "new_string": new}
    )


def test_utf16_be_reste_big_endian(tmp_path):
    p = tmp_path / "a.ps1"
    p.write_bytes(b"\xfe\xff" + "Write-Host 'été'\r\n".encode("utf-16-be"))
    _edit(tmp_path, "a.ps1", "été", "hiver")
    data = p.read_bytes()
    assert data[:2] == b"\xfe\xff"
    assert data[2:].decode("utf-16-be") == "Write-Host 'hiver'\r\n"


def test_utf16_le_reste_little_endian(tmp_path):
    p = tmp_path / "b.ps1"
    p.write_bytes(b"\xff\xfe" + "a = 1\r\n".encode("utf-16-le"))
    _edit(tmp_path, "b.ps1", "a = 1", "a = 2")
    assert p.read_bytes() == b"\xff\xfe" + "a = 2\r\n".encode("utf-16-le")


def test_octets_invalides_apres_bom_refuse_au_lieu_de_corrompre(tmp_path):
    p = tmp_path / "c.txt"
    original = b"\xef\xbb\xbf" + b"garde \xff\xfe ici\nmodifie-moi\n"
    p.write_bytes(original)
    with pytest.raises(ToolError):
        _edit(tmp_path, "c.txt", "modifie-moi", "fait")
    assert p.read_bytes() == original


def test_fins_de_ligne_mixtes_seule_la_zone_editee_change(tmp_path):
    p = tmp_path / "d.py"
    p.write_bytes(b"a = 1\r\nb = 2\nc = 3\r\n")
    _edit(tmp_path, "d.py", "b = 2", "b = 20")
    assert p.read_bytes() == b"a = 1\r\nb = 20\nc = 3\r\n"


def test_fichier_crlf_pur_inchange_en_style(tmp_path):
    p = tmp_path / "e.py"
    p.write_bytes(b"x = 1\r\ny = 2\r\n")
    _edit(tmp_path, "e.py", "x = 1\ny = 2", "x = 1\ny = 3\nz = 4")
    assert p.read_bytes() == b"x = 1\r\ny = 3\r\nz = 4\r\n"
