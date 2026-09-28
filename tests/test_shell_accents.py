# Sous Windows FR, PowerShell et ses exe natifs écrivent en page OEM (cp850) alors
# que run_shell décode en UTF-8 : les accents revenaient en U+FFFD.
import sys

import pytest

from loom.tools.shell import make_run_shell

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell Windows")


@pytest.mark.parametrize(
    "command", ["Write-Output 'éèà ç'", "cmd /c echo éèà ç"], ids=["cmdlet", "natif"]
)
def test_run_shell_rend_les_accents(tmp_path, command):
    out = make_run_shell(str(tmp_path)).run({"command": command})
    assert "�" not in out
    assert "éèà ç" in out


def test_run_shell_garde_le_code_de_sortie(tmp_path):
    out = make_run_shell(str(tmp_path)).run({"command": "exit 3"})
    assert "exit 3" in out
