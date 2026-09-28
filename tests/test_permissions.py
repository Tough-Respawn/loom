# Politique de permissions : allowlist de chemins, deny-list dure des commandes.
import pytest

from loom.permissions import PermissionConfig, evaluate


def _write(path, allow):
    cfg = PermissionConfig(mode="allowlist", allow_paths=allow)
    return evaluate("write_file", {"path": path, "content": "x"}, cfg).action


def test_allowlist_accepte_les_chemins_sous_le_dossier():
    assert _write("src/a.py", ["src"]) == "allow"
    assert _write(r"src\pkg\b.py", ["src"]) == "allow"
    assert _write("src/./c.py", ["src"]) == "allow"


@pytest.mark.parametrize(
    "path",
    [
        "src/../../Users/me/AppData/Roaming/Microsoft/Windows/Start Menu/x.bat",
        "src/../secrets.txt",
        r"src\..\..\x",
        "srcx/a.py",
    ],
)
def test_allowlist_ne_se_contourne_pas(path):
    assert _write(path, ["src"]) == "ask"


def _shell(command, mode="allow"):
    return evaluate("run_shell", {"command": command}, PermissionConfig(mode=mode))


@pytest.mark.parametrize(
    "command",
    [
        r"rd /s /q C:\x",
        r"rmdir /s C:\x",
        r"rd C:\x /s /q",
        "del /s /q *",
        "del /f x",
        "erase /s *.txt",
        r"ri -r -fo C:\x",
        r"rm -r -fo C:\x",
        r"Remove-Item -Recurse -Force C:\x",
        "rm -rf /",
        "format C:",
        "format.com D: /q",
        "git reset --hard HEAD~3",
    ],
)
def test_deny_list_bloque_meme_en_mode_allow(command):
    assert _shell(command).action == "deny", command


@pytest.mark.parametrize(
    "command",
    [
        "ruff format .",
        "uv run ruff format loom",
        "npm run format",
        "clang-format -i a.c",
        "dir /s",
        "git status",
        "rm build.log",
        "Remove-Item x.txt",
    ],
)
def test_deny_list_sans_faux_positif(command):
    assert _shell(command).action == "allow", command
