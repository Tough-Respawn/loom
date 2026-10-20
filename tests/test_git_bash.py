# Git Bash = shell de run_shell sous Windows (2026-10-20) : les modèles écrivent du
# bash par réflexe, PowerShell 5.1 refusait `&&`, `head`, `ls -la`, `python3`.
import sys
from pathlib import Path

import pytest

from loom.runtime import platform_info
from loom.runtime.platform_info import _git_bash, detect
from loom.tools.shell import make_run_shell

win = pytest.mark.skipif(sys.platform != "win32", reason="Windows seulement")


@pytest.fixture
def fresh_detect():
    detect.cache_clear()
    yield
    detect.cache_clear()


@win
def test_jamais_le_bash_de_wsl():
    found = _git_bash()
    if found:
        assert "system32" not in found.lower()
        assert Path(found).parts[-2:] == ("bin", "bash.exe")


@win
def test_loom_shell_powershell_force_l_ancien_shell(monkeypatch, fresh_detect):
    monkeypatch.setenv("LOOM_SHELL", "powershell")
    p = detect()
    assert p.shell_family == "powershell" and p.is_windows
    assert "Pas de `&&`" in p.prompt_block()


def test_argv_et_prompt_git_bash():
    p = platform_info.PlatformInfo(
        key="windows",
        label="Windows 11",
        shell_kind="git-bash",
        shell_family="posix",
        is_windows=True,
        is_macos=False,
        is_linux=False,
        shell_path="C:/Git/bin/bash.exe",
    )
    argv = p.shell_argv("ls")
    assert argv[:2] == ["C:/Git/bin/bash.exe", "-c"] and argv[2].endswith("ls")
    block = p.prompt_block()
    assert "Git Bash" in block and "C:/Users/" in block and "cmd //c" in block
    assert "Pas de `&&`" not in block


@win
def test_run_shell_execute_du_bash(tmp_path, fresh_detect):
    if detect().shell_kind != "git-bash":
        pytest.skip("Git for Windows absent")
    out = make_run_shell(str(tmp_path)).run(
        {"command": "echo a && echo b | head -1; python3 --version; pwd -W"}
    )
    lines = out.splitlines()
    assert lines[:2] == ["a", "b"]
    assert lines[2].startswith("Python 3")  # alias python3 -> python
    assert Path(lines[3]) == tmp_path  # dossier de travail respecté
