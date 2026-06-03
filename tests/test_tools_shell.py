# tests/test_tools_shell.py
import sys

import pytest

import loom.tools.shell as tools_shell
from loom.tools import ToolError, build_registry
from loom.tools.shell import make_run_shell


def _sreg(tmp_path, **kw):
    reg = build_registry(
        workspace_dir=str(tmp_path),
        extensions=[".py", ".txt"],
        max_bytes=10000,
        enabled=["run_shell"],
    )
    return reg


# --- exécution réelle (UN seul echo) -----------------------------------


def test_run_shell_simple_echo(tmp_path):
    spec = make_run_shell(str(tmp_path))
    out = spec.run({"command": "echo bonjour"})
    assert "exit=0" in out
    assert "bonjour" in out
    # succès (exit 0) => PAS marqué comme erreur (ok=True côté boucle)
    assert not out.startswith("erreur")


# --- exit non-nul (monkeypatch subprocess) -----------------------------


class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_shell_nonzero_exit(tmp_path, monkeypatch):
    def fake_run(*a, **k):
        return _FakeCompleted(3, stdout="out", stderr="boom")

    monkeypatch.setattr(tools_shell.subprocess, "run", fake_run)
    spec = make_run_shell(str(tmp_path))
    out = spec.run({"command": "whatever"})
    # exit non-nul => signalé comme erreur (ok=False côté boucle), cause racine
    # des faux 'OK' du reviewer. Le corps (exit/stdout/stderr) reste exploitable.
    assert out.startswith("erreur")
    assert "exit=3" in out
    assert "out" in out
    assert "boom" in out


def test_run_shell_andand_without_pwsh_gives_usable_error(tmp_path, monkeypatch):
    import pytest

    from loom.tools import ToolError

    monkeypatch.setattr(tools_shell.sys, "platform", "win32")
    monkeypatch.setattr(tools_shell.shutil, "which", lambda name: None)
    spec = make_run_shell(str(tmp_path))
    # run() lève ToolError ; ToolRegistry.run la convertit ensuite en 'erreur: ...'
    with pytest.raises(ToolError) as exc:
        spec.run({"command": "cd x && npm test"})
    msg = str(exc.value)
    assert ";" in msg or "LASTEXITCODE" in msg  # message exploitable par le modèle


def test_shell_argv_prefers_pwsh_when_available(monkeypatch):
    monkeypatch.setattr(tools_shell.sys, "platform", "win32")
    monkeypatch.setattr(
        tools_shell.shutil,
        "which",
        lambda name: "C:/pwsh.exe" if name == "pwsh" else None,
    )
    argv = tools_shell._shell_argv("echo hi")
    assert argv[0] == "C:/pwsh.exe"


# --- timeout respecté --------------------------------------------------


def test_run_shell_timeout(tmp_path, monkeypatch):
    import subprocess as real_sp

    def fake_run(*a, **k):
        raise real_sp.TimeoutExpired(cmd="x", timeout=k.get("timeout", 1))

    monkeypatch.setattr(tools_shell.subprocess, "run", fake_run)
    spec = make_run_shell(str(tmp_path), timeout=1)
    out = spec.run({"command": "sleep 100"})
    assert "timeout" in out.lower()


# --- troncature --------------------------------------------------------


def test_run_shell_truncates(tmp_path, monkeypatch):
    def fake_run(*a, **k):
        return _FakeCompleted(0, stdout="A" * 5000, stderr="")

    monkeypatch.setattr(tools_shell.subprocess, "run", fake_run)
    spec = make_run_shell(str(tmp_path), max_output=100)
    out = spec.run({"command": "x"})
    assert "[tronqué]" in out
    assert len(out) < 5000


# --- détection OS : Windows -> powershell ------------------------------


def test_run_shell_windows_uses_powershell(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        captured["cwd"] = k.get("cwd")
        return _FakeCompleted(0, stdout="", stderr="")

    monkeypatch.setattr(tools_shell.subprocess, "run", fake_run)
    monkeypatch.setattr(tools_shell.sys, "platform", "win32")
    spec = make_run_shell(str(tmp_path))
    spec.run({"command": "Get-Date"})
    assert captured["cmd"][0] == "powershell"
    assert "Get-Date" in captured["cmd"]
    assert captured["cwd"] == str(tmp_path)


def test_run_shell_posix_uses_bash(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return _FakeCompleted(0, stdout="", stderr="")

    monkeypatch.setattr(tools_shell.subprocess, "run", fake_run)
    monkeypatch.setattr(tools_shell.sys, "platform", "linux")
    spec = make_run_shell(str(tmp_path))
    spec.run({"command": "ls"})
    assert captured["cmd"][0] == "/bin/bash"
    assert "ls" in captured["cmd"]


# --- deny-list dure : subprocess JAMAIS appelé -------------------------


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", "Remove-Item -Recurse -Force C:\\", "format c:"],
)
def test_run_shell_denylist_blocks_before_subprocess(tmp_path, monkeypatch, command):
    calls = {"n": 0}

    def spy_run(*a, **k):
        calls["n"] += 1
        return _FakeCompleted(0)

    monkeypatch.setattr(tools_shell.subprocess, "run", spy_run)
    spec = make_run_shell(str(tmp_path))
    with pytest.raises(ToolError):
        spec.run({"command": command})
    assert calls["n"] == 0


def test_run_shell_denylist_raises_tool_error_direct(tmp_path, monkeypatch):
    calls = {"n": 0}

    def spy_run(*a, **k):
        calls["n"] += 1
        return _FakeCompleted(0)

    monkeypatch.setattr(tools_shell.subprocess, "run", spy_run)
    spec = make_run_shell(str(tmp_path))
    with pytest.raises(ToolError):
        spec.run({"command": "rm -rf /"})
    assert calls["n"] == 0


# --- argument manquant -------------------------------------------------


def test_run_shell_missing_command(tmp_path):
    spec = make_run_shell(str(tmp_path))
    with pytest.raises(ToolError):
        spec.run({})


# --- registre + schéma -------------------------------------------------


def test_build_registry_registers_run_shell(tmp_path):
    reg = _sreg(tmp_path)
    assert "run_shell" in reg
    assert len(reg) == 1


def test_run_shell_schema(tmp_path):
    spec = make_run_shell(str(tmp_path))
    props = spec.to_openai()["function"]["parameters"]["properties"]
    assert "command" in props
    assert spec.name == "run_shell"


def test_run_shell_uses_sys_platform_import():
    # garantit que le module expose bien sys (pour monkeypatch.setattr)
    assert tools_shell.sys is sys
