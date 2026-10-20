# Les commandes du modèle ne doivent pas hériter du venv de Loom (vécu 2026-10-20,
# session 8ff2be6f7e1b : avertissement uv à chaque commande, `python` = celui de Loom),
# et la ligne d'erreur résumée doit dire la vraie erreur, pas un avertissement.
import os
import sys
from pathlib import Path

from loom.tools.shell import _child_env, _error_head, make_run_shell


def test_child_env_retire_le_venv_de_loom(monkeypatch, tmp_path):
    venv = tmp_path / ".venv"
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    other = tmp_path / "ailleurs"
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setenv("PATH", os.pathsep.join([str(scripts), str(other)]))
    env = _child_env()
    assert "VIRTUAL_ENV" not in env
    assert env["PATH"].split(os.pathsep) == [str(other)]
    assert env["PYTHONUTF8"] == "1"


def test_run_shell_ne_voit_pas_le_venv_de_loom(tmp_path):
    # Les tests tournent dans le venv de Loom : le python de la commande doit être un autre.
    out = make_run_shell(str(tmp_path)).run(
        {"command": 'python -c "import sys; print(sys.prefix)"'}
    )
    assert Path(out.strip()) != Path(sys.prefix)


def test_error_head_prend_la_vraie_erreur():
    uv_warn = (
        "warning: `VIRTUAL_ENV=C:\\x\\.venv` does not match the project environment"
    )
    tb = (
        f"{uv_warn}\nTraceback (most recent call last):\n"
        '  File "a.py", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'numpy'"
    )
    assert _error_head(tb) == "ModuleNotFoundError: No module named 'numpy'"
    assert _error_head(f"{uv_warn}\nfatal: not a git repository") == (
        "fatal: not a git repository"
    )
    assert _error_head("") == "commande échouée"
