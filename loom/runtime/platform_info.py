"""Détection de l'OS et de ses conventions — SOURCE DE VÉRITÉ unique pour Loom.

Le shell (run_shell) ET le prompt système en dérivent : Loom identifie tout seul sur quel
système il tourne et se comporte selon SES standards (PowerShell + cmdlets sous Windows,
bash + commandes unix sous macOS/Linux, séparateurs de chemin, variables d'env…). Un seul
endroit décide -> le modèle n'est jamais orienté vers les mauvaises conventions."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Préfixe PowerShell : sortie en UTF-8 pour le processus ET ses exe natifs (le
# try/catch couvre un hôte sans console, où l'affectation peut lever).
_PS_UTF8 = (
    "try{[Console]::OutputEncoding=[System.Text.Encoding]::UTF8}catch{}; "
    "$OutputEncoding=[System.Text.Encoding]::UTF8; "
)

# Git Bash : `python3` n'existe pas sous Windows (l'alias du Microsoft Store répond
# « Python est introuvable ») et les modèles l'écrivent par réflexe Linux.
_PY3_ALIAS = 'python3() { python "$@"; }; '
# Même piège d'accents que _PS_UTF8 : les exe natifs (cmd, ipconfig…) écrivent dans la
# page OEM de la console (cp850) ; on la passe en UTF-8, comme le fait le préfixe PS.
_BASH_UTF8 = "chcp.com 65001 >/dev/null 2>&1; "


def _git_bash() -> str:
    """bash.exe de Git for Windows, ou ''. Jamais `shutil.which("bash")` : sous Windows
    il renvoie System32\\bash.exe, le lanceur WSL (autre système de fichiers, autre
    Python, sans les outils de la machine)."""
    candidates = []
    git = shutil.which("git")
    if git:
        # Git\cmd\git.exe ou Git\mingw64\bin\git.exe -> Git\bin\bash.exe
        candidates += [p / "bin" / "bash.exe" for p in Path(git).resolve().parents[:3]]
    for env, sub in (
        ("ProgramFiles", "Git"),
        ("ProgramW6432", "Git"),
        ("LOCALAPPDATA", "Programs/Git"),
    ):
        if os.environ.get(env):
            candidates.append(Path(os.environ[env]) / sub / "bin" / "bash.exe")
    return next((str(c) for c in candidates if c.is_file()), "")


@dataclass(frozen=True)
class PlatformInfo:
    key: str  # "windows" | "macos" | "linux"
    label: str  # ex. "Windows 11", "macOS 14.5 (Apple Silicon)", "Ubuntu 24.04 LTS"
    shell_kind: str  # "pwsh" | "powershell" | "bash" | "git-bash"
    shell_family: str  # "powershell" | "posix"
    is_windows: bool
    is_macos: bool
    is_linux: bool
    shell_path: str = ""  # git-bash : chemin de bash.exe

    @property
    def shell_label(self) -> str:
        return {
            "pwsh": "PowerShell 7 (pwsh)",
            "powershell": "PowerShell",
            "bash": "bash",
            "git-bash": "bash (Git Bash)",
        }[self.shell_kind]

    def shell_argv(self, command: str) -> list[str]:
        """argv pour exécuter `command` dans le shell natif du système."""
        if self.shell_kind == "git-bash":
            # -c et pas -lc : le lanceur Git\bin\bash.exe pose déjà le PATH unix, et un
            # shell de login coûte ~600 ms par commande (mesuré 2026-10-20 : 60 contre 650).
            return [self.shell_path, "-c", _BASH_UTF8 + _PY3_ALIAS + command]
        if self.shell_family == "powershell":
            exe = "pwsh" if self.shell_kind == "pwsh" else "powershell"
            # Loom lit la sortie en UTF-8, mais PowerShell (et les exe natifs qu'il
            # lance) écrit dans la page OEM (cp850 sur un Windows FR) : tout accent
            # revenait en U+FFFD, jusqu'aux messages d'erreur système.
            return [
                exe,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _PS_UTF8 + command,
            ]
        bash = shutil.which("bash") or "/bin/bash"
        return [bash, "-lc", command]

    def prompt_block(self) -> str:
        """Bloc « Système » injecté au prompt : dit au modèle l'OS, son shell et les
        conventions à respecter, pour qu'il produise les BONNES commandes/chemins."""
        if self.shell_family == "powershell":
            conv = (
                f"run_shell exécute du **{self.shell_label}**. N'écris JAMAIS d'unix "
                "(`grep`, `ls`, `cat`, `find`, `wc`, `2>/dev/null`, `rm -rf`…) : ils "
                "n'existent pas ici — utilise les cmdlets (`Get-ChildItem`, `Select-String`, "
                "`Get-Content`, `Measure-Object`, `Remove-Item`). Variables `$env:VAR`. "
                "Pas de `&&` ni `||` (PowerShell 5.1) : `;` entre les commandes, ou un appel "
                "run_shell par commande. `python3` n'existe pas : `python`. `/tmp` n'existe "
                "pas non plus, y compris DANS tes scripts (`open('/tmp/x')`, `save('/tmp/x')`) "
                ": écris tes sorties dans le bac à sable de la session."
            )
        elif self.shell_kind == "git-bash":
            conv = (
                "run_shell exécute du **bash (Git Bash)** : commandes unix (`ls`, `grep`, "
                "`cat`, `head`, `find`, `&&`, `||`, `2>/dev/null`…) et programmes Windows "
                "(`python`, `uv`, `git`, `node`). PAS de cmdlets PowerShell. Chemins : écris "
                "`C:/Users/...` (lecteur et barres obliques), compris par le shell, les autres "
                "outils et tes scripts ; `/c/Users/...` ne marche que dans le shell. `cmd /c` "
                "s'écrit `cmd //c`. Python est celui de Windows : dans tes scripts, `/tmp` "
                "devient `C:\\tmp` ; écris tes sorties dans le bac à sable de la session."
            )
        else:
            conv = (
                f"run_shell exécute du **bash** ({self.label}). Utilise les commandes unix "
                "standard (`ls`, `grep`, `cat`, `find`, `sed`, `awk`, `rm`, `2>/dev/null`…), "
                "chemins en `/`, variables `$VAR`. PAS de cmdlets PowerShell."
            )
        return (
            f"# Système\nTu tournes sur **{self.label}**. {conv} Ne réimplémente pas en "
            "shell ce qu'un outil dédié fait déjà (chercher, lister, lire un fichier)."
        )


def _windows_label() -> str:
    rel = platform.release() or ""  # renvoie "10" même sur Windows 11
    try:
        build = int((platform.version() or "0.0.0").split(".")[-1])
        if build >= 22000:  # seuil Windows 11
            rel = "11"
    except (ValueError, IndexError):
        pass
    return f"Windows {rel}".strip()


def _macos_label() -> str:
    ver = platform.mac_ver()[0] or ""
    arch = "Apple Silicon" if platform.machine() == "arm64" else "Intel"
    base = f"macOS {ver}".strip()  # "macOS" seul si la version est indisponible
    return f"{base} ({arch})"


def _linux_label() -> str:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"') or "Linux"
    except OSError:
        pass
    return "Linux"


@lru_cache(maxsize=1)
def detect() -> PlatformInfo:
    """Profil du système courant (mis en cache : l'OS ne change pas en cours d'exécution)."""
    p = sys.platform
    if p.startswith("win"):
        # Git Bash par défaut : les modèles écrivent du bash bien plus sûrement que du
        # PowerShell (session dc23325088d6 : 13 échecs sur 17 commandes, dont 9 réflexes
        # bash refusés par PowerShell 5.1). LOOM_SHELL=powershell force l'ancien shell.
        bash = (
            ""
            if os.environ.get("LOOM_SHELL", "").lower() == "powershell"
            else _git_bash()
        )
        if bash:
            return PlatformInfo(
                key="windows",
                label=_windows_label(),
                shell_kind="git-bash",
                shell_family="posix",
                is_windows=True,
                is_macos=False,
                is_linux=False,
                shell_path=bash,
            )
        return PlatformInfo(
            key="windows",
            label=_windows_label(),
            shell_kind="pwsh" if shutil.which("pwsh") else "powershell",
            shell_family="powershell",
            is_windows=True,
            is_macos=False,
            is_linux=False,
        )
    if p == "darwin":
        return PlatformInfo(
            key="macos",
            label=_macos_label(),
            shell_kind="bash",
            shell_family="posix",
            is_windows=False,
            is_macos=True,
            is_linux=False,
        )
    return PlatformInfo(
        key="linux",
        label=_linux_label(),
        shell_kind="bash",
        shell_family="posix",
        is_windows=False,
        is_macos=False,
        is_linux=True,
    )
