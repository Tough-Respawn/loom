"""Détection de l'OS et de ses conventions — SOURCE DE VÉRITÉ unique pour Loom.

Le shell (run_shell) ET le prompt système en dérivent : Loom identifie tout seul sur quel
système il tourne et se comporte selon SES standards (PowerShell + cmdlets sous Windows,
bash + commandes unix sous macOS/Linux, séparateurs de chemin, variables d'env…). Un seul
endroit décide -> le modèle n'est jamais orienté vers les mauvaises conventions."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class PlatformInfo:
    key: str  # "windows" | "macos" | "linux"
    label: str  # ex. "Windows 11", "macOS 14.5 (Apple Silicon)", "Ubuntu 24.04 LTS"
    shell_kind: str  # "pwsh" | "powershell" | "bash"
    shell_family: str  # "powershell" | "posix"
    is_windows: bool
    is_macos: bool
    is_linux: bool

    @property
    def shell_label(self) -> str:
        return {
            "pwsh": "PowerShell 7 (pwsh)",
            "powershell": "PowerShell",
            "bash": "bash",
        }[self.shell_kind]

    def shell_argv(self, command: str) -> list[str]:
        """argv pour exécuter `command` dans le shell natif du système."""
        if self.shell_family == "powershell":
            exe = "pwsh" if self.shell_kind == "pwsh" else "powershell"
            return [exe, "-NoProfile", "-NonInteractive", "-Command", command]
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
                "`Get-Content`, `Measure-Object`, `Remove-Item`). Variables `$env:VAR`."
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
