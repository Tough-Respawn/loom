# loom/tools/shell.py
"""Outil run_shell : exécute une commande shell bornée au workspace.

Le plus dangereux des outils. Double barrière de sécurité :
1. la deny-list DURE de `loom.permissions` (`_is_hard_denied`) est évaluée
   AVANT tout `subprocess.run` — une commande destructrice lève `ToolError`
   et le sous-processus n'est JAMAIS lancé (indépendant de l'UI) ;
2. l'exécution se fait dans `workspace_dir`, sortie tronquée, timeout dur.

Détection d'OS via `sys.platform` : PowerShell sous Windows, bash ailleurs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from loom.permissions import _is_hard_denied
from loom.tools.base import ToolError, ToolSpec


def _shell_argv(command: str) -> list[str]:
    """Renvoie l'argv adapté à l'OS. Préfère pwsh (PowerShell 7+, supporte `&&`)."""
    if sys.platform.startswith("win"):
        exe = shutil.which("pwsh") or "powershell"
        return [exe, "-NoProfile", "-NonInteractive", "-Command", command]
    return ["/bin/bash", "-lc", command]


def _truncate(text: str, max_output: int) -> str:
    """Tronque `text` à `max_output` caractères avec un marqueur explicite."""
    if len(text) > max_output:
        return text[:max_output] + "\n...[tronqué]"
    return text


def make_run_shell(
    workspace_dir: str, timeout: int = 30, max_output: int = 8000
) -> ToolSpec:
    """Outil run_shell borné au workspace, deny-list dure, timeout et troncature."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            raise ToolError("argument 'command' manquant")
        # Barrière de sécurité non contournable : refus AVANT subprocess.
        if _is_hard_denied(command, []):
            raise ToolError("commande interdite par la politique de sécurité")
        # PowerShell 5.1 (pas de pwsh) ne supporte pas '&&'/'||' -> erreur exploitable
        # par le modèle plutôt qu'un parse error opaque.
        if (
            sys.platform.startswith("win")
            and not shutil.which("pwsh")
            and ("&&" in command or "||" in command)
        ):
            raise ToolError(
                "PowerShell 5.1 ne supporte pas '&&'/'||'. Utilise ';' entre les "
                "commandes (et teste $LASTEXITCODE), ou fais des appels run_shell séparés."
            )
        try:
            proc = subprocess.run(
                _shell_argv(command),
                cwd=str(root),
                capture_output=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return f"timeout : commande interrompue après {timeout}s"
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        body = _truncate(
            f"exit={proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}", max_output
        )
        # Statut fiable : un exit non-nul = ÉCHEC. On rejoint la convention "erreur:"
        # des autres outils pour que la boucle calcule ok=False (sinon le reviewer
        # croit qu'une commande de test ratée a réussi). Le corps reste exploitable.
        if proc.returncode != 0:
            return f"erreur: commande échouée (exit {proc.returncode}).\n{body}"
        return body

    return ToolSpec(
        name="run_shell",
        description=(
            "Exécute une commande shell dans le workspace (PowerShell sous "
            "Windows, bash sinon) et renvoie le code de sortie, stdout et "
            "stderr. Les commandes destructrices sont refusées."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Commande shell à exécuter dans le workspace.",
                }
            },
            "required": ["command"],
        },
        run=run,
    )
