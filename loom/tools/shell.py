# loom/tools/shell.py
"""Outil run_shell : exécute une commande shell bornée au workspace.

Le plus dangereux des outils. Double barrière de sécurité :
1. la deny-list DURE de `loom.permissions` (`_is_hard_denied`) est évaluée
   AVANT tout lancement — une commande destructrice lève `ToolError`
   et le sous-processus n'est JAMAIS lancé (indépendant de l'UI) ;
2. l'exécution se fait dans `workspace_dir`, sortie tronquée, timeout dur QUI TUE
   tout l'arbre de process (une GUI/serveur qui ne rend pas la main ne doit pas
   figer la boucle ni laisser d'orphelin).

Détection d'OS via `sys.platform` : PowerShell sous Windows, bash ailleurs.
"""

from __future__ import annotations

import os
import shutil
import signal
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


def _kill_tree(proc: subprocess.Popen) -> None:
    """Tue le process ET toute sa descendance. Au timeout, tuer le seul shell laisserait
    ses petits-enfants orphelins (ex. une GUI Tkinter lancée par powershell continue de
    tourner et bloque). On tue tout l'arbre : `taskkill /T` sous Windows, le groupe de
    process (start_new_session) sinon. Best-effort : ne lève jamais."""
    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 - best-effort, on ne casse jamais la boucle
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def make_run_shell(
    workspace_dir: str, timeout: int = 30, max_output: int = 8000
) -> ToolSpec:
    """Outil run_shell borné au workspace, deny-list dure, timeout (tue l'arbre) et troncature."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            raise ToolError("argument 'command' manquant")
        # Barrière de sécurité non contournable : refus AVANT lancement.
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
        # Popen + groupe de process isolé : permet de tuer TOUTE la descendance au timeout
        # (subprocess.run ne tue que le process direct, pas une GUI lancée par le shell).
        popen_kwargs: dict = {}
        if not sys.platform.startswith("win"):
            popen_kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(
                _shell_argv(command),
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )
        except OSError as exc:
            raise ToolError(f"impossible de lancer le shell : {exc}") from exc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa: BLE001 - reap best-effort
                pass
            # Message ACTIONNABLE : une commande qui ne rend jamais la main est presque
            # toujours une GUI ou un serveur. On dit explicitement au modèle de NE PAS
            # relancer (sinon il boucle) et comment vérifier autrement.
            return (
                f"erreur: « {command[:80]} » n'a pas rendu la main après {timeout}s et a été "
                "ARRÊTÉE (process et descendance tués). Une commande qui ne se termine jamais "
                "est presque toujours une appli à FENÊTRE (Tkinter/PyQt/jeu/GUI) ou un SERVEUR "
                "qui tourne en boucle. run_shell ne peut NI afficher NI piloter une fenêtre : "
                "NE RELANCE PAS cette commande à l'identique. Pour vérifier un programme à "
                "interface, teste-le SANS lancer sa boucle d'événements (ex. "
                '`python -c "import minesweeper"` pour confirmer qu\'il se construit sans '
                "erreur, ou un test des fonctions de logique), ou laisse l'utilisateur "
                "l'ouvrir lui-même. Pour une page web, utilise check_page."
            )
        stdout = stdout or ""
        stderr = stderr or ""
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
            "stderr. La commande doit TERMINER (pas de GUI ni de serveur qui tourne en "
            "boucle : ils seront tués au timeout). Les commandes destructrices sont refusées."
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
