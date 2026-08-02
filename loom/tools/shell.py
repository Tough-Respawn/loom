"""Outil run_shell : exécute une commande shell bornée au workspace.

Le plus dangereux des outils. Double barrière de sécurité :
1. la deny-list DURE de `loom.permissions` (`_is_hard_denied`) est évaluée
   AVANT tout lancement — une commande destructrice lève `ToolError`
   et le sous-processus n'est JAMAIS lancé (indépendant de l'UI) ;
2. l'exécution se fait dans `workspace_dir`, sortie tronquée, timeout dur QUI TUE
   tout l'arbre de process (une GUI/serveur qui ne rend pas la main ne doit pas
   figer la boucle ni laisser d'orphelin).

Détection d'OS centralisée dans `loom.runtime.platform_info` (source de vérité unique,
partagée avec le prompt système) : PowerShell sous Windows, bash sous macOS/Linux.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from loom.permissions import _is_hard_denied
from loom.runtime.platform_info import detect
from loom.tools.base import AVAILABLE_TOOLS, ToolError, ToolSpec

# Détecter les noms d'outils saisis comme commandes pour produire une erreur actionnable.
_LOOM_TOOL_NAMES = frozenset(t["name"] for t in AVAILABLE_TOOLS)

# Suggérer un équivalent PowerShell seulement après un échec réel de résolution.
_UNIX_EQUIV = {
    "grep": "Select-String -Pattern <motif> <fichiers>",
    "sed": "(Get-Content f) -replace 'a','b' | Set-Content f",
    "awk": "Get-Content f | ForEach-Object { ($_ -split ' ')[0] }",
    "head": "Get-Content f -TotalCount N   (ou | Select-Object -First N)",
    "tail": "Get-Content f -Tail N",
    "wc": "(Get-Content f | Measure-Object -Line).Lines",
    "touch": "New-Item -ItemType File f   (ou : if (!(Test-Path f)) { New-Item f })",
    "which": "(Get-Command nom).Source",
    "export": "$env:NOM = 'valeur'",
    "man": "Get-Help <cmdlet>",
    "df": "Get-PSDrive",
    "sudo": "(pas de sudo : lance un terminal en administrateur)",
}


def _unix_ism_hint(command: str, stderr: str) -> str:
    """Si la commande a échoué en « commande inconnue » et que son 1er mot est un
    unix-isme connu, renvoie l'équivalent PowerShell (préfixé d'un saut de ligne).
    Sinon ''. Non intrusif : n'agit qu'APRÈS un vrai échec, jamais en préemption."""
    if detect().shell_kind != "powershell":
        return ""
    low = (stderr or "").lower()
    first = command.split(maxsplit=1)[0].strip("'\"`").lower() if command else ""
    # Sous PowerShell 5.1, curl/wget sont des alias et échouent différemment des inconnues.
    if first in ("curl", "wget") and (
        "paramètre" in low or "parameter" in low or "invoke-webrequest" in low
    ):
        return (
            f"\nAstuce PowerShell : « {first} » est ici un alias d'Invoke-WebRequest "
            "(flags unix incompatibles) -> utilise curl.exe (le vrai curl) ou "
            "Invoke-RestMethod -Uri <url>"
        )
    # Couvrir aussi le message localisé de Windows.
    if (
        "not recognized" not in low
        and "commandnotfoundexception" not in low
        and "pas reconnu" not in low
    ):
        return ""
    equiv = _UNIX_EQUIV.get(first)
    return (
        f"\nAstuce PowerShell : « {first} » n'existe pas ici -> {equiv}"
        if equiv
        else ""
    )


def _shell_argv(command: str) -> list[str]:
    """argv adapté à l'OS courant (délègue à la détection centrale)."""
    return detect().shell_argv(command)


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
        if detect().is_windows:
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
    workspace_dir: str, timeout: int = 180, max_output: int = 8000
) -> ToolSpec:
    """Outil run_shell borné au workspace, deny-list dure, timeout (tue l'arbre) et troncature."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            raise ToolError("argument 'command' manquant")
        # Refuser les commandes destructrices avant tout lancement.
        if _is_hard_denied(command, []):
            raise ToolError("commande interdite par la politique de sécurité")
        # Un outil Loom n'est pas un exécutable shell.
        first = command.split(maxsplit=1)[0].strip("'\"`").lower()
        if first in _LOOM_TOOL_NAMES:
            raise ToolError(
                f"« {first} » est un OUTIL Loom, PAS une commande shell : ne le lance pas "
                "via run_shell. Appelle-le DIRECTEMENT comme outil, avec ses arguments "
                "(ex. check_page avec url=<chemin .html> ; format_code avec path=<fichier>). "
                "run_shell est réservé aux VRAIES commandes système (python, git, npm, …)."
            )
        # PowerShell 5.1 ne comprend pas `&&` ni `||`.
        if detect().shell_kind == "powershell" and ("&&" in command or "||" in command):
            raise ToolError(
                "PowerShell 5.1 ne supporte pas '&&'/'||'. Utilise ';' entre les "
                "commandes (et teste $LASTEXITCODE), ou fais des appels run_shell séparés."
            )
        # Un groupe isolé permet de tuer toute la descendance au timeout.
        popen_kwargs: dict = {}
        if not detect().is_windows:
            popen_kwargs["start_new_session"] = True
        # Forcer UTF-8 évite le mojibake des scripts Python sous Windows localisé.
        _env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.Popen(
                _shell_argv(command),
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                env=_env,
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
            # Un timeout de GUI ou serveur doit décourager une relance identique.
            return (
                f"erreur: « {command[:80]} » n'a pas rendu la main après {timeout}s et a été "
                "ARRÊTÉE (process et descendance tués). Une commande qui ne se termine jamais "
                "est presque toujours une appli à FENÊTRE (Tkinter/PyQt/jeu/GUI) ou un SERVEUR "
                "qui tourne en boucle. run_shell ne peut NI afficher NI piloter une fenêtre : "
                "NE RELANCE PAS cette commande à l'identique. Pour vérifier un programme à "
                "interface, teste-le SANS lancer sa boucle d'événements (ex. "
                '`python -c "import minesweeper"` pour confirmer qu\'il se construit sans '
                "erreur, ou un test des fonctions de logique), ou laisse l'utilisateur "
                "l'ouvrir lui-même. Pour une page web servie par un SERVEUR (Next.js/Vite/Flask), utilise serve_and_check (il demarre le serveur, verifie, puis l'arrete). Pour une page .html statique, utilise check_page."
            )
        stdout = (stdout or "").strip("\r\n")
        stderr = (stderr or "").strip("\r\n")
        # Montrer la sortie avant le statut rend l'aperçu immédiatement utile.
        if stdout and stderr:
            body = f"{stdout}\n--- stderr ---\n{stderr}"
        elif stderr:
            body = stderr
        else:
            body = stdout or "(aucune sortie)"
        body = _truncate(body, max_output)
        # Le préfixe d'erreur pilote le statut; le vrai diagnostic doit rester en tête.
        if proc.returncode != 0:
            head = (stderr or stdout or "").strip().split("\n")[0][
                :200
            ] or "commande échouée"
            hint = _unix_ism_hint(command, stderr)
            return f"erreur: exit {proc.returncode} — {head}{hint}\n{body}"
        return body

    return ToolSpec(
        name="run_shell",
        description=(
            f"Runs a shell command ({detect().shell_label}) in the workspace and "
            "returns the exit code, stdout and stderr. Write for THAT shell only — "
            "on Windows that means PowerShell syntax, never unix-isms (ls|wc, grep, "
            "dir /b). To START-CHECK-STOP a web server (Next.js/Vite/Flask), use "
            "serve_and_check. The command must TERMINATE (no GUI or server loop: "
            "killed at the timeout). Destructive commands are refused."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run in the workspace.",
                }
            },
            "required": ["command"],
        },
        run=run,
    )
