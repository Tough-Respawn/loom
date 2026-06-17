"""Eval set de Loom : cas figés + graders déterministes (code-based).

Chaque cas cible UN travers documenté du petit modèle (cf. mémoires : confabulation
d'exécution, thrash replace_lines, unix-ismes, recherche inutile, sur-outillage). Le
grader code donne le poids OBJECTIF ; un juge LLM (run_eval.py) complète sur le « la tâche
est-elle accomplie » ouvert. Tous les cas sont bénins (aucun shell destructeur).

Un cas = (prompt utilisateur, rubrique pour le juge, setup du workspace, check code).
`check(traj, ws)` renvoie un dict {nom_du_check: bool} ; TOUS doivent passer pour que le
cas soit « réussi » sur ce run. La trajectoire (outils appelés, réponse finale) vient de
run_eval.Trajectory.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class EvalCase:
    id: str
    prompt: str
    rubric: str
    setup: Callable[[Path], None]
    check: Callable[[object, Path], dict]


# --- helpers de lecture de trajectoire ---------------------------------------

_SUCCESS_RE = re.compile(
    r"\b(ça marche|ca marche|fonctionne|c'est bon|terminé|termine|ça tourne|"
    r"sans erreur|0 erreur|aucune erreur|fait\b|réussi|reussi)\b",
    re.IGNORECASE,
)
# unix-ismes appelés DANS run_shell sous Windows (le prompt l'interdit)
_UNIX_RE = re.compile(r"\b(grep|ls|cat|sed|awk|head|tail|touch|rm|find)\b")
# idiomes PowerShell attendus à la place
_PS_RE = re.compile(
    r"(Get-ChildItem|\bgci\b|\bdir\b|Measure-Object|Select-String|Get-Content|Where-Object)",
    re.IGNORECASE,
)


def used(traj, name: str) -> bool:
    return any(n == name for n, _ in traj.tool_calls)


def calls_to(traj, name: str):
    return [a for n, a in traj.tool_calls if n == name]


def shell_cmds(traj) -> list[str]:
    out = []
    for a in calls_to(traj, "run_shell"):
        c = a.get("command") if isinstance(a, dict) else None
        if c:
            out.append(str(c))
    return out


def claims_success(traj) -> bool:
    return bool(_SUCCESS_RE.search(traj.final_text or ""))


def ran_proof(traj) -> bool:
    """Une vraie preuve d'exécution a-t-elle tourné (run_shell / check_page) ?"""
    for r in traj.tool_results:
        if r.get("name") in ("run_shell", "check_page", "check_interactive") and r.get(
            "ok"
        ):
            return True
    return False


def replace_lines_failures(traj) -> int:
    return sum(
        1
        for r in traj.tool_results
        if r.get("name") == "replace_lines" and not r.get("ok")
    )


# --- setups ------------------------------------------------------------------

_CALC_PY = """\
def add(a, b):
    # BUG : soustrait au lieu d'additionner
    return a - b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b
"""

_GREET_PY = 'name = "monde"\nprint(f"bonjour {name}")\n'

_NOTES_MD = (
    "# Notes projet\n\n"
    "Le serveur tourne sur le port 8080. La base est PostgreSQL 16. "
    "Le déploiement se fait via le script deploy.sh chaque vendredi.\n"
)


def _noop(ws: Path) -> None:
    pass


def _seed_calc(ws: Path) -> None:
    (ws / "calc.py").write_text(_CALC_PY, encoding="utf-8")


def _seed_greet(ws: Path) -> None:
    (ws / "greet.py").write_text(_GREET_PY, encoding="utf-8")


def _seed_txt(ws: Path) -> None:
    for n in ("a.txt", "b.txt", "notes.md", "c.txt"):
        (ws / n).write_text("x", encoding="utf-8")


def _seed_notes(ws: Path) -> None:
    d = ws / "docs"
    d.mkdir(exist_ok=True)
    (d / "notes.md").write_text(_NOTES_MD, encoding="utf-8")


# --- checks ------------------------------------------------------------------


def _check_edit_block(traj, ws: Path) -> dict:
    edited_by_block = used(traj, "replace_lines") or used(traj, "edit_file")
    not_lazy_rewrite = not any(
        (a.get("path", "").endswith("calc.py")) for a in calls_to(traj, "write_file")
    )
    # E2E : la fonction add additionne-t-elle vraiment maintenant ?
    fixed = False
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import calc; assert calc.add(2,3)==5; print('OK')"],
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=20,
        )
        fixed = r.returncode == 0
    except Exception:
        fixed = False
    return {
        "édite par bloc (replace_lines/edit_file)": edited_by_block,
        "ne réécrit pas tout au write_file": not_lazy_rewrite,
        "zéro échec replace_lines": replace_lines_failures(traj) == 0,
        "E2E: add(2,3)==5": fixed,
    }


def _check_html(traj, ws: Path) -> dict:
    html = next(ws.glob("*.html"), None)
    content = html.read_text(encoding="utf-8", errors="replace") if html else ""
    return {
        "crée un fichier .html": html is not None,
        "appelle check_page": used(traj, "check_page"),
        "succès affirmé => prouvé par check_page": (
            (not claims_success(traj)) or used(traj, "check_page")
        ),
        "HTML contient un bouton interactif": bool(
            re.search(r"<button", content, re.I)
            and re.search(r"(onclick|addEventListener)", content, re.I)
        ),
    }


def _check_does_it_work(traj, ws: Path) -> dict:
    return {
        "lance run_shell": used(traj, "run_shell"),
        "pas de confabulation (succès => preuve exécutée)": (
            (not claims_success(traj)) or ran_proof(traj)
        ),
    }


def _check_windows_shell(traj, ws: Path) -> dict:
    cmds = shell_cmds(traj)
    used_shell = bool(cmds)
    no_unix = not any(_UNIX_RE.search(c) for c in cmds)
    # bonne voie : soit PowerShell idiomatique, soit l'outil dédié (list_dir/search_text)
    good_path = (
        any(_PS_RE.search(c) for c in cmds)
        or used(traj, "list_dir")
        or used(traj, "find_files")
        or used(traj, "search_text")
    )
    return {
        "pas d'unix-isme dans run_shell": no_unix,
        "voie correcte (PowerShell ou outil dédié)": good_path,
        "_a utilisé le shell (info)": used_shell,  # informatif, non bloquant si False
    }


def _check_path_given(traj, ws: Path) -> dict:
    read_calls = calls_to(traj, "read_file") + calls_to(traj, "read_document")
    read_target = any("notes.md" in str(a.get("path", "")) for a in read_calls)
    no_search = not (
        used(traj, "find_files") or used(traj, "search_text") or used(traj, "list_dir")
    )
    return {
        "lit directement le fichier donné": read_target,
        "ne cherche pas (chemin déjà fourni)": no_search,
    }


def _check_direct_answer(traj, ws: Path) -> dict:
    return {
        "répond sans outil (calibrage d'effort)": len(traj.tool_calls) == 0,
        "réponse pertinente (sorted/key)": bool(
            re.search(r"sorted|key\s*=|\.sort\(", traj.final_text or "")
        ),
    }


# --- l'eval set --------------------------------------------------------------

CASES: list[EvalCase] = [
    EvalCase(
        id="edit_block",
        prompt=(
            "Dans le fichier calc.py de ce dossier, la fonction add est buguée : elle "
            "soustrait au lieu d'additionner. Corrige-la pour qu'elle retourne a + b."
        ),
        rubric="La fonction add de calc.py retourne a + b, sans casser le reste du fichier.",
        setup=_seed_calc,
        check=_check_edit_block,
    ),
    EvalCase(
        id="html_counter",
        prompt=(
            "Crée dans ce dossier une page counter.html : un bouton qui, à chaque clic, "
            "incrémente un compteur affiché à l'écran. Vérifie qu'elle s'affiche sans erreur."
        ),
        rubric="Une page counter.html existe, avec un bouton qui incrémente un compteur, "
        "et l'agent a vérifié son rendu (0 erreur console) au lieu de le supposer.",
        setup=_noop,
        check=_check_html,
    ),
    EvalCase(
        id="does_it_work",
        prompt="Le script greet.py de ce dossier marche-t-il ? Lance-le et dis-moi le résultat.",
        rubric="L'agent a réellement exécuté greet.py (run_shell) et rapporte sa sortie, "
        "sans prétendre que ça marche sans l'avoir lancé.",
        setup=_seed_greet,
        check=_check_does_it_work,
    ),
    EvalCase(
        id="windows_shell",
        prompt="Avec le shell, compte combien de fichiers .txt il y a dans ce dossier.",
        rubric="L'agent compte les .txt avec une commande PowerShell valide (pas d'unix "
        "grep/ls/cat) ou via un outil dédié, et donne le bon nombre (3).",
        setup=_seed_txt,
        check=_check_windows_shell,
    ),
    EvalCase(
        id="path_given",
        prompt="Résume en une phrase le fichier {NOTES_PATH}.",
        rubric="L'agent lit directement le chemin fourni et résume, sans étape de recherche.",
        setup=_seed_notes,
        check=_check_path_given,
    ),
    EvalCase(
        id="direct_answer",
        prompt=(
            "En Python, comment trier une liste de dictionnaires par la clé 'age' ? "
            "Réponds directement, pas besoin de toucher au disque."
        ),
        rubric="L'agent répond directement (sorted avec key=lambda) sans appeler d'outil.",
        setup=_noop,
        check=_check_direct_answer,
    ),
]
