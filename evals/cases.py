"""Eval set de Loom : cas figés + graders déterministes (code-based).

Chaque cas cible UN travers documenté du petit modèle (cf. mémoires : confabulation
d'exécution, échecs edit_file, unix-ismes, recherche inutile, sur-outillage). Le
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
    # L'historique synthétique simule la saturation sans payer sa génération.
    history: list | None = None
    # Un seuil bas force le chemin de compaction pour ce cas.
    compact_tokens: int | None = None



_SUCCESS_RE = re.compile(
    r"\b(ça marche|ca marche|fonctionne|c'est bon|terminé|termine|ça tourne|"
    r"sans erreur|0 erreur|aucune erreur|fait\b|réussi|reussi)\b",
    re.IGNORECASE,
)
# Détecter les unixismes envoyés au shell Windows malgré le prompt.
_UNIX_RE = re.compile(r"\b(grep|ls|cat|sed|awk|head|tail|touch|rm|find)\b")
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
        if r.get("name") in ("run_shell", "check_page") and r.get("ok"):
            return True
    return False


def edit_file_failures(traj) -> int:
    """Compte les échecs d'edit_file (ex. old_string non unique ou introuvable)."""
    return sum(
        1 for r in traj.tool_results if r.get("name") == "edit_file" and not r.get("ok")
    )



_CALC_PY = """\
def add(a, b):
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


# Écrire en bytes garantit le CRLF indépendamment de Git et de l'OS.
_CONFIG_PY_CRLF = (
    "TIMEOUT = 30\r\n"
    "RETRIES = 3\r\n"
    "\r\n"
    "\r\n"
    "def effective_timeout():\r\n"
    "    # délai effectif avant abandon\r\n"
    "    return TIMEOUT * RETRIES\r\n"
)

_INVENTORY_FILES = {
    "src/alpha.py": "def alpha_load():\n    pass\n\n\ndef alpha_save():\n    pass\n",
    "src/beta.py": "def beta_run():\n    pass\n",
    "lib/gamma.py": "def gamma_parse():\n    pass\n",
    "lib/delta.py": "def delta_merge():\n    pass\n",
}
_INVENTORY_FUNCS = (
    "alpha_load",
    "alpha_save",
    "beta_run",
    "gamma_parse",
    "delta_merge",
)

# Le ballast est volontairement jetable afin que la compaction conserve la tâche.
_BALLAST_TXT = (
    "Compte-rendu détaillé de l'étape précédente du projet (archivable) : nous avons "
    "passé en revue la structure des dossiers, discuté des conventions de nommage, "
    "évalué plusieurs pistes d'optimisation qui n'ont finalement pas été retenues, et "
    "consigné de longues listes de vérifications intermédiaires sans impact sur la "
    "suite. Rien dans ce paragraphe n'est nécessaire pour la prochaine tâche. "
) * 8
_SQUEEZE_HISTORY = [
    {
        "role": "user" if i % 2 == 0 else "assistant",
        "content": f"[tour archivé n°{i + 1}] {_BALLAST_TXT}",
    }
    for i in range(8)
]

_FACTS_TXT = (
    "Référence interne du projet.\nLe code d'accès est 4732.\nNe pas diffuser.\n"
)


def _noop(ws: Path) -> None:
    pass


def _seed_config_crlf(ws: Path) -> None:
    (ws / "config.py").write_bytes(_CONFIG_PY_CRLF.encode("utf-8"))


def _seed_inventory(ws: Path) -> None:
    for rel, content in _INVENTORY_FILES.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _seed_facts(ws: Path) -> None:
    (ws / "facts.txt").write_text(_FACTS_TXT, encoding="utf-8")


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


# Le BOM UTF-16 reproduit les fichiers générés par défaut avec PowerShell.
_SETTINGS_UTF16 = "MaxRetries=3\r\nTimeout=30\r\nVerbose=false\r\n"


def _seed_settings_utf16(ws: Path) -> None:
    (ws / "settings.txt").write_bytes(_SETTINGS_UTF16.encode("utf-16"))


# Les sous-dossiers vérifient qu'un motif nu reste récursif.
_PYPROJ_FILES = {
    "main.py": "print('root')\n",
    "src/core.py": "def core():\n    pass\n",
    "src/util/helpers.py": "def helper():\n    pass\n",
    "README.md": "# projet\n",
}


def _seed_pyproj(ws: Path) -> None:
    for rel, content in _PYPROJ_FILES.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")




def _check_edit_block(traj, ws: Path) -> dict:
    edited_by_block = used(traj, "edit_file")
    not_lazy_rewrite = not any(
        (a.get("path", "").endswith("calc.py")) for a in calls_to(traj, "write_file")
    )
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
        "édite par bloc (edit_file)": edited_by_block,
        "ne réécrit pas tout au write_file": not_lazy_rewrite,
        # Informatif: juger le résultat final, pas un ré-édit redondant raté.
        "_zéro échec edit_file": edit_file_failures(traj) == 0,
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
    # Accepter PowerShell idiomatique ou l'outil dédié.
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
    read_calls = calls_to(traj, "read_file")
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


def _check_crlf_edit(traj, ws: Path) -> dict:
    edited_by_block = used(traj, "edit_file")
    not_lazy_rewrite = not any(
        (a.get("path", "").endswith("config.py")) for a in calls_to(traj, "write_file")
    )
    fixed = False
    try:
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "import config; assert config.effective_timeout()==95; print('OK')",
            ],
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=20,
        )
        fixed = r.returncode == 0
    except Exception:
        fixed = False
    # L'édition doit préserver les fins de ligne CRLF d'origine.
    crlf_kept = False
    try:
        crlf_kept = b"\r\n" in (ws / "config.py").read_bytes()
    except OSError:
        crlf_kept = False
    return {
        "édite par bloc (edit_file)": edited_by_block,
        "ne réécrit pas tout au write_file": not_lazy_rewrite,
        "E2E: effective_timeout()==95": fixed,
        "fins de ligne CRLF conservées": crlf_kept,
        "_zéro échec edit_file": edit_file_failures(traj) == 0,
    }


def _check_dispatch(traj, ws: Path) -> dict:
    # Tolérer une omission, pas un inventaire largement inventé.
    text = traj.final_text or ""
    named = sum(1 for f in _INVENTORY_FUNCS if f in text)
    explored = (
        used(traj, "dispatch_agent")
        or used(traj, "read_file")
        or used(traj, "search_text")
        or used(traj, "list_dir")
        or used(traj, "find_files")
    )
    return {
        "a exploré le projet (outils)": explored,
        "synthèse correcte (>=3 fonctions nommées)": named >= 3,
        # Informatif: la bonne réponse prime sur le chemin de délégation attendu.
        "_a délégué (dispatch_agent)": used(traj, "dispatch_agent"),
    }


def _check_context_squeeze(traj, ws: Path) -> dict:
    # Le ballast dépasse volontairement le seuil de compaction.
    compacted = any(
        str(r.get("name", "")).startswith("(compaction")
        or str(r.get("name", "")) in ("(résumé de session)",)
        for r in traj.tool_results
    )
    read_facts = any(
        "facts" in str(a.get("path", "")) or "facts" in str(a.get("command", ""))
        for _, a in traj.tool_calls
    )
    return {
        "compaction déclenchée (historique saturé)": compacted,
        "lit facts.txt malgré la compaction": read_facts,
        "E2E: répond 4732": "4732" in (traj.final_text or ""),
    }


def _check_edit_utf16(traj, ws: Path) -> dict:
    # Vérifier à la fois le contenu et la conservation de l'UTF-16.
    new_value = False
    still_utf16 = False
    try:
        raw = (ws / "settings.txt").read_bytes()
        still_utf16 = raw[:2] in (b"\xff\xfe", b"\xfe\xff")  # BOM UTF-16 conservé
        text = raw.decode("utf-16", errors="replace")
        new_value = "MaxRetries=5" in text and "Timeout=30" in text
    except OSError:
        pass
    return {
        "édite par bloc (edit_file)": used(traj, "edit_file"),
        "ne réécrit pas tout au write_file": not any(
            a.get("path", "").endswith("settings.txt")
            for a in calls_to(traj, "write_file")
        ),
        "E2E: MaxRetries passé à 5": new_value,
        "fichier reste un UTF-16 lisible (pas corrompu)": still_utf16,
        "_zéro échec edit_file": edit_file_failures(traj) == 0,
    }


def _check_calc_power(traj, ws: Path) -> dict:
    # Normaliser les séparateurs de milliers avant d'accepter l'arrondi 1628/1629.
    text = re.sub(r"[\s \xa0]", "", traj.final_text or "")
    correct = bool(re.search(r"1628|1629", text))
    return {
        "utilise l'outil calculate": used(traj, "calculate"),
        "E2E: résultat correct (~1628.89)": correct,
    }


def _check_search_recursive(traj, ws: Path) -> dict:
    text = traj.final_text or ""
    # Accepter le compte total ou la mention explicite des fichiers imbriqués.
    found_all = bool(re.search(r"\b3\b", text)) or (
        "core.py" in text and "helpers.py" in text
    )
    explored = (
        used(traj, "find_files")
        or used(traj, "search_text")
        or used(traj, "list_dir")
        or used(traj, "run_shell")
    )
    return {
        "a cherché (outil)": explored,
        "E2E: trouve les .py en sous-dossiers (récursif)": found_all,
    }



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
        id="crlf_edit",
        prompt=(
            "Dans le fichier config.py de ce dossier, modifie effective_timeout pour "
            "ajouter une marge de 5 secondes au résultat (TIMEOUT * RETRIES + 5). "
            "Ne change rien d'autre."
        ),
        rubric=(
            "effective_timeout() de config.py retourne TIMEOUT * RETRIES + 5 (soit 95), "
            "via une édition chirurgicale, sans réécrire le fichier ni casser le reste."
        ),
        setup=_seed_config_crlf,
        check=_check_crlf_edit,
    ),
    EvalCase(
        id="dispatch_probe",
        prompt=(
            "Fais l'inventaire de ce projet : pour chacun des dossiers src/ et lib/, "
            "liste les fonctions définies dans les fichiers .py, puis rends-moi une "
            "synthèse courte (dossier -> fonctions). C'est un travail d'exploration "
            "volumineux : délègue-le si c'est plus efficace."
        ),
        rubric=(
            "La synthèse liste correctement les fonctions des deux dossiers (alpha_load, "
            "alpha_save, beta_run dans src/ ; gamma_parse, delta_merge dans lib/)."
        ),
        setup=_seed_inventory,
        check=_check_dispatch,
    ),
    EvalCase(
        id="context_squeeze",
        prompt=(
            "Lis le fichier facts.txt de ce dossier et donne-moi le code d'accès, "
            "rien d'autre."
        ),
        rubric=(
            "Malgré un long historique de conversation saturé (compacté), l'agent lit "
            "facts.txt et répond le code 4732."
        ),
        setup=_seed_facts,
        check=_check_context_squeeze,
        history=_SQUEEZE_HISTORY,
        compact_tokens=1500,
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
    EvalCase(
        id="edit_utf16",
        prompt=(
            "Dans le fichier settings.txt de ce dossier, passe MaxRetries de 3 à 5. "
            "Ne change rien d'autre."
        ),
        rubric=(
            "settings.txt (encodé UTF-16, comme un fichier PowerShell) contient "
            "MaxRetries=5, via une édition chirurgicale, sans corrompre l'encodage."
        ),
        setup=_seed_settings_utf16,
        check=_check_edit_utf16,
    ),
    EvalCase(
        id="calc_power",
        prompt=(
            "Un capital de 1000€ placé à 5% par an pendant 10 ans vaut 1000 × 1.05^10. "
            "Calcule ce montant avec l'outil et donne-le."
        ),
        rubric="L'agent calcule 1000 × 1.05^10 ≈ 1628.89 via l'outil calculate.",
        setup=_noop,
        check=_check_calc_power,
    ),
    EvalCase(
        id="search_recursive",
        prompt=(
            "Combien de fichiers .py y a-t-il dans ce projet (dossier et sous-dossiers) ? "
            "Liste-les."
        ),
        rubric=(
            "L'agent trouve les 3 fichiers .py (main.py, src/core.py, "
            "src/util/helpers.py), y compris ceux en sous-dossiers."
        ),
        setup=_seed_pyproj,
        check=_check_search_recursive,
    ),
]
