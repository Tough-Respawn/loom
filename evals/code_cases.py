"""ST-05 — cas du banc A/B d'intelligence de code (hors production).

Six cas, graders DÉTERMINISTES, état disque final vérifiable, Python + JS
(langage web déjà utilisé par Loom). Chaque cas cible une classe de tâche de la
story : retrouver définition/usages, résumer un gros fichier, renommer
multi-fichiers, détecter une erreur de type/import, corriger puis PROUVER.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from evals.cases import EvalCase, calls_to, used

# ------------------------------- workspaces -----------------------------------

_BILLING = '''\
"""Facturation."""


def compute_total(items, tax=0.2):
    base = sum(items)
    return base * (1 + tax)


def compute_discount(total, rate):
    return total * rate
'''

_REPORT = """\
from billing import compute_total


def monthly(items):
    return compute_total(items, tax=0.2)
"""

_API = """\
from billing import compute_total


def quote(payload):
    items = payload.get("items", [])
    return {"total": compute_total(items)}
"""

_CLI = """\
from billing import compute_total

if __name__ == "__main__":
    print(compute_total([1, 2, 3]))
"""


def _seed_usages(ws: Path) -> None:
    (ws / "billing.py").write_text(_BILLING, encoding="utf-8")
    (ws / "report.py").write_text(_REPORT, encoding="utf-8")
    (ws / "api.py").write_text(_API, encoding="utf-8")
    (ws / "cli.py").write_text(_CLI, encoding="utf-8")


_BIG_SYMBOLS = [f"handler_{chr(97 + i)}{i:02d}" for i in range(24)]


def _seed_big(ws: Path) -> None:
    parts = ['"""Module volumineux généré pour le banc."""\n\n']
    for i, name in enumerate(_BIG_SYMBOLS):
        body = "\n".join(f"    x{j} = {j} * {i}" for j in range(12))
        parts.append(f"def {name}(a, b={i}):\n{body}\n    return a + b\n\n")
    parts.append(
        "class Pipeline:\n    def run(self):\n        pass\n\n"
        "class Registry:\n    def add(self, item):\n        pass\n"
    )
    (ws / "big.py").write_text("\n".join(parts), encoding="utf-8")


_DATE_JS = """\
export function fmtDate(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}
"""

_APP_JS = """\
import { fmtDate } from "./date.js";

export function renderHeader(now) {
  return `<h1>${fmtDate(now)}</h1>`;
}

export function renderFooter(now) {
  return `<footer>${fmtDate(now)}</footer>`;
}
"""

_REPORT_JS = """\
import { fmtDate } from "./date.js";

export function reportLine(entry) {
  return fmtDate(entry.date) + " - " + entry.label;
}
"""


def _seed_rename_js(ws: Path) -> None:
    (ws / "date.js").write_text(_DATE_JS, encoding="utf-8")
    (ws / "app.js").write_text(_APP_JS, encoding="utf-8")
    (ws / "report.js").write_text(_REPORT_JS, encoding="utf-8")


_APP_PY_BROKEN = """\
def surface(rayon):
    return math.pi * rayon ** 2


def resume(valeurs):
    total = sum(valeurs)
    return f"total={totl}"
"""


def _seed_detect_py(ws: Path) -> None:
    (ws / "app.py").write_text(_APP_PY_BROKEN, encoding="utf-8")


_SVC_PY_BROKEN = """\
def charge(chemin):
    with open(chemin, encoding="utf-8") as fh:
        return json.loads(fh.read())


def resume(data):
    return {"clefs": list(data.keys()), "n": len(data)}
"""


def _seed_fix_py(ws: Path) -> None:
    (ws / "svc.py").write_text(_SVC_PY_BROKEN, encoding="utf-8")


_STORE_JS_BROKEN = """\
export function saveEntry(entry) {
  const key = buildKey(entry.id);
  localStorage.setItem(key, JSON.stringify(entry));
  return key;
}
"""


def _seed_detect_js(ws: Path) -> None:
    (ws / "store.js").write_text(_STORE_JS_BROKEN, encoding="utf-8")


# --------------------------------- graders ------------------------------------


def _final(traj) -> str:
    return traj.final_text or ""


def _check_find_usages(traj, ws: Path) -> dict:
    text = _final(traj)
    return {
        "définition localisée (billing.py)": "billing.py" in text,
        "les 3 usages nommés (report/api/cli)": all(
            f in text for f in ("report.py", "api.py", "cli.py")
        ),
        "aucun fichier modifié": not used(traj, "edit_file")
        and not used(traj, "write_file"),
    }


def _check_big_outline(traj, ws: Path) -> dict:
    text = _final(traj)
    named = sum(1 for s in _BIG_SYMBOLS if s in text)
    # RÈGLE (décision revue ST-05, 2026-08-09) : une restitution COMPACTE d'une
    # série régulière est EXACTE si elle nomme les deux BORNES de la série
    # (handler_a00 … handler_x23) et son NOMBRE (24) — exiger les 24 noms
    # verbatim pénalisait une compression fidèle (constaté : run proto rejeté à
    # tort alors que l'inventaire était juste). L'énumération explicite (>=8
    # noms exacts) reste acceptée à l'identique.
    compact = _BIG_SYMBOLS[0] in text and _BIG_SYMBOLS[-1] in text and "24" in text
    return {
        "structure restituée (>=8 symboles OU série compacte exacte)": named >= 8
        or compact,
        "classes citées (Pipeline, Registry)": "Pipeline" in text
        and "Registry" in text,
        "des lignes sont citées": bool(re.search(r"\d{2,}", text)),
    }


def _check_rename_js(traj, ws: Path) -> dict:
    contents = {
        n: (ws / n).read_text(encoding="utf-8", errors="replace")
        for n in ("date.js", "app.js", "report.js")
    }
    old_left = sum(c.count("fmtDate") for c in contents.values())
    new_count = sum(c.count("formatDate") for c in contents.values())
    return {
        "plus AUCUN fmtDate": old_left == 0,
        # 6 occurrences seedées : 1 définition + import et 2 usages (app.js)
        # + import et 1 usage (report.js). (Grader corrigé : attendait 5 à tort,
        # tous les runs de la 1re campagne tombaient rouges sur ce seul check.)
        "formatDate partout (6 occurrences)": new_count == 6,
        "les 3 fichiers touchés cohérents": all(
            "formatDate" in c for c in contents.values()
        ),
    }


def _check_detect_py(traj, ws: Path) -> dict:
    text = _final(traj)
    return {
        "erreur import math détectée": "math" in text,
        "nom indéfini totl détecté": "totl" in text,
        "fichier NON modifié (consigne)": (ws / "app.py").read_text(encoding="utf-8")
        == _APP_PY_BROKEN,
    }


def _ruff_clean(path: Path) -> bool:
    ruff = shutil.which("ruff")
    if not ruff:  # repli : la correction minimale attendue est l'import manquant
        return "import json" in path.read_text(encoding="utf-8", errors="replace")
    proc = subprocess.run(
        [ruff, "check", "--no-fix", "--isolated", "--select", "E9,F", str(path)],
        capture_output=True,
        timeout=20,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode == 0


def _proved_after_last_edit(traj, filename: str) -> bool:
    """Vrai si une VÉRIFICATION outillée (diagnostics, shell, format) a réussi
    APRÈS la dernière écriture du fichier — la « preuve » exigée par le cas."""
    last_edit = -1
    for i, (name, args) in enumerate(traj.tool_calls):
        if name in ("edit_file", "write_file", "append_file") and filename in str(
            args.get("path", "")
        ):
            last_edit = i
    if last_edit < 0:
        return False
    for r in traj.tool_results[last_edit + 1 :]:
        if r.get("name") in ("code_diagnostics", "run_shell", "format_code") and r.get(
            "ok"
        ):
            return True
    return False


def _check_fix_prove(traj, ws: Path) -> dict:
    return {
        "E2E: svc.py réparé (diagnostics à zéro)": _ruff_clean(ws / "svc.py"),
        "preuve outillée APRÈS la dernière édition": _proved_after_last_edit(
            traj, "svc.py"
        ),
        "édition chirurgicale (pas de réécriture write_file)": not any(
            "svc.py" in str(a.get("path", "")) for a in calls_to(traj, "write_file")
        ),
    }


def _check_detect_js(traj, ws: Path) -> dict:
    text = _final(traj)
    return {
        "identifiant indéfini buildKey détecté": "buildKey" in text,
        "fichier NON modifié (consigne)": (ws / "store.js").read_text(encoding="utf-8")
        == _STORE_JS_BROKEN,
    }


CODE_CASES: list[EvalCase] = [
    EvalCase(
        id="code_find_usages",
        prompt=(
            "Dans ce projet Python : où est DÉFINIE la fonction compute_total, et "
            "où est-elle UTILISÉE ? Donne le fichier de définition et chaque "
            "fichier d'usage (avec les lignes si possible). Ne modifie rien."
        ),
        rubric="Définition (billing.py) et les trois usages (report.py, api.py, cli.py).",
        setup=_seed_usages,
        check=_check_find_usages,
    ),
    EvalCase(
        id="code_big_outline",
        prompt=(
            "Donne-moi la STRUCTURE du fichier big.py : ses classes et fonctions "
            "avec leurs lignes. Pas le code, juste l'inventaire des symboles."
        ),
        rubric="Inventaire fidèle des symboles de big.py avec leurs lignes.",
        setup=_seed_big,
        check=_check_big_outline,
    ),
    EvalCase(
        id="code_rename_js",
        prompt=(
            "Renomme la fonction fmtDate en formatDate PARTOUT dans ce projet JS "
            "(sa définition dans date.js et TOUS ses usages/imports), sans rien "
            "casser d'autre."
        ),
        rubric="Plus aucun fmtDate ; formatDate en définition, imports et usages.",
        setup=_seed_rename_js,
        check=_check_rename_js,
    ),
    EvalCase(
        id="code_detect_py",
        prompt=(
            "Y a-t-il des erreurs dans app.py ? Liste-les PRÉCISÉMENT (ligne et "
            "nature de chaque problème) SANS corriger le fichier."
        ),
        rubric="Import math manquant et nom indéfini totl, fichier intact.",
        setup=_seed_detect_py,
        check=_check_detect_py,
    ),
    EvalCase(
        id="code_fix_prove",
        prompt=(
            "Corrige les erreurs de svc.py, puis PROUVE que le diagnostic est "
            "revenu à zéro (vérification outillée APRÈS ta correction)."
        ),
        rubric="svc.py réparé et preuve outillée post-correction.",
        setup=_seed_fix_py,
        check=_check_fix_prove,
    ),
    EvalCase(
        id="code_detect_js",
        prompt=(
            "Y a-t-il un problème dans store.js ? Décris-le précisément SANS "
            "modifier le fichier."
        ),
        rubric="buildKey n'est ni défini ni importé ; fichier intact.",
        setup=_seed_detect_js,
        check=_check_detect_js,
    ),
]
