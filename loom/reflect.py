# loom/reflect.py
"""Harnais de réflexion (tranche mince) : décompose -> exécute -> vérifie -> intègre.

Rail déterministe sur le PROCESS (séquence, isolation, preuve), JAMAIS le contenu : le
modèle remplit le plan, le code, les critères. `run_reflective` yield les MÊMES events que
`stream_chat_tools` (+ ('phase', {...})) pour être streamé à l'identique par la web app.

Reporté au palier 2 (absent ici) : contrats partagés, accumulateur de leçons, re-découpage
auto d'une tâche bloquée, triage auto code/Q&A (ici : engagé par toggle UI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loom.tools.base import ToolError


@dataclass
class Task:
    goal: str
    files: list[str] = field(default_factory=list)
    acceptance: str = ""
    status: str = "pending"  # pending | in_progress | done | blocked
    evidence: str = ""


@dataclass
class Plan:
    goal: str
    success_check: str
    tasks: list[Task] = field(default_factory=list)


def parse_plan(args: dict) -> Plan:
    """Construit un `Plan` depuis les arguments de submit_plan. Lève `ToolError` (message
    actionnable) si la forme de base est invalide ; les règles métier sont au gate."""
    raw_tasks = args.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ToolError("'tasks' doit être une liste de tâches")
    tasks: list[Task] = []
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            raise ToolError("chaque tâche est un objet {goal, files, acceptance}")
        files = raw.get("files") or []
        if isinstance(files, str):  # tolère une string seule
            files = [files]
        tasks.append(
            Task(
                goal=(raw.get("goal") or "").strip(),
                files=[str(f).strip() for f in files if str(f).strip()],
                acceptance=(raw.get("acceptance") or "").strip(),
            )
        )
    return Plan(
        goal=(args.get("goal") or "").strip(),
        success_check=(args.get("success_check") or "").strip(),
        tasks=tasks,
    )


# Marqueurs d'une acceptation EXÉCUTABLE (heuristique du gate) : commande, outil de preuve,
# extension de fichier, ou notion de sortie observable. Une acceptation vague (« code
# propre ») n'en contient aucun -> refusée.
_PROOF_HINTS = (
    "run_shell",
    "check_page",
    "pytest",
    "compile",
    "python",
    "node",
    "npm",
    ".py",
    ".html",
    ".js",
    ".css",
    "exit",
    "console",
    "http",
    "sortie",
    "renvoie",
    "affiche",
    "erreur",
    "0 ",
    "cellule",
)


def validate_plan(plan: Plan, max_tasks: int = 30) -> str | None:
    """Gate structurel DUR. Renvoie None si le plan passe, sinon un message ACTIONNABLE
    qui dit exactement quoi corriger (réémis au modèle pour une nouvelle décomposition)."""
    if not plan.goal:
        return "objectif global ('goal') manquant"
    if not plan.success_check:
        return (
            "critère de succès final ('success_check') manquant : comment prouvera-t-on "
            "que l'objectif d'origine est atteint de bout en bout ?"
        )
    if not plan.tasks:
        return "aucune tâche : découpe l'objectif en plusieurs petites tâches atomiques"
    if len(plan.tasks) > max_tasks:
        return f"trop de tâches ({len(plan.tasks)}) : garde un plan <= {max_tasks}"
    for i, t in enumerate(plan.tasks, 1):
        if not t.goal:
            return f"tâche {i} sans objectif ('goal')"
        if not t.acceptance:
            return (
                f"tâche {i} sans critère d'acceptation : donne une commande/observable "
                "qui PROUVE qu'elle est finie"
            )
        if not any(h in t.acceptance.lower() for h in _PROOF_HINTS):
            return (
                f"tâche {i} : l'acceptation « {t.acceptance[:60]} » n'est pas une preuve "
                "exécutable (cite une commande run_shell/check_page, un fichier, un "
                "nombre attendu, une sortie console)"
            )
    return None
