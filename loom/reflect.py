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

from loom.tools.base import ToolError, ToolSpec


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


# --- Anti-bluff : un verdict positif n'est cru que s'il est PROUVÉ ---------------------
_PROOF_TOOLS = frozenset({"run_shell", "check_page"})


def verdict_proven(verdict_ok: bool, saw_proof: bool) -> tuple[bool, str]:
    """Le vérificateur EST le même petit modèle : on l'empêche de se déclarer vert tout
    seul. Un `ok` positif n'est accepté QUE si une preuve outillée réelle a réussi pendant
    la vérif (un run_shell/check_page ok). Sinon -> non prouvé."""
    if verdict_ok and not saw_proof:
        return (
            False,
            "verdict positif REJETÉ : aucune preuve exécutée (ni run_shell ni check_page "
            "n'a tourné avec succès). « ça marche » sans preuve = non prouvé.",
        )
    return (verdict_ok, "")


def execute_prompt(task: Task) -> str:
    """Consigne autonome d'exécution d'UNE tâche, en proof-first."""
    files = (
        "\n".join(f"- {f}" for f in task.files) or "(localise-les toi-même si besoin)"
    )
    return (
        f"TÂCHE UNIQUE (ne fais QUE ça, ne touche à rien d'autre) :\n{task.goal}\n\n"
        f"Fichiers concernés — lis d'abord leur ÉTAT COURANT (ne suppose pas le "
        f"contenu) :\n{files}\n\n"
        f"Critère d'acceptation (la PREUVE que c'est fini) :\n{task.acceptance}\n\n"
        "Procède en PROOF-FIRST :\n"
        "1) lance d'abord le critère d'acceptation et CONSTATE qu'il échoue (sauf tâche "
        "triviale) ;\n"
        "2) écris le code minimal pour le satisfaire, puis appelle format_code ;\n"
        "3) relance le critère et CONSTATE qu'il passe, sortie réelle à l'appui.\n"
        "Termine en rapportant la sortie réelle constatée, sans rien inventer."
    )


def fix_prompt(task: Task) -> str:
    """Consigne de correction après un échec de vérification (l'evidence = la vraie sortie)."""
    return (
        f"La tâche suivante a ÉCHOUÉ à sa vérification :\n{task.goal}\n\n"
        f"Critère d'acceptation :\n{task.acceptance}\n\n"
        f"Preuve d'échec observée :\n{task.evidence[:800]}\n\n"
        "Corrige la cause RÉELLE de cet échec (lis l'état courant des fichiers, ne "
        "suppose rien), appelle format_code, puis relance le critère et CONSTATE qu'il "
        "passe. Ne fais que cette correction."
    )


def verify_prompt(task: Task) -> str:
    """Consigne du vérificateur frais : lance la preuve, puis report_verdict."""
    return (
        "Tu es un VÉRIFICATEUR à regard neuf : tu n'as pas écrit ce code, ne le juge pas "
        "de confiance.\n\n"
        f"Objectif de la tâche : {task.goal}\n"
        f"Critère d'acceptation à PROUVER : {task.acceptance}\n\n"
        "Lance RÉELLEMENT ce critère (run_shell / check_page selon le cas) et observe la "
        "sortie. Puis appelle report_verdict(ok, evidence) où `evidence` est la sortie "
        "RÉELLE observée (pas une paraphrase). N'appelle report_verdict qu'APRÈS avoir "
        "exécuté la preuve : un « ok » sans commande lancée sera rejeté."
    )


def integration_prompt(plan) -> str:
    """Consigne de la Phase 4 : prouver l'objectif d'origine de bout en bout."""
    return (
        "Toutes les tâches sont faites. VÉRIFIE maintenant l'objectif D'ORIGINE de bout "
        "en bout, comme un utilisateur final, à regard neuf.\n\n"
        f"Objectif : {plan.goal}\n"
        f"Critère de succès final à PROUVER : {plan.success_check}\n\n"
        "Lance RÉELLEMENT la preuve (run_shell / check_page), observe la sortie, puis "
        "appelle report_verdict(ok, evidence) avec la sortie RÉELLE. N'invente rien."
    )


# --- Outils INTERNES du rail (jamais dans AVAILABLE_TOOLS, non cochables) ---------------
# Ils ne donnent aucune capacité d'action : juste un canal de RETOUR typé. On les monte
# dans un registre éphémère pour récupérer du JSON validé plutôt que de parser le texte
# libre d'un petit modèle.


def make_submit_plan(holder: dict) -> ToolSpec:
    """Outil submit_plan : range le plan parsé dans holder['plan'] (ou lève ToolError ->
    message actionnable que le modèle corrige)."""

    def run(args: dict) -> str:
        plan = parse_plan(args)
        holder["plan"] = plan
        return f"Plan reçu : {len(plan.tasks)} tâche(s). (le rail prend la suite)"

    return ToolSpec(
        name="submit_plan",
        description=(
            "Rends ton plan structuré : l'objectif global ('goal'), le critère de succès "
            "final ('success_check' : comment prouver l'objectif de bout en bout), et "
            "'tasks' = la liste des petites tâches atomiques, chacune avec un critère "
            "'acceptance' EXÉCUTABLE."
        ),
        parameters={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Objectif global reformulé."},
                "success_check": {
                    "type": "string",
                    "description": "Preuve runnable de l'objectif d'origine de bout en bout.",
                },
                "tasks": {
                    "type": "array",
                    "description": "Tâches atomiques, dans l'ordre d'exécution.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "L'unique chose à faire.",
                            },
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Fichiers concernés (chemins).",
                            },
                            "acceptance": {
                                "type": "string",
                                "description": "Preuve runnable que la tâche est finie.",
                            },
                        },
                        "required": ["goal", "acceptance"],
                    },
                },
            },
            "required": ["goal", "success_check", "tasks"],
        },
        run=run,
    )


def make_report_verdict(holder: dict) -> ToolSpec:
    """Outil report_verdict : range le verdict du vérificateur dans holder['verdict']."""

    def run(args: dict) -> str:
        holder["verdict"] = {
            "ok": bool(args.get("ok")),
            "evidence": (args.get("evidence") or "").strip(),
        }
        return "Verdict enregistré."

    return ToolSpec(
        name="report_verdict",
        description=(
            "Rends ton verdict de vérification APRÈS avoir lancé la preuve : ok (true/false) "
            "et evidence = la sortie RÉELLE observée."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ok": {"type": "boolean", "description": "La preuve passe-t-elle ?"},
                "evidence": {
                    "type": "string",
                    "description": "Sortie réelle observée.",
                },
            },
            "required": ["ok", "evidence"],
        },
        run=run,
    )
