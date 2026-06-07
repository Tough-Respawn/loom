# loom/reflect.py
"""Harnais de réflexion (tranche mince) : décompose -> exécute -> vérifie -> intègre.

Rail déterministe sur le PROCESS (séquence, isolation, preuve), JAMAIS le contenu : le
modèle remplit le plan, le code, les critères. `run_reflective` yield les MÊMES events que
`stream_chat_tools` (+ ('phase', {...})) pour être streamé à l'identique par la web app.

Reporté au palier 2 (absent ici) : contrats partagés, accumulateur de leçons, re-découpage
auto d'une tâche bloquée, triage auto code/Q&A (ici : engagé par toggle UI).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from loom.prompts import REFLECT_DECOMPOSE, SUBAGENT_SYSTEM
from loom.tools.base import ToolError, ToolRegistry, ToolSpec


@dataclass
class Task:
    goal: str
    files: list[str] = field(default_factory=list)
    acceptance: str = ""
    status: str = "pending"  # pending | in_progress | done | blocked
    evidence: str = ""


@dataclass
class Behavior:
    desc: str  # "cliquer une cellule la révèle"
    step: dict = field(
        default_factory=dict
    )  # {op, selector, text?, expect:{selector,check,value,cmp?}}


@dataclass
class Plan:
    goal: str
    success_check: str
    tasks: list[Task] = field(default_factory=list)
    program_type: str = (
        "script"  # html_game | web_page | cli | python_lib | api | script
    )
    launch: str = ""  # page .html / commande de lancement
    behaviors: list[Behavior] = field(default_factory=list)


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


_WEB_TYPES = frozenset({"html_game", "web_page"})


def parse_spec(args: dict) -> Plan:
    """Construit un Plan enrichi depuis les arguments de submit_spec. Réutilise parse_plan
    pour goal/success_check/tasks, ajoute program_type/launch/behaviors."""
    plan = parse_plan(args)
    plan.program_type = (args.get("program_type") or "script").strip()
    plan.launch = (args.get("launch") or "").strip()
    raw = args.get("behaviors") or []
    behaviors: list[Behavior] = []
    if isinstance(raw, list):
        for b in raw:
            if isinstance(b, dict):
                step = b.get("step") if isinstance(b.get("step"), dict) else {}
                behaviors.append(
                    Behavior(desc=(b.get("desc") or "").strip(), step=step)
                )
    plan.behaviors = behaviors
    return plan


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
        low = t.acceptance.lower()
        if any(p in low for p in ("wc -l", "nombre de lignes", "compte de lignes")):
            return (
                f"tâche {i} : l'acceptation se base sur un COMPTE DE LIGNES du fichier — "
                "c'est un proxy trompeur (le nombre de lignes ne prouve pas que ça marche). "
                "Donne un critère de COMPORTEMENT : check_page (0 erreur console + nombre "
                "d'éléments attendus, ex. 81 cellules) ou la sortie réelle d'une commande."
            )
        if not any(h in t.acceptance.lower() for h in _PROOF_HINTS):
            return (
                f"tâche {i} : l'acceptation « {t.acceptance[:60]} » n'est pas une preuve "
                "exécutable (cite une commande run_shell/check_page, un fichier, un "
                "nombre attendu, une sortie console)"
            )
    return None


def validate_spec(plan: Plan, max_tasks: int = 30) -> str | None:
    """Gate de submit_spec : d'abord le gate de plan, puis les exigences du contrat —
    program_type connu, et pour un programme web des behaviors avec post-condition
    OBSERVABLE (un step.expect testable). Message actionnable sinon."""
    base = validate_plan(plan, max_tasks)
    if base is not None:
        return base
    types = {"html_game", "web_page", "cli", "python_lib", "api", "script"}
    if plan.program_type not in types:
        return (
            f"program_type '{plan.program_type}' inconnu : choisis parmi "
            f"{', '.join(sorted(types))}."
        )
    if plan.program_type in _WEB_TYPES:
        if not plan.launch:
            return "launch manquant : donne le fichier .html à ouvrir (la page testée)."
        if not plan.behaviors:
            return (
                "aucun behavior : déclare des comportements PROUVABLES (ex. cliquer une "
                "cellule la révèle), chacun avec un step {op, selector, expect}."
            )
        for i, b in enumerate(plan.behaviors, 1):
            exp = b.step.get("expect") if isinstance(b.step, dict) else None
            if (
                not isinstance(exp, dict)
                or not exp.get("selector")
                or not exp.get("check")
            ):
                return (
                    f"behavior {i} (« {b.desc[:40]} ») sans post-condition observable : "
                    "ajoute step.expect = {selector, check (count|class|text|absent), value}."
                )
    return None


# --- Anti-bluff : un verdict positif n'est cru que s'il est PROUVÉ ---------------------
_PROOF_TOOLS = frozenset({"run_shell", "check_page"})
# Outils qui MODIFIENT le fichier : s'ils tournent APRÈS le dernier check_page, la preuve
# est périmée (on ne valide pas un état non re-vérifié).
_WRITE_TOOLS_REFLECT = frozenset(
    {"write_file", "append_file", "edit_file", "replace_lines", "insert_lines"}
)


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


def evaluate_executor_proof(
    stats: dict, acceptance: str
) -> tuple[bool, str, str | None]:
    """Porte de validation DÉTERMINISTE (remplace le sous-agent vérificateur). Lit le DERNIER
    check_page lancé par l'exécuteur (capté dans `stats`) et décide, par le CODE et non par un
    jugement du modèle, si la tâche est prouvée :
      - un check_page a bien tourné ;
      - aucun fichier modifié APRÈS ce check_page (sinon la preuve est périmée) ;
      - 0 erreur console ;
      - si l'`acceptance` exprime un compte d'éléments attendu ET que check_page l'a compté,
        au moins ce nombre d'éléments.
    Renvoie (ok, evidence, nudge). `nudge` : consigne à ajouter à la prochaine tentative de
    fix quand l'échec vient d'une preuve MANQUANTE / PÉRIMÉE / incomplète (None si l'échec
    vient de vraies erreurs : la sortie parle d'elle-même)."""
    import re

    out = (stats.get("last_check") or "").strip()
    if not stats.get("check_ran"):
        return (
            False,
            "(aucun check_page lancé)",
            "Tu n'as PAS prouvé via check_page. Appelle l'OUTIL check_page sur la PAGE .html "
            "(avec count_selectors pour le compte attendu) et constate le résultat.",
        )
    if stats.get("dirty_after_check"):
        return (
            False,
            out[:600],
            "Tu as modifié le fichier APRÈS ton dernier check_page : la preuve est périmée. "
            "Relance check_page pour prouver l'état FINAL.",
        )
    m = re.search(r"console\s*:\s*(\d+)\s*erreur", out)
    if m is not None:
        n_err = int(m.group(1))
    elif "aucune erreur console" in out:
        n_err = 0
    else:
        return (
            False,
            out[:600],
            "Relance check_page et laisse-le rapporter sa sortie complète "
            "(format attendu : « console : N erreur(s) »).",
        )
    if n_err > 0:
        return (
            False,
            out[:600],
            None,
        )  # vraies erreurs console -> fix (sortie parlante)
    # Compte d'éléments attendu : un entier de l'acceptance NON suivi de « erreur/warning ».
    nums = [
        int(n)
        for n in re.findall(r"\b(\d{1,4})\b", acceptance)
        if not re.search(rf"\b{n}\s*(?:erreur|warning)", acceptance)
    ]
    expected = max(nums) if nums else None
    if expected:
        # Sélecteur CSS ISOLÉ (lookbehind : évite « .html » dans « index.html »).
        sel_m = re.search(r"(?<![\w])[.#][A-Za-z][\w-]*", acceptance)
        sel = sel_m.group(0) if sel_m else None
        count_m = (
            re.search(rf"{re.escape(sel)}\s*×\s*(\d+)", out)
            if sel
            else re.search(r"×\s*(\d+)", out)
        )
        if count_m is None:
            return (
                False,
                out[:600],
                f"Relance check_page avec count_selectors='{sel or '.cell'}' pour PROUVER "
                f"le compte attendu ({expected} éléments).",
            )
        if int(count_m.group(1)) < expected:
            return (False, out[:600], None)  # pas assez d'éléments rendus -> fix
    return (True, out[:600], None)


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
        "`check_page` et `format_code` sont des OUTILS : appelle-les DIRECTEMENT "
        "(check_page avec l'URL de la PAGE .html ; jamais un .css/.js seul), ne les tape "
        "JAMAIS comme commande dans run_shell. N'invente aucun résultat : sans appel "
        "d'outil réel, tu n'as pas de preuve."
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
        "Lance RÉELLEMENT la preuve en APPELANT l'outil adéquat DIRECTEMENT : `check_page` "
        "(sur la PAGE .html) pour une page web, `run_shell` pour une vraie commande système. "
        "N'écris JAMAIS « check_page » dans run_shell (c'est un outil, pas une commande). "
        "Puis appelle report_verdict(ok, evidence) où `evidence` est la sortie RÉELLE "
        "observée (pas une paraphrase). N'appelle report_verdict qu'APRÈS avoir exécuté la "
        "preuve : un « ok » sans appel d'outil réel sera rejeté."
    )


def integration_prompt(plan) -> str:
    """Consigne de la Phase 4 : prouver l'objectif d'origine de bout en bout."""
    return (
        "Toutes les tâches sont faites. PROUVE l'objectif D'ORIGINE de bout en bout.\n\n"
        f"Objectif : {plan.goal}\n"
        f"Critère de succès final : {plan.success_check}\n\n"
        "Appelle l'OUTIL check_page sur la PAGE .html finale (avec count_selectors pour le "
        "compte attendu, ex. '.cell') et constate le résultat. TERMINE sur ce check_page : "
        "c'est lui qui fait foi, ne modifie plus rien après. N'invente aucun résultat."
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


def make_submit_spec(holder: dict) -> ToolSpec:
    """Outil submit_spec : contrat de succès structuré rangé dans holder['plan']. Étend
    submit_plan (goal/success_check/tasks) avec program_type, launch, behaviors prouvables."""

    def run(args: dict) -> str:
        plan = parse_spec(args)
        holder["plan"] = plan
        return (
            f"Spec reçue : type={plan.program_type}, {len(plan.tasks)} tâche(s), "
            f"{len(plan.behaviors)} comportement(s). (le rail prend la suite)"
        )

    return ToolSpec(
        name="submit_spec",
        description=(
            "Rends ton CONTRAT de succès AVANT d'écrire : program_type "
            "(html_game|web_page|cli|python_lib|api|script), files (manifest), launch "
            "(fichier .html à ouvrir / commande), tasks (petites tâches atomiques avec "
            "acceptance exécutable), et behaviors = comportements PROUVABLES, chacun avec un "
            "`step` jouable {op, selector, expect:{selector, check, value}}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Objectif global reformulé."},
                "success_check": {
                    "type": "string",
                    "description": "Preuve finale de bout en bout.",
                },
                "program_type": {
                    "type": "string",
                    "enum": [
                        "html_game",
                        "web_page",
                        "cli",
                        "python_lib",
                        "api",
                        "script",
                    ],
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Manifest des fichiers attendus.",
                },
                "launch": {
                    "type": "string",
                    "description": "Fichier .html à ouvrir, ou commande de lancement.",
                },
                "tasks": {
                    "type": "array",
                    "description": "Tâches atomiques, dans l'ordre.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "files": {"type": "array", "items": {"type": "string"}},
                            "acceptance": {"type": "string"},
                        },
                        "required": ["goal", "acceptance"],
                    },
                },
                "behaviors": {
                    "type": "array",
                    "description": "Comportements à prouver par interaction réelle.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "desc": {"type": "string"},
                            "step": {
                                "type": "object",
                                "properties": {
                                    "op": {
                                        "type": "string",
                                        "enum": [
                                            "click",
                                            "rightclick",
                                            "dblclick",
                                            "hover",
                                            "type",
                                            "none",
                                        ],
                                    },
                                    "selector": {"type": "string"},
                                    "text": {"type": "string"},
                                    "expect": {
                                        "type": "object",
                                        "properties": {
                                            "selector": {"type": "string"},
                                            "check": {
                                                "type": "string",
                                                "enum": [
                                                    "count",
                                                    "class",
                                                    "text",
                                                    "absent",
                                                ],
                                            },
                                            "value": {"type": "string"},
                                            "cmp": {
                                                "type": "string",
                                                "enum": ["min", "eq"],
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "required": ["desc", "step"],
                    },
                },
            },
            "required": ["goal", "success_check", "program_type", "tasks"],
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


# --- Orchestrateur ---------------------------------------------------------------------


def _drive_subloop(
    client,
    sub_messages,
    registry,
    stats,
    *,
    system_prompt,
    model,
    max_tokens,
    permission,
    thinking=False,
    done=None,
):
    """Générateur : lance une sous-boucle stream_chat_tools, RELAIE ses events vers le haut
    et remplit `stats` en place : stats['saw_proof'] (un run_shell/check_page a réussi),
    stats['text'] (texte final accumulé).

    `done` (optionnel) : callable sans argument testé APRÈS chaque event. Dès qu'il renvoie
    vrai, on ARRÊTE la sous-boucle (et on ferme le flux). Sert aux sous-boucles à outil
    interne : une fois le plan/verdict capté dans le holder, inutile de laisser le modèle
    re-émettre submit_plan/report_verdict en boucle jusqu'au repeat_limit (tours gaspillés)."""
    stats["saw_proof"] = False
    stats["text"] = ""
    stats["check_ran"] = False  # un check_page a-t-il tourné ?
    stats["last_check"] = (
        ""  # sortie du DERNIER check_page (pour la porte déterministe)
    )
    stats["dirty_after_check"] = False  # un fichier modifié APRÈS ce check_page ?
    stream = client.stream_chat_tools(
        sub_messages,
        system_prompt,
        max_tokens,
        model=model,
        registry=registry,
        thinking=thinking,
        permission=permission,
    )
    try:
        for kind, payload in stream:
            if kind == "content" and isinstance(payload, str):
                stats["text"] += payload
            elif kind == "tool_result" and isinstance(payload, dict):
                name = payload.get("name")
                if payload.get("ok") and name in _PROOF_TOOLS:
                    stats["saw_proof"] = True
                if name == "check_page":
                    stats["check_ran"] = True
                    stats["last_check"] = (
                        payload.get("detail") or payload.get("preview") or ""
                    )
                    stats["dirty_after_check"] = False
                elif name in _WRITE_TOOLS_REFLECT and payload.get("ok"):
                    stats["dirty_after_check"] = True
            yield (kind, payload)
            if done is not None and done():
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def _plan_summary(plan: Plan) -> str:
    lines = [f"\n**Plan ({len(plan.tasks)} tâches)** — objectif : {plan.goal}"]
    lines += [f"{i}. {t.goal}" for i, t in enumerate(plan.tasks, 1)]
    lines.append(f"Critère final : {plan.success_check}")
    return "\n".join(lines)


def _blocked_report(idx: int, total: int, task: Task) -> str:
    return (
        f"\n**STOP — tâche {idx}/{total} bloquée.** Je n'avance pas sur une base non "
        f"prouvée.\nObjectif : {task.goal}\nCritère : {task.acceptance}\n"
        f"Dernière preuve d'échec :\n{task.evidence[:800]}"
    )


def _final_report(plan: Plan, evidence: str, success: bool) -> str:
    head = (
        "\n**✓ Objectif atteint et prouvé de bout en bout.**"
        if success
        else "\n**✗ Intégration NON prouvée** (les tâches passent isolément mais "
        "l'ensemble ne valide pas le critère d'origine)."
    )
    return f"{head}\nCritère : {plan.success_check}\nPreuve :\n{evidence[:1000]}"


def run_reflective(
    client,
    messages,
    system_prompt,
    *,
    make_sub_registry: Callable[..., ToolRegistry],
    model=None,
    max_tokens=2048,
    permission=None,
    max_tasks: int = 30,
    max_fix_attempts: int = 3,
    max_decompose_retries: int = 2,
) -> Iterator[tuple[str, object]]:
    """Rail de réflexion. `make_sub_registry(extra_specs=None)` fabrique un registre frais
    de sous-agent (outils complets + extras éventuels). Yield les events de stream_chat_tools
    (+ 'phase'). Sur blocage d'une tâche : STOP + rapport (re-découpage = palier 2)."""
    holder: dict = {}

    # --- Phase 1 : décomposition (bornée + repli en mode direct) ---
    yield (
        "phase",
        {"name": "décomposition", "task": None, "detail": "découpe la demande"},
    )
    plan: Plan | None = None
    convo = list(messages)
    for _ in range(max_decompose_retries + 1):
        holder.pop("plan", None)
        reg = ToolRegistry([make_submit_spec(holder)])
        stats: dict = {}
        yield from _drive_subloop(
            client,
            convo,
            reg,
            stats,
            system_prompt=REFLECT_DECOMPOSE,
            model=model,
            max_tokens=max_tokens,
            permission=permission,
            thinking=True,
            done=lambda: "plan" in holder,
        )
        plan = holder.get("plan")
        if plan is None:
            convo = list(messages) + [
                {
                    "role": "user",
                    "content": "Tu n'as pas appelé submit_spec. Appelle "
                    "submit_spec MAINTENANT avec le plan structuré (goal, success_check, program_type, tasks).",
                }
            ]
            continue
        err = validate_spec(plan, max_tasks)
        if err is None:
            break
        convo = list(messages) + [
            {
                "role": "user",
                "content": f"Plan refusé : {err}. Réémets submit_spec corrigé.",
            }
        ]
        plan = None

    if plan is None:
        yield ("content", "\n[réflexion : plan inexploitable, repli en mode direct.]")
        yield from client.stream_chat_tools(
            messages,
            system_prompt,
            max_tokens,
            model=model,
            registry=make_sub_registry(),
            thinking=False,
            permission=permission,
        )
        return

    yield ("content", _plan_summary(plan))
    # --- Phases 2+3 : exécuter puis vérifier chaque tâche (contexte frais, anti-bluff) ---
    total = len(plan.tasks)
    for idx, task in enumerate(plan.tasks, 1):
        task.status = "in_progress"
        proven = False
        for fix_attempt in range(max_fix_attempts + 1):
            # Phase exécution (1er essai) ou fix (essais suivants).
            if fix_attempt == 0:
                yield (
                    "phase",
                    {"name": "exécution", "task": idx, "detail": task.goal[:70]},
                )
                exec_msg = execute_prompt(task)
            else:
                yield (
                    "phase",
                    {
                        "name": "fix",
                        "task": idx,
                        "detail": f"correction {fix_attempt}/{max_fix_attempts}",
                    },
                )
                exec_msg = fix_prompt(task)
            es: dict = {}
            yield from _drive_subloop(
                client,
                [{"role": "user", "content": exec_msg}],
                make_sub_registry(),
                es,
                system_prompt=SUBAGENT_SYSTEM,
                model=model,
                max_tokens=max_tokens,
                permission=permission,
            )

            # Vérification DÉTERMINISTE (plus de sous-agent vérificateur séparé) : le harnais
            # LIT le dernier check_page de l'exécuteur et tranche par le CODE (0 erreur +
            # compte attendu + preuve non périmée). ÷2 les boucles, et le verdict ne peut
            # plus être auto-jugé/bluffé par le modèle.
            yield (
                "phase",
                {"name": "vérification", "task": idx, "detail": task.acceptance[:70]},
            )
            ok, evidence, nudge = evaluate_executor_proof(es, task.acceptance)
            task.evidence = f"{evidence}\n{nudge}" if nudge else evidence
            if ok:
                proven = True
                break

        if proven:
            task.status = "done"
            yield ("content", f"\n✓ Tâche {idx}/{total} prouvée : {task.goal[:80]}")
        else:
            task.status = "blocked"
            yield (
                "phase",
                {"name": "blocage", "task": idx, "detail": "tâche non prouvée"},
            )
            yield ("content", _blocked_report(idx, total, task))
            return

    # --- Phase 4 : intégration (1 boucle) — l'agent lance check_page sur la page finale,
    # le harnais valide le success_check de façon DÉTERMINISTE (même porte que les tâches).
    yield (
        "phase",
        {"name": "intégration", "task": None, "detail": plan.success_check[:70]},
    )
    isum: dict = {}
    yield from _drive_subloop(
        client,
        [{"role": "user", "content": integration_prompt(plan)}],
        make_sub_registry(),
        isum,
        system_prompt=SUBAGENT_SYSTEM,
        model=model,
        max_tokens=max_tokens,
        permission=permission,
    )
    ok, evidence, _ = evaluate_executor_proof(isum, plan.success_check)
    yield ("content", _final_report(plan, evidence, success=ok))
