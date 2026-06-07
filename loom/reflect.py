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
                if payload.get("ok") and payload.get("name") in _PROOF_TOOLS:
                    stats["saw_proof"] = True
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
        reg = ToolRegistry([make_submit_plan(holder)])
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
                    "content": "Tu n'as pas appelé submit_plan. Appelle "
                    "submit_plan MAINTENANT avec le plan structuré (goal, success_check, tasks).",
                }
            ]
            continue
        err = validate_plan(plan, max_tasks)
        if err is None:
            break
        convo = list(messages) + [
            {
                "role": "user",
                "content": f"Plan refusé : {err}. Réémets submit_plan corrigé.",
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

            # Phase vérification : un sous-agent FRAIS lance la preuve.
            yield (
                "phase",
                {"name": "vérification", "task": idx, "detail": task.acceptance[:70]},
            )
            holder.pop("verdict", None)
            vreg = make_sub_registry([make_report_verdict(holder)])
            vs: dict = {}
            yield from _drive_subloop(
                client,
                [{"role": "user", "content": verify_prompt(task)}],
                vreg,
                vs,
                system_prompt=SUBAGENT_SYSTEM,
                model=model,
                max_tokens=max_tokens,
                permission=permission,
                done=lambda: "verdict" in holder,
            )
            verdict = holder.get("verdict") or {
                "ok": False,
                "evidence": vs.get("text", ""),
            }
            ok, note = verdict_proven(verdict["ok"], vs.get("saw_proof", False))
            task.evidence = verdict["evidence"] or note
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

    # --- Phase 4 : intégration (prouver l'objectif d'origine de bout en bout) ---
    yield (
        "phase",
        {"name": "intégration", "task": None, "detail": plan.success_check[:70]},
    )
    holder.pop("verdict", None)
    ireg = make_sub_registry([make_report_verdict(holder)])
    isum: dict = {}
    yield from _drive_subloop(
        client,
        [{"role": "user", "content": integration_prompt(plan)}],
        ireg,
        isum,
        system_prompt=SUBAGENT_SYSTEM,
        model=model,
        max_tokens=max_tokens,
        permission=permission,
        done=lambda: "verdict" in holder,
    )
    verdict = holder.get("verdict") or {"ok": False, "evidence": isum.get("text", "")}
    ok, note = verdict_proven(verdict["ok"], isum.get("saw_proof", False))
    yield ("content", _final_report(plan, verdict["evidence"] or note, success=ok))
