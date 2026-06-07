# Harnais de réflexion (tranche mince) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forcer un petit modèle local à décomposer une demande de build en tâches atomiques, exécuter chacune en contexte frais (proof-first), vérifier chacune par une preuve réelle, puis prouver l'objectif d'origine de bout en bout.

**Architecture:** Un nouveau module `loom/reflect.py` expose `run_reflective(...)`, un orchestrateur générateur qui yield les MÊMES events que `stream_chat_tools` (+ un event `phase`), donc streamé à l'identique par la web app. Il pilote des sous-boucles `stream_chat_tools` à contexte frais (mécanisme de `dispatch_agent`) et capture le plan / les verdicts via deux outils internes (`submit_plan`, `report_verdict`) qui ne font que renvoyer du JSON validé. Engagé par un toggle UI (le triage auto code/Q&A est reporté au palier 2).

**Tech Stack:** Python 3 (dataclasses, générateurs), Flask + SSE, SDK openai, llama.cpp/llama-swap. Outillage : `uv` (run/smoke), `ruff` (format/lint). Pas de suite pytest (cf. mémoire `loom-pas-de-tests`) : vérif par smokes `uv run python -c` + un smoke à client factice jetable + run live.

**Hors périmètre (palier 2, déjà spécifiés) :** contrats partagés, accumulateur de leçons, re-découpage auto d'une tâche bloquée, triage automatique. Ici : sur blocage → STOP + rapport.

---

## Structure des fichiers

| Fichier | Responsabilité | Action |
|---|---|---|
| `loom/reflect.py` | Dataclasses `Task`/`Plan`, fonctions pures (parse/validate/anti-bluff/prompts), outils internes (`submit_plan`/`report_verdict`), orchestrateur `run_reflective` | Créer |
| `loom/prompts/reflect.decompose.md` | Prompt système de la phase décomposition | Créer |
| `loom/prompts/__init__.py` | Charger le nouveau prompt (`REFLECT_DECOMPOSE`) | Modifier |
| `loom/tools/base.py` | Param `extra_specs` sur la construction du registre (via `build_registry`) | (inchangé — voir `__init__`) |
| `loom/tools/__init__.py` | `extra_specs` sur `build_registry` + helper exporté `build_subagent_registry` | Modifier |
| `loom/conversation.py` | Flag `reflect` (persisté) | Modifier |
| `loom/web/__main__.py` | Fabrique de sous-registre `make_sub_registry` passée à `create_app` | Modifier |
| `loom/web/app.py` | Branche `run_reflective` quand `conv.reflect`, route `/reflect`, relais de l'event `phase` | Modifier |
| `loom/web/templates/*` + `loom/web/static/app.js` | Toggle réflexion + (optionnel) rendu de l'event `phase` | Modifier |

Chaque tâche produit un changement autonome et vérifiable. Les fonctions pures (Tasks 1-3) sont smoke-testées sans modèle ; l'orchestrateur (Tasks 5-7) est smoke-testé avec un client FACTICE déterministe (Task 8), puis validé en live (Task 11).

---

## Task 1 : Dataclasses + parse + gate de validation

**Files:**
- Create: `loom/reflect.py`

- [ ] **Step 1 : Créer `loom/reflect.py` avec l'en-tête, les dataclasses, `parse_plan`, `validate_plan`**

```python
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
    "run_shell", "check_page", "pytest", "compile", "python", "node", "npm",
    ".py", ".html", ".js", ".css", "exit", "console", "http", "sortie", "renvoie",
    "affiche", "erreur", "0 ", "cellule",
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
```

- [ ] **Step 2 : Smoke — le gate accepte un bon plan et refuse les mauvais**

Run :
```bash
uv run python -c "from loom.reflect import parse_plan, validate_plan; ok=parse_plan({'goal':'jeu','success_check':'check_page montre 0 erreur','tasks':[{'goal':'grille','acceptance':'run_shell python -c affiche OK','files':['a.py']}]}); print('OK_PLAN', validate_plan(ok)); bad=parse_plan({'goal':'g','success_check':'','tasks':[]}); print('NO_SC', validate_plan(bad)); vague=parse_plan({'goal':'g','success_check':'check_page','tasks':[{'goal':'x','acceptance':'le code est propre'}]}); print('VAGUE', validate_plan(vague))"
```
Expected :
```
OK_PLAN None
NO_SC critère de succès final ('success_check') manquant : ...
VAGUE tâche 1 : l'acceptation « le code est propre » n'est pas une preuve exécutable ...
```

- [ ] **Step 3 : format + commit**

```bash
uv run ruff format loom/reflect.py && uv run ruff check loom/reflect.py
git add loom/reflect.py && git commit -m "feat(reflect): dataclasses Plan/Task + gate de validation"
```

---

## Task 2 : Anti-bluff du verdict + builders de prompts (fonctions pures)

**Files:**
- Modify: `loom/reflect.py` (ajout en fin de fichier, avant les outils internes)

- [ ] **Step 1 : Ajouter `verdict_proven` et les builders de prompts**

```python
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
    files = "\n".join(f"- {f}" for f in task.files) or "(localise-les toi-même si besoin)"
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


def integration_prompt(plan: Plan) -> str:
    """Consigne de la Phase 4 : prouver l'objectif d'origine de bout en bout."""
    return (
        "Toutes les tâches sont faites. VÉRIFIE maintenant l'objectif D'ORIGINE de bout "
        "en bout, comme un utilisateur final, à regard neuf.\n\n"
        f"Objectif : {plan.goal}\n"
        f"Critère de succès final à PROUVER : {plan.success_check}\n\n"
        "Lance RÉELLEMENT la preuve (run_shell / check_page), observe la sortie, puis "
        "appelle report_verdict(ok, evidence) avec la sortie RÉELLE. N'invente rien."
    )
```

- [ ] **Step 2 : Smoke — anti-bluff + prompts contiennent les consignes clés**

Run :
```bash
uv run python -c "from loom.reflect import verdict_proven, execute_prompt, verify_prompt, Task; print('REJECT', verdict_proven(True, False)); print('ACCEPT', verdict_proven(True, True)); t=Task(goal='g', acceptance='run_shell python x.py', files=['x.py']); print('EXEC', 'PROOF-FIRST' in execute_prompt(t)); print('VERIF', 'report_verdict' in verify_prompt(t))"
```
Expected :
```
REJECT (False, 'verdict positif REJETÉ ...')
ACCEPT (True, '')
EXEC True
VERIF True
```

- [ ] **Step 3 : format + commit**

```bash
uv run ruff format loom/reflect.py && uv run ruff check loom/reflect.py
git add loom/reflect.py && git commit -m "feat(reflect): anti-bluff du verdict + builders de prompts"
```

---

## Task 3 : Outils internes `submit_plan` / `report_verdict`

**Files:**
- Modify: `loom/reflect.py` (ajout après les builders de prompts)

- [ ] **Step 1 : Ajouter les deux outils de capture JSON**

```python
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
                            "goal": {"type": "string", "description": "L'unique chose à faire."},
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
                "evidence": {"type": "string", "description": "Sortie réelle observée."},
            },
            "required": ["ok", "evidence"],
        },
        run=run,
    )
```

- [ ] **Step 2 : Smoke — les outils rangent bien dans le holder**

Run :
```bash
uv run python -c "from loom.reflect import make_submit_plan, make_report_verdict; h={}; sp=make_submit_plan(h); print(sp.run({'goal':'g','success_check':'check_page','tasks':[{'goal':'t','acceptance':'run_shell x'}]})); print('PLAN_TASKS', len(h['plan'].tasks)); rv=make_report_verdict(h); print(rv.run({'ok':True,'evidence':'exit=0'})); print('VERDICT', h['verdict'])"
```
Expected :
```
Plan reçu : 1 tâche(s). (le rail prend la suite)
PLAN_TASKS 1
Verdict enregistré.
VERDICT {'ok': True, 'evidence': 'exit=0'}
```

- [ ] **Step 3 : format + commit**

```bash
uv run ruff format loom/reflect.py && uv run ruff check loom/reflect.py
git add loom/reflect.py && git commit -m "feat(reflect): outils internes submit_plan/report_verdict"
```

---

## Task 4 : Prompt de décomposition + chargement

**Files:**
- Create: `loom/prompts/reflect.decompose.md`
- Modify: `loom/prompts/__init__.py`

- [ ] **Step 1 : Créer `loom/prompts/reflect.decompose.md`**

```markdown
Tu es le PLANIFICATEUR de Loom. Ta SEULE mission ce tour-ci : transformer la demande de l'utilisateur en un plan de petites tâches atomiques, puis appeler l'outil `submit_plan`. Tu N'ÉCRIS PAS de code et tu n'exécutes rien maintenant — un ouvrier le fera tâche par tâche ensuite.

Raisonne en ENTONNOIR, du global au minuscule :
1. GLOBAL : reformule l'objectif en 1-2 phrases, et définis `success_check` — la preuve RUNNABLE qui montrera, à la toute fin, que l'objectif d'origine est atteint de bout en bout (ex. « check_page sur index.html : 0 erreur console ET 81 cellules cliquables », « run_shell python app.py : sort sans erreur »).
2. MOYEN : liste les grands morceaux du travail.
3. COURT : casse chaque morceau en tâches ATOMIQUES. Une tâche = UNE seule chose (une fonction, un fichier, un fix). Si une tâche contient « et » entre deux actions, coupe-la en deux.

Chaque tâche doit avoir :
- `goal` : l'unique chose à faire, précise ;
- `files` : le(s) fichier(s) qu'elle touche ;
- `acceptance` : un critère EXÉCUTABLE qui prouve qu'elle est finie — une commande `run_shell`, une vérif `check_page`, un nombre attendu, une sortie console. JAMAIS « le code est propre / ça marche » : donne la commande qui le démontre.

Vise BEAUCOUP de petites tâches plutôt que peu de grosses : plus une tâche est petite et vérifiable, moins elle peut échouer.

Quand ton plan est prêt, appelle `submit_plan(goal, success_check, tasks)`. N'écris pas le plan en texte : émets directement l'appel d'outil.
```

- [ ] **Step 2 : Charger le prompt dans `loom/prompts/__init__.py`**

Modify `loom/prompts/__init__.py` — ajouter après la ligne `SUBAGENT_SYSTEM = _load("subagent.system.md")` :

```python
REFLECT_DECOMPOSE = _load("reflect.decompose.md")
```

- [ ] **Step 3 : Smoke — le prompt se charge et `loom.reflect` s'importe**

Run :
```bash
uv run python -c "from loom.prompts import REFLECT_DECOMPOSE; print('LEN', len(REFLECT_DECOMPOSE) > 100, 'submit_plan' in REFLECT_DECOMPOSE); import loom.reflect; print('IMPORT_OK')"
```
Expected :
```
LEN True True
IMPORT_OK
```

- [ ] **Step 4 : commit**

```bash
git add loom/prompts/reflect.decompose.md loom/prompts/__init__.py
git commit -m "feat(reflect): prompt de decomposition (entonnoir + submit_plan)"
```

---

## Task 5 : Helper de sous-registre + param `extra_specs`

**Files:**
- Modify: `loom/tools/__init__.py`

- [ ] **Step 1 : Ajouter `extra_specs` à `build_registry`**

Modify `loom/tools/__init__.py` — la signature de `build_registry` : ajouter le paramètre `extra_specs` à la fin du bloc keyword-only :

```python
    permission=None,
    active_model: str | None = None,
    extra_specs: list[ToolSpec] | None = None,
) -> ToolRegistry:
```

Puis, juste avant `from loom.models_profile import load_profile` en fin de fonction, insérer :

```python
    # Outils SUPPLÉMENTAIRES injectés par l'appelant (ex. report_verdict du harnais de
    # réflexion) : montés tels quels, hors univers AVAILABLE_TOOLS (non cochables).
    if extra_specs:
        specs.extend(extra_specs)
    from loom.models_profile import load_profile
```

- [ ] **Step 2 : Ajouter le helper `build_subagent_registry` et l'exporter**

Modify `loom/tools/__init__.py` — ajouter cette fonction à la fin du fichier :

```python
def build_subagent_registry(
    workspace_dir: str,
    max_bytes: int,
    web_cfg=None,
    *,
    active_model: str | None = None,
    extra_specs: list[ToolSpec] | None = None,
) -> ToolRegistry:
    """Registre d'un sous-agent du HARNAIS de réflexion : les outils _SUBAGENT_TOOLS (donc
    PAS dispatch_agent ni manage_todos) + d'éventuels outils internes (report_verdict).
    Fabriqué frais à chaque tâche pour ne partager aucun état."""
    return build_registry(
        workspace_dir,
        max_bytes,
        _SUBAGENT_TOOLS,
        web_cfg=web_cfg,
        active_model=active_model,
        extra_specs=extra_specs,
    )
```

Et l'ajouter à `__all__` (après `"build_registry",`) :

```python
    "build_registry",
    "build_subagent_registry",
```

- [ ] **Step 3 : Smoke — un registre de sous-agent porte l'outil extra**

Run :
```bash
uv run python -c "from loom.tools import build_subagent_registry; from loom.reflect import make_report_verdict; r=build_subagent_registry('.', 100000, extra_specs=[make_report_verdict({})]); print('HAS_VERDICT', 'report_verdict' in r); print('HAS_RUNSHELL', 'run_shell' in r); print('NO_DISPATCH', 'dispatch_agent' not in r)"
```
Expected :
```
HAS_VERDICT True
HAS_RUNSHELL True
NO_DISPATCH True
```

- [ ] **Step 4 : commit**

```bash
git add loom/tools/__init__.py
git commit -m "feat(tools): extra_specs + build_subagent_registry pour le harnais"
```

---

## Task 6 : Orchestrateur — phase décomposition (avec repli)

**Files:**
- Modify: `loom/reflect.py` (ajout après les outils internes)

- [ ] **Step 1 : Ajouter le driver de sous-boucle et les helpers de texte**

```python
# --- Orchestrateur ---------------------------------------------------------------------


def _drive_subloop(
    client, sub_messages, registry, stats, *, system_prompt, model, max_tokens,
    permission, thinking=False,
):
    """Générateur : lance une sous-boucle stream_chat_tools, RELAIE ses events vers le haut
    et remplit `stats` en place : stats['saw_proof'] (un run_shell/check_page a réussi),
    stats['text'] (texte final accumulé)."""
    stats["saw_proof"] = False
    stats["text"] = ""
    for kind, payload in client.stream_chat_tools(
        sub_messages,
        system_prompt,
        max_tokens,
        model=model,
        registry=registry,
        thinking=thinking,
        permission=permission,
    ):
        if kind == "content" and isinstance(payload, str):
            stats["text"] += payload
        elif kind == "tool_result" and isinstance(payload, dict):
            if payload.get("ok") and payload.get("name") in _PROOF_TOOLS:
                stats["saw_proof"] = True
        yield (kind, payload)


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
```

- [ ] **Step 2 : Ajouter `run_reflective` — squelette + phase 1 (décompose + gate + repli)**

```python
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
    yield ("phase", {"name": "décomposition", "task": None, "detail": "découpe la demande"})
    plan: Plan | None = None
    convo = list(messages)
    for _ in range(max_decompose_retries + 1):
        holder.pop("plan", None)
        reg = ToolRegistry([make_submit_plan(holder)])
        stats: dict = {}
        yield from _drive_subloop(
            client, convo, reg, stats, system_prompt=REFLECT_DECOMPOSE,
            model=model, max_tokens=max_tokens, permission=permission, thinking=True,
        )
        plan = holder.get("plan")
        if plan is None:
            convo = list(messages) + [
                {"role": "user", "content": "Tu n'as pas appelé submit_plan. Appelle "
                 "submit_plan MAINTENANT avec le plan structuré (goal, success_check, tasks)."}
            ]
            continue
        err = validate_plan(plan, max_tasks)
        if err is None:
            break
        convo = list(messages) + [
            {"role": "user", "content": f"Plan refusé : {err}. Réémets submit_plan corrigé."}
        ]
        plan = None

    if plan is None:
        yield ("content", "\n[réflexion : plan inexploitable, repli en mode direct.]")
        yield from client.stream_chat_tools(
            messages, system_prompt, max_tokens, model=model,
            registry=make_sub_registry(), thinking=False, permission=permission,
        )
        return

    yield ("content", _plan_summary(plan))
    # (Phases 2-4 ajoutées dans les tâches suivantes)
    yield ("content", "\n[exécution des tâches : à implémenter — Task 7]")
```

- [ ] **Step 3 : Smoke — l'import et la signature tiennent**

Run :
```bash
uv run python -c "import inspect; from loom.reflect import run_reflective, _drive_subloop, _plan_summary; print('PARAMS', list(inspect.signature(run_reflective).parameters)[:3]); print('OK')"
```
Expected :
```
PARAMS ['client', 'messages', 'system_prompt']
OK
```

- [ ] **Step 4 : format + commit**

```bash
uv run ruff format loom/reflect.py && uv run ruff check loom/reflect.py
git add loom/reflect.py && git commit -m "feat(reflect): orchestrateur phase 1 (decompose + gate + repli)"
```

---

## Task 7 : Orchestrateur — phases 2-3-4 (exécute / vérifie / fixe / intègre)

**Files:**
- Modify: `loom/reflect.py` (remplace les 2 lignes placeholder de la fin de `run_reflective`)

- [ ] **Step 1 : Remplacer le placeholder par les phases 2-3-4**

Dans `run_reflective`, remplacer ces deux lignes :

```python
    # (Phases 2-4 ajoutées dans les tâches suivantes)
    yield ("content", "\n[exécution des tâches : à implémenter — Task 7]")
```

par :

```python
    # --- Phases 2+3 : exécuter puis vérifier chaque tâche (contexte frais, anti-bluff) ---
    total = len(plan.tasks)
    for idx, task in enumerate(plan.tasks, 1):
        task.status = "in_progress"
        proven = False
        for fix_attempt in range(max_fix_attempts + 1):
            # Phase exécution (1er essai) ou fix (essais suivants).
            if fix_attempt == 0:
                yield ("phase", {"name": "exécution", "task": idx, "detail": task.goal[:70]})
                exec_msg = execute_prompt(task)
            else:
                yield ("phase", {"name": "fix", "task": idx,
                                 "detail": f"correction {fix_attempt}/{max_fix_attempts}"})
                exec_msg = fix_prompt(task)
            es: dict = {}
            yield from _drive_subloop(
                client, [{"role": "user", "content": exec_msg}], make_sub_registry(), es,
                system_prompt=SUBAGENT_SYSTEM, model=model, max_tokens=max_tokens,
                permission=permission,
            )

            # Phase vérification : un sous-agent FRAIS lance la preuve.
            yield ("phase", {"name": "vérification", "task": idx, "detail": task.acceptance[:70]})
            holder.pop("verdict", None)
            vreg = make_sub_registry([make_report_verdict(holder)])
            vs: dict = {}
            yield from _drive_subloop(
                client, [{"role": "user", "content": verify_prompt(task)}], vreg, vs,
                system_prompt=SUBAGENT_SYSTEM, model=model, max_tokens=max_tokens,
                permission=permission,
            )
            verdict = holder.get("verdict") or {"ok": False, "evidence": vs.get("text", "")}
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
            yield ("phase", {"name": "blocage", "task": idx, "detail": "tâche non prouvée"})
            yield ("content", _blocked_report(idx, total, task))
            return

    # --- Phase 4 : intégration (prouver l'objectif d'origine de bout en bout) ---
    yield ("phase", {"name": "intégration", "task": None, "detail": plan.success_check[:70]})
    holder.pop("verdict", None)
    ireg = make_sub_registry([make_report_verdict(holder)])
    isum: dict = {}
    yield from _drive_subloop(
        client, [{"role": "user", "content": integration_prompt(plan)}], ireg, isum,
        system_prompt=SUBAGENT_SYSTEM, model=model, max_tokens=max_tokens,
        permission=permission,
    )
    verdict = holder.get("verdict") or {"ok": False, "evidence": isum.get("text", "")}
    ok, note = verdict_proven(verdict["ok"], isum.get("saw_proof", False))
    yield ("content", _final_report(plan, verdict["evidence"] or note, success=ok))
```

- [ ] **Step 2 : Smoke — le fichier compile et `run_reflective` est un générateur**

Run :
```bash
uv run python -c "import inspect; from loom.reflect import run_reflective; print('GEN', inspect.isgeneratorfunction(run_reflective))"
```
Expected :
```
GEN True
```

- [ ] **Step 3 : format + commit**

```bash
uv run ruff format loom/reflect.py && uv run ruff check loom/reflect.py
git add loom/reflect.py && git commit -m "feat(reflect): orchestrateur phases 2-4 (execute/verifie/fixe/integre)"
```

---

## Task 8 : Smoke à client FACTICE (valide tout le rail sans modèle)

**Files:**
- Create (jetable, NON commité): `tmp_smoke_reflect.py`

- [ ] **Step 1 : Écrire le smoke à client factice**

Create `tmp_smoke_reflect.py` :

```python
"""Smoke jetable : pilote run_reflective avec un client FACTICE déterministe (aucun modèle).
Le faux client inspecte le registre pour décider quoi faire : appeler submit_plan en phase
décompose, simuler une preuve + report_verdict en phase vérif/intégration, sinon produire
du texte d'exécution. Vérifie le déroulé bout en bout : plan -> exécute -> vérifie -> intègre."""

from loom.reflect import run_reflective
from loom.tools.base import ToolRegistry


class FakeClient:
    def stream_chat_tools(self, messages, system_prompt, max_tokens, *, model=None,
                          registry=None, thinking=True, permission=None, **kw):
        names = list(registry._specs) if registry else []
        if "submit_plan" in names:
            yield ("reasoning", "je découpe…")
            registry.run("submit_plan", {
                "goal": "faire X",
                "success_check": "run_shell python -c print(ok) affiche ok",
                "tasks": [
                    {"goal": "tâche A", "files": ["a.py"], "acceptance": "run_shell python a.py exit=0"},
                    {"goal": "tâche B", "files": ["b.py"], "acceptance": "check_page 0 erreur"},
                ],
            })
            yield ("tool_result", {"id": "1", "name": "submit_plan", "ok": True, "preview": "ok"})
        elif "report_verdict" in names:
            # simule une PREUVE réelle réussie (saw_proof=True) puis le verdict positif
            yield ("tool_result", {"id": "p", "name": "run_shell", "ok": True, "preview": "exit=0"})
            registry.run("report_verdict", {"ok": True, "evidence": "exit=0, sortie réelle"})
            yield ("tool_result", {"id": "v", "name": "report_verdict", "ok": True, "preview": "ok"})
        else:
            yield ("content", "j'ai écrit le code et lancé la preuve.")
            yield ("tool_result", {"id": "e", "name": "run_shell", "ok": True, "preview": "exit=0"})


def make_sub(extra=None):
    return ToolRegistry(list(extra or []))


events = list(run_reflective(
    FakeClient(), [{"role": "user", "content": "fais X"}], "SYS",
    make_sub_registry=make_sub, model="fake", max_tokens=256,
))
phases = [p["name"] for k, p in events if k == "phase"]
texts = "".join(p for k, p in events if k == "content")
print("PHASES", phases)
assert "décomposition" in phases and "exécution" in phases, phases
assert "vérification" in phases and "intégration" in phases, phases
assert "Objectif atteint et prouvé" in texts, texts[-300:]
print("SMOKE_OK")
```

- [ ] **Step 2 : Lancer le smoke**

Run :
```bash
uv run python tmp_smoke_reflect.py
```
Expected (fin) :
```
PHASES ['décomposition', 'exécution', 'vérification', ..., 'intégration']
SMOKE_OK
```

- [ ] **Step 3 : (anti-bluff) vérifier qu'un verdict SANS preuve est rejeté**

Run :
```bash
uv run python -c "from loom.reflect import verdict_proven; assert verdict_proven(True, False)[0] is False; assert verdict_proven(True, True)[0] is True; assert verdict_proven(False, True)[0] is False; print('ANTIBLUFF_OK')"
```
Expected :
```
ANTIBLUFF_OK
```

- [ ] **Step 4 : Supprimer le smoke jetable (pas de suite de tests persistée)**

Run :
```bash
rm tmp_smoke_reflect.py
```

Aucun commit (fichier jetable).

---

## Task 9 : Flag `reflect` sur la conversation (persisté)

**Files:**
- Modify: `loom/conversation.py`

- [ ] **Step 1 : Ajouter le champ, le setter, la (dé)sérialisation**

Modify `loom/conversation.py` :

1. Après `thinking: bool = True` (ligne ~19) ajouter :
```python
    # Mode harnais de réflexion (découpage forcé) : OFF par défaut, activé par toggle UI.
    reflect: bool = False
```

2. Après `set_thinking` ajouter :
```python
    def set_reflect(self, reflect: bool) -> None:
        self.reflect = bool(reflect)
```

3. Dans `to_dict`, après `"thinking": self.thinking,` ajouter :
```python
            "reflect": self.reflect,
```

4. Dans `from_dict`, après `thinking=bool(data.get("thinking", True)),` ajouter :
```python
            reflect=bool(data.get("reflect", False)),
```

- [ ] **Step 2 : Smoke — round-trip de persistance**

Run :
```bash
uv run python -c "from loom.conversation import Conversation; c=Conversation(system_prompt='s'); c.set_reflect(True); d=c.to_dict(); print('SER', d['reflect']); c2=Conversation.from_dict(d,'s'); print('DESER', c2.reflect); print('DEFAULT', Conversation.from_dict({},'s').reflect)"
```
Expected :
```
SER True
DESER True
DEFAULT False
```

- [ ] **Step 3 : commit**

```bash
git add loom/conversation.py
git commit -m "feat(reflect): flag reflect persiste par conversation"
```

---

## Task 10 : Câblage web — fabrique de sous-registre, route, branche, event phase

**Files:**
- Modify: `loom/web/__main__.py`
- Modify: `loom/web/app.py`

- [ ] **Step 1 : Fabrique de sous-registre dans `__main__.py`**

Modify `loom/web/__main__.py` — dans `build_app`, après la définition de `make_registry` (vers ligne 54), ajouter :

```python
    from loom.tools import build_subagent_registry

    def make_sub_registry(workspace=None, conversation=None, extra_specs=None):
        """Registre frais d'un sous-agent du harnais de réflexion (outils complets sans
        dispatch/todos, + extras comme report_verdict). Workspace = celui de la session."""
        return build_subagent_registry(
            workspace or cfg.chat.workspace_dir,
            cfg.chat.read_file_max_bytes,
            cfg.chat.web_search,
            active_model=(conversation.model if conversation else cfg.default_model),
            extra_specs=extra_specs,
        )
```

Puis dans l'appel `create_app(...)`, ajouter l'argument (après `tool_factory=make_registry,`) :

```python
        reflect_factory=make_sub_registry,
```

- [ ] **Step 2 : `create_app` accepte `reflect_factory`**

Modify `loom/web/app.py` — ajouter `reflect_factory=None` à la signature de `create_app` (à côté de `tool_factory`). Repérer la ligne `tool_factory,` (ou `tool_factory=None`) dans la signature et ajouter en dessous :

```python
    reflect_factory=None,
```

- [ ] **Step 3 : Importer `run_reflective` et brancher dans `generate()`**

Modify `loom/web/app.py` :

1. En tête du fichier (avec les autres imports `from loom...`), ajouter :
```python
from loom.reflect import run_reflective
```

2. Dans `generate()`, remplacer le bloc actuel (lignes ~354-374) :
```python
            use_tools = registry is not None and len(registry)
            if use_tools:
                source = client.stream_chat_tools(
                    conv.to_messages(),
                    system_prompt,
                    max_tokens,
                    model=conv.model or None,
                    registry=registry,
                    thinking=conv.thinking,
                    permission=permission,
                    confirm=_confirm,
                    compact_after_tokens=compact_after_tokens,
                )
            else:
                source = client.stream_chat(
```
par :
```python
            use_tools = registry is not None and len(registry)
            if use_tools and conv.reflect and reflect_factory is not None:
                # Harnais de réflexion : décompose -> exécute -> vérifie -> intègre.
                # Les sous-agents tournent sur un registre frais (sans dispatch/todos).
                def _make_sub(extra=None):
                    return reflect_factory(ws, conv, extra)

                source = run_reflective(
                    client,
                    conv.to_messages(),
                    system_prompt,
                    make_sub_registry=_make_sub,
                    model=conv.model or None,
                    max_tokens=max_tokens,
                    permission=permission,
                )
            elif use_tools:
                source = client.stream_chat_tools(
                    conv.to_messages(),
                    system_prompt,
                    max_tokens,
                    model=conv.model or None,
                    registry=registry,
                    thinking=conv.thinking,
                    permission=permission,
                    confirm=_confirm,
                    compact_after_tokens=compact_after_tokens,
                )
            else:
                source = client.stream_chat(
```

3. Dans la boucle de relais des events (après le `elif kind == "usage":` bloc, vers ligne 402), ajouter un relais pour `phase` — repérer :
```python
                    elif kind == "usage":
                        yield _sse("usage", **payload)
```
et ajouter juste après :
```python
                    elif kind == "phase":
                        yield _sse("phase", **payload)
```

- [ ] **Step 4 : Route `/reflect` (calquée sur `/thinking`)**

Modify `loom/web/app.py` — après la route `thinking_update` (vers ligne 467), ajouter :

```python
    @app.post("/reflect")
    def reflect_update():
        conv, save = _ctx()
        conv.set_reflect(request.form.get("reflect") == "1")
        save()
        return Response(str(int(conv.reflect)), mimetype="text/plain")
```

- [ ] **Step 5 : Smoke — l'app se construit avec la nouvelle fabrique**

Run :
```bash
uv run python -c "from loom.web.__main__ import build_app; from loom.config import load_config; from pathlib import Path; cfg=load_config(Path('loom/loom.config.toml')); app=build_app(cfg); print('APP_OK', any(r.rule=='/reflect' for r in app.url_map.iter_rules()))"
```
Expected :
```
APP_OK True
```

- [ ] **Step 6 : commit**

```bash
git add loom/web/__main__.py loom/web/app.py
git commit -m "feat(reflect): cablage web (sous-registre, branche run_reflective, route, event phase)"
```

---

## Task 11 : Toggle UI + rendu de l'event `phase`

**Files:**
- Modify: `loom/web/static/app.js`
- Modify: `loom/web/templates/` (le template du panneau réglages portant la case `thinking`)

- [ ] **Step 1 : Localiser la case `thinking` dans les templates**

Run :
```bash
uv run python -c "import pathlib; [print(p) for p in pathlib.Path('loom/web/templates').rglob('*.html') if 'thinking' in p.read_text(encoding='utf-8')]"
```
Expected : le(s) fichier(s) contenant la case à cocher `thinking` (ex. `index.html` ou `_settings.html`).

- [ ] **Step 2 : Ajouter la case `reflect` à côté de `thinking`**

Dans le fichier trouvé, repérer le label/checkbox `thinking` (un `<input type="checkbox" id="thinking-cb" ...>`) et ajouter juste après un bloc analogue :

```html
<label class="toggle">
  <input type="checkbox" id="reflect-cb" />
  <span>Mode réflexion (découpage forcé)</span>
</label>
```

(Adapter les classes/markup à ceux réellement utilisés par la case `thinking` voisine.)

- [ ] **Step 3 : Brancher le toggle en JS (calqué sur `thinkingCb`)**

Modify `loom/web/static/app.js` — repérer le bloc `thinkingCb` (vers ligne 565) :
```javascript
if (thinkingCb) {
  thinkingCb.addEventListener("change", () => {
    ...
  });
}
```
et ajouter en dessous un bloc analogue :
```javascript
const reflectCb = document.getElementById("reflect-cb");
if (reflectCb) {
  reflectCb.addEventListener("change", async () => {
    const fd = new FormData();
    fd.append("reflect", reflectCb.checked ? "1" : "0");
    await fetch("/reflect", { method: "POST", body: fd });
  });
}
```

- [ ] **Step 4 : Rendre l'event `phase` (séparateur de section)**

Modify `loom/web/static/app.js` — dans le `switch (evt.type)` de `onEvent`, après le `case "tool_result"` (vers ligne 315), ajouter :
```javascript
      case "phase":
        // Séparateur de phase du harnais de réflexion. Si on ignore cet event, le run
        // reste lisible (les lignes ✓/Plan/STOP sont déjà dans le texte) -> non bloquant.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "phase", name: evt.name, task: evt.task, detail: evt.detail });
        break;
```

Puis, dans la fonction de rendu d'un item (le mapping `kind` -> vue ; repérer où `kind === "tool"` est rendu), ajouter une branche minimale pour `phase` qui affiche une ligne `— <name> <detail>` (réutiliser une classe d'aspect discret existante). Si le moteur de rendu ignore un `kind` inconnu sans casser, cette branche reste optionnelle (l'event est déjà non bloquant).

- [ ] **Step 5 : Vérif visuelle rapide (lint JS via format_code si dispo, sinon visuel)**

Run :
```bash
uv run python -c "import pathlib; s=pathlib.Path('loom/web/static/app.js').read_text(encoding='utf-8'); print('PHASE_CASE', 'case \"phase\"' in s); print('REFLECT_CB', 'reflect-cb' in s)"
```
Expected :
```
PHASE_CASE True
REFLECT_CB True
```

- [ ] **Step 6 : commit**

```bash
git add loom/web/static/app.js loom/web/templates/
git commit -m "feat(reflect): toggle UI + rendu de l'event phase"
```

---

## Task 12 : Validation live (le litmus test)

> Pas de pytest : on valide le rail en conditions réelles. Le démineur n'est qu'un véhicule
> (succès visible) ; le harnais est généraliste (cf. mémoire `loom-but-harnais-generaliste`).

- [ ] **Step 1 : Lancer la stack (l'utilisateur lance lui-même, cf. mémoire `loom-pas-de-background`)**

Demander à l'utilisateur de lancer la stack (serveur modèle + `uv run python -m loom.web`), ou le faire en avant-plan court si convenu. NE PAS lancer en arrière-plan persistant.

- [ ] **Step 2 : Activer le mode réflexion dans l'UI, puis piloter via Playwright**

- Cocher « Mode réflexion (découpage forcé) ».
- Donner une demande de build à succès visible, ex. : « Crée un démineur HTML jouable à la souris dans C:/tmp/mines-reflect/ ».
- Observer via Playwright : phase `décomposition` → un plan de plusieurs petites tâches s'affiche → chaque tâche `exécution`/`vérification` → preuve réelle (`run_shell`/`check_page`) → ligne `✓ Tâche i/n prouvée` → phase `intégration` → rapport final.

- [ ] **Step 3 : Critères d'observation (le plan tient ses promesses)**

Vérifier en live :
1. Le plan a plusieurs tâches atomiques, chacune avec une acceptation exécutable.
2. Aucune tâche n'est marquée prouvée sans une vraie sortie d'outil (sinon `verdict_proven` la rejette → fix).
3. Sur une tâche qui casse au-delà de `max_fix_attempts`, le run STOPPE avec `_blocked_report` (et n'enchaîne pas).
4. La Phase 4 lance le `success_check` et conclut sur preuve réelle.
5. Une demande pure Q&A (mode réflexion OFF) passe toujours par le mode direct, inchangé.

- [ ] **Step 4 : Noter le constat live en mémoire**

Mettre à jour la mémoire `loom-harnais-reflexion-design` avec le résultat (le 4B décompose-t-il utilement ? coût/latence observés ? faut-il enchaîner le palier 2 — contrats/leçons/re-découpage — ou ajuster le prompt de décomposition ?).

---

## Self-Review (plan vs spec)

**1. Couverture du spec (tranche mince) :**
- Décomposition en entonnoir + gate structurel → Tasks 1, 4, 6. ✓
- Exécution contexte frais + proof-first → Tasks 2, 7. ✓
- Vérification preuve réelle + anti-bluff → Tasks 2, 3, 7. ✓
- Fix borné par tâche, STOP sur blocage (pas de re-découpage auto = palier 2) → Task 7. ✓
- Phase 4 intégration (success_check) → Tasks 2, 7. ✓
- Events `phase` streamés à l'identique → Tasks 6 (relais), 11 (rendu). ✓
- Engagement par toggle (triage auto = palier 2) → Tasks 9, 10, 11. ✓
- Outils communs inchangés, sous-registre sans dispatch/todos → Task 5. ✓
- Reporté explicitement : contrats, leçons, re-découpage auto, triage. ✓ (noté en tête)

**2. Placeholders :** aucun « TBD/à compléter » ; chaque step de code montre le code complet. Le seul flou assumé est Task 11 (markup UI dépendant du template réel) — instruction = calquer sur la case `thinking` voisine, avec smoke de présence.

**3. Cohérence des types/signatures :**
- `make_sub_registry(extra_specs=None)` : défini Task 10 (`_make_sub(extra=None)` → `reflect_factory(ws, conv, extra)`), consommé Task 6/7 via `make_sub_registry()` et `make_sub_registry([spec])`. ✓
- `build_subagent_registry(workspace_dir, max_bytes, web_cfg=None, *, active_model, extra_specs)` : défini Task 5, appelé Task 10. ✓
- `holder['plan']` (Plan) / `holder['verdict']` ({ok, evidence}) : produits par `make_submit_plan`/`make_report_verdict` (Task 3), lus dans `run_reflective` (Tasks 6, 7). ✓
- `_drive_subloop(... , system_prompt, ...)` remplit `stats['saw_proof']`/`stats['text']` : défini Task 6, utilisé Task 7. ✓
- Events relayés (`phase`, `content`, `tool_result`, …) : produits par `run_reflective`, relayés app.py (Task 10), rendus app.js (Task 11). ✓

---

**Plan complet.** Options d'exécution : (1) **Subagent-Driven** (un sous-agent frais par tâche, revue entre tâches) ou (2) **Inline** (exécution dans cette session, checkpoints). Laquelle ?
