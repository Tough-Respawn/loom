# Complete-program MVP (preuve interactive + submit_spec) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Donner au harnais une PREUVE INTERACTIVE (clics/saisie réels + lecture du DOM après action) et un contrat `submit_spec` (program_type + behaviors testables) ; à l'intégration, le harnais joue lui-même les behaviors et tranche de façon déterministe — « jouable » prouvé, pas « 0 erreur ».

**Architecture:** `loom/tools/browser.py` gagne `run_interactive(url, steps)` (Playwright headless qui exécute une séquence d'actions et évalue une post-condition DOM par étape) + un outil `check_interactive`. `loom/reflect.py` : `submit_plan` devient `submit_spec` (ajoute `program_type`, `launch`, `behaviors[{desc, step}]`) avec gate anti-vague ; la **Phase 4 intégration** compile les behaviors en steps et appelle `run_interactive` directement (preuve possédée par le harnais, lue par le code). La décomposition en tâches + la porte déterministe par tâche (palier 1) restent inchangées.

**Tech Stack:** Python 3.12, Playwright sync (déjà dépendance + chromium installé), Flask. Outillage : `uv run` pour Python, `uvx ruff` pour lint/format (PAS `uv run ruff`... — si, désormais ruff est dans le venv, donc `uv run ruff` marche aussi ; les deux sont OK). Pas de pytest sur Loom : smokes `uv run python -c` + Playwright réel sur des fichiers HTML temporaires + client factice.

**Contraintes :** branche `feat/harness-reflexion`. Un hook PostToolUse (autoflake/ruff) retire les imports non utilisés — écrire imports + usage dans la même écriture (ou écrire le fichier d'un coup), puis relancer le smoke.

---

## Structure des fichiers

| Fichier | Responsabilité | Action |
|---|---|---|
| `loom/tools/browser.py` | `run_interactive` (moteur d'actions Playwright + éval DOM) + outil `check_interactive` ; `check_page` inchangé | Modifier |
| `loom/tools/base.py` | `check_interactive` dans `AVAILABLE_TOOLS` | Modifier |
| `loom/tools/__init__.py` | montage de `check_interactive` dans `build_registry` + `_SUBAGENT_TOOLS` | Modifier |
| `loom/permissions.py` | `check_interactive` dans `READ_TOOLS` (charge une page, pas d'effet de bord) | Modifier |
| `loom/loom.config.toml` | `check_interactive` dans `[tools] enabled` | Modifier |
| `loom/reflect.py` | `Behavior`, champs Spec sur `Plan`, `parse_spec`/`validate_spec`, `make_submit_spec`, `_compile_behaviors`, Phase 4 → `run_interactive` | Modifier |
| `loom/prompts/reflect.decompose.md` | consignes `submit_spec` (program_type, launch, behaviors avec step exécutable) | Modifier |

---

## Task 1 : moteur `run_interactive` (Playwright : actions + éval DOM)

**Files:**
- Modify: `loom/tools/browser.py`

- [ ] **Step 1 : Ajouter `_eval_expect`, `_run_step`, `run_interactive` dans `browser.py`**

Insérer ces fonctions au niveau module (après `make_check_page`, en fin de fichier). Elles réutilisent `_resolve_in_root` et `_INSTALL_HINT` déjà importés/définis dans le fichier.

```python
def _eval_expect(page, expect: dict) -> tuple[bool, str]:
    """Évalue une post-condition DANS le DOM courant. Renvoie (ok, observé)."""
    sel = (expect.get("selector") or "").strip()
    check = (expect.get("check") or "").strip().lower()
    val = expect.get("value")
    if not sel or not check:
        return True, "(aucune post-condition)"
    try:
        if check == "count":
            n = len(page.query_selector_all(sel))
            cmp = (expect.get("cmp") or "min").lower()
            ok = n >= int(val) if cmp == "min" else n == int(val)
            return ok, f"{sel} ×{n} (attendu {cmp} {val})"
        el = page.query_selector(sel)
        if check == "absent":
            return el is None, f"{sel} {'absent' if el is None else 'présent'}"
        if el is None:
            return False, f"{sel} introuvable"
        if check == "class":
            classes = (el.get_attribute("class") or "").split()
            return str(val) in classes, f"{sel} classes={classes}"
        if check == "text":
            txt = el.inner_text()
            return str(val).lower() in txt.lower(), f"{sel} texte≈{txt[:60]!r}"
        return False, f"check inconnu '{check}'"
    except Exception as exc:  # noqa: BLE001 - une éval ratée = step en échec, pas un crash
        return False, f"évaluation échouée : {str(exc)[:120]}"


def _run_step(page, step: dict) -> dict:
    """Joue UNE action puis évalue sa post-condition. Ne lève jamais."""
    op = (step.get("op") or "none").strip().lower()
    selector = (step.get("selector") or "").strip()
    res = {"op": op, "selector": selector, "ok": False, "observed": ""}
    try:
        if op == "click":
            page.click(selector, timeout=4000)
        elif op == "rightclick":
            page.click(selector, button="right", timeout=4000)
        elif op == "dblclick":
            page.dblclick(selector, timeout=4000)
        elif op == "hover":
            page.hover(selector, timeout=4000)
        elif op == "type":
            page.fill(selector, step.get("text") or "", timeout=4000)
        elif op in ("none", "load", ""):
            pass
        else:
            res["observed"] = f"op inconnu '{op}'"
            return res
        page.wait_for_timeout(300)  # laisse le JS réagir à l'action
    except Exception as exc:  # noqa: BLE001 - action ratée = step en échec
        res["observed"] = f"action '{op}' échouée : {str(exc)[:120]}"
        return res
    res["ok"], res["observed"] = _eval_expect(page, step.get("expect") or {})
    return res


def run_interactive(workspace_dir: str, target: str, steps: list[dict]) -> dict:
    """Charge une page, JOUE `steps` (clics/saisie réels) et évalue une post-condition DOM
    après chaque action. Renvoie un dict STRUCTURÉ lu par le harnais (jamais par le modèle) :
    {url, ok, console_errors, steps:[{op,selector,ok,observed}], error}. `ok` global = 0 erreur
    console ET toutes les étapes ok. Ne lève jamais (toute panne -> ok=False + error)."""
    from pathlib import Path

    root = Path(workspace_dir)
    if target.startswith(("http://", "https://", "file://")):
        url = target
    else:
        path = _resolve_in_root(root, target)
        if not path.exists():
            return {"url": target, "ok": False, "error": f"fichier introuvable : {target}",
                    "console_errors": [], "steps": []}
        url = path.as_uri()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"url": url, "ok": False, "error": _INSTALL_HINT, "console_errors": [], "steps": []}

    console: list[tuple[str, str]] = []
    page_errors: list[str] = []
    results: list[dict] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("console", lambda m: console.append((m.type, m.text)))
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.goto(url, wait_until="load", timeout=15000)
            page.wait_for_timeout(800)
            for step in steps:
                results.append(_run_step(page, step))
            browser.close()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            return {"url": url, "ok": False, "error": _INSTALL_HINT, "console_errors": [], "steps": results}
        return {"url": url, "ok": False, "error": f"échec du chargement : {msg[:200]}",
                "console_errors": [], "steps": results}

    errors = [t for (k, t) in console if k == "error"] + page_errors
    ok = not errors and all(r["ok"] for r in results)
    return {"url": url, "ok": ok, "console_errors": errors[:8], "steps": results, "error": ""}
```

- [ ] **Step 2 : Smoke — Playwright réel sur un HTML temporaire (clic + lecture DOM)**

Run :
```bash
uv run python - <<'PY'
import tempfile, os
from loom.tools.browser import run_interactive
html = """<!doctype html><html><body>
<div class="cell"></div><div class="cell"></div><div class="cell"></div>
<button id="go" onclick="document.querySelector('.cell').classList.add('revealed')">go</button>
</body></html>"""
d = tempfile.mkdtemp(); p = os.path.join(d, "t.html"); open(p, "w").write(html)
# 1) compte initial OK ; 2) clic révèle la 1re cellule
r = run_interactive(d, "t.html", [
    {"op": "none", "expect": {"selector": ".cell", "check": "count", "value": 3, "cmp": "min"}},
    {"op": "click", "selector": "#go", "expect": {"selector": ".cell", "check": "class", "value": "revealed"}},
])
print("OK_GLOBAL", r["ok"])
print("STEPS", [(s["op"], s["ok"]) for s in r["steps"]])
# négatif : une classe qui n'apparaîtra jamais -> step en échec
r2 = run_interactive(d, "t.html", [{"op": "none", "expect": {"selector": ".cell", "check": "class", "value": "nope"}}])
print("NEG_OK_FALSE", r2["ok"] is False)
PY
```
Expected :
```
OK_GLOBAL True
STEPS [('none', True), ('click', True)]
NEG_OK_FALSE True
```

- [ ] **Step 3 : format + commit**

```bash
uvx ruff format loom/tools/browser.py && uvx ruff check loom/tools/browser.py
git add loom/tools/browser.py && git commit -m "feat(browser): run_interactive (actions Playwright reelles + eval DOM)"
```

---

## Task 2 : outil `check_interactive` + enregistrement

**Files:**
- Modify: `loom/tools/browser.py`, `loom/tools/base.py`, `loom/tools/__init__.py`, `loom/permissions.py`, `loom/loom.config.toml`

- [ ] **Step 1 : Ajouter le `ToolSpec` `make_check_interactive` (fin de `browser.py`)**

```python
def make_check_interactive(workspace_dir: str) -> ToolSpec:
    """Outil check_interactive : joue une séquence d'actions sur une page et vérifie le DOM
    après chaque action. Pour PROUVER qu'une page est jouable (pas seulement « 0 erreur »)."""

    def run(args: dict) -> str:
        target = (args.get("url") or "").strip()
        if not target:
            raise ToolError("argument 'url' manquant (page HTML à tester)")
        steps = args.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ToolError("argument 'steps' : liste non vide d'actions {op, selector, expect}")
        res = run_interactive(workspace_dir, target, steps)
        lines = [f"page : {res['url']}"]
        if res.get("error"):
            lines.append(f"erreur: {res['error']}")
        lines.append(f"console : {len(res.get('console_errors', []))} erreur(s)")
        for e in res.get("console_errors", [])[:5]:
            lines.append(f"  [erreur] {e[:160]}")
        for i, s in enumerate(res.get("steps", []), 1):
            mark = "ok" if s["ok"] else "ÉCHEC"
            lines.append(f"  étape {i} [{mark}] {s['op']} {s['selector']} -> {s['observed']}")
        lines.append("VERDICT : " + ("toutes les actions passent, 0 erreur" if res["ok"]
                                      else "au moins une action/post-condition échoue"))
        return "\n".join(lines)

    return ToolSpec(
        name="check_interactive",
        description=(
            "Prouve qu'une page HTML est JOUABLE : joue une séquence d'actions réelles "
            "(click, rightclick, dblclick, hover, type) sur des sélecteurs CSS et vérifie, "
            "APRÈS chaque action, une post-condition dans le DOM. Va plus loin que check_page "
            "(qui ne fait que charger). Utilise-le pour prouver « cliquer une cellule la "
            "révèle », « clic droit pose un drapeau », « restart réinitialise »."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Page HTML (chemin .html ou URL)."},
                "steps": {
                    "type": "array",
                    "description": "Actions à jouer dans l'ordre.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["click", "rightclick", "dblclick", "hover", "type", "none"]},
                            "selector": {"type": "string", "description": "Cible CSS de l'action."},
                            "text": {"type": "string", "description": "Texte à saisir (op=type)."},
                            "expect": {
                                "type": "object",
                                "description": "Post-condition DOM après l'action.",
                                "properties": {
                                    "selector": {"type": "string"},
                                    "check": {"type": "string", "enum": ["count", "class", "text", "absent"]},
                                    "value": {"type": "string"},
                                    "cmp": {"type": "string", "enum": ["min", "eq"]},
                                },
                            },
                        },
                        "required": ["op"],
                    },
                },
            },
            "required": ["url", "steps"],
        },
        run=run,
    )
```
Note : `value` est déclaré `string` dans le schéma (un 4B mélange types) ; `_eval_expect` fait `int(val)` pour `count`, ce qui accepte `"81"` comme `81`.

- [ ] **Step 2 : Enregistrer l'outil**

`loom/tools/base.py` — dans `AVAILABLE_TOOLS`, après l'entrée `check_page` :
```python
    {"name": "check_interactive", "label": "check_interactive", "danger": False},
```

`loom/tools/__init__.py` — dans `_SUBAGENT_TOOLS`, après `"check_page",` :
```python
    "check_interactive",
```
Et dans `build_registry`, après le bloc qui monte `check_page` :
```python
    if "check_interactive" in enabled:
        from loom.tools.browser import make_check_interactive

        specs.append(make_check_interactive(workspace_dir))
```

`loom/permissions.py` — dans `READ_TOOLS`, ajouter `"check_interactive",` (à côté de `"check_page"`).

`loom/loom.config.toml` — dans `[tools] enabled`, ajouter `"check_interactive"` à côté de `"check_page"`.

- [ ] **Step 3 : Smoke — l'outil est monté et bien typé read-only**

Run :
```bash
uv run python -c "
from loom.tools import build_registry
from loom.permissions import evaluate, PermissionConfig
r = build_registry('.', 40000, ['check_interactive'])
print('MOUNTED', 'check_interactive' in r)
print('READONLY', evaluate('check_interactive', {}, PermissionConfig(mode='ask')).action == 'allow')
"
```
Expected :
```
MOUNTED True
READONLY allow
```

- [ ] **Step 4 : commit**

```bash
uvx ruff check loom/tools/browser.py loom/tools/__init__.py loom/permissions.py
git add loom/tools/browser.py loom/tools/base.py loom/tools/__init__.py loom/permissions.py loom/loom.config.toml
git commit -m "feat(tools): outil check_interactive (monte + read-only)"
```

---

## Task 3 : `Behavior` + champs spec sur `Plan` + gate

**Files:**
- Modify: `loom/reflect.py`

- [ ] **Step 1 : Ajouter `Behavior` et étendre `Plan` (dataclasses)**

Dans `loom/reflect.py`, remplacer la dataclass `Plan` par (et ajouter `Behavior` juste avant) :
```python
@dataclass
class Behavior:
    desc: str  # "cliquer une cellule la révèle"
    step: dict = field(default_factory=dict)  # {op, selector, text?, expect:{selector,check,value,cmp?}}


@dataclass
class Plan:
    goal: str
    success_check: str
    tasks: list[Task] = field(default_factory=list)
    program_type: str = "script"  # html_game | web_page | cli | python_lib | api | script
    launch: str = ""  # page .html / commande de lancement
    behaviors: list[Behavior] = field(default_factory=list)
```

- [ ] **Step 2 : `parse_spec` (parse les args submit_spec) — ajouter après `parse_plan`**

```python
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
                behaviors.append(Behavior(desc=(b.get("desc") or "").strip(), step=step))
    plan.behaviors = behaviors
    return plan
```

- [ ] **Step 3 : `validate_spec` (gate dur) — ajouter après `validate_plan`**

```python
def validate_spec(plan: Plan, max_tasks: int = 30) -> str | None:
    """Gate de submit_spec : d'abord le gate de plan, puis les exigences du contrat —
    program_type connu, et pour un programme web des behaviors avec post-condition
    OBSERVABLE (un step.expect testable). Message actionnable sinon."""
    base = validate_plan(plan, max_tasks)
    if base is not None:
        return base
    types = {"html_game", "web_page", "cli", "python_lib", "api", "script"}
    if plan.program_type not in types:
        return (f"program_type '{plan.program_type}' inconnu : choisis parmi "
                f"{', '.join(sorted(types))}.")
    if plan.program_type in _WEB_TYPES:
        if not plan.launch:
            return "launch manquant : donne le fichier .html à ouvrir (la page testée)."
        if not plan.behaviors:
            return ("aucun behavior : déclare des comportements PROUVABLES (ex. cliquer une "
                    "cellule la révèle), chacun avec un step {op, selector, expect}.")
        for i, b in enumerate(plan.behaviors, 1):
            exp = b.step.get("expect") if isinstance(b.step, dict) else None
            if not isinstance(exp, dict) or not exp.get("selector") or not exp.get("check"):
                return (f"behavior {i} (« {b.desc[:40]} ») sans post-condition observable : "
                        "ajoute step.expect = {selector, check (count|class|text|absent), value}.")
    return None
```

- [ ] **Step 4 : Smoke — gate accepte un bon spec, refuse les mauvais**

Run :
```bash
uv run python -c "
from loom.reflect import parse_spec, validate_spec
good = parse_spec({'goal':'demineur','success_check':'check_interactive index.html','program_type':'html_game','launch':'index.html','tasks':[{'goal':'t','acceptance':'check_page index.html 81 .cell'}],'behaviors':[{'desc':'cliquer revele','step':{'op':'click','selector':'.cell','expect':{'selector':'.cell.revealed','check':'count','value':'1','cmp':'min'}}}]})
print('GOOD', validate_spec(good))
bad_type = parse_spec({'goal':'g','success_check':'x','program_type':'jeu','launch':'i.html','tasks':[{'goal':'t','acceptance':'check_page x'}],'behaviors':[{'desc':'d','step':{'op':'click','expect':{'selector':'.c','check':'class','value':'r'}}}]})
print('BAD_TYPE', validate_spec(bad_type)[:30])
no_beh = parse_spec({'goal':'g','success_check':'x','program_type':'html_game','launch':'i.html','tasks':[{'goal':'t','acceptance':'check_page x'}],'behaviors':[]})
print('NO_BEH', validate_spec(no_beh)[:25])
"
```
Expected :
```
GOOD None
BAD_TYPE program_type 'jeu' inconnu
NO_BEH aucun behavior
```

- [ ] **Step 5 : format + commit**

```bash
uvx ruff format loom/reflect.py && uvx ruff check loom/reflect.py
git add loom/reflect.py && git commit -m "feat(reflect): Behavior + parse_spec/validate_spec (gate contrat)"
```

---

## Task 4 : outil `submit_spec` + prompt de décomposition

**Files:**
- Modify: `loom/reflect.py`, `loom/prompts/reflect.decompose.md`

- [ ] **Step 1 : `make_submit_spec` (après `make_submit_plan`)**

```python
def make_submit_spec(holder: dict) -> ToolSpec:
    """Outil submit_spec : contrat de succès structuré rangé dans holder['plan']. Étend
    submit_plan (goal/success_check/tasks) avec program_type, launch, behaviors prouvables."""

    def run(args: dict) -> str:
        plan = parse_spec(args)
        holder["plan"] = plan
        return (f"Spec reçue : type={plan.program_type}, {len(plan.tasks)} tâche(s), "
                f"{len(plan.behaviors)} comportement(s). (le rail prend la suite)")

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
                "success_check": {"type": "string", "description": "Preuve finale de bout en bout."},
                "program_type": {"type": "string", "enum": ["html_game", "web_page", "cli", "python_lib", "api", "script"]},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Manifest des fichiers attendus."},
                "launch": {"type": "string", "description": "Fichier .html à ouvrir, ou commande de lancement."},
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
                                    "op": {"type": "string", "enum": ["click", "rightclick", "dblclick", "hover", "type", "none"]},
                                    "selector": {"type": "string"},
                                    "text": {"type": "string"},
                                    "expect": {
                                        "type": "object",
                                        "properties": {
                                            "selector": {"type": "string"},
                                            "check": {"type": "string", "enum": ["count", "class", "text", "absent"]},
                                            "value": {"type": "string"},
                                            "cmp": {"type": "string", "enum": ["min", "eq"]},
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
```

- [ ] **Step 2 : Brancher `submit_spec` dans la décomposition**

Dans `run_reflective`, la phase 1 utilise `make_submit_plan` + `validate_plan`. Remplacer :
- `reg = ToolRegistry([make_submit_plan(holder)])` → `reg = ToolRegistry([make_submit_spec(holder)])`
- `err = validate_plan(plan, max_tasks)` → `err = validate_spec(plan, max_tasks)`
- le nudge « Tu n'as pas appelé submit_plan… » → remplacer `submit_plan` par `submit_spec` (2 occurrences dans les messages de la phase 1).

- [ ] **Step 3 : Mettre à jour `loom/prompts/reflect.decompose.md`**

Remplacer la dernière ligne (« Quand ton plan est prêt, appelle `submit_plan(goal, success_check, tasks)`… ») par :
```markdown
Déclare aussi le CONTRAT du programme : `program_type` (html_game | web_page | cli | python_lib | api | script), `launch` (le fichier .html à ouvrir pour une page web, ou la commande de lancement), et `behaviors` — les comportements à PROUVER par interaction réelle. Chaque behavior a un `step` jouable : `{op: click|rightclick|dblclick|hover|type|none, selector: "<css>", expect: {selector, check: count|class|text|absent, value}}`.

Exemple (démineur) : `{desc:"cliquer une cellule la révèle", step:{op:"click", selector:".cell:first-child", expect:{selector:".cell.revealed", check:"count", value:"1", cmp:"min"}}}` ; `{desc:"clic droit pose un drapeau", step:{op:"rightclick", selector:".cell:nth-child(2)", expect:{selector:".flag", check:"count", value:"1", cmp:"min"}}}`.

Quand ton contrat est prêt, appelle `submit_spec(goal, success_check, program_type, files, launch, tasks, behaviors)`. N'écris pas le plan en texte : émets directement l'appel d'outil.
```

- [ ] **Step 4 : Smoke — submit_spec capte un Plan enrichi**

Run :
```bash
uv run python -c "
from loom.reflect import make_submit_spec
h={}; t=make_submit_spec(h)
print(t.run({'goal':'g','success_check':'x','program_type':'html_game','launch':'i.html','tasks':[{'goal':'t','acceptance':'check_page x'}],'behaviors':[{'desc':'d','step':{'op':'click','selector':'.c','expect':{'selector':'.c.r','check':'count','value':'1'}}}]}))
print('TYPE', h['plan'].program_type, 'BEH', len(h['plan'].behaviors), 'LAUNCH', h['plan'].launch)
"
```
Expected :
```
Spec reçue : type=html_game, 1 tâche(s), 1 comportement(s). (le rail prend la suite)
TYPE html_game BEH 1 LAUNCH i.html
```

- [ ] **Step 5 : format + commit**

```bash
uvx ruff format loom/reflect.py && uvx ruff check loom/reflect.py
git add loom/reflect.py loom/prompts/reflect.decompose.md
git commit -m "feat(reflect): submit_spec (contrat + behaviors) branche dans la decomposition"
```

---

## Task 5 : compiler les behaviors + Phase 4 intégration interactive

**Files:**
- Modify: `loom/reflect.py`

- [ ] **Step 1 : `_compile_behaviors` + `_interactive_report` (helpers, après `_final_report`)**

```python
def _compile_behaviors(plan: Plan) -> list[dict]:
    """Traduit les behaviors du contrat en steps pour run_interactive (ne garde que ceux
    qui ont un step exploitable)."""
    return [b.step for b in plan.behaviors if isinstance(b.step, dict) and b.step]


def _interactive_report(plan: Plan, res: dict) -> str:
    head = (
        "\n**✓ Objectif atteint : tous les comportements sont PROUVÉS par interaction réelle.**"
        if res.get("ok")
        else "\n**✗ Comportements NON prouvés** (au moins une action/post-condition échoue)."
    )
    lines = [head, f"Page : {res.get('url', plan.launch)}"]
    if res.get("error"):
        lines.append(f"erreur : {res['error']}")
    if res.get("console_errors"):
        lines.append(f"erreurs console : {len(res['console_errors'])}")
    for i, s in enumerate(res.get("steps", []), 1):
        b = plan.behaviors[i - 1].desc if i - 1 < len(plan.behaviors) else s.get("op", "")
        lines.append(f"  {i}. [{'ok' if s['ok'] else 'ÉCHEC'}] {b} → {s['observed']}")
    return "\n".join(lines)
```

- [ ] **Step 2 : Phase 4 — preuve interactive déterministe pour les types web**

Dans `run_reflective`, repérer la Phase 4 (« --- Phase 4 : intégration … »). Remplacer tout le bloc Phase 4 actuel par :
```python
    # --- Phase 4 : intégration. Pour un programme WEB, le HARNAIS joue lui-même les
    # behaviors (clics réels) via run_interactive et tranche déterministiquement. Sinon,
    # repli sur la preuve du sous-agent (porte evaluate_executor_proof, palier 1).
    yield (
        "phase",
        {"name": "intégration", "task": None, "detail": plan.success_check[:70]},
    )
    steps = _compile_behaviors(plan)
    if plan.program_type in _WEB_TYPES and plan.launch and steps:
        from loom.tools.browser import run_interactive

        res = run_interactive(workspace_dir_for(make_sub_registry), plan.launch, steps)
        yield ("content", _interactive_report(plan, res))
        return
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
```

`run_interactive` a besoin du `workspace_dir`. `run_reflective` ne le connaît pas directement (il reçoit `make_sub_registry`). Ajouter un paramètre explicite : dans la signature de `run_reflective`, ajouter `workspace_dir: str = "."` (keyword-only, après `permission=None`). Puis remplacer `workspace_dir_for(make_sub_registry)` ci-dessus par simplement `workspace_dir`. Et dans `loom/web/app.py`, à l'appel `run_reflective(...)`, passer `workspace_dir=ws`.

(Concrètement : Step 2a édite la signature, Step 2b l'appel web. Voir sous-étapes.)

- [ ] **Step 2a : signature `run_reflective`**

Repérer dans `loom/reflect.py` :
```python
    permission=None,
    max_tasks: int = 30,
```
et insérer entre les deux :
```python
    permission=None,
    workspace_dir: str = ".",
    max_tasks: int = 30,
```
Puis dans le bloc Phase 4 ci-dessus, utiliser `run_interactive(workspace_dir, plan.launch, steps)`.

- [ ] **Step 2b : appel web**

Dans `loom/web/app.py`, à l'appel `source = run_reflective(...)`, ajouter l'argument `workspace_dir=ws,` (la variable `ws` = workspace de la session, déjà définie juste au-dessus dans `generate()`).

- [ ] **Step 3 : Smoke — compiler + intégration interactive de bout en bout (Playwright réel + client factice)**

Créer `tmp_smoke_mvp.py` :
```python
import tempfile, os
from loom.reflect import run_reflective
from loom.tools.base import ToolRegistry

HTML = """<!doctype html><html><body>
<div class="cell"></div><div class="cell"></div>
<button id="go" onclick="document.querySelector('.cell').classList.add('revealed')">go</button>
</body></html>"""
d = tempfile.mkdtemp(); open(os.path.join(d, "index.html"), "w").write(HTML)


class FakeClient:
    def stream_chat_tools(self, messages, system_prompt, max_tokens, *, model=None,
                          registry=None, thinking=True, permission=None, **kw):
        names = list(registry._specs) if registry else []
        if "submit_spec" in names:
            registry.run("submit_spec", {
                "goal": "demineur", "success_check": "clic révèle", "program_type": "html_game",
                "launch": "index.html",
                "tasks": [{"goal": "page", "files": ["index.html"], "acceptance": "check_page index.html .cell"}],
                "behaviors": [
                    {"desc": "compte des cellules", "step": {"op": "none", "expect": {"selector": ".cell", "check": "count", "value": "2", "cmp": "min"}}},
                    {"desc": "cliquer révèle", "step": {"op": "click", "selector": "#go", "expect": {"selector": ".cell.revealed", "check": "count", "value": "1", "cmp": "min"}}},
                ],
            })
            yield ("tool_result", {"id": "1", "name": "submit_spec", "ok": True, "preview": "ok"})
        else:  # exécuteur de tâche : termine sur un check_page vert
            yield ("tool_result", {"id": "c", "name": "check_page", "ok": True, "preview": "ok",
                                   "detail": "console : 0 erreur(s), 0 warning(s)\néléments : .cell ×2"})


def make_sub(extra=None):
    return ToolRegistry(list(extra or []))


events = list(run_reflective(FakeClient(), [{"role": "user", "content": "demineur"}], "SYS",
                             make_sub_registry=make_sub, model="fake", max_tokens=256, workspace_dir=d))
txt = "".join(p for k, p in events if k == "content")
print("INTERACTIVE_PROVEN", "comportements sont PROUVÉS" in txt)
print(txt[-400:])
```
Run :
```bash
uv run python tmp_smoke_mvp.py && rm tmp_smoke_mvp.py
```
Expected : `INTERACTIVE_PROVEN True` puis un rapport listant les 2 comportements `[ok]`.

- [ ] **Step 4 : format + commit**

```bash
uvx ruff format loom/reflect.py && uvx ruff check loom/reflect.py loom/web/app.py
git add loom/reflect.py loom/web/app.py
git commit -m "feat(reflect): Phase 4 interactive (le harnais joue les behaviors, verdict deterministe)"
```

---

## Task 6 : validation live (le litmus, piloté par l'utilisateur)

- [ ] **Step 1** : l'utilisateur lance la stack (serveur modèle + `uv run python -m loom.web`).
- [ ] **Step 2** : Mode réflexion activé, demande de build à succès visible (démineur HTML). Piloter via Playwright.
- [ ] **Step 3** : Observer : la décomposition produit un `submit_spec` avec `program_type=html_game`, `launch`, et des `behaviors` (clics) ; à l'intégration, le harnais **joue les clics** et rend un verdict « PROUVÉ / NON prouvé » par comportement — pas un simple « 0 erreur ».
- [ ] **Step 4** : Noter le constat en mémoire `loom-harnais-reflexion-design` (le modèle produit-il des behaviors jouables exploitables ? le verdict interactif est-il fiable ?).

---

## Self-Review (plan vs spec)

**1. Couverture du spec (périmètre MVP) :**
- submit_spec (program_type, files, launch, behaviors, checks/tasks, gate anti-vague) → Tasks 3, 4. ✓ (`checks` non distinct des `tasks.acceptance` au MVP — assumé.)
- Preuve interactive (clics réels + DOM après) → Tasks 1, 2. ✓
- Runner typé : web traité (run_interactive) ; pytest/CLI/HTTP → **déférés** (le repli `evaluate_executor_proof` couvre le non-web au MVP). ✓ noté.
- Registre de preuves, classifieur/repair, snapshots, relecture fraîche, toggle dédié → **hors MVP** (spec §, palier 2 suite). ✓
- Frontière déterminisme (le harnais joue + lit la preuve, le modèle écrit) → Task 5. ✓

**2. Placeholders :** aucun « TBD ». Le seul flou volontaire : `value` typé string dans les schémas (coercition `int()` côté éval) — documenté.

**3. Cohérence des types/signatures :**
- `Behavior{desc, step}` : défini Task 3, produit par `parse_spec` (Task 3) / `make_submit_spec` (Task 4), consommé par `_compile_behaviors` (Task 5). ✓
- `Plan` étendu (program_type, launch, behaviors) : Task 3, lu Tasks 4-5. ✓
- `run_interactive(workspace_dir, target, steps) -> dict{ok,steps,console_errors,error,url}` : Task 1, appelé Task 2 (outil) et Task 5 (Phase 4). ✓
- step schema `{op, selector, text?, expect:{selector, check, value, cmp?}}` identique partout (run_interactive, check_interactive, submit_spec, prompt, smokes). ✓
- `run_reflective(..., workspace_dir=".")` : Task 5 (signature + appel web). ✓
- `_WEB_TYPES` : défini Task 3, utilisé Tasks 3 et 5. ✓

---

**Plan complet.** Options d'exécution : **(1) Subagent-Driven** (un sous-agent frais par tâche, revue entre tâches — recommandé) ou **(2) Inline**. Laquelle ?
