# loom/workflow/runtime.py
"""Moteur des workflows : exécute un script Python qui orchestre des sous-agents.

POURQUOI un script plutôt que le modèle tour par tour. `dispatch_agent` sait déjà
fan-out (le modèle émet N appels, ils partent en threads). Ce qu'il ne sait pas
faire, structurellement :
- SORTIR LES RÉSULTATS DU CONTEXTE. Chaque synthèse de sous-agent atterrit dans la
  conversation du parent. En local le contexte est 24576 tokens (borné par 6 Go de
  VRAM) : un audit de 100 fichiers est physiquement impossible. Ici les synthèses
  vivent dans des variables du script ; seul le retour final remonte au modèle.
  C'est le gain décisif, et il vaut pour le LOCAL autant que pour le distant.
- BOUCLER JUSQU'À CONVERGENCE. « Jusqu'à ce que deux tours ne trouvent rien de neuf »
  demande au modèle de tenir un compteur à travers des tours qui mangent son
  contexte. Un `while` ne coûte rien.

CE N'EST PAS le retour de l'orchestrateur déterministe supprimé le 2026-06-04. Ce
qui bridait le modèle, c'était un orchestrateur FIXE, écrit à l'avance, qui imposait
sa forme à toute tâche. Ici le script est écrit PAR le modèle POUR la tâche du
moment : il garde la décision, il l'exprime en code au lieu de tour par tour.

POURQUOI PYTHON et pas le JS de Claude Code : Loom n'a aucun moteur JS, et en
embarquer un pour la seule raison qu'Anthropic l'a choisi coûterait une dépendance
node à un projet offline plus un pont IPC pour que `agent()` retraverse vers Python.
Bonus non négligeable : `agent()` BLOQUE, donc le script est du Python synchrone
ordinaire — pas d'async/await à écrire juste, ce qui compte quand c'est un modèle
moyen qui l'écrit.

`exec()` de code écrit par le modèle n'ouvre aucune porte que `run_shell("python -c
…")` n'ouvre déjà : le bac à sable a été supprimé délibérément (2026-06-04), la
deny-list de loom.permissions reste le garde-fou assumé, et les sous-agents héritent
de la même politique de permission que le fil principal.
"""

from __future__ import annotations

import ast
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

# Backstops anti-runaway (mêmes ordres de grandeur que Claude Code) : un script qui
# part en boucle ne doit pas pouvoir brûler la machine ou le quota d'API.
MAX_AGENTS = 1000  # agents sur la VIE d'un run
MAX_ITEMS = 4096  # items d'un seul parallel()/pipeline() — erreur explicite, pas
# une troncature silencieuse (un run tronqué qui se dit complet, c'est pire)

_FN_NAME = "__loom_workflow__"


class WorkflowError(Exception):
    """Erreur de workflow (script invalide, cap dépassé) — message montrable."""


def _default_workers(is_remote: bool) -> int:
    """Concurrence réelle. RÈGLE CARDINALE « on bride le local, on exploite le
    distant » : le serveur local n'a qu'UN slot llama-swap, donc tout s'y sérialise —
    lancer 16 threads dessus ne ferait que les faire attendre. C'est exactement la
    condition `is_remote` qu'applique déjà _run_tools_parallel dans client.py.
    Un `parallel()` garde donc la même SÉMANTIQUE en local, seule la perf change."""
    if not is_remote:
        return 1
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def parse_meta(source: str) -> dict:
    """Extrait le bloc `meta` d'un script SANS l'exécuter.

    L'appelant a besoin du nom/des phases AVANT de lancer quoi que ce soit (affichage,
    validation). D'où l'exigence d'un LITTÉRAL pur (comme Claude Code) : `ast.literal_eval`
    ne peut pas lire un meta calculé, et l'évaluer pour de vrai reviendrait à exécuter le
    script pour savoir ce qu'il fait."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkflowError(_syntax_message(exc)) from exc
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "meta" and node.value is not None:
                try:
                    value = ast.literal_eval(node.value)
                except ValueError as exc:
                    raise WorkflowError(
                        "`meta` doit être un littéral pur (pas de variable, d'appel de "
                        "fonction ni de f-string) : il est lu sans exécuter le script."
                    ) from exc
                if not isinstance(value, dict):
                    raise WorkflowError("`meta` doit être un dict.")
                return value
    raise WorkflowError(
        "script sans bloc `meta` : commence par "
        "`meta = {'name': ..., 'description': ...}`."
    )


def _syntax_message(exc: SyntaxError) -> str:
    """Erreur de syntaxe ACTIONNABLE : le modèle doit pouvoir corriger SON script sans
    deviner. On nomme la ligne et on la montre (cf. frontière d'entrée des outils)."""
    line = f" ligne {exc.lineno}" if exc.lineno else ""
    text = (exc.text or "").strip()
    shown = f"\n  {text}" if text else ""
    return f"script Python invalide{line} : {exc.msg}.{shown}"


def _compile(source: str):
    """Compile le script en une FONCTION, pour que `return` marche au niveau du script.

    L'enveloppe se fait au niveau AST, PAS par ré-indentation du texte : indenter les
    lignes corromprait le contenu de toute chaîne multi-lignes (docstring, prompt
    d'agent) — et un script de workflow est fait de prompts multi-lignes."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkflowError(_syntax_message(exc)) from exc
    fn = ast.FunctionDef(
        name=_FN_NAME,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=tree.body or [ast.Pass()],
        decorator_list=[],
        returns=None,
    )
    fn.type_params = []  # requis depuis Python 3.12
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        return compile(module, "<workflow>", "exec")
    except SyntaxError as exc:
        # Piège connu : `import *` est légal au niveau module, illégal dans une fonction.
        raise WorkflowError(_syntax_message(exc)) from exc


def _validate_schema(schema) -> None:
    """Refuse un `schema` malformé AVANT l'appel API, avec un message que le modèle
    peut corriger.

    Pourquoi ça mérite du code : `schema` devient les `parameters` de l'outil
    submit_result, donc un schéma invalide part tel quel à l'API, qui rejette l'appel
    -> chaque agent meurt en api_error et rend None. Un échec GLOBAL et SILENCIEUX, qui
    ressemble à « la feature est cassée » alors que c'est une faute du script : c'est
    exactement ce que la frontière d'entrée des outils (base.py) existe pour empêcher.
    On nomme la faute au lieu de la subir.

    NB : ce n'est PAS ce qui a causé l'incident E2E du 2026-07-16 (agents tous à None) —
    la cause racine était que submit_result manquait aux catégories de permissions et
    tombait en 'ask', refusé d'office côté sous-agent (cf. loom/permissions.py). Cette
    validation est un durcissement d'une autre voie d'échec, pas le correctif."""
    if not isinstance(schema, dict):
        raise WorkflowError(
            f"agent(schema=…) : un dict JSON Schema est attendu, reçu {type(schema).__name__}. "
            'Exemple : schema={"type": "object", "properties": {"bug": {"type": "string"}}, '
            '"required": ["bug"]}'
        )
    if schema.get("type") != "object":
        raise WorkflowError(
            'agent(schema=…) : le schéma racine doit être {"type": "object", …} — '
            "c'est le schéma des ARGUMENTS d'un outil, pas une valeur libre. "
            "Enveloppe une liste dans une propriété : "
            '{"type": "object", "properties": {"bugs": {"type": "array", …}}}'
        )
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        raise WorkflowError(
            "agent(schema=…) : `properties` manquant ou vide — sans champ à remplir, "
            "l'agent n'a rien à renvoyer. Déclare au moins une propriété."
        )
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise WorkflowError("agent(schema=…) : `required` doit être une liste de noms.")
    unknown = [r for r in required if r not in props]
    if unknown:
        raise WorkflowError(
            f"agent(schema=…) : `required` cite des champs absents de `properties` : "
            f"{', '.join(map(str, unknown))}."
        )


class _Run:
    """État d'un run : compteur d'agents, phase courante, throttle de concurrence."""

    def __init__(self, agent_fn: Callable, on_event: Callable, workers: int) -> None:
        self.agent_fn = agent_fn
        self.on_event = on_event
        self.workers = workers
        # Le VRAI throttle. Les pools ci-dessous ne servent qu'à la structure : un
        # pipeline imbriquant un parallel a besoin de son propre pool pour ne pas
        # s'auto-affamer (les tâches externes attendraient des tâches internes qui
        # n'auraient plus de slot). Avec un sémaphore global, la concurrence réelle
        # reste bornée quelle que soit l'imbrication, sans risque d'interblocage.
        self._slots = threading.Semaphore(workers)
        self._lock = threading.Lock()
        self._count = 0
        self._phase: str | None = None

    def _pool(self, n: int) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=max(1, min(self.workers, n)))

    # --- primitives exposées au script ---------------------------------------

    def agent(
        self,
        prompt: str,
        *,
        schema: dict | None = None,
        label: str | None = None,
        phase: str | None = None,
    ):
        """Lance UN sous-agent et renvoie son résultat (str, ou dict si `schema`).

        Renvoie None si le sous-agent échoue : un ouvrier mort ne doit pas tuer le run
        (contrat identique à Claude Code — l'appelant filtre). `phase` explicite évite
        la course sur la phase globale quand des étapes tournent en concurrence.

        Un `schema` malformé, lui, ARRÊTE le run (WorkflowError) au lieu de rendre None :
        c'est une faute du script, identique pour tous les agents. La rendre en None la
        déguiserait en « les ouvriers ont échoué » et enverrait le modèle debugger la
        mauvaise chose (constaté en E2E, cf. _validate_schema).
        """
        if schema is not None:
            _validate_schema(schema)
        with self._lock:
            self._count += 1
            if self._count > MAX_AGENTS:
                raise WorkflowError(
                    f"plafond de {MAX_AGENTS} agents dépassé sur ce run "
                    "(boucle probablement infinie dans le script)."
                )
            n = self._count
        name = label or (prompt or "").strip().split("\n")[0][:60]
        where = phase or self._phase
        self.on_event("agent_start", {"n": n, "label": name, "phase": where})
        try:
            with self._slots:
                result = self.agent_fn(prompt, schema=schema, label=name)
            ok = result is not None
        except WorkflowError:
            raise
        except Exception as exc:  # noqa: BLE001 - un ouvrier mort ne tue pas le run
            self.on_event(
                "agent_end", {"n": n, "label": name, "ok": False, "error": str(exc)}
            )
            return None
        self.on_event("agent_end", {"n": n, "label": name, "ok": ok})
        return result

    def parallel(self, thunks) -> list:
        """Exécute des thunks (callables sans argument) concurremment. BARRIÈRE : rend
        la main quand TOUS ont fini. Un thunk qui lève donne None — l'appel lui-même ne
        lève jamais, donc filtre les None avant d'utiliser les résultats."""
        thunks = list(thunks)
        if len(thunks) > MAX_ITEMS:
            raise WorkflowError(
                f"parallel() : {len(thunks)} items > plafond {MAX_ITEMS}."
            )
        if not thunks:
            return []

        def _safe(t):
            try:
                return t()
            except WorkflowError:
                raise
            except Exception:  # noqa: BLE001
                return None

        with self._pool(len(thunks)) as pool:
            return list(pool.map(_safe, thunks))

    def pipeline(self, items, *stages) -> list:
        """Fait passer chaque item par TOUTES les étapes, indépendamment — PAS de
        barrière entre étapes : l'item A peut être en étape 3 pendant que B est en
        étape 1. Chaque étape reçoit (résultat_précédent, item_initial, index). Une
        étape qui lève fait tomber CET item à None et saute ses étapes restantes."""
        items = list(items)
        if len(items) > MAX_ITEMS:
            raise WorkflowError(
                f"pipeline() : {len(items)} items > plafond {MAX_ITEMS}."
            )
        if not items or not stages:
            return list(items)

        def _chain(pair):
            index, item = pair
            value = item
            for stage in stages:
                try:
                    value = _call_stage(stage, value, item, index)
                except WorkflowError:
                    raise
                except Exception:  # noqa: BLE001
                    return None
            return value

        with self._pool(len(items)) as pool:
            return list(pool.map(_chain, enumerate(items)))

    def phase(self, title: str) -> None:
        """Ouvre une phase : les agents suivants y sont rattachés (affichage)."""
        self._phase = title
        self.on_event("phase", {"title": title})

    def log(self, message: str) -> None:
        """Message de progression pour l'utilisateur (pas pour le modèle)."""
        self.on_event("log", {"message": str(message)})


def _call_stage(stage, value, item, index):
    """Appelle une étape de pipeline en tolérant sa SIGNATURE : le script peut écrire
    `lambda r: ...` aussi bien que `lambda r, item, i: ...`. On essaie la forme longue
    et on retombe sur les plus courtes — sans quoi la forme la plus naturelle (1
    argument) planterait, et un TypeError venu de l'INTÉRIEUR de l'étape serait
    confondu avec une erreur d'arité. D'où l'inspection de la signature plutôt qu'un
    try/except TypeError."""
    import inspect

    try:
        sig = inspect.signature(stage)
        n = len(
            [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]
        )
        if any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()):
            n = 3
    except (TypeError, ValueError):
        n = 1
    if n >= 3:
        return stage(value, item, index)
    if n == 2:
        return stage(value, item)
    return stage(value)


def run_workflow(
    source: str,
    *,
    agent_fn: Callable,
    args=None,
    is_remote: bool = False,
    on_event: Callable | None = None,
    workers: int | None = None,
    path: str | None = None,
):
    """Exécute un script de workflow et renvoie sa valeur de retour.

    `agent_fn(prompt, *, schema, label)` lance un sous-agent (injecté : le runtime ne
    connaît ni le client ni les modèles). `on_event(kind, payload)` reçoit la
    progression — appelé DEPUIS DES THREADS, il doit être thread-safe. `is_remote`
    décide de la concurrence réelle (cf. _default_workers).
    """
    meta = parse_meta(source)  # valide le script avant tout effet de bord
    code = _compile(source)
    run = _Run(
        agent_fn, on_event or (lambda *_: None), workers or _default_workers(is_remote)
    )
    ns: dict = {
        "__name__": "__loom_workflow__",
        "__builtins__": __builtins__,
        # `__file__` = chemin du script. Un script Python se repère par __file__ (réflexe
        # universel) : sans lui, `Path(__file__).parent` lève un NameError et le modèle
        # perd un tour à coder son chemin en dur. Constaté en E2E UI le 2026-07-16, dès
        # le premier run réel. Le fournir coûte une ligne ; ne pas le fournir coûte un
        # aller-retour modèle à chaque script qui suit ce réflexe.
        "__file__": path,
        "meta": meta,
        "args": args,
        "agent": run.agent,
        "parallel": run.parallel,
        "pipeline": run.pipeline,
        "phase": run.phase,
        "log": run.log,
    }
    exec(code, ns)  # noqa: S102 - cf. docstring du module (pas de sandbox, par décision)
    return ns[_FN_NAME]()
