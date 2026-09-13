"""Sous-agents : le RUNNER (machinerie) et l'outil dispatch_agent (sa façade).

Pourquoi : certaines tâches demandent de lire/chercher/agir BEAUCOUP pour ne
ramener qu'une conclusion. Tout faire dans le fil principal noie son contexte
(un petit modèle s'y perd). Le sous-agent fait ce gros travail dans SA propre
boucle tool-use, puis ne renvoie qu'une synthèse -> le contexte principal reste
propre.

Deux consommateurs de la MÊME machinerie (`SubAgentRunner`) :
- `dispatch_agent` : le modèle délègue une tâche, tour par tour, et la synthèse
  revient dans SON contexte ;
- `run_workflow` (loom/workflow) : un SCRIPT délègue N tâches, et les synthèses
  restent dans des variables du script — le contexte du modèle ne les voit jamais.
D'où l'extraction : le runner ne connaît ni l'un ni l'autre.

Le sous-agent dispose des MÊMES outils que le principal (lecture, écriture,
shell) : un ouvrier en lecture seule ne sert à rien. Garde-fous :
- son registre N'INCLUT PAS dispatch_agent -> pas de récursion ;
- il hérite de la MÊME politique de permission (deny-list dure de run_shell
  incluse) ; en mode « ask » sans confirmation interactive, l'action est refusée
  par défaut (le sous-agent tourne sans UI) ;
- `thinking=False` ; l'arrêt suit le stop naturel du modèle, borné par les
  garde-fous de stream_chat_tools ET par les budgets propres à l'ouvrier
  (durée, appels API, tokens cumulés, contexte et cycles de résultats).

Plafond de tours (`max_iters`) selon LOCAL vs DISTANT — règle cardinale « on
bride le local, on exploite le distant » :
- LOCAL : 30. Le coupe-circuit anti-boucle + le non-progrès (repeat_limit) sont
  actifs ; 30 n'est qu'un plafond dur d'appoint. Un slot VRAM, on limite.
- DISTANT : 500 reste un backstop anti-runaway, pas une politique de budget. Les
  bornes d'ouvrier ci-dessus coupent bien avant un emballement coûteux.
"""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator

from loom.tools.base import ToolError, ToolRegistry, ToolSpec


class _CancelRegistry:
    """ST-03 : annulation CIBLÉE d'un sous-agent, par (session_id, agent_id).

    Point de rendez-vous entre la route HTTP (thread serveur) et la sous-boucle
    (thread de génération). Coopératif : `request` pose un Event que le runner
    observe entre deux événements et que run_shell observe pendant une commande
    longue — aucun thread n'est tué. Idempotent : re-demander une annulation
    (ou viser un ouvrier inconnu/terminé) rend un état, jamais une erreur.
    Un ouvrier terminé se DÉSENREGISTRE (finally du runner) : le registre ne
    garde aucune entrée zombie après la fin ou la fermeture de session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (sid, aid) -> {"event": Event, "state": "running"|"cancelling"}
        self._active: dict[tuple[str, str], dict] = {}

    def register(
        self, sid: str, aid: str, event: threading.Event | None = None
    ) -> threading.Event:
        ev = event or threading.Event()
        with self._lock:
            self._active[(sid, aid)] = {"event": ev, "state": "running"}
        return ev

    def unregister(self, sid: str, aid: str) -> None:
        with self._lock:
            self._active.pop((sid, aid), None)

    def request(self, sid: str, aid: str) -> str:
        """Demande l'annulation. 'cancelling' si l'ouvrier est actif (ou déjà en
        annulation — idempotent), 'unknown' s'il est inconnu ou déjà terminé."""
        with self._lock:
            entry = self._active.get((sid, aid))
            if entry is None:
                return "unknown"
            entry["state"] = "cancelling"
            entry["event"].set()
            return "cancelling"

    def cancel_session(self, sid: str) -> int:
        """Annule TOUS les ouvriers actifs de `sid` — le chemin de l'arrêt GLOBAL
        d'une session (/cancel) : le parent s'arrête par son mécanisme historique,
        ses ouvriers par ici. Les ouvriers des AUTRES sessions ne sont jamais
        touchés (clé par session). Idempotent ; rend le nombre d'ouvriers visés."""
        n = 0
        with self._lock:
            for (s, _aid), entry in self._active.items():
                if s == sid:
                    entry["state"] = "cancelling"
                    entry["event"].set()
                    n += 1
        return n

    def state(self, sid: str, aid: str) -> str | None:
        with self._lock:
            entry = self._active.get((sid, aid))
            return entry["state"] if entry else None


# Registre process-global : la route web et les runners partagent la même vue.
CANCELLATIONS = _CancelRegistry()


# ST-02 : chronologie BORNÉE d'un sous-agent (subagent_tool_call/result confondus).
# Au-delà, les mirrors sont comptés (`events_dropped` dans subagent_end) mais plus
# émis — pas de conservation illimitée ; usage et fin passent toujours.
SUBAGENT_EVENT_CAP = 200
# Aperçu borné d'un résultat d'outil dans la chronologie (mêmes règles d'affichage
# que les pastilles : jamais les arguments, seulement le début du résultat).
_SUB_PREVIEW = 160
# Bornes de TRAVAIL d'un ouvrier, distinctes du backstop `max_iters=500` de la
# boucle distante. Elles bornent le coût et la durée même lorsqu'un modèle fait
# varier ses arguments et échappe ainsi à repeat_stop.
SUBAGENT_MAX_DURATION_S = 15 * 60
SUBAGENT_MAX_API_CALLS = 100
SUBAGENT_MAX_PROMPT_TOKENS = 4_000_000

# Détection de non-progrès sur les RÉSULTATS des outils d'exploration. Trois
# résultats byte-identiques dans une fenêtre courte signalent un cycle même si
# les arguments (curseur/start_line/pattern) changent.
SUBAGENT_RESULT_REPEAT_LIMIT = 3
SUBAGENT_RESULT_WINDOW = 12
SUBAGENT_CONSECUTIVE_TOOL_ERRORS = 5
_RESULT_CYCLE_TOOLS = frozenset({"read_file", "search_text", "list_dir", "find_files"})
# Table des ÉTATS TERMINAUX d'une délégation (ST-02, revue user) :
# - natural (ou stop vide hérité) -> completed ;
# - tout arrêt de garde-fou ou d'API ci-dessous -> failed ;
# - exception de la sous-boucle -> failed (émis avant de remonter) ;
# - cancelled : RÉSERVÉ à ST-03, jamais émis ici.
_SUB_FAILED_STOPS = frozenset(
    {
        "api_error",
        "empty_response",
        "repeat_stop",
        "loop_degenerate",
        "max_iters",
        "context_irreducible",
        "output_overflow",
        "worker_timeout",
        "api_call_budget",
        "prompt_budget",
        "context_budget",
        "result_cycle",
        "tool_error_budget",
    }
)

# Le schéma porte la forme; cette consigne impose seulement l'appel final.
_SUBMIT_INSTRUCTION = (
    "\n\nWhen you are done, you MUST report your result by calling the "
    "`submit_result` tool. Its schema defines exactly what to provide. Do not "
    "write the result as plain text: only the `submit_result` call is recorded."
)

# Validation STRICTE de submit_result — sous-ensemble JSON Schema NOMMÉ, pas un
# validateur complet. Supporté (= ce que les scripts de workflow exposent) :
# - type : string | integer | number | boolean | object | array | null ;
# - object : properties, required (PRÉSENCE de la clé, comme en JSON Schema —
#   un requis déclaré `type: "null"` accepte explicitement None) ;
# - additionalProperties : SEULE la forme `false` est supportée (la forme
#   schéma ne l'est pas) ; absent -> champs inconnus permis, comme en JSON Schema ;
# - array : items ;
# - enum (liste non vide — `enum: []` est un schéma invalide, refusé en amont
#   par loom.workflow.runtime._validate_schema).
# Tout le reste (anyOf/oneOf/allOf, format, pattern, minimum…) est IGNORÉ.
# La FORME du schéma est validée récursivement AVANT l'appel au sous-agent
# (_validate_schema) ; _schema_faults reste néanmoins défensif : un schéma
# malformé qui passerait quand même est ignoré champ à champ, JAMAIS une exception.
# Réservé à submit_result : les autres outils gardent la tolérance de
# validate_and_coerce (coercition best-effort, jamais de refus de type).
_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _schema_faults(value: object, schema: dict, path: str) -> list[str]:
    """Écarts de `value` au sous-ensemble supporté de `schema`, en chemins
    exploitables (ex. `bugs[2].confidence`). Ne lève jamais : liste vide = conforme."""
    if not isinstance(schema, dict):
        return []
    faults: list[str] = []
    jtype = schema.get("type")
    if (
        isinstance(jtype, str)
        and jtype in _TYPE_CHECKS
        and not _TYPE_CHECKS[jtype](value)
    ):
        # Type faux : inutile de descendre dans une structure qui n'en est pas une.
        return [f"{path} : {jtype} attendu, reçu {value!r} ({type(value).__name__})"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        faults.append(f"{path} : valeur hors enum {enum}, reçu {value!r}")
    if isinstance(value, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        required = schema.get("required")
        # `required` = PRÉSENCE de la clé (sémantique JSON Schema) : un None
        # explicite est présent — c'est le check de type qui tranchera sa validité.
        for r in required if isinstance(required, list) else []:
            if isinstance(r, str) and r not in value:
                faults.append(
                    f"{path}.{r} : champ requis manquant"
                    if path
                    else f"{r} : champ requis manquant"
                )
        for k, v in value.items():
            child = f"{path}.{k}" if path else str(k)
            if k in props:
                faults.extend(_schema_faults(v, props[k], child))
            elif schema.get("additionalProperties") is False:
                faults.append(
                    f"{child} : champ non déclaré (additionalProperties=false)"
                )
    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(value):
                faults.extend(_schema_faults(v, items, f"{path}[{i}]"))
    return faults


def make_submit_result(schema: dict, sink: list) -> ToolSpec:
    """Outil de SORTIE : ses `parameters` SONT le schéma demandé par l'appelant.

    C'est le mécanisme de sortie structurée de Loom. Pas de `response_format` :
    llama.cpp ne le supporte pas uniformément selon le modèle. La coercition de
    premier niveau de `validate_and_coerce` s'applique d'abord ("5"->5,
    '{"a":1}'->dict), PUIS `_schema_faults` valide strictement le résultat final
    (récursif, sous-ensemble nommé ci-dessus). Une non-conformité lève ToolError :
    l'erreur repart dans la boucle du sous-agent, qui corrige et rappelle —
    aucune boucle de retry dédiée, ce sont les garde-fous de tours existants
    qui bornent.

    CONTRAT :
    - le PREMIER appel valide gagne ; tout appel suivant est refusé (ToolError)
      sans écraser le résultat enregistré ;
    - aucun appel valide -> sink vide -> l'appelant (run_workflow) rend None,
      contrat d'échec inchangé.
    """

    def run(args: dict) -> str:
        if sink:
            raise ToolError(
                "résultat déjà enregistré (le premier appel valide fait foi) — "
                "ne rappelle plus submit_result, termine ta réponse."
            )
        faults = _schema_faults(args, schema, "")
        if faults:
            shown = " ; ".join(faults[:6])
            more = f" (+{len(faults) - 6} autres)" if len(faults) > 6 else ""
            raise ToolError(
                f"résultat non conforme au schéma : {shown}{more}. "
                "Corrige ces champs et rappelle submit_result."
            )
        sink.append(args)
        return "ok: résultat enregistré. Termine maintenant (ne réémets aucun appel)."

    return ToolSpec(
        name="submit_result",
        description=(
            "Reports your final result in structured form. Call this exactly once, "
            "when your task is complete: the FIRST valid call is recorded, any "
            "further call is rejected. A non-conforming result returns an error "
            "naming the faulty fields — fix them and call again. This is the ONLY "
            "way your result is recorded — plain text is discarded."
        ),
        parameters=schema,
        run=run,
    )


class SubAgentRunner:
    """Machinerie d'un sous-agent : tiers de modèles, cache KV, sortie structurée.

    ROUTAGE DÉTERMINISTE (décision user 2026-07-15) : `model_chain` = tiers de
    modèles essayés DANS L'ORDRE (ex. gratuit -> payant) avant le repli final sur
    `model` (le fil parent). Un tier qui meurt en 'api_error' (429 free tier, 5xx,
    timeout) passe la main au suivant — le modèle appelant ne choisit RIEN.
    `local_only` (session privée) court-circuite la chaîne : tout reste sur `model`,
    aucun octet ne part vers une API. `compact_for(tier)` rend le seuil de
    compaction de CE tier (fenêtre du modèle qui bosse, pas du parent) ; à défaut,
    `compact_after_tokens`. `max_iters` None -> résolu PAR TIER : 30 en local
    (bridé), 500 en distant (backstop seul).

    `build_sub_registry` est un thunk qui fabrique le registre du sous-agent (tous
    les outils SAUF dispatch_agent) : on le (re)construit à chaque appel pour ne pas
    partager d'état mutable entre délégations.
    """

    def __init__(
        self,
        client,
        build_sub_registry: Callable[[], ToolRegistry],
        *,
        system_prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        max_iters: int | None = None,
        permission=None,
        compact_after_tokens: int | None = None,
        model_chain: list[str] | None = None,
        local_only: bool = False,
        compact_for: Callable[[str | None], int | None] | None = None,
        model_roles: dict[str, str] | None = None,
        session_id: str | None = None,
        max_duration_s: float | None = SUBAGENT_MAX_DURATION_S,
        max_api_calls: int | None = SUBAGENT_MAX_API_CALLS,
        max_prompt_tokens: int | None = SUBAGENT_MAX_PROMPT_TOKENS,
        max_context_tokens: int | None = None,
        result_repeat_limit: int = SUBAGENT_RESULT_REPEAT_LIMIT,
        result_window: int = SUBAGENT_RESULT_WINDOW,
        max_consecutive_tool_errors: int = SUBAGENT_CONSECUTIVE_TOOL_ERRORS,
    ) -> None:
        self.client = client
        self.build_sub_registry = build_sub_registry
        # ST-03 : identité de session — le ciblage (session_id, agent_id) d'une
        # annulation passe par le registre global CANCELLATIONS. Sans session
        # (évals, tests), l'annulation ciblée est simplement indisponible.
        self.session_id = session_id or ""
        # Le thunk historique ne prend pas d'argument ; un thunk moderne accepte
        # cancel_event pour armer run_shell. Détection UNE fois, pas par appel.
        try:
            self._sub_takes_cancel = (
                "cancel_event" in inspect.signature(build_sub_registry).parameters
            )
        except (TypeError, ValueError):
            self._sub_takes_cancel = False
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.max_iters = max_iters
        self.permission = permission
        self.compact_after_tokens = compact_after_tokens
        self.compact_for = compact_for
        self.max_duration_s = max_duration_s
        self.max_api_calls = max_api_calls
        self.max_prompt_tokens = max_prompt_tokens
        self.max_context_tokens = max_context_tokens
        self.result_repeat_limit = max(2, result_repeat_limit)
        self.result_window = max(self.result_repeat_limit, result_window)
        self.max_consecutive_tool_errors = max(2, max_consecutive_tool_errors)
        # Les rôles abstraits gardent les workflows portables entre configurations.
        self.model_roles = model_roles or {}
        # Une session privée ignore tout override susceptible d'envoyer des données ailleurs.
        self.local_only = local_only
        chain = [] if local_only else [m for m in (model_chain or []) if m != model]
        self.tiers = [*chain, model]

    def _limits(self, tier):
        iters = self.max_iters
        if iters is None:
            iters = 500 if self.client.is_remote(tier) else 30
        threshold = (
            self.compact_for(tier) if self.compact_for else self.compact_after_tokens
        )
        return iters, threshold

    def stream(
        self,
        task: str,
        *,
        schema: dict | None = None,
        sink: list | None = None,
        model: str | None = None,
        label: str | None = None,
        parent: str | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Yield les events de la sous-boucle EN DIRECT (pour que l'UI voie l'ouvrier
        agir). Lève ToolError tout de suite si la tâche manque.

        `schema` : demande une sortie STRUCTURÉE — on injecte `submit_result` dans le
        registre et on pousse les arguments capturés dans `sink`. L'appelant lit
        `sink[-1]` après épuisement du générateur.

        `model` : ÉPINGLE cet appel sur un modèle au lieu de la chaîne (cas d'usage :
        un workflow route ses vérificateurs sur le modèle FORT et laisse ses
        chercheurs sur le tier gratuit). Accepte un id concret ("glm-zai") ou un RÔLE
        ("cheap"/"strong") résolu via `model_roles` — la forme à préférer dans un
        script réutilisable. Le modèle de session reste en repli si le tier épinglé
        meurt en api_error. IGNORÉ en session privée (local_only) : un override ne
        doit jamais faire fuir des octets vers une API — la confidentialité prime sur
        le routage. Modèle/rôle inconnu -> ignoré, la chaîne normale s'applique.

        ST-02 — CONTRAT D'ÉVÉNEMENTS commun (dispatch_agent ET run_workflow) :
        chaque délégation porte un `agent_id` stable et émet, EN PLUS du flux
        historique (inchangé pour les consommateurs existants) :
        - subagent_start  : agent_id, parent, label, model_requested, model_resolved ;
        - subagent_tool_call / subagent_tool_result : outil, statut, aperçu borné
          (chronologie plafonnée à SUBAGENT_EVENT_CAP, surplus compté) ;
        - subagent_usage  : tokens CUMULÉS + modèle du tier courant (une relève de
          tier se lit ici et dans subagent_end.model) ;
        - subagent_end    : status completed|failed (cancelled : réservé ST-03),
          stop_reason, duration_s, model final, events_dropped.
        Ces événements restent de la TÉLÉMÉTRIE : aucun message de prompt n'est
        ajouté et le cache n'est pas touché. Le consommateur de dispatch traduit
        toutefois `subagent_end.status/stop_reason` dans l'enveloppe du résultat
        parent afin qu'un échec ne soit jamais présenté comme `ok=true`.
        `label`/`parent` identifient la délégation ; à défaut : première ligne de
        la tâche / "dispatch".
        """
        task = (task or "").strip()
        if not task:
            raise ToolError("argument 'task' manquant (décris la tâche à déléguer)")
        tiers = self.tiers
        # ST-02 : préserver le modèle/RÔLE BRUT demandé ("cheap", "glm-zai"…)
        # AVANT toute résolution — c'est lui que subagent_start.model_requested
        # doit montrer ; la résolution ne concerne que model_resolved.
        requested = model
        if model:
            model = self.model_roles.get(model, model)
        if model and not self.local_only and self.client.is_remote(model):
            fallback = [self.model] if self.model and self.model != model else []
            tiers = [model, *fallback]
        # ST-03 : l'Event d'annulation naît AVANT le registre d'outils, pour que
        # run_shell (commande longue) l'observe PENDANT son exécution — c'est ce
        # qui permet de tuer l'arbre de processus via le mécanisme existant au
        # lieu d'attendre le timeout. L'identité (agent_id) naît ici aussi.
        aid = uuid.uuid4().hex[:8]
        cancel_ev = threading.Event()
        sub_registry = (
            self.build_sub_registry(cancel_event=cancel_ev)
            if self._sub_takes_cancel
            else self.build_sub_registry()
        )
        if schema is not None:
            if sink is None:
                raise ToolError("sink requis avec schema (bug interne)")
            sub_registry.add(make_submit_result(schema, sink))
            task = task + _SUBMIT_INSTRUCTION

        def _run_tier(tier):
            """Sous-boucle sur UN tier. Yield ses events ; l'échec se lit dans 'done'."""
            iters, threshold = self._limits(tier)
            # Seul un sous-agent local peut écraser le slot KV du parent.
            saved = (
                self.client.save_slot(self.model, "dispatch.kv")
                if not self.client.is_remote(tier)
                else False
            )
            try:
                yield from self.client.stream_chat_tools(
                    [{"role": "user", "content": task}],
                    self.system_prompt,
                    self.max_tokens,
                    model=tier,
                    registry=sub_registry,
                    thinking=False,
                    max_iters=iters,
                    permission=self.permission,
                    # Compacter avant saturation évite les appels d'outils tronqués.
                    compact_after_tokens=threshold,
                    # Slot annexe : le cache du parent (slot 0) reste intact.
                    id_slot=self.client.annex_slot(tier),
                )
            finally:
                if saved:
                    self.client.restore_slot(self.model, "dispatch.kv")

        def _mname(tier) -> str:
            return tier or self.model or "local"

        def _stream():
            t0 = time.monotonic()
            meta = {
                "agent_id": aid,
                "parent": parent or "dispatch",
                "label": (label or task.split("\n", 1)[0][:60]).strip(),
            }
            # ST-03 : enregistrement PAREsseux (au 1er next), désenregistrement
            # GARANTI (finally) — jamais d'entrée zombie, même si le parent
            # abandonne le générateur (fermeture de session, stop global).
            if self.session_id:
                CANCELLATIONS.register(self.session_id, aid, cancel_ev)
            tok_in = tok_out = chron = dropped = api_calls = 0
            stop = ""
            cur = _mname(tiers[0])
            cancelled = False
            recent_results: deque[tuple[str, str]] = deque(maxlen=self.result_window)
            tool_error_streaks: dict[str, int] = {}

            def _budget_stop(reason: str):
                labels = {
                    "worker_timeout": "durée maximale de l'ouvrier atteinte",
                    "api_call_budget": "budget d'appels API de l'ouvrier atteint",
                    "prompt_budget": "budget cumulé de tokens prompt atteint",
                    "context_budget": "contexte d'un appel devenu trop volumineux",
                    "result_cycle": "cycle de résultats d'outils identiques détecté",
                    "tool_error_budget": "trop d'erreurs consécutives du même outil",
                }
                yield ("content", f"\n[arrêt ouvrier : {labels[reason]}]\n")
                yield ("done", {"reason": reason})

            try:
                yield (
                    "subagent_start",
                    {
                        **meta,
                        # Brut demandé (rôle compris) ; sans demande explicite, le
                        # routage automatique = la tête de chaîne (requested == resolved).
                        "model_requested": requested or _mname(tiers[0]),
                        "model_resolved": _mname(tiers[0]),
                    },
                )
                try:
                    for i, tier in enumerate(tiers):
                        cur = _mname(tier)
                        failed = False
                        _, compact_threshold = self._limits(tier)
                        context_limit = self.max_context_tokens
                        if context_limit is None and compact_threshold:
                            # Laisser la compaction agir à son seuil, mais couper si
                            # une requête le dépasse malgré la marge de sortie.
                            context_limit = compact_threshold + self.max_tokens
                        gen = _run_tier(tier)
                        try:
                            while True:
                                # Observation COOPÉRATIVE de l'annulation : entre
                                # deux événements de la sous-boucle (= entre tours
                                # et entre outils), jamais au milieu d'un appel.
                                # Aucun thread tué : on FERME le générateur, le
                                # flux modèle se termine proprement (finally de
                                # _run_tier inclus). La course avec une fin
                                # naturelle est saine : si le flux s'est terminé
                                # avant ce check, StopIteration gagne et le
                                # résultat est conservé (status completed).
                                if cancel_ev.is_set():
                                    cancelled = True
                                    yield (
                                        "content",
                                        "\n[annulé : ouvrier arrêté par "
                                        "l'utilisateur avant la fin]\n",
                                    )
                                    break
                                if (
                                    self.max_duration_s is not None
                                    and time.monotonic() - t0 >= self.max_duration_s
                                ):
                                    stop = "worker_timeout"
                                    yield from _budget_stop(stop)
                                    break
                                try:
                                    kind, payload = next(gen)
                                except StopIteration:
                                    break
                                if (
                                    kind == "done"
                                    and isinstance(payload, dict)
                                    and payload.get("reason") == "api_error"
                                    and i + 1 < len(tiers)
                                ):
                                    # Un tier indisponible cède la main au suivant avec une trace visible.
                                    failed = True
                                    yield (
                                        "content",
                                        f"\n[relève : {tier} indisponible -> {tiers[i + 1]}]\n",
                                    )
                                    break
                                forced_stop = ""
                                if kind == "done" and isinstance(payload, dict):
                                    stop = payload.get("reason") or stop
                                elif kind == "tool_call" and isinstance(payload, dict):
                                    if chron < SUBAGENT_EVENT_CAP:
                                        chron += 1
                                        yield (
                                            "subagent_tool_call",
                                            {
                                                **meta,
                                                "name": payload.get("name"),
                                                "model": cur,
                                            },
                                        )
                                    else:
                                        dropped += 1
                                elif kind == "tool_result" and isinstance(
                                    payload, dict
                                ):
                                    tool_name = str(payload.get("name") or "")
                                    tool_ok = bool(payload.get("ok"))
                                    result_text = str(
                                        payload.get("out_full")
                                        or payload.get("detail")
                                        or payload.get("preview")
                                        or ""
                                    ).strip()
                                    if tool_name in _RESULT_CYCLE_TOOLS and result_text:
                                        fingerprint = (tool_name, result_text)
                                        recent_results.append(fingerprint)
                                        if (
                                            sum(
                                                x == fingerprint for x in recent_results
                                            )
                                            >= self.result_repeat_limit
                                        ):
                                            forced_stop = "result_cycle"
                                    if tool_ok:
                                        tool_error_streaks[tool_name] = 0
                                    else:
                                        tool_error_streaks[tool_name] = (
                                            tool_error_streaks.get(tool_name, 0) + 1
                                        )
                                        if (
                                            tool_error_streaks[tool_name]
                                            >= self.max_consecutive_tool_errors
                                        ):
                                            forced_stop = "tool_error_budget"
                                    if chron < SUBAGENT_EVENT_CAP:
                                        chron += 1
                                        yield (
                                            "subagent_tool_result",
                                            {
                                                **meta,
                                                "name": payload.get("name"),
                                                "ok": bool(payload.get("ok")),
                                                "preview": str(
                                                    payload.get("preview", "")
                                                )[:_SUB_PREVIEW],
                                                "model": cur,
                                            },
                                        )
                                    else:
                                        dropped += 1
                                elif kind == "usage" and isinstance(payload, dict):
                                    api_calls += 1
                                    tok_in += payload.get("prompt_tokens") or 0
                                    tok_out += payload.get("completion_tokens") or 0
                                    if (
                                        self.max_api_calls is not None
                                        and api_calls >= self.max_api_calls
                                    ):
                                        forced_stop = "api_call_budget"
                                    if (
                                        self.max_prompt_tokens is not None
                                        and tok_in > self.max_prompt_tokens
                                    ):
                                        forced_stop = "prompt_budget"
                                    if (
                                        context_limit is not None
                                        and (payload.get("prompt_tokens") or 0)
                                        > context_limit
                                    ):
                                        forced_stop = "context_budget"
                                    yield (
                                        "subagent_usage",
                                        {
                                            **meta,
                                            "prompt_tokens": tok_in,
                                            "completion_tokens": tok_out,
                                            "model": cur,
                                        },
                                    )
                                yield (kind, payload)
                                if forced_stop:
                                    stop = forced_stop
                                    yield from _budget_stop(stop)
                                    break
                                if kind == "done":
                                    # Fin naturelle émise : la sous-boucle est
                                    # close — une annulation qui arrive APRÈS
                                    # perd la course, le résultat est conservé.
                                    break
                        finally:
                            gen.close()
                        if cancelled or not failed:
                            break
                except Exception:
                    # Un crash de sous-boucle reste visible et corrélé avant de remonter.
                    yield (
                        "subagent_end",
                        {
                            **meta,
                            "status": "failed",
                            "stop_reason": stop or "exception",
                            "duration_s": round(time.monotonic() - t0, 1),
                            "model": cur,
                            "events_dropped": dropped,
                            "api_calls": api_calls,
                            "prompt_tokens": tok_in,
                            "completion_tokens": tok_out,
                        },
                    )
                    raise
                # UN SEUL subagent_end, point d'émission unique — y compris en
                # course annulation / fin naturelle (cancelled tranche).
                yield (
                    "subagent_end",
                    {
                        **meta,
                        # cancelled > table des stops (_SUB_FAILED_STOPS) > completed.
                        "status": (
                            "cancelled"
                            if cancelled
                            else (
                                "failed" if stop in _SUB_FAILED_STOPS else "completed"
                            )
                        ),
                        "stop_reason": "cancelled" if cancelled else stop,
                        "duration_s": round(time.monotonic() - t0, 1),
                        "model": cur,
                        "events_dropped": dropped,
                        "api_calls": api_calls,
                        "prompt_tokens": tok_in,
                        "completion_tokens": tok_out,
                    },
                )
            finally:
                if self.session_id:
                    CANCELLATIONS.unregister(self.session_id, aid)

        return _stream()

    def run(self, task: str, *, model: str | None = None) -> str:
        """Repli non-streamant : draine `stream` et garde la synthèse (content)."""
        chunks = [p for kind, p in self.stream(task, model=model) if kind == "content"]
        return "".join(chunks).strip() or "(le sous-agent n'a rien renvoyé)"


def make_dispatch_agent(
    client,
    build_sub_registry: Callable[[], ToolRegistry],
    *,
    system_prompt: str,
    model: str | None = None,
    max_tokens: int = 2048,
    max_iters: int | None = None,
    permission=None,
    compact_after_tokens: int | None = None,
    model_chain: list[str] | None = None,
    local_only: bool = False,
    compact_for: Callable[[str | None], int | None] | None = None,
    runner: SubAgentRunner | None = None,
) -> ToolSpec:
    """Outil dispatch_agent : façade mince sur SubAgentRunner (une tâche -> synthèse).

    `runner` : réutilise une machinerie déjà construite (build_registry la partage avec
    run_workflow). Absent -> on en fabrique une depuis les autres arguments."""
    runner = runner or SubAgentRunner(
        client,
        build_sub_registry,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        max_iters=max_iters,
        permission=permission,
        compact_after_tokens=compact_after_tokens,
        model_chain=model_chain,
        local_only=local_only,
        compact_for=compact_for,
    )

    def _requested_role(args: dict) -> str | None:
        explicit = args.get("model")
        if explicit in ("cheap", "strong"):
            return explicit
        # Un dispatch libre peut devenir un audit multi-fichiers non borné. Quand
        # un rôle fort existe, il est donc le défaut ; `cheap` reste disponible
        # explicitement pour une recherche courte et précisément découpée.
        return "strong" if runner.model_roles.get("strong") else None

    def run_stream(args: dict):
        # Valider avant de créer le générateur pour remonter immédiatement les erreurs.
        return runner.stream(args.get("task") or "", model=_requested_role(args))

    def run(args: dict) -> str:
        return runner.run(args.get("task") or "", model=_requested_role(args))

    return ToolSpec(
        name="dispatch_agent",
        description=(
            "Delegates a self-contained TASK to a sub-agent with an isolated context (it "
            "has the same tools as you: read, write, shell). Use it when the task "
            "requires exploring/reading/modifying a lot and you only want a SYNTHESIS "
            "back, not all the detail in your context. Give a precise, self-contained "
            "instruction (objective + done criterion); the sub-agent acts then returns "
            "what it did. It CANNOT delegate in turn. Broad/multi-file tasks use the "
            "strong worker by default; request cheap only for a short, bounded lookup."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Precise, self-contained question or research task to delegate."
                    ),
                },
                "model": {
                    "type": "string",
                    "enum": ["cheap", "strong"],
                    "description": (
                        "Worker role. Omit for strong when configured; use cheap only "
                        "for a short and tightly bounded lookup."
                    ),
                },
            },
            "required": ["task"],
        },
        run=run,
        run_stream=run_stream,
    )
