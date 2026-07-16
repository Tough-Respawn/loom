# loom/tools/agent.py
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
  garde-fous de stream_chat_tools (plafond de tours + non-progrès ; le mur de
  temps a été retiré).

Plafond de tours (`max_iters`) selon LOCAL vs DISTANT — règle cardinale « on
bride le local, on exploite le distant » :
- LOCAL : 30. Le coupe-circuit anti-boucle + le non-progrès (repeat_limit) sont
  actifs ; 30 n'est qu'un plafond dur d'appoint. Un slot VRAM, on limite.
- DISTANT : 500, comme le fil principal. Pour un modèle fort, l'anti-boucle est
  COUPÉ (cf. stream_chat_tools) -> `max_iters` est le SEUL backstop : il doit
  être un vrai seuil de runaway, pas un cap de progression. Un ouvrier
  multi-fichiers distant a besoin de marge, sinon on le décapite en pleine tâche.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from loom.tools.base import ToolError, ToolRegistry, ToolSpec

# Consigne ajoutée à la tâche quand une SORTIE STRUCTURÉE est demandée. La mécanique
# (les champs, leurs types) vit dans le SCHÉMA de submit_result, pas ici — un schéma
# d'outil est mieux respecté qu'une description en prose (leçon 2026-07). Cette
# phrase ne dit donc que l'OBLIGATION d'appeler l'outil, pas la forme du résultat.
_SUBMIT_INSTRUCTION = (
    "\n\nWhen you are done, you MUST report your result by calling the "
    "`submit_result` tool. Its schema defines exactly what to provide. Do not "
    "write the result as plain text: only the `submit_result` call is recorded."
)


def make_submit_result(schema: dict, sink: list) -> ToolSpec:
    """Outil de SORTIE : ses `parameters` SONT le schéma demandé par l'appelant.

    C'est le mécanisme de sortie structurée de Loom. Pas de `response_format` :
    llama.cpp ne le supporte pas uniformément selon le modèle, et ça dupliquerait
    une validation qu'on a déjà — `validate_and_coerce` valide et coerce le schéma
    d'un outil gratuitement, y compris les fautes de type d'un petit modèle
    ("5"->5, '{"a":1}'->dict). Le sous-agent remplit l'outil, on capture les args.
    """

    def run(args: dict) -> str:
        sink.append(args)
        return "ok: résultat enregistré. Termine maintenant (ne réémets aucun appel)."

    return ToolSpec(
        name="submit_result",
        description=(
            "Reports your final result in structured form. Call this exactly once, "
            "when your task is complete. This is the ONLY way your result is "
            "recorded — plain text is discarded."
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
    ) -> None:
        self.client = client
        self.build_sub_registry = build_sub_registry
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.max_iters = max_iters
        self.permission = permission
        self.compact_after_tokens = compact_after_tokens
        self.compact_for = compact_for
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
        self, task: str, *, schema: dict | None = None, sink: list | None = None
    ) -> Iterator[tuple[str, object]]:
        """Yield les events de la sous-boucle EN DIRECT (pour que l'UI voie l'ouvrier
        agir). Lève ToolError tout de suite si la tâche manque.

        `schema` : demande une sortie STRUCTURÉE — on injecte `submit_result` dans le
        registre et on pousse les arguments capturés dans `sink`. L'appelant lit
        `sink[-1]` après épuisement du générateur.
        """
        task = (task or "").strip()
        if not task:
            raise ToolError("argument 'task' manquant (décris la tâche à déléguer)")
        sub_registry = self.build_sub_registry()
        if schema is not None:
            if sink is None:
                raise ToolError("sink requis avec schema (bug interne)")
            sub_registry.add(make_submit_result(schema, sink))
            task = task + _SUBMIT_INSTRUCTION

        def _run_tier(tier):
            """Sous-boucle sur UN tier. Yield ses events ; l'échec se lit dans 'done'."""
            iters, threshold = self._limits(tier)
            # Cache souverain : une sous-boucle LOCALE écrase le slot KV du parent
            # -> save/restore autour (~ms, re-prefill évité, constaté 2026-07-10).
            # Tier DISTANT : le serveur local n'est jamais touché, on ne sauve rien
            # (le dispatch devient gratuit pour le cache du fil principal).
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
                    # Sans seuil, la sous-boucle saturait sa fenêtre (session
                    # 2026-07-14 : completion étranglée à ~129 tokens, tool calls
                    # tronqués en boucle) — le sous-agent compacte comme le principal.
                    compact_after_tokens=threshold,
                )
            finally:
                if saved:
                    self.client.restore_slot(self.model, "dispatch.kv")

        def _stream():
            for i, tier in enumerate(self.tiers):
                failed = False
                for kind, payload in _run_tier(tier):
                    if (
                        kind == "done"
                        and isinstance(payload, dict)
                        and payload.get("reason") == "api_error"
                        and i + 1 < len(self.tiers)
                    ):
                        # Tier mort (429/5xx/timeout) : on passe la main au suivant
                        # au lieu de rendre un échec — marqueur visible dans la synthèse.
                        failed = True
                        yield (
                            "content",
                            f"\n[relève : {tier} indisponible -> {self.tiers[i + 1]}]\n",
                        )
                        break
                    yield (kind, payload)
                if not failed:
                    return

        return _stream()

    def run(self, task: str) -> str:
        """Repli non-streamant : draine `stream` et garde la synthèse (content)."""
        chunks = [p for kind, p in self.stream(task) if kind == "content"]
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

    def run_stream(args: dict):
        # Validation eagerly : appeler run_stream(args) lève ToolError tout de suite
        # si la tâche manque (le registre la convertit en event d'erreur).
        return runner.stream(args.get("task") or "")

    def run(args: dict) -> str:
        return runner.run(args.get("task") or "")

    return ToolSpec(
        name="dispatch_agent",
        description=(
            "Delegates a self-contained TASK to a sub-agent with an isolated context (it "
            "has the same tools as you: read, write, shell). Use it when the task "
            "requires exploring/reading/modifying a lot and you only want a SYNTHESIS "
            "back, not all the detail in your context. Give a precise, self-contained "
            "instruction (objective + done criterion); the sub-agent acts then returns "
            "what it did. It CANNOT delegate in turn."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Precise, self-contained question or research task to delegate."
                    ),
                }
            },
            "required": ["task"],
        },
        run=run,
        run_stream=run_stream,
    )
