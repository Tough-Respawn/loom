# loom/tools/agent.py
"""Outil dispatch_agent : délègue une tâche à un SOUS-AGENT à contexte isolé.

Pourquoi : certaines tâches demandent de lire/chercher/agir BEAUCOUP pour ne
ramener qu'une conclusion. Tout faire dans le fil principal noie son contexte
(un petit modèle s'y perd). Le sous-agent fait ce gros travail dans SA propre
boucle tool-use, puis ne renvoie qu'une synthèse -> le contexte principal reste
propre.

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

from collections.abc import Callable

from loom.tools.base import ToolError, ToolRegistry, ToolSpec


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
) -> ToolSpec:
    """Outil dispatch_agent : lance une boucle tool-use isolée et renvoie sa synthèse.

    `build_sub_registry` est un thunk qui fabrique le registre du sous-agent (tous
    les outils SAUF dispatch_agent) : on le (re)construit à chaque appel pour ne pas
    partager d'état mutable entre délégations. `permission` est relayée telle quelle
    à la sous-boucle (même politique de sécurité que le fil principal).

    ROUTAGE DÉTERMINISTE (décision user 2026-07-15) : `model_chain` = tiers de
    modèles essayés DANS L'ORDRE (ex. gratuit -> payant) avant le repli final sur
    `model` (le fil parent). Un tier qui meurt en 'api_error' (429 free tier, 5xx,
    timeout) passe la main au suivant — le modèle appelant ne choisit RIEN.
    `local_only` (session privée) court-circuite la chaîne : tout reste sur `model`,
    aucun octet ne part vers une API. `compact_for(tier)` rend le seuil de
    compaction de CE tier (fenêtre du modèle qui bosse, pas du parent) ; à défaut,
    `compact_after_tokens`. `max_iters` None -> résolu PAR TIER : 30 en local
    (bridé), 500 en distant (backstop seul).
    """
    tiers = [] if local_only else [m for m in (model_chain or []) if m != model]
    tiers = [*tiers, model]

    def _limits(tier):
        iters = max_iters
        if iters is None:
            iters = 500 if client.is_remote(tier) else 30
        threshold = compact_for(tier) if compact_for else compact_after_tokens
        return iters, threshold

    def run_stream(args: dict):
        """Yield les events de la sous-boucle EN DIRECT (pour que l'UI voie l'ouvrier
        agir). Validation eagerly : appeler run_stream(args) lève ToolError tout de
        suite si la tâche manque (le registre la convertit en event d'erreur)."""
        task = (args.get("task") or "").strip()
        if not task:
            raise ToolError("argument 'task' manquant (décris la tâche à déléguer)")
        sub_registry = build_sub_registry()

        def _run_tier(tier):
            """Sous-boucle sur UN tier. Yield ses events ; renvoie (via StopIteration
            impossible en génération) rien — l'échec se lit dans l'event 'done'."""
            iters, threshold = _limits(tier)
            # Cache souverain : une sous-boucle LOCALE écrase le slot KV du parent
            # -> save/restore autour (~ms, re-prefill évité, constaté 2026-07-10).
            # Tier DISTANT : le serveur local n'est jamais touché, on ne sauve rien
            # (le dispatch devient gratuit pour le cache du fil principal).
            saved = (
                client.save_slot(model, "dispatch.kv")
                if not client.is_remote(tier)
                else False
            )
            try:
                yield from client.stream_chat_tools(
                    [{"role": "user", "content": task}],
                    system_prompt,
                    max_tokens,
                    model=tier,
                    registry=sub_registry,
                    thinking=False,
                    max_iters=iters,
                    permission=permission,
                    # Sans seuil, la sous-boucle saturait sa fenêtre (session
                    # 2026-07-14 : completion étranglée à ~129 tokens, tool calls
                    # tronqués en boucle) — le sous-agent compacte comme le principal.
                    compact_after_tokens=threshold,
                )
            finally:
                if saved:
                    client.restore_slot(model, "dispatch.kv")

        def _stream():
            for i, tier in enumerate(tiers):
                failed = False
                for kind, payload in _run_tier(tier):
                    if (
                        kind == "done"
                        and isinstance(payload, dict)
                        and payload.get("reason") == "api_error"
                        and i + 1 < len(tiers)
                    ):
                        # Tier mort (429/5xx/timeout) : on passe la main au suivant
                        # au lieu de rendre un échec — marqueur visible dans la synthèse.
                        failed = True
                        yield (
                            "content",
                            f"\n[relève : {tier} indisponible -> {tiers[i + 1]}]\n",
                        )
                        break
                    yield (kind, payload)
                if not failed:
                    return

        return _stream()

    def run(args: dict) -> str:
        # Repli non-streamant : on draine run_stream et on garde la synthèse (content).
        chunks = [payload for kind, payload in run_stream(args) if kind == "content"]
        return "".join(chunks).strip() or "(le sous-agent n'a rien renvoyé)"

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
