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
) -> ToolSpec:
    """Outil dispatch_agent : lance une boucle tool-use isolée et renvoie sa synthèse.

    `build_sub_registry` est un thunk qui fabrique le registre du sous-agent (tous
    les outils SAUF dispatch_agent) : on le (re)construit à chaque appel pour ne pas
    partager d'état mutable entre délégations. `permission` est relayée telle quelle
    à la sous-boucle (même politique de sécurité que le fil principal).

    `max_iters` None (défaut) -> résolu selon local/distant : 30 en local (bridé),
    500 en distant (comme le principal ; seul backstop car l'anti-boucle est coupé).
    """
    # Le sous-agent hérite du modèle du fil parent -> on décide le plafond MAINTENANT.
    if max_iters is None:
        max_iters = 500 if client.is_remote(model) else 30

    def run_stream(args: dict):
        """Yield les events de la sous-boucle EN DIRECT (pour que l'UI voie l'ouvrier
        agir). Validation eagerly : appeler run_stream(args) lève ToolError tout de
        suite si la tâche manque (le registre la convertit en event d'erreur)."""
        task = (args.get("task") or "").strip()
        if not task:
            raise ToolError("argument 'task' manquant (décris la tâche à déléguer)")
        sub_registry = build_sub_registry()
        return client.stream_chat_tools(
            [{"role": "user", "content": task}],
            system_prompt,
            max_tokens,
            model=model,
            registry=sub_registry,
            thinking=False,
            max_iters=max_iters,
            permission=permission,
        )

    def run(args: dict) -> str:
        # Repli non-streamant : on draine run_stream et on garde la synthèse (content).
        chunks = [payload for kind, payload in run_stream(args) if kind == "content"]
        return "".join(chunks).strip() or "(le sous-agent n'a rien renvoyé)"

    return ToolSpec(
        name="dispatch_agent",
        description=(
            "Délègue une TÂCHE autonome à un sous-agent à contexte isolé (il a les "
            "mêmes outils que toi : lecture, écriture, shell). Utilise-le quand la "
            "tâche suppose d'explorer/lire/modifier beaucoup et que tu ne veux qu'une "
            "SYNTHÈSE en retour, pas tout le détail dans ton contexte. Donne une "
            "consigne précise et autonome (objectif + critère de fini) ; le sous-agent "
            "agit puis te renvoie ce qu'il a fait. Il ne peut PAS déléguer à son tour."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Question ou tâche de recherche précise et autonome à déléguer."
                    ),
                }
            },
            "required": ["task"],
        },
        run=run,
        run_stream=run_stream,
    )
