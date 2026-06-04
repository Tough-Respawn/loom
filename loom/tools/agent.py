# loom/tools/agent.py
"""Outil dispatch_agent : délègue une enquête à un SOUS-AGENT à contexte isolé.

Pourquoi : certaines tâches demandent de lire/chercher BEAUCOUP pour ne ramener
qu'une conclusion. Tout faire dans le fil principal noie son contexte (un petit
modèle s'y perd). Le sous-agent fait ce gros travail dans SA propre boucle
tool-use, puis ne renvoie qu'une synthèse -> le contexte principal reste propre.

Garde-fous :
- le sous-agent reçoit un registre en LECTURE SEULE (pas d'écriture/shell, donc
  rien d'irréversible sans supervision) ;
- ce registre N'INCLUT PAS dispatch_agent -> pas de récursion ;
- `thinking=False` et un `max_iters` court bornent le coût (single GPU).
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
    max_iters: int = 6,
) -> ToolSpec:
    """Outil dispatch_agent : lance une boucle tool-use isolée et renvoie sa synthèse.

    `build_sub_registry` est un thunk qui fabrique le registre LECTURE SEULE du
    sous-agent (sans dispatch_agent) : on le (re)construit à chaque appel pour ne
    pas partager d'état mutable entre délégations.
    """

    def run(args: dict) -> str:
        task = (args.get("task") or "").strip()
        if not task:
            raise ToolError("argument 'task' manquant (décris l'enquête à déléguer)")
        sub_registry = build_sub_registry()
        chunks: list[str] = []
        for kind, payload in client.stream_chat_tools(
            [{"role": "user", "content": task}],
            system_prompt,
            max_tokens,
            model=model,
            registry=sub_registry,
            thinking=False,
            max_iters=max_iters,
        ):
            if kind == "content":
                chunks.append(payload)
        return "".join(chunks).strip() or "(le sous-agent n'a rien renvoyé)"

    return ToolSpec(
        name="dispatch_agent",
        description=(
            "Délègue une ENQUÊTE de lecture/recherche à un sous-agent à contexte "
            "isolé (outils en lecture seule). Utilise-le quand répondre suppose "
            "d'explorer/lire beaucoup de fichiers ou pages et que tu ne veux qu'une "
            "SYNTHÈSE, pas tout le détail dans ton contexte. Donne une tâche précise "
            "et autonome ; le sous-agent te renvoie sa conclusion. Ne lui confie PAS "
            "d'écriture ni d'exécution (il ne peut pas), fais-les toi-même ensuite."
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
    )
