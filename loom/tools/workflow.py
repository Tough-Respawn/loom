# loom/tools/workflow.py
"""Outil run_workflow : exécute un script Python qui orchestre N sous-agents.

Le modèle ÉCRIT le script (write_file) puis appelle run_workflow(path=...).

Pourquoi un CHEMIN et pas le script en argument : faire passer 50 lignes de Python
dans une string JSON obligerait le modèle à échapper guillemets et sauts de ligne sur
tout le script — le piège d'échappement le ferait tomber, et une seule faute rend
l'appel entier illisible. Un chemin est trivial à émettre, réutilise write_file (qui
sait déjà écrire du code), et donne gratuitement la propriété que Claude Code met en
avant : le script est un FICHIER, qu'on peut relire, diffter et relancer. Un workflow
« sauvegardé » n'est alors qu'un fichier qu'on garde.

L'outil est STREAMANT : la progression (phases, agents, logs) remonte en direct dans
la pastille, comme pour dispatch_agent. Les synthèses des sous-agents, elles, ne
remontent PAS — elles vivent dans les variables du script. C'est tout l'intérêt :
seul le retour final du script entre dans le contexte du modèle.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.workflow import WorkflowError, parse_meta, run_workflow

# Le retour du script entre dans le contexte du modèle : on le borne comme n'importe
# quelle sortie d'outil. Un workflow qui rend 200 ko annulerait sa raison d'être.
_MAX_RESULT_CHARS = 20000


def _render(value) -> str:
    """Rend la valeur de retour du script pour le modèle."""
    if value is None:
        return "(le workflow n'a rien renvoyé — le script s'est-il terminé sans `return` ?)"
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > _MAX_RESULT_CHARS:
        text = (
            text[:_MAX_RESULT_CHARS]
            + f"\n… (tronqué à {_MAX_RESULT_CHARS} caractères ; fais renvoyer moins au script)"
        )
    return text


def make_run_workflow(
    runner, workspace_dir: str, *, is_remote: bool = False
) -> ToolSpec:
    """`runner` : un loom.tools.agent.SubAgentRunner (machinerie de sous-agent partagée
    avec dispatch_agent). `is_remote` : le modèle actif tourne-t-il sur une API ? ->
    décide de la concurrence réelle (1 en local, un slot llama-swap)."""

    def run_stream(args: dict):
        rel = (args.get("path") or "").strip()
        if not rel:
            raise ToolError("argument 'path' manquant (chemin du script du workflow)")
        p: Path = _resolve_in_root(Path(workspace_dir), rel)
        try:
            source = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ToolError(
                f"script introuvable : {p}. Écris-le d'abord avec write_file."
            ) from None
        except OSError as exc:
            raise ToolError(f"lecture impossible de {p} : {exc}") from None
        try:
            meta = parse_meta(source)  # échoue AVANT tout effet de bord
        except WorkflowError as exc:
            raise ToolError(str(exc)) from None

        events: queue.Queue = queue.Queue()
        _DONE = object()

        def agent_fn(prompt, *, schema=None, label=None, model=None):
            """Un sous-agent. Sa synthèse est RENVOYÉE au script, pas yieldée : le
            contexte du modèle appelant ne doit jamais la voir. `model` : épinglage
            par agent, relayé au runner (qui l'ignore en session privée)."""
            sink: list = []
            parts: list[str] = []
            for kind, payload in runner.stream(
                prompt, schema=schema, sink=sink, model=model
            ):
                if kind == "content" and isinstance(payload, str):
                    parts.append(payload)
                elif kind in ("tool_call", "tool_result", "usage"):
                    # Activité de l'ouvrier : visible en direct. 'usage' remonte aussi
                    # pour que la conso du run compte dans les totaux de session.
                    events.put((kind, payload))
            if schema is not None:
                return sink[-1] if sink else None
            return "".join(parts).strip() or None

        def on_event(kind, payload):
            # Appelé DEPUIS LES THREADS du runtime -> Queue (thread-safe) obligatoire.
            events.put((kind, payload))

        result: dict = {}

        def _work():
            try:
                result["value"] = run_workflow(
                    source,
                    agent_fn=agent_fn,
                    args=args.get("args"),
                    is_remote=is_remote,
                    on_event=on_event,
                    path=str(p),
                )
            except WorkflowError as exc:
                result["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - remonté comme erreur d'outil
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                events.put(_DONE)

        thread = threading.Thread(target=_work, daemon=True, name="loom-workflow")
        thread.start()

        name = meta.get("name") or p.name
        yield ("tool_call", {"name": f"workflow: {name}"})
        while True:
            item = events.get()
            if item is _DONE:
                break
            kind, payload = item
            # Traduction vers les events que la boucle sait déjà relayer en lignes
            # d'activité (_sub_activity_line) : aucun changement de frontend requis.
            if kind == "phase":
                yield ("tool_call", {"name": f"phase — {payload.get('title', '')}"})
            elif kind == "log":
                yield (
                    "tool_result",
                    {"ok": True, "preview": payload.get("message", "")},
                )
            elif kind == "agent_start":
                yield (
                    "tool_call",
                    {"name": f"agent {payload.get('n')} — {payload.get('label', '')}"},
                )
            elif kind == "agent_end":
                yield (
                    "tool_result",
                    {
                        "ok": bool(payload.get("ok")),
                        "preview": payload.get("error") or "terminé",
                    },
                )
            else:
                yield (kind, payload)
        thread.join()

        if "error" in result:
            yield ("content", f"erreur: workflow '{name}' — {result['error']}")
            return
        yield ("content", _render(result.get("value")))

    def run(args: dict) -> str:
        chunks = [p for kind, p in run_stream(args) if kind == "content"]
        return "".join(chunks).strip() or "(le workflow n'a rien renvoyé)"

    return ToolSpec(
        name="run_workflow",
        description=(
            "Runs a Python script that orchestrates MANY sub-agents, and returns only "
            "the script's return value. Use it when a task needs more sub-agents than "
            "your context can hold (audit every file in a directory, migrate many "
            "files, cross-check a question from several angles), or needs a loop "
            "('keep going until nothing new is found'). Write the script with "
            "write_file first, then pass its path here.\n"
            "The script is plain SYNCHRONOUS Python. It must start with a literal "
            "`meta = {'name': ..., 'description': ...}` and `return` its result. "
            "Available globals (do not import them):\n"
            "- agent(prompt, schema=None, label=None, model=None) -> the sub-agent's "
            "answer as a string, or a dict matching `schema` (a JSON Schema) if you "
            "pass one, or None if it failed. It BLOCKS; no async/await anywhere. "
            '`schema` must have a `{"type": "object", "properties": {...}}` root, like '
            "a tool's arguments; to get a list back, wrap it in a property. `model` "
            "pins this agent to a specific remote model id (e.g. verification stages "
            "on the strong model); omit it for the default routing chain.\n"
            "- parallel([lambda: agent(...), ...]) -> list of results; waits for all; "
            "a failed item is None. In a comprehension you MUST bind loop variables by "
            "value: `parallel([lambda f=f: agent(f) for f in files])`. Without `f=f` "
            "every lambda sees the LAST value and you run the same task N times. "
            "Prefer pipeline() over a comprehension of lambdas: it has no such trap.\n"
            "- pipeline(items, stage1, stage2, ...) -> runs each item through every "
            "stage independently, no barrier between stages. Each stage is called with "
            "(previous_result, original_item, index).\n"
            "- phase(title), log(message) -> progress shown to the user.\n"
            "- args -> the `args` value you passed to this tool.\n"
            "Sub-agent answers stay in script variables, so keep the return value "
            "small: return the conclusion, not every intermediate result."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the Python workflow script to run.",
                },
                "args": {
                    "description": (
                        "Optional value exposed to the script as the global `args`."
                    ),
                },
            },
            "required": ["path"],
        },
        run=run,
        run_stream=run_stream,
    )
