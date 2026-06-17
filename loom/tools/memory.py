"""Outils mémoire : `remember` (écrit) et `recall` (lit) la mémoire persistante de Loom.

`remember(text, kind)` : kind='episodic' -> store épisodique cherchable (provider) ;
kind ∈ {'memory','profile','soul'} -> append dédup dans MEMORY.md / USER.md / SOUL.md.
`recall(query)` : top-K épisodes pertinents (FTS5), rendus en texte borné. Un résumeur
LLM optionnel (`summarize`, câblé au Plan 2) condense les hits au-delà d'un seuil ; sans
lui, rendu brut borné.
"""

from __future__ import annotations

from loom.memory import identity as _id
from loom.tools.base import ToolError, ToolSpec

_KIND_TO_PATH = {
    "memory": "memory_md_path",
    "profile": "user_path",
    "soul": "soul_path",
}
_MAX_RECALL_CHARS = 1500


def make_remember(provider, paths: dict) -> ToolSpec:
    """`paths` : dict {memory_md_path, user_path, soul_path} (chemins identité)."""

    def run(args: dict) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            raise ToolError("argument 'text' : un contenu non vide est attendu")
        kind = (args.get("kind") or "episodic").strip().lower()
        if kind == "episodic":
            provider.remember(text, kind="episodic", source="remember")
            return "mémorisé (épisode cherchable via recall)."
        key = _KIND_TO_PATH.get(kind)
        if not key:
            raise ToolError(
                "kind invalide : attendu 'episodic' | 'memory' | 'profile' | 'soul'."
            )
        _id.append_unique(paths[key], text)
        return f"consigné dans {kind} (mémoire durable always-on)."

    return ToolSpec(
        name="remember",
        description=(
            "Capitalise un fait DURABLE et de haute valeur dans ta mémoire persistante "
            "(elle survit à la session). kind='episodic' (défaut) : range une "
            "leçon/observation dans le store cherchable (à retrouver via recall). "
            "kind='memory' : fait général durable (convention projet, détail "
            "d'environnement). kind='profile' : fait stable sur l'utilisateur. kind='soul' : "
            "trait de ta propre persona. Écris la LEÇON dense, pas le log brut. Ne mémorise "
            "que ce qui resservira."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Le fait à mémoriser (synthèse dense).",
                },
                "kind": {
                    "type": "string",
                    "enum": ["episodic", "memory", "profile", "soul"],
                    "description": "Où ranger : episodic (store) | memory | profile | soul.",
                },
            },
            "required": ["text"],
        },
        run=run,
    )


def make_recall(provider, *, summarize=None, threshold: int = 5) -> ToolSpec:
    """`summarize` : callable optionnel (query, hits) -> str. Au-delà de `threshold` hits,
    condense ; sinon rendu brut borné. Le résumeur (LLM) est câblé au Plan 2."""

    def run(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            raise ToolError("argument 'query' : décris ce que tu cherches en mémoire")
        k = max(1, min(int(args.get("k", 5) or 5), 10))
        hits = provider.recall(query, k=k)
        if not hits:
            return "(aucun souvenir pertinent)"
        if summarize is not None and len(hits) >= threshold:
            return summarize(query, hits)
        out, total = [], 0
        for h in hits:
            line = f"- {h.text}" + (f"  [{h.source}]" if h.source else "")
            if total + len(line) > _MAX_RECALL_CHARS:
                break
            out.append(line)
            total += len(line)
        return "Souvenirs pertinents :\n" + "\n".join(out)

    return ToolSpec(
        name="recall",
        description=(
            "Interroge ta mémoire persistante (épisodes des sessions passées) par recherche "
            "plein-texte. À utiliser quand une tâche ressemble à du déjà-vu : tu peux avoir "
            "une leçon ou une procédure mémorisée. Argument : query (mots-clés)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Mots-clés de ce que tu cherches.",
                },
                "k": {
                    "type": "integer",
                    "description": "Nombre de souvenirs (défaut 5, max 10).",
                },
            },
            "required": ["query"],
        },
        run=run,
    )
