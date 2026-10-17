"""Readable execution journal from persisted events, never from model assertions."""

from __future__ import annotations

import json
import re


def render_transcript(events: list[dict]) -> str:
    out = [
        "# Journal de session\n\nEntrées et sorties enregistrées des outils. "
        "Un appel réussi ne garantit pas que la tâche demandée soit accomplie.\n"
    ]
    pending = {}
    text = []

    def flush():
        if text:
            out.append("\n## Réponse\n\n" + "".join(text) + "\n")
            text.clear()

    def block(value):
        value = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, indent=2)
        )
        # Longer than any backtick run in the output: preserve arbitrary tool text.
        fence = "`" * max(
            3, 1 + max((len(m[0]) for m in re.finditer(r"`+", value)), default=0)
        )
        return f"\n{fence}\n{value}\n{fence}\n"

    for event in events:
        kind, d = event.get("event"), event.get("data") or {}
        if kind == "text":
            text.append(d.get("text", ""))
            continue
        if kind == "reasoning":
            continue
        flush()
        if kind == "user":
            out.append("\n## Demande\n" + block(d.get("content", "")))
        elif kind == "harness":
            out.append("\n## Relance du harnais\n" + block(d.get("text", "")))
        elif kind == "tool_call":
            pending[d.get("id")] = d
        elif kind == "tool_result":
            pending.pop(d.get("id"), None)
            status = "succès de l’appel" if d.get("ok") else "échec de l’appel"
            out.append(f"\n## Outil : {d.get('name', '?')} ({status})\n")
            out.append(
                "\nEntrée :\n"
                + block(
                    d.get("in_full", d.get("cmd") or d.get("path") or "non enregistrée")
                )
            )
            output = d.get(
                "out_full", d.get("detail", d.get("preview", "non enregistrée"))
            )
            out.append("\nSortie enregistrée :\n" + block(output))
    flush()
    for call in pending.values():
        out.append(
            f"\nOutil {call.get('name', '?')} : aucun résultat enregistré (en cours ou interrompu).\n"
        )
    return "".join(out)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys

    parser = argparse.ArgumentParser(
        description="Afficher les preuves d’exécution d’une session Loom"
    )
    parser.add_argument(
        "session", type=Path, help="Dossier de session ou timeline.jsonl"
    )
    args = parser.parse_args()
    path = args.session / "timeline.jsonl" if args.session.is_dir() else args.session
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # an active writer may not have finished its final event
    sys.stdout.reconfigure(encoding="utf-8")
    print(render_transcript(rows))
