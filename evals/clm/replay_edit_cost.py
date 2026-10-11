"""Coût de re-prefill d'une ÉDITION DE CONTEXTE sur llama-server, avec une vraie requête Loom.

Étude CLM (Context Language Models), cf. docs/clm-chiffrage-2026-10-08.md.

usage : uv run python evals/clm/replay_edit_cost.py <dump.json> [port=8080] [slot=1]

`dump.json` est une requête capturée par le hook `LOOM_DUMP_REQUESTS` (loom/agent/calltrace.py).
Chaque variante : renvoyer le payload de base (cache plein sur le slot), puis envoyer le payload
édité avec max_tokens=1 et lire les `timings` de llama-server (`prompt_n` = tokens re-préfillés).
Une ligne JSON par mesure sur stdout (séries archivées : evals/clm/replay_ornith*.jsonl).
"""

from __future__ import annotations

import copy
import json
import sys
import time
import urllib.request

CLEARED = "[cleared by the model: no longer needed]"


def main(argv: list[str]) -> None:
    dump = argv[1]
    port = int(argv[2]) if len(argv) > 2 else 8080
    slot = int(argv[3]) if len(argv) > 3 else 1
    base = json.load(open(dump, encoding="utf-8"))
    msgs = base["messages"]
    tools = base.get("tools")
    model = base["model"]

    def send(messages: list[dict], label: str) -> dict:
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "cache_prompt": True,
            "id_slot": slot,
        }
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=3600) as r:
            res = json.loads(r.read())
        t = res.get("timings", {})
        row = {
            "variant": label,
            "prompt_n": t.get("prompt_n"),
            "cache_n": t.get("cache_n"),
            "prompt_ms": round(t.get("prompt_ms", 0)),
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    def tool_idx() -> list[int]:
        return [i for i, x in enumerate(msgs) if x["role"] == "tool"]

    def tail(m: list[dict]) -> None:
        """Un mot ajouté au dernier message utilisateur (édition en fin de contexte)."""
        for x in reversed(m):
            if x["role"] == "user" and isinstance(x.get("content"), str):
                x["content"] += " (merci)"
                return
        m[-1]["content"] = (m[-1].get("content") or "") + " (merci)"

    def middle(m: list[dict]) -> None:
        """Le résultat d'outil du MILIEU remplacé par une note (une édition style CLM)."""
        ti = tool_idx()
        if not ti:
            raise SystemExit("aucun message tool dans ce payload")
        m[ti[len(ti) // 2]]["content"] = CLEARED

    def oldest(m: list[dict]) -> None:
        """Le plus ancien résultat d'outil remplacé."""
        m[tool_idx()[0]]["content"] = CLEARED

    def compact_all_but_last2(m: list[dict]) -> None:
        """Style microcompact de Loom : tous les résultats d'outils vidés sauf les 2 derniers."""
        for i in tool_idx()[:-2]:
            m[i]["content"] = "[cleared]"

    def head(m: list[dict]) -> None:
        """Un mot changé dans le system prompt (édition du préfixe stable)."""
        m[0]["content"] = m[0]["content"].replace("Loom", "Loom ", 1)

    def edited(fn) -> list[dict]:
        m = copy.deepcopy(msgs)
        fn(m)
        return m

    print(
        json.dumps(
            {
                "dump": dump.replace("\\", "/").split("/")[-1],
                "messages": len(msgs),
                "tool_msgs": len(tool_idx()),
                "tools": len(tools or []),
                "slot": slot,
            }
        )
    )
    send(msgs, "base_cold_or_whatever_slot_had")
    send(msgs, "base_again_warm")
    variants = [
        ("edit_tail", tail),
        ("edit_middle_tool", middle),
        ("edit_oldest_tool", oldest),
        ("compact_tools_keep2", compact_all_but_last2),
        ("edit_head_system", head),
    ]
    for label, fn in variants:
        send(msgs, "restore_base")
        send(edited(fn), label)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv)
