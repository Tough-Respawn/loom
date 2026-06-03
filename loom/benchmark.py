# loom/benchmark.py
"""Benchmark du runtime local : débit, latence, validité JSON contraint.

Prérequis : un modèle servi sur l'endpoint (uv run loom/serve.py).
Usage : uv run loom/benchmark.py
Sort en code != 0 si l'endpoint ne répond pas, si le débit est sous le seuil,
ou si la sortie JSON contrainte n'est pas un JSON valide.
"""

from __future__ import annotations

import json

import requests

BASE_URL = "http://127.0.0.1:8080"
MIN_TOKENS_PER_SEC = 5.0  # plancher (CPU-friendly) ; bien plus haut en GPU


def _chat(messages: list[dict], **extra) -> dict:
    payload = {"model": "local", "messages": messages, "stream": False, **extra}
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def bench_throughput() -> float:
    data = _chat(
        [
            {
                "role": "user",
                "content": "Écris une fonction Python qui inverse une liste.",
            }
        ],
        max_tokens=128,
    )
    timings = data.get("timings", {})
    tps = float(timings.get("predicted_per_second", 0.0))
    n = timings.get("predicted_n", 0)
    print(
        f"[bench] {n} tokens générés -> {tps:.1f} tok/s "
        f"(prompt {timings.get('prompt_per_second', 0):.1f} tok/s)"
    )
    return tps


def bench_json() -> bool:
    """Force une sortie JSON via response_format et vérifie qu'elle parse."""
    data = _chat(
        [
            {
                "role": "user",
                "content": 'Donne un objet JSON {"langage": ..., "annee": ...}.',
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=128,
    )
    content = data["choices"][0]["message"]["content"]
    try:
        json.loads(content)
        print("[bench] sortie JSON contrainte : VALIDE")
        return True
    except json.JSONDecodeError:
        print(f"[bench] sortie JSON contrainte : INVALIDE -> {content!r}")
        return False


def main() -> int:
    try:
        tps = bench_throughput()
        json_ok = bench_json()
    except requests.RequestException as exc:
        print(f"[bench] ERREUR : endpoint injoignable ({exc}). serve.py tourne ?")
        return 1

    ok = tps >= MIN_TOKENS_PER_SEC and json_ok
    print(
        f"[bench] RÉSULTAT : {'OK' if ok else 'ÉCHEC'} (seuil {MIN_TOKENS_PER_SEC} tok/s)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
