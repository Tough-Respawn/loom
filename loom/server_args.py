# loom/server_args.py
"""Construction (pure) de la ligne de commande llama-server."""

from __future__ import annotations


def build_server_args(
    server_bin: str,
    model_path: str,
    port: int,
    context: int,
    n_gpu_layers: int,
    threads: int,
    mmproj_path: str | None = None,
    gpu_tuning: bool = False,
    n_parallel: int = 1,
) -> list[str]:
    """Liste d'arguments pour lancer llama-server en API OpenAI-compatible local.

    `n_parallel` ÉPINGLE le nombre de slots de batching continu : c'est la SOURCE DE
    VÉRITÉ unique du parallélisme. Le harness (compute_budget) dérive sa concurrence et
    ses tailles de génération de cette même valeur → on ne peut plus déborder le pool KV
    partagé (cf. docs/plan-harness-robustesse). Auparavant implicite/hand-tuné.
    """
    args = [
        server_bin,
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(context),
        "--parallel",
        str(max(1, n_parallel)),
        "-ngl",
        str(n_gpu_layers),
        "-t",
        str(threads),
        # Active le chat-template Jinja tool-aware : nécessaire pour que le modèle
        # émette des tool_calls structurés (boucle tool-use). Sans effet si aucun
        # outil n'est passé dans la requête.
        "--jinja",
    ]
    if gpu_tuning:
        # Réglages GPU prouvés au benchmark (RTX 2060) : Flash-Attention (attention
        # plus rapide + cache KV ~÷2), cache KV quantifié q8_0, gros batch de prompt,
        # priorité process. Couplé au continuous batching (n_parallel auto) : ~+47%
        # en single-stream et ~×3 en parallèle. Cf. docs/perf-gpu.md.
        args += [
            "-fa",
            "on",
            "-b",
            "2048",
            "-ub",
            "512",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--prio",
            "2",
        ]
    if mmproj_path:
        args += ["--mmproj", str(mmproj_path), "--no-mmproj-offload"]
    return args
