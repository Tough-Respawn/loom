"""Construction (pure) de la ligne de commande llama-server."""

from __future__ import annotations


def resolve_parallel(n_parallel: int, cache_isolation: bool) -> int:
    """Nombre de slots EFFECTIF pour un modèle : le global [server] n_parallel,
    monté à 2 minimum quand le bench a mesuré que le cache du modèle ne survit
    pas à la pollution du slot (model.toml cache_isolation = true — mémoire
    hybride/SWA exclue du prompt-cache RAM natif). Les appels annexes s'isolent
    alors dans le 2e slot et la conversation garde son cache."""
    base = max(1, n_parallel)
    return max(base, 2) if cache_isolation else base


def build_server_args(
    server_bin: str,
    model_path: str,
    port: int,
    context: int,
    n_gpu_layers: int,
    threads: int,
    mmproj_path: str | None = None,
    gpu_tuning: bool = False,
    unified_memory: bool = False,
    n_parallel: int = 1,
    cpu_moe: bool = False,
    n_cpu_moe: int | None = None,
    slot_save_dir: str | None = None,
    ubatch: int | None = None,
    batch: int | None = None,
    checkpoint_min_step: int | None = None,
) -> list[str]:
    """Liste d'arguments pour lancer llama-server en API OpenAI-compatible local.

    `n_parallel` fixe le nombre de slots (--parallel) de llama-server. Loom étant
    mono-flux, 1 suffit et laisse tout le pool KV (-c) au seul échange en cours.

    `cpu_moe` offloade TOUS les experts MoE en RAM (`--cpu-moe`), `n_cpu_moe` n'en
    offloade que N (`--n-cpu-moe N`, garde le reste sur GPU) ; incompatibles entre
    eux, `n_cpu_moe` prioritaire. `gpu_tuning` active le profil GPU benchmarké
    (Flash-Attention, cache KV q8_0, gros batch prompt). `mmproj_path` ajoute un
    projet multimodal (`--mmproj`) pour les modèles vision.
    """
    args = [
        server_bin,
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        # `context` est par slot, tandis que llama-server attend le total.
        "-c",
        str(context * max(1, n_parallel)),
        "--parallel",
        str(max(1, n_parallel)),
        "-ngl",
        str(n_gpu_layers),
        "-t",
        str(threads),
        # Requis pour les appels d'outils structurés des templates llama.cpp.
        "--jinja",
    ]
    # L'offload partiel conserve plus d'experts sur GPU et reste donc plus rapide.
    if n_cpu_moe is not None:
        args += ["--n-cpu-moe", str(n_cpu_moe)]
    elif cpu_moe:
        args.append("--cpu-moe")
    if gpu_tuning:
        # Ces valeurs mesurées privilégient le débit tout en bornant le cache KV.
        # Les batchs par modèle permettent d'ajuster le compromis débit/VRAM.
        args += [
            "-fa",
            "on",
            "-b",
            str(batch or 2048),
            "-ub",
            str(ubatch or 512),
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--prio",
            "2",
        ]
        # `--no-mmap` accélère les dGPU, mais peut épuiser la mémoire unifiée Vulkan.
        if not unified_memory:
            args.append("--no-mmap")
    if mmproj_path:
        args += ["--mmproj", str(mmproj_path), "--no-mmproj-offload"]
    # Les appels annexes écrasent le slot unique; sa sauvegarde évite un nouveau prefill.
    if slot_save_dir:
        args += ["--slot-save-path", str(slot_save_dir)]
    # Un maillage plus serré borne le retraitement des modèles hybrides après compaction.
    if checkpoint_min_step is not None:
        args += ["--checkpoint-min-step", str(checkpoint_min_step)]
    return args
