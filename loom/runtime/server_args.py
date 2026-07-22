# loom/runtime/server_args.py
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
        # SÉMANTIQUE : `context` = fenêtre PAR SLOT (celle que le bench calibre
        # et que l'utilisateur voit). llama-server répartit -c entre les slots
        # -> on multiplie ici. Vécu 2026-07-22 : un -c doublé posé en dur dans
        # la config avait été re-mesuré par /rebench comme une fenêtre mono-slot
        # -> spill mémoire (0,8 t/s) et faux verdict de réduction.
        "-c",
        str(context * max(1, n_parallel)),
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
    # Offload MoE : garde attention + FFN dense sur GPU (-ngl élevé), bascule les experts
    # ROUTÉS en RAM. `--n-cpu-moe N` n'offloade que N couches (garde le reste sur GPU ->
    # remplit la VRAM, plus rapide) ; `--cpu-moe` les offloade TOUTES. Indispensable pour
    # faire tenir un MoE 26-35B sur 6 Go.
    if n_cpu_moe is not None:
        args += ["--n-cpu-moe", str(n_cpu_moe)]
    elif cpu_moe:
        args.append("--cpu-moe")
    if gpu_tuning:
        # Réglages GPU prouvés au banc : Flash-Attention (attention plus rapide +
        # cache KV ~÷2), cache KV quantifié q8_0, gros batch de prompt, priorité
        # process. Couplé au continuous batching (n_parallel auto), gain net en
        # single-stream comme en parallèle.
        # `ubatch`/`batch` PAR MODÈLE (model.toml) : sur un MoE offloadé, le prefill
        # est borné par l'amortissement des passes d'experts CPU sur la taille du
        # microbatch — un -ub plus gros le multiplie (mesuré au banc), au prix de
        # buffers VRAM plus gros (à valider par modèle vs sa marge). Défauts inchangés.
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
        # Poids CPU (experts MoE) chargés en mémoire hôte PINNÉE au lieu de mmap :
        # uploads GPU en DMA -> gain de prefill net, mesuré au banc — mais sur les
        # dGPU CUDA du parc UNIQUEMENT. Sur mémoire UNIFIÉE (iGPU Vulkan), c'est
        # un bug upstream (ggml-org/llama.cpp #18317 « Cannot Run Model with
        # mmap = 0 », #14999 --no-mmap + MoE) et un crash vécu (Ornith 35B /
        # Radeon 860M : ErrorOutOfDeviceMemory au CHARGEMENT) : mmap, le défaut
        # officiel llama.cpp, est conservé.
        if not unified_memory:
            args.append("--no-mmap")
    if mmproj_path:
        args += ["--mmproj", str(mmproj_path), "--no-mmproj-offload"]
    # Sauvegarde/restauration du cache KV du slot (API POST /slots/0?action=save|restore,
    # fichiers sous ce dossier). Pilier du « cache souverain » : le slot est UNIQUE, tout
    # appel non-conversationnel (sous-agent, reflect, titre) écrase le cache du fil ->
    # on le sauve avant, on le restaure après (~ms au lieu de minutes de re-prefill ;
    # mesuré 2026-07-10 : save 42 ms / restore 22 ms / reprise 11 tokens au lieu de 925,
    # KV q8_0 supporté).
    if slot_save_dir:
        args += ["--slot-save-path", str(slot_save_dir)]
    return args
