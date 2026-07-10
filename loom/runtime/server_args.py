# loom/runtime/server_args.py
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
    cpu_moe: bool = False,
    n_cpu_moe: int | None = None,
    slot_save_dir: str | None = None,
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
    # Offload MoE : garde attention + FFN dense sur GPU (-ngl élevé), bascule les experts
    # ROUTÉS en RAM. `--n-cpu-moe N` n'offloade que N couches (garde le reste sur GPU ->
    # remplit la VRAM, plus rapide) ; `--cpu-moe` les offloade TOUTES. Indispensable pour
    # faire tenir un MoE 26-35B sur 6 Go.
    if n_cpu_moe is not None:
        args += ["--n-cpu-moe", str(n_cpu_moe)]
    elif cpu_moe:
        args.append("--cpu-moe")
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
            # Poids CPU (experts MoE) chargés en mémoire hôte PINNÉE CUDA au lieu de mmap :
            # les uploads vers le GPU passent en DMA -> +21% de prefill (gemma), +89% (qwen)
            # au bench 2026-07-06. Contrepartie : chargement plus long et RAM non-paginable
            # (~taille du modèle). cf. docs/bench-llama.md.
            "--no-mmap",
        ]
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
