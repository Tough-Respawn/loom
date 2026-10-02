"""Construction de la ligne de commande llama-server (pure, sauf la sonde mise en
cache des flags du binaire : `no_mmap_args`)."""

from __future__ import annotations

import subprocess
from functools import lru_cache


@lru_cache(maxsize=16)
def no_mmap_args(server_bin: str) -> list[str]:
    """Flag « pas de mmap » compris par CE binaire. llama.cpp a retiré `--no-mmap`
    le 2026-09-09 (#28334) au profit de `--load-mode none` : un binaire récent
    refuse l'ancien flag (le serveur ne démarre pas), un ancien ignore le nouveau.
    Sonde `--help` une fois par chemin ; binaire injoignable ou macro llama-swap
    (`${...}`) -> ancien flag, comportement historique."""
    if "${" in server_bin:
        return ["--no-mmap"]
    try:
        res = subprocess.run(
            [server_bin, "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
        )
        out = (res.stdout or "") + (res.stderr or "")
    except Exception:  # noqa: BLE001 - sonde best-effort, jamais bloquante
        return ["--no-mmap"]
    return ["--load-mode", "none"] if "--load-mode" in out else ["--no-mmap"]


def resolve_parallel(n_parallel: int, cache_isolation: bool) -> int:
    """Nombre de slots EFFECTIF pour un modèle : le global [server] n_parallel,
    monté à 2 minimum quand le bench a mesuré que le cache du modèle ne survit
    pas à la pollution du slot (model.toml cache_isolation = true — mémoire
    hybride/SWA exclue du prompt-cache RAM natif). Les appels annexes s'isolent
    alors dans le 2e slot et la conversation garde son cache."""
    base = max(1, n_parallel)
    return max(base, 2) if cache_isolation else base


def compute_slot_counts(models, n_parallel: int) -> dict[str, int]:
    """Slots effectifs par modèle (= --parallel émis dans le yaml). Sert au client
    pour router les appels annexes sur le slot 1 quand il existe."""
    return {m.id: resolve_parallel(n_parallel, m.cache_isolation) for m in models}


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
    ctx_checkpoints: int | None = None,
    log_file: str | None = None,
    log_verbosity: int | None = None,
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
        # Sans mmap : accélère les dGPU, mais peut épuiser la mémoire unifiée Vulkan.
        if not unified_memory:
            args += no_mmap_args(str(server_bin))
    else:
        # Hors profil GPU, honorer quand même des batchs EXPLICITES (mesurés par la
        # sonde ou posés dans model.toml) : ils étaient silencieusement ignorés sur
        # les topologies CPU, sonde comprise — mesurer deux configs identiques.
        if batch:
            args += ["-b", str(batch)]
        if ubatch:
            args += ["-ub", str(ubatch)]
    if mmproj_path:
        args += ["--mmproj", str(mmproj_path), "--no-mmproj-offload"]
    # Les appels annexes écrasent le slot unique; sa sauvegarde évite un nouveau prefill.
    if slot_save_dir:
        args += ["--slot-save-path", str(slot_save_dir)]
    # Un maillage plus serré borne le retraitement des modèles hybrides après compaction.
    if checkpoint_min_step is not None:
        args += ["--checkpoint-min-step", str(checkpoint_min_step)]
    # Moins de checkpoints = moins de RAM et un save de fin de tour plus léger.
    if ctx_checkpoints is not None:
        args += ["--ctx-checkpoints", str(ctx_checkpoints)]
    # Journal fichier : sans lui, les décisions de cache ne sont visibles nulle part.
    if log_file and log_verbosity:
        args += ["--log-file", str(log_file), "-lv", str(log_verbosity)]
    return args
