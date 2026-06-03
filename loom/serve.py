# loom/serve.py
"""Lanceur cross-platform et auto-adaptatif de llama-server.

Usage : uv run loom/serve.py
Auto-détecte le hardware (GPU NVIDIA sinon CPU), résout la config,
télécharge les GGUF du registre si absents, génère le llama-swap.yaml
et démarre llama-swap (routeur multi-modèles, API OpenAI-compatible).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from loom.config import RuntimeConfig, load_config
from loom.hardware import HardwareProfile, detect_hardware, recommend_gpu_layers
from loom.models_fetch import ensure_model
from loom.server_args import build_server_args
from loom.swap import build_swap_config, write_swap_yaml

LOOM_DIR = Path(__file__).resolve().parent
CONFIG_PATH = LOOM_DIR / "loom.config.toml"
LOCAL_CONFIG_PATH = LOOM_DIR / "loom.config.local.toml"
MODELS_DIR = LOOM_DIR / "models"
SWAP_YAML = MODELS_DIR.parent / "llama-swap.yaml"


def resolve_n_gpu_layers(
    profile: HardwareProfile,
    override: int | None,
    model_size_mb: int,
    total_layers: int,
) -> int:
    """Override prioritaire ; sinon 0 en CPU, sinon recommandation auto."""
    if override is not None:
        return override
    if not profile.has_gpu:
        return 0
    return recommend_gpu_layers(profile.vram_free_mb, model_size_mb, total_layers)


def resolve_mmproj_path(
    mmproj_filename: str, models_dir: Path, repo: str = ""
) -> str | None:
    """Télécharge le mmproj si configuré, renvoie son chemin local (ou None)."""
    if not mmproj_filename:
        return None
    path = ensure_model(repo, mmproj_filename, models_dir)
    return str(path)


def ensure_all_models(models, models_dir: Path) -> None:
    """Télécharge le GGUF (et le mmproj) de chaque modèle du registre s'il manque."""
    for m in models:
        ensure_model(m.repo, m.filename, models_dir)
        if m.mmproj_filename:
            ensure_model(m.repo, m.mmproj_filename, models_dir)


def build_launch(
    cfg: RuntimeConfig,
    profile: HardwareProfile,
    model_path: Path,
    mmproj_path: str | None = None,
) -> list[str]:
    n_gpu = resolve_n_gpu_layers(
        profile, cfg.override_n_gpu_layers, cfg.model.size_mb, cfg.model.n_layers
    )
    # En mode GPU, threads = cœurs PHYSIQUES (≈ logiques/2 si HyperThreading) : au-delà,
    # la contention HT ralentit la passe CPU (PLE de Gemma 3n). En CPU-only, tous les
    # threads. cf. benchmark docs/perf-gpu.md.
    if cfg.override_threads:
        threads = cfg.override_threads
    elif profile.has_gpu:
        threads = max(1, profile.cpu_threads // 2)
    else:
        threads = profile.cpu_threads
    return build_server_args(
        server_bin=cfg.server_bin,
        model_path=str(model_path),
        port=cfg.port,
        context=cfg.context,
        n_gpu_layers=n_gpu,
        threads=threads,
        mmproj_path=mmproj_path,
        gpu_tuning=profile.has_gpu,
        n_parallel=cfg.n_parallel,
    )


def _run(args: list[str], bin_name: str, hint: str) -> int:
    """Lance un binaire externe ; message clair s'il est introuvable."""
    print(f"[loom] Lancement : {' '.join(args)}", file=sys.stderr)
    try:
        return subprocess.run(args).returncode
    except FileNotFoundError:
        print(
            f"[loom] ERREUR : binaire '{bin_name}' introuvable. {hint}",
            file=sys.stderr,
        )
        return 1


def launch_direct(cfg: RuntimeConfig, profile: HardwareProfile) -> int:
    """Un seul modèle : llama-server directement, pas besoin de llama-swap."""
    model = cfg.model  # = modèle par défaut
    model_path = MODELS_DIR / model.filename
    mmproj_path = resolve_mmproj_path(
        model.mmproj_filename, MODELS_DIR, repo=model.repo
    )
    args = build_launch(cfg, profile, model_path, mmproj_path)
    return _run(
        args,
        cfg.server_bin,
        "Renseigne 'bin' dans loom.config.local.toml (voir docs/install-windows.md).",
    )


def launch_swap(cfg: RuntimeConfig, profile: HardwareProfile) -> int:
    """Plusieurs modèles : llama-swap route vers le bon selon le champ 'model'."""
    swap = build_swap_config(
        cfg.models,
        profile,
        llama_bin=cfg.server_bin,
        models_dir=str(MODELS_DIR),
        context=cfg.context,
    )
    write_swap_yaml(swap, SWAP_YAML)
    args = [
        cfg.swap_bin,
        "--config",
        str(SWAP_YAML),
        "--listen",
        f"127.0.0.1:{cfg.port}",
    ]
    return _run(
        args,
        cfg.swap_bin,
        "Télécharge llama-swap (voir docs/install-windows.md), ou garde un seul "
        "modèle pour lancer llama-server en direct.",
    )


def main() -> int:
    cfg = load_config(CONFIG_PATH, LOCAL_CONFIG_PATH)
    profile = detect_hardware()
    print(f"[loom] Profil détecté : {profile}", file=sys.stderr)

    ensure_all_models(cfg.models, MODELS_DIR)
    print(
        f"[loom] {len(cfg.models)} modèle(s), défaut={cfg.default_model}",
        file=sys.stderr,
    )

    # Un seul modèle : pas de routeur, llama-server direct (zéro dépendance externe).
    # Plusieurs : llama-swap pour le hot-swap par le champ 'model' de la requête.
    if len(cfg.models) <= 1:
        return launch_direct(cfg, profile)
    return launch_swap(cfg, profile)


if __name__ == "__main__":
    raise SystemExit(main())
