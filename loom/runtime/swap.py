"""Génération de la config llama-swap (un modèle = une commande llama-server)."""

from __future__ import annotations

from pathlib import Path

import yaml

from loom.config import ModelConfig
from loom.runtime.hardware import HardwareProfile
from loom.runtime.ngl import resolve_ngl
from loom.runtime.server_args import build_server_args, resolve_parallel


def _model_cmd(
    model: ModelConfig,
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
    override_n_gpu_layers: int | None = None,
    slot_save_dir: str | None = None,
    n_parallel: int = 1,
) -> str:
    base = (
        model.dir or models_dir
    )  # dossier du modèle (découverte) sinon racine partagée
    model_path = f"{base}/{model.filename}"
    # Garder la même précédence d'offload que le chemin mono-modèle.
    ngl = resolve_ngl(model, profile, override_n_gpu_layers)
    ctx = model.context or context
    mmproj = f"{base}/{model.mmproj_filename}" if model.mmproj_filename else None
    # Reprendre les réglages mono-modèle évite des performances différentes via le routeur.
    threads = (
        max(1, profile.cpu_threads // 2) if profile.has_gpu else profile.cpu_threads
    )
    args = build_server_args(
        server_bin=llama_bin,
        model_path=model_path,
        port="${PORT}",
        context=ctx,
        n_gpu_layers=ngl,
        threads=threads,
        mmproj_path=mmproj,
        gpu_tuning=profile.has_gpu,
        unified_memory=not profile.vram_is_discrete,
        cpu_moe=model.cpu_moe,
        n_cpu_moe=model.n_cpu_moe,
        slot_save_dir=slot_save_dir,
        ubatch=model.ubatch,
        batch=model.batch,
        checkpoint_min_step=model.checkpoint_min_step,
        # L'isolation du cache est une propriété du modèle, pas de la machine.
        n_parallel=resolve_parallel(n_parallel, model.cache_isolation),
    )
    return " ".join(str(a) for a in args).replace("\\", "/")


def build_swap_config(
    models: list[ModelConfig],
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
    override_n_gpu_layers: int | None = None,
    slot_save_dir: str | None = None,
    n_parallel: int = 1,
) -> dict:
    return {
        "models": {
            m.id: {
                "cmd": _model_cmd(
                    m,
                    profile,
                    llama_bin,
                    models_dir,
                    context,
                    override_n_gpu_layers,
                    slot_save_dir=slot_save_dir,
                    n_parallel=n_parallel,
                )
            }
            for m in models
        }
    }


def dump_yaml(config: dict) -> str:
    """Sérialise la structure {models: {id: {cmd: str}}} en YAML (PyYAML)."""
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def write_swap_yaml(config: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # var/cache/ absent sur un clone neuf
    p.write_text(dump_yaml(config), encoding="utf-8")
