"""Génération de la config llama-swap (un modèle = une commande llama-server)."""

from __future__ import annotations

from pathlib import Path

import yaml

from loom.config import ModelConfig
from loom.runtime.effective import launch_flags
from loom.runtime.hardware import HardwareProfile
from loom.runtime.ngl import resolve_ngl
from loom.runtime.server_args import build_server_args, resolve_parallel
from loom.runtime.serverlog import server_log_path
from loom.utils import atomic_write_text


def _model_cmd(
    model: ModelConfig,
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
    override_n_gpu_layers: int | None = None,
    slot_save_dir: str | None = None,
    n_parallel: int = 1,
    # Repli MACHINE, même précédence que `context` : le modèle gagne, la machine
    # sert de défaut mesuré, la constante aveugle ne sert qu'en dernier recours.
    default_ubatch: int | None = None,
    default_batch: int | None = None,
    default_checkpoint_min_step: int | None = None,
    log_verbosity: int | None = None,
    override_threads: int | None = None,
) -> str:
    base = (
        model.dir or models_dir
    )  # dossier du modèle (découverte) sinon racine partagée
    model_path = f"{base}/{model.filename}"
    # Garder la même précédence d'offload que le chemin mono-modèle.
    ngl = resolve_ngl(model, profile, override_n_gpu_layers)
    ctx = model.context or context
    mmproj = f"{base}/{model.mmproj_filename}" if model.mmproj_filename else None
    # Mêmes flags machine que serve.py et que la sonde : une seule dérivation
    # (effective.launch_flags), sinon le routeur mesure/sert une autre configuration.
    flags = launch_flags(profile, override_threads)
    args = build_server_args(
        server_bin=model.server_bin or llama_bin,
        model_path=model_path,
        port="${PORT}",
        context=ctx,
        n_gpu_layers=ngl,
        threads=flags.threads,
        mmproj_path=mmproj,
        gpu_tuning=flags.gpu_tuning,
        unified_memory=flags.unified_memory,
        cpu_moe=model.cpu_moe,
        n_cpu_moe=model.n_cpu_moe,
        slot_save_dir=slot_save_dir,
        ubatch=model.ubatch or default_ubatch,
        batch=model.batch or default_batch,
        checkpoint_min_step=model.checkpoint_min_step or default_checkpoint_min_step,
        ctx_checkpoints=model.ctx_checkpoints,
        # L'isolation du cache est une propriété du modèle, pas de la machine.
        n_parallel=resolve_parallel(n_parallel, model.cache_isolation),
        log_file=server_log_path(model.id) if log_verbosity else None,
        log_verbosity=log_verbosity,
    )
    return " ".join(_quote(str(a).replace("\\", "/")) for a in args)


def _quote(arg: str) -> str:
    """llama-swap redécoupe `cmd` (shlex Windows ou POSIX) : un argument avec espace
    doit être entre guillemets doubles, compris des deux découpages."""
    return f'"{arg}"' if any(c.isspace() for c in arg) else arg


# Délai /health accordé par llama-swap à un llama-server qui charge (défaut llama-swap :
# 120 s, au-delà il le tue et le relance). Un MoE Q8 de 36 Go lu en RAM sur un PC chargé
# l'a dépassé (vécu 2026-10-02 : 138 s perdues + une requête en 500). Large exprès : un
# vrai blocage est déjà signalé par le disjoncteur et le journal llama-server.
SWAP_HEALTH_TIMEOUT_S = 600


def build_swap_config(
    models: list[ModelConfig],
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
    override_n_gpu_layers: int | None = None,
    slot_save_dir: str | None = None,
    n_parallel: int = 1,
    default_ubatch: int | None = None,
    default_batch: int | None = None,
    default_checkpoint_min_step: int | None = None,
    log_verbosity: int | None = None,
    override_threads: int | None = None,
) -> dict:
    return {
        "healthCheckTimeout": SWAP_HEALTH_TIMEOUT_S,
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
                    default_ubatch=default_ubatch,
                    default_batch=default_batch,
                    default_checkpoint_min_step=default_checkpoint_min_step,
                    log_verbosity=log_verbosity,
                    override_threads=override_threads,
                )
            }
            for m in models
        },
    }


def dump_yaml(config: dict) -> str:
    """Sérialise la structure {models: {id: {cmd: str}}} en YAML (PyYAML)."""
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def write_swap_yaml(config: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # var/cache/ absent sur un clone neuf
    atomic_write_text(p, dump_yaml(config))
