# loom/swap.py
"""Génération de la config llama-swap (un modèle = une commande llama-server)."""

from __future__ import annotations

from pathlib import Path

from loom.config import ModelConfig
from loom.hardware import HardwareProfile, recommend_gpu_layers
from loom.server_args import build_server_args


def _model_cmd(
    model: ModelConfig,
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
) -> str:
    model_path = f"{models_dir}/{model.filename}"
    if model.n_gpu_layers is not None:
        ngl = model.n_gpu_layers
    elif profile.has_gpu:
        ngl = recommend_gpu_layers(profile.vram_free_mb, model.size_mb, model.n_layers)
    else:
        ngl = 0
    mmproj = f"{models_dir}/{model.mmproj_filename}" if model.mmproj_filename else None
    args = build_server_args(
        server_bin=llama_bin,
        model_path=model_path,
        port="${PORT}",
        context=context,
        n_gpu_layers=ngl,
        threads=profile.cpu_threads,
        mmproj_path=mmproj,
    )
    return " ".join(str(a) for a in args).replace("\\", "/")


def build_swap_config(
    models: list[ModelConfig],
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
) -> dict:
    return {
        "models": {
            m.id: {"cmd": _model_cmd(m, profile, llama_bin, models_dir, context)}
            for m in models
        }
    }


def dump_yaml(config: dict) -> str:
    """Sérialise la structure simple {models: {id: {cmd: str}}} en YAML."""
    lines = ["models:"]
    for model_id, entry in config["models"].items():
        lines.append(f'  "{model_id}":')
        cmd = entry["cmd"].replace('"', '\\"')
        lines.append(f'    cmd: "{cmd}"')
    return "\n".join(lines) + "\n"


def write_swap_yaml(config: dict, path: str | Path) -> None:
    Path(path).write_text(dump_yaml(config), encoding="utf-8")
