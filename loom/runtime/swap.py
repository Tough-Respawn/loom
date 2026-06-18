# loom/runtime/swap.py
"""Génération de la config llama-swap (un modèle = une commande llama-server)."""

from __future__ import annotations

from pathlib import Path

from loom.config import ModelConfig
from loom.runtime.hardware import HardwareProfile, recommend_gpu_layers
from loom.runtime.server_args import build_server_args


def _model_cmd(
    model: ModelConfig,
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
    override_n_gpu_layers: int | None = None,
) -> str:
    base = (
        model.dir or models_dir
    )  # dossier du modèle (découverte) sinon racine partagée
    model_path = f"{base}/{model.filename}"
    # Précédence (identique au chemin mono-modèle resolve_n_gpu_layers) : champ par
    # modèle > override global ([override] n_gpu_layers, ex. 99 = offload total) >
    # recommandation auto. SANS l'override ici, llama-swap laissait des couches sur CPU
    # (-ngl 30/35 pour Gemma) -> plus lent que l'offload total (33 tok/s).
    if model.cpu_moe or model.n_cpu_moe is not None:
        # MoE offloadé : toutes les couches DENSES sur GPU (-ngl 999), les experts routés
        # partent en RAM via --cpu-moe / --n-cpu-moe (build_server_args). On ignore
        # l'override global n_gpu_layers (pensé pour les petits modèles denses).
        ngl = 999
    elif model.n_gpu_layers is not None:
        ngl = model.n_gpu_layers
    elif override_n_gpu_layers is not None:
        ngl = override_n_gpu_layers
    elif profile.has_gpu:
        ngl = recommend_gpu_layers(profile.vram_free_mb, model.size_mb, model.n_layers)
    else:
        ngl = 0
    # Contexte propre au modèle si défini (gros MoE -> KV plus lourd -> on raccourcit).
    ctx = model.context or context
    mmproj = f"{base}/{model.mmproj_filename}" if model.mmproj_filename else None
    # Mêmes réglages perf que le chemin mono-modèle (serve.py) : Flash-Attn + KV q8_0
    # (gpu_tuning) divisent le KV par 2 -> indispensable sur 6 Go, sinon spill. Threads =
    # cœurs PHYSIQUES en GPU (logiques/2) pour ne pas pénaliser la passe CPU (PLE Gemma).
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
        cpu_moe=model.cpu_moe,
        n_cpu_moe=model.n_cpu_moe,
    )
    return " ".join(str(a) for a in args).replace("\\", "/")


def build_swap_config(
    models: list[ModelConfig],
    profile: HardwareProfile,
    llama_bin: str,
    models_dir: str,
    context: int,
    override_n_gpu_layers: int | None = None,
) -> dict:
    return {
        "models": {
            m.id: {
                "cmd": _model_cmd(
                    m, profile, llama_bin, models_dir, context, override_n_gpu_layers
                )
            }
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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # var/cache/ absent sur un clone neuf
    p.write_text(dump_yaml(config), encoding="utf-8")
