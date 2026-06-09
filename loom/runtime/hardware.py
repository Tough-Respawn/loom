# loom/runtime/hardware.py
"""Détection hardware et recommandation de réglages d'offload GPU."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


def recommend_gpu_layers(
    vram_free_mb: int,
    model_size_mb: int,
    total_layers: int,
    kv_headroom_mb: int = 1024,
) -> int:
    """Nombre de couches à offloader sur GPU selon la VRAM libre.

    Renvoie 0 (CPU-only) si le budget VRAM ne dépasse pas la marge réservée
    au cache KV. Sinon offload proportionnel, plafonné à toutes les couches.
    """
    budget_mb = vram_free_mb - kv_headroom_mb
    if budget_mb <= 0:
        return 0
    if budget_mb >= model_size_mb:
        return total_layers
    return max(0, round(total_layers * budget_mb / model_size_mb))


def parse_nvidia_smi(output: str) -> tuple[str, int] | None:
    """Parse la 1re ligne de `nvidia-smi --query-gpu=name,memory.free
    --format=csv,noheader,nounits`. Renvoie (nom, vram_libre_mb) ou None.
    """
    line = output.strip().splitlines()[0] if output.strip() else ""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1])


@dataclass
class HardwareProfile:
    has_gpu: bool
    gpu_name: str | None
    vram_free_mb: int
    cpu_threads: int


def _run_nvidia_smi() -> str | None:
    """Exécute nvidia-smi si présent, renvoie sa sortie CSV ou None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return res.stdout
    except (subprocess.SubprocessError, OSError):
        return None


def detect_hardware() -> HardwareProfile:
    """Détecte le meilleur profil disponible : GPU NVIDIA si présent, sinon CPU."""
    threads = os.cpu_count() or 4
    raw = _run_nvidia_smi()
    parsed = parse_nvidia_smi(raw) if raw else None
    if parsed is None:
        return HardwareProfile(False, None, 0, threads)
    name, free_mb = parsed
    return HardwareProfile(True, name, free_mb, threads)
