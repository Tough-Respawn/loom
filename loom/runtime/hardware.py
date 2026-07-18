# loom/runtime/hardware.py
"""Détection hardware et recommandation de réglages d'offload GPU."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


def ram_available_mb() -> int:
    """RAM physique DISPONIBLE (Mo), 0 si indéterminable.

    Windows : GlobalMemoryStatusEx (ullAvailPhys). POSIX : MemAvailable de
    /proc/meminfo. Sert aux arbitrages de co-résidence (ex. garder le cache
    RAM du moteur image à côté du LLM quand la machine est assez large)."""
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys // (1024 * 1024))
        return 0
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def top_ram_processes(limit: int = 8) -> list[dict]:
    """Plus gros consommateurs de RAM, GROUPÉS par nom de process (un Chrome =
    des dizaines de process : la somme est la seule vue actionnable). Renvoie
    [{name, mb, count}] triés décroissants. Liste vide si psutil indisponible —
    affichage best-effort, jamais bloquant."""
    try:
        import psutil
    except ImportError:
        return []
    totals: dict[str, list[int]] = {}
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = p.info["name"] or "?"
            mem = p.info["memory_info"]
            rss = mem.rss if mem else 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        t = totals.setdefault(name, [0, 0])
        t[0] += rss
        t[1] += 1
    rows = [
        {"name": n, "mb": rss // (1024 * 1024), "count": c}
        for n, (rss, c) in totals.items()
    ]
    rows.sort(key=lambda r: r["mb"], reverse=True)
    return rows[:limit]


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
