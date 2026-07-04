# loom/runtime/sysmon.py
"""Monitoring système LIVE (CPU / RAM / GPU) pour l'indicateur affiché quand un modèle
LOCAL est sélectionné. CPU/RAM via psutil (dégradation propre s'il manque) ; GPU via
nvidia-smi (comme la détection matérielle). Lecture GPU mise en cache ~0.8 s : nvidia-smi
coûte ~50-200 ms, on ne le relance pas à chaque client/poll rapproché."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from functools import lru_cache

try:  # psutil = dépendance ; absente -> métriques CPU/RAM à None, le reste marche.
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

# CPU % : mesuré par un thread échantillonneur DÉDIÉ, pas à chaque requête. `cpu_percent`
# a un état global UNIQUE par process : si plusieurs clients (onglets, polls) l'appellent en
# `interval=None`, la fenêtre de mesure tombe à ~0 -> 0% faux. Un seul sampler qui boucle en
# `interval=1.0` donne une valeur fiable toutes les ~1 s, que lisent tous les lecteurs.
_CPU_LOCK = threading.Lock()
_cpu_val: dict = {"v": None}
_cpu_started = {"on": False}


def _cpu_sampler() -> None:  # pragma: no cover - boucle daemon
    while True:
        try:
            v = psutil.cpu_percent(interval=1.0)
            with _CPU_LOCK:
                _cpu_val["v"] = v
        except Exception:  # noqa: BLE001
            time.sleep(1.0)


def _ensure_cpu_sampler() -> None:
    if psutil is None or _cpu_started["on"]:
        return
    _cpu_started["on"] = True
    threading.Thread(target=_cpu_sampler, daemon=True, name="loom-cpu-sampler").start()


_GPU_LOCK = threading.Lock()
_gpu_cache: dict = {"ts": 0.0, "data": None}
_GPU_TTL = 0.8


@lru_cache(maxsize=1)
def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _read_gpu_raw() -> dict | None:
    if not _has_nvidia_smi():
        return None
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    lines = (res.stdout or "").strip().splitlines()
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 6:
        return None
    return {
        "name": parts[0],
        "util": _num(parts[1]),  # %
        "mem_used": _num(parts[2]),  # Mo
        "mem_total": _num(parts[3]),  # Mo
        "temp": _num(parts[4]),  # °C
        "power": _num(parts[5]),  # W
    }


def _read_gpu() -> dict | None:
    """Lecture GPU avec cache court (thread-safe) pour ne pas spammer nvidia-smi."""
    now = time.monotonic()
    with _GPU_LOCK:
        if now - _gpu_cache["ts"] < _GPU_TTL:
            return _gpu_cache["data"]
    data = _read_gpu_raw()
    with _GPU_LOCK:
        _gpu_cache["ts"] = time.monotonic()
        _gpu_cache["data"] = data
    return data


def read_metrics() -> dict:
    """Instantané des métriques système. Champs à None si la source est indisponible."""
    cpu = None
    ram = None
    if psutil is not None:
        _ensure_cpu_sampler()
        with _CPU_LOCK:
            cpu = _cpu_val["v"]
        try:
            vm = psutil.virtual_memory()
            ram = {"used": int(vm.used), "total": int(vm.total), "percent": vm.percent}
        except Exception:  # noqa: BLE001 - best-effort
            ram = None
    gpu = _read_gpu()
    return {
        "cpu": cpu,
        "ram": ram,
        "gpu": gpu,
        "available": (cpu is not None) or (gpu is not None),
    }
