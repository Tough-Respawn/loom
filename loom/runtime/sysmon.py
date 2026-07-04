# loom/runtime/sysmon.py
"""Monitoring système LIVE (CPU / RAM / GPU) pour l'indicateur affiché quand un modèle
LOCAL est sélectionné. CPU/RAM via psutil (dégradation propre s'il manque).

GPU, deux sources, dans l'ordre :
1. `nvidia-smi` (NVIDIA) — le plus riche (util, VRAM, température, puissance).
2. Repli GÉNÉRIQUE Windows (AMD / Intel / NVIDIA) via les COMPTEURS DE PERFORMANCE Windows
   (ceux du Gestionnaire des tâches, agnostiques du vendeur) : nom + util + VRAM used/total.
   Température/puissance indisponibles par cette voie -> None (le front affiche ce qu'il a).

Lecture GPU mise en cache court : ces sondes coûtent 50-500 ms, on ne les relance pas à
chaque poll rapproché."""

from __future__ import annotations

import base64
import json
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


# --- Repli GÉNÉRIQUE Windows (AMD / Intel) via les compteurs de performance -----------------
# Script PowerShell auto-suffisant -> JSON {name, mem_total_mb, mem_used_mb, util}. Nom +
# VRAM totale : WMI + registre (qwMemorySize, fiable au-delà de 4 Go, contrairement à
# AdapterRAM). VRAM utilisée + utilisation : compteurs `GPU Adapter Memory` / `GPU Engine`
# (agnostiques du vendeur, comme le Gestionnaire des tâches). Passé en -EncodedCommand
# (base64 UTF-16LE) pour éviter tout souci de quoting.
_WIN_GPU_PS = r"""
$ErrorActionPreference='SilentlyContinue'
$best = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Basic Display|Remote|Meta|Parsec' } | Select-Object -First 1
$name = if($best){$best.Name}else{''}
$total = 0
Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' | ForEach-Object {
  $v = (Get-ItemProperty $_.PSPath -Name 'HardwareInformation.qwMemorySize').'HardwareInformation.qwMemorySize'
  if($v -and $v -gt $total){ $total = [int64]$v }
}
$used = 0
try { $used = ((Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -EA Stop).CounterSamples | Measure-Object CookedValue -Sum).Sum } catch {}
$util = 0
try { $util = ((Get-Counter '\GPU Engine(*)\Utilization Percentage' -EA Stop).CounterSamples | Measure-Object CookedValue -Sum).Sum } catch {}
if($util -gt 100){ $util = 100 }
[pscustomobject]@{ name=$name; mem_total_mb=[math]::Round($total/1MB); mem_used_mb=[math]::Round($used/1MB); util=[math]::Round($util,1) } | ConvertTo-Json -Compress
"""

_win_gpu: dict = {"ts": 0.0, "data": None}
_win_last_req = {"t": 0.0}
_win_started = {"on": False}


def _read_gpu_windows_raw() -> dict | None:
    from loom.runtime.platform_info import detect

    if not detect().is_windows:
        return None
    enc = base64.b64encode(_WIN_GPU_PS.encode("utf-16-le")).decode()
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
            capture_output=True,
            text=True,
            timeout=12,
        )
        out = (res.stdout or "").strip()
        d = json.loads(out) if out else None
    except (subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError):
        return None
    if not d:
        return None
    mt = d.get("mem_total_mb")
    if not mt:  # sans VRAM totale fiable, l'indicateur perd son sens -> pas de GPU
        return None
    util = d.get("util")
    mu = d.get("mem_used_mb")
    return {
        "name": d.get("name") or "GPU",
        "util": float(util) if util is not None else None,
        "mem_used": float(mu) if mu is not None else None,
        "mem_total": float(mt),
        "temp": None,  # indisponible via les compteurs Windows
        "power": None,
    }


def _win_gpu_sampler() -> None:  # pragma: no cover - boucle daemon
    # Échantillonne toutes les ~2,5 s, mais SEULEMENT si le widget est actif (une lecture a eu
    # lieu depuis < 8 s) : pas de PowerShell perpétuel quand aucun modèle local n'est affiché.
    while True:
        if time.monotonic() - _win_last_req["t"] < 8.0:
            data = _read_gpu_windows_raw()
            with _GPU_LOCK:
                _win_gpu["data"] = data
                _win_gpu["ts"] = time.monotonic()
        time.sleep(2.5)


def _read_gpu_windows() -> dict | None:
    """GPU générique Windows via un sampler gaté (démarré à la 1re demande). La 1re lecture
    est synchrone pour ne pas afficher un GPU vide au premier coup."""
    _win_last_req["t"] = time.monotonic()
    if not _win_started["on"]:
        _win_started["on"] = True
        data = _read_gpu_windows_raw()
        with _GPU_LOCK:
            _win_gpu["data"] = data
            _win_gpu["ts"] = time.monotonic()
        threading.Thread(
            target=_win_gpu_sampler, daemon=True, name="loom-gpu-win"
        ).start()
        return data
    with _GPU_LOCK:
        return _win_gpu["data"]


def _read_gpu() -> dict | None:
    """Lecture GPU avec cache court (thread-safe). nvidia-smi d'abord (riche), sinon repli
    générique Windows (AMD/Intel). Le cache évite de spammer les sondes."""
    now = time.monotonic()
    with _GPU_LOCK:
        if now - _gpu_cache["ts"] < _GPU_TTL:
            return _gpu_cache["data"]
    data = _read_gpu_raw()
    if data is None:
        data = _read_gpu_windows()
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
