"""Détection hardware et recommandation de réglages d'offload GPU.

Détection GPU AGNOSTIQUE : la source de vérité est le binaire llama.cpp installé
(`llama-server --list-devices`) — il liste exactement les devices que CE binaire
sait exploiter (Vulkan AMD/Intel/NVIDIA, CUDA, Metal…). nvidia-smi ne sert que
de repli d'AVANT-binaire (étape 1 du setup : choisir le build CUDA vs Vulkan)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def ram_total_mb() -> int:
    """RAM physique TOTALE (Mo), 0 si indéterminable. Sert aux budgets de
    CAPACITÉ (« ce modèle tiendra-t-il ? ») : la mémoire d'un modèle DÉJÀ chargé
    (déchargé par llama-swap avant le nouveau) ou de Loom/navigateur ne doit PAS
    rétrécir le budget — sinon /add-model masque des quants qui tiennent (vécu
    2026-07-23 : ornith Q8 34 Go tourne, mais un Q4 19 Go marqué « ne tiendra
    pas » car mesuré sur la RAM DISPO pendant qu'ornith occupait la mémoire).
    Même leçon P3 que topology.py (budgets sur le TOTAL, jamais la dispo)."""
    if sys.platform == "win32":
        import ctypes

        class _Mem(ctypes.Structure):
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

        stat = _Mem()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys // (1024 * 1024))
        return 0
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil

        return int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:  # noqa: BLE001 - jamais bloquant, 0 = indéterminé
        return 0


def ram_available_mb() -> int:
    """RAM physique DISPONIBLE (Mo), 0 si indéterminable.

    Windows : GlobalMemoryStatusEx (ullAvailPhys). Linux : MemAvailable de
    /proc/meminfo. macOS (pas de /proc) : psutil. Sert aux arbitrages de
    co-résidence (ex. garder le cache RAM du moteur image à côté du LLM
    quand la machine est assez large)."""
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
    try:
        import psutil

        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:  # noqa: BLE001 - jamais bloquant, 0 = indéterminé
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
    # Renseignés par la détection agnostique `--list-devices`.
    vram_total_mb: int = 0
    backend: str | None = None
    # La VRAM partagée d'un iGPU ne doit pas être ajoutée une seconde fois à la RAM.
    vram_is_discrete: bool = False

    @property
    def budget_vram_mb(self) -> int:
        """Contribution VRAM à un budget mémoire VRAM+RAM : 0 si non discrète."""
        return self.vram_free_mb if self.vram_is_discrete else 0


_DEVICE_RE = re.compile(
    r"^\s*([A-Za-z]+)(\d+):\s+(.+?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB free\)\s*$"
)


def parse_list_devices(output: str) -> list[dict]:
    """Devices GPU de `llama-server --list-devices` : [{backend, name, total_mb,
    free_mb}], liste vide si aucun (build CPU-only) ou sortie inattendue."""
    devices = []
    for line in output.splitlines():
        m = _DEVICE_RE.match(line)
        if m:
            devices.append(
                {
                    "backend": m.group(1),
                    "name": m.group(3),
                    "total_mb": int(m.group(4)),
                    "free_mb": int(m.group(5)),
                }
            )
    return devices


def _run_list_devices(server_bin) -> str | None:
    """Sortie de `<server_bin> --list-devices`, None si le binaire ne répond pas.
    stdout+stderr concaténés : llama.cpp logge selon les builds sur l'un ou l'autre.
    Path() normalise les séparateurs : CreateProcess (Windows) refuse un chemin
    relatif à slashs avant."""
    try:
        res = subprocess.run(
            [str(Path(server_bin)), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return (res.stdout or "") + "\n" + (res.stderr or "")


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


def detect_hardware(server_bin=None) -> HardwareProfile:
    """Détecte le profil GPU/CPU. AGNOSTIQUE quand `server_bin` est fourni : le
    binaire installé fait foi (`--list-devices`) — s'il liste un device, on le
    prend ; s'il n'en liste aucun (build CPU-only), on n'offloadera rien même si
    un GPU physique existe. Repli nvidia-smi si le binaire manque ou ne répond
    pas (seul cas où il faut savoir AVANT le binaire : choisir le build CUDA)."""
    threads = os.cpu_count() or 4
    if server_bin is not None:
        out = _run_list_devices(server_bin)
        if out is not None:
            devices = parse_list_devices(out)
            if not devices:
                return HardwareProfile(False, None, 0, threads)
            d = devices[0]
            return HardwareProfile(
                True,
                d["name"],
                d["free_mb"],
                threads,
                vram_total_mb=d["total_mb"],
                backend=d["backend"],
                vram_is_discrete=(
                    d["backend"].lower() == "cuda"
                    or shutil.which("nvidia-smi") is not None
                ),
            )
    raw = _run_nvidia_smi()
    parsed = parse_nvidia_smi(raw) if raw else None
    if parsed is None:
        return HardwareProfile(False, None, 0, threads)
    name, free_mb = parsed
    return HardwareProfile(True, name, free_mb, threads, vram_is_discrete=True)
