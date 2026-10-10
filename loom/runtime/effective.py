"""Flags MACHINE de l'exécutant, résolus UNE fois pour tous les lanceurs.

Trois chemins lancent llama-server par le même `build_server_args` : serve.py
(mono-modèle), swap.py (llama-swap) et la sonde de bench (topology.ServerProbe).
Chacun dérivait seul threads / profil GPU / mémoire unifiée depuis le profil
matériel, et ils divergeaient (constaté le 2026-10-09 sur la Radeon 860M) :
- llama-swap ignorait [override] threads que le bench venait d'écrire ;
- la sonde tirait son profil GPU d'une topologie fondée sur nvidia-smi (absent
  sur AMD -> « ram ») et mesurait SANS -fa on / q8_0 / --prio 2, ni mémoire
  unifiée, alors que l'exécutant (profil `--list-devices`) les pose.
Le conseilleur doit simuler l'exécutant : une seule dérivation, partagée.
"""

from __future__ import annotations

from dataclasses import dataclass

from loom.runtime.hardware import HardwareProfile


@dataclass(frozen=True)
class LaunchFlags:
    threads: int
    gpu_tuning: bool  # profil GPU benchmarké : -fa on, KV q8_0, gros batch, --prio 2
    unified_memory: (
        bool  # iGPU : la VRAM est la RAM -> pas de no-mmap (Vulkan le refuse)
    )


def launch_flags(profile: HardwareProfile, override_threads: int | None) -> LaunchFlags:
    """Flags machine depuis le profil matériel (`--list-devices`) et l'override.

    threads : [override] threads (mesuré par le bench) s'il est renseigné ; sinon
    ~cœurs physiques (logiques / 2) avec un GPU — l'hyperthreading ralentit la
    passe CPU quand le GPU traite le reste du modèle — ; sinon tous les threads."""
    if override_threads:
        threads = int(override_threads)
    elif profile.has_gpu:
        threads = max(1, profile.cpu_threads // 2)
    else:
        threads = profile.cpu_threads
    return LaunchFlags(
        threads=threads,
        gpu_tuning=profile.has_gpu,
        unified_memory=not profile.vram_is_discrete,
    )
