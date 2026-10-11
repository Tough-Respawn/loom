# tests/test_gpu_backend.py
"""« Y a-t-il un GPU exploitable ? » — la vérité est le binaire (`--list-devices`).

Vécu 2026-10-10 sur la Radeon 860M : `has_gpu_backend` cherchait un ggml-vulkan.dll à
côté de llama-server.exe, mais le build maison intègre Vulkan en statique (aucune DLL).
loom-setup et /rebench faisaient `has_gpu_backend AND hw.has_gpu` : la machine passait
pour sans GPU — candidats llama-bench ngl 0, topologie « ram », placement « CPU seul »
validé à 32 768 tokens. Le profil matériel issu de `--list-devices` (backend renseigné)
fait foi ; l'heuristique DLL ne sert plus qu'au repli nvidia-smi, qui ne sait pas si le
BUILD offloade."""

from __future__ import annotations

from loom.runtime.hardware import HardwareProfile
from loom.setup.bench import gpu_backend_available

_VULKAN = HardwareProfile(
    True, "Radeon 860M", 46_350, 16, vram_total_mb=48_789, backend="Vulkan"
)
_NVSMI = HardwareProfile(True, "RTX 2060", 6_000, 16, vram_is_discrete=True)  # repli
_CPU = HardwareProfile(False, None, 0, 16)


def test_build_statique_sans_dll_le_binaire_fait_foi():
    assert gpu_backend_available(
        _VULKAN, "x/llama-server.exe", has_dll=lambda sb: False
    )


def test_repli_nvidia_smi_exige_l_heuristique_dll():
    # Le binaire n'a pas répondu : nvidia-smi voit un GPU, mais le build est-il GPU ?
    assert gpu_backend_available(_NVSMI, "x", has_dll=lambda sb: True) is True
    assert gpu_backend_available(_NVSMI, "x", has_dll=lambda sb: False) is False


def test_sans_gpu_jamais():
    assert gpu_backend_available(_CPU, "x", has_dll=lambda sb: True) is False
