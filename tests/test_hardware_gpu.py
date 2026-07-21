# Détection GPU AGNOSTIQUE : le binaire llama.cpp installé (`--list-devices`) est
# la source de vérité (Vulkan AMD/Intel/NVIDIA, CUDA, Metal…) ; nvidia-smi n'est
# qu'un repli d'avant-binaire. SANS réseau ni vrai binaire (sorties rejouées).
from loom.runtime import hardware
from loom.runtime.hardware import (
    HardwareProfile,
    detect_hardware,
    parse_list_devices,
)
from loom.setup.bench import ngl_candidates

# Sortie réelle observée (Radeon 860M, build Vulkan b10075).
_VULKAN_OUT = """Available devices:
  Vulkan0: AMD Radeon(TM) 860M Graphics (36682 MiB, 34848 MiB free)
"""
_CUDA_OUT = """Available devices:
  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 24000 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (12288 MiB, 12000 MiB free)
"""


def test_parse_list_devices_vulkan_reel():
    devs = parse_list_devices(_VULKAN_OUT)
    assert devs == [
        {
            "backend": "Vulkan",
            "name": "AMD Radeon(TM) 860M Graphics",
            "total_mb": 36682,
            "free_mb": 34848,
        }
    ]


def test_parse_list_devices_multi_cuda():
    devs = parse_list_devices(_CUDA_OUT)
    assert [d["name"] for d in devs] == [
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3060",
    ]
    assert devs[0]["backend"] == "CUDA" and devs[0]["free_mb"] == 24000


def test_parse_list_devices_vide_ou_malforme():
    assert parse_list_devices("") == []
    assert parse_list_devices("Available devices:\n") == []
    assert parse_list_devices("garbage\nAvailable devices: soon\n") == []


def test_detect_hardware_via_binaire_est_agnostique(monkeypatch):
    monkeypatch.setattr(hardware, "_run_list_devices", lambda b: _VULKAN_OUT)
    hw = detect_hardware(server_bin="fake/llama-server")
    assert hw.has_gpu is True
    assert hw.gpu_name == "AMD Radeon(TM) 860M Graphics"
    assert hw.vram_free_mb == 34848
    assert hw.vram_total_mb == 36682
    assert hw.backend == "Vulkan"


def test_detect_hardware_binaire_sans_device_fait_foi(monkeypatch):
    # Build CPU-only sur machine à GPU : le binaire ne PEUT pas offloader ->
    # CPU-only, même si nvidia-smi existe (on ne doit pas le consulter).
    monkeypatch.setattr(hardware, "_run_list_devices", lambda b: "Available devices:\n")
    monkeypatch.setattr(hardware, "_run_nvidia_smi", lambda: "RTX 4090, 24000")
    hw = detect_hardware(server_bin="fake/llama-server")
    assert hw.has_gpu is False and hw.vram_free_mb == 0


def test_detect_hardware_sans_binaire_replie_sur_nvidia(monkeypatch):
    monkeypatch.setattr(hardware, "_run_nvidia_smi", lambda: "RTX 2060, 6000")
    hw = detect_hardware()
    assert hw.has_gpu is True and hw.gpu_name == "RTX 2060"
    assert hw.vram_is_discrete is True


def test_budget_vram_discrete_vs_unifiee():
    # iGPU Vulkan : sa « VRAM » est la RAM partagée -> ne JAMAIS l'additionner
    # à la RAM dans un budget (double-comptage).
    unified = HardwareProfile(
        True, "Radeon 860M", 34848, 16, vram_total_mb=36682, backend="Vulkan"
    )
    assert unified.budget_vram_mb == 0
    discrete = HardwareProfile(True, "RTX 2060", 6000, 16, vram_is_discrete=True)
    assert discrete.budget_vram_mb == 6000


def test_ngl_candidates():
    # Sans backend GPU : CPU seul.
    assert ngl_candidates(False, 34848, 1000, 36) == [0]
    # VRAM couvre tout le modèle : offload total suffit, pas d'intermédiaire.
    assert ngl_candidates(True, 34848, 1000, 36) == [0, 99]
    # VRAM partielle : la recommandation proportionnelle est MESURÉE aussi.
    cands = ngl_candidates(True, 6000, 16000, 48)
    assert cands[0] == 0 and cands[-1] == 99 and len(cands) == 3
    assert 0 < cands[1] < 48
    # n_layers inconnu (GGUF illisible) : on garde 0/99.
    assert ngl_candidates(True, 6000, 16000, None) == [0, 99]
