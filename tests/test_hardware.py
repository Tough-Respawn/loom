# tests/test_hardware.py
from unittest.mock import patch

from loom.hardware import (
    HardwareProfile,
    detect_hardware,
    parse_nvidia_smi,
    recommend_gpu_layers,
)


def test_cpu_only_when_no_vram_budget():
    # VRAM dispo <= marge KV -> tout sur CPU
    assert (
        recommend_gpu_layers(
            vram_free_mb=512, model_size_mb=4700, total_layers=28, kv_headroom_mb=1024
        )
        == 0
    )


def test_all_layers_when_model_fits():
    # 6 Go libres, modèle 4.7 Go + marge -> toutes les couches sur GPU
    assert (
        recommend_gpu_layers(
            vram_free_mb=6000, model_size_mb=4700, total_layers=28, kv_headroom_mb=1024
        )
        == 28
    )


def test_partial_offload_when_tight():
    # budget = 4000-1000 = 3000 ; 3000/6000 * 32 = 16 couches
    assert (
        recommend_gpu_layers(
            vram_free_mb=4000, model_size_mb=6000, total_layers=32, kv_headroom_mb=1024
        )
        == 16
    )


def test_parse_nvidia_smi_valid():
    # format CSV sans header : "name, memory.free [MiB]"
    out = "NVIDIA GeForce RTX 2060, 5800\n"
    name, free_mb = parse_nvidia_smi(out)
    assert name == "NVIDIA GeForce RTX 2060"
    assert free_mb == 5800


def test_parse_nvidia_smi_empty_returns_none():
    assert parse_nvidia_smi("") is None


def test_detect_hardware_with_gpu():
    fake = "NVIDIA GeForce RTX 2060, 5800\n"
    with (
        patch("loom.hardware._run_nvidia_smi", return_value=fake),
        patch("loom.hardware.os.cpu_count", return_value=12),
    ):
        prof = detect_hardware()
    assert isinstance(prof, HardwareProfile)
    assert prof.has_gpu is True
    assert prof.gpu_name == "NVIDIA GeForce RTX 2060"
    assert prof.vram_free_mb == 5800
    assert prof.cpu_threads == 12


def test_detect_hardware_cpu_only():
    with (
        patch("loom.hardware._run_nvidia_smi", return_value=None),
        patch("loom.hardware.os.cpu_count", return_value=16),
    ):
        prof = detect_hardware()
    assert prof.has_gpu is False
    assert prof.gpu_name is None
    assert prof.vram_free_mb == 0
    assert prof.cpu_threads == 16
