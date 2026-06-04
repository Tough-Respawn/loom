# tests/test_serve.py
from pathlib import Path
from unittest.mock import patch

from loom.config import ChatConfig, ModelConfig, RuntimeConfig
from loom.hardware import HardwareProfile
from loom.serve import (
    ensure_all_models,
    main,
    resolve_mmproj_path,
    resolve_n_gpu_layers,
)


def _cfg(n_models: int) -> RuntimeConfig:
    models = [
        ModelConfig(repo="r", filename=f"m{i}.gguf", n_layers=4, size_mb=10, id=f"m{i}")
        for i in range(n_models)
    ]
    return RuntimeConfig(
        models=models,
        default_model="m0",
        context=8192,
        port=8080,
        server_bin="llama-server",
        swap_bin="llama-swap",
        override_n_gpu_layers=None,
        override_threads=None,
        chat=ChatConfig(system_prompt="x", history_path="h.json", web_port=8000),
    )


def test_main_single_model_launches_direct_not_swap():
    prof = HardwareProfile(False, None, 0, 8)
    with (
        patch("loom.serve.load_config", return_value=_cfg(1)),
        patch("loom.serve.detect_hardware", return_value=prof),
        patch("loom.serve.ensure_all_models"),
        patch("loom.serve.launch_direct", return_value=0) as direct,
        patch("loom.serve.launch_swap") as swap,
    ):
        assert main() == 0
    direct.assert_called_once()
    swap.assert_not_called()


def test_main_multi_model_launches_swap():
    prof = HardwareProfile(False, None, 0, 8)
    with (
        patch("loom.serve.load_config", return_value=_cfg(2)),
        patch("loom.serve.detect_hardware", return_value=prof),
        patch("loom.serve.ensure_all_models"),
        patch("loom.serve.launch_direct") as direct,
        patch("loom.serve.launch_swap", return_value=0) as swap,
    ):
        assert main() == 0
    swap.assert_called_once()
    direct.assert_not_called()


def test_resolve_uses_override_when_present():
    prof = HardwareProfile(True, "GPU", 6000, 12)
    assert (
        resolve_n_gpu_layers(prof, override=10, model_size_mb=4700, total_layers=28)
        == 10
    )


def test_resolve_cpu_profile_gives_zero():
    prof = HardwareProfile(False, None, 0, 16)
    assert (
        resolve_n_gpu_layers(prof, override=None, model_size_mb=4700, total_layers=28)
        == 0
    )


def test_resolve_gpu_auto_recommends_all_when_fits():
    prof = HardwareProfile(True, "GPU", 6000, 12)
    assert (
        resolve_n_gpu_layers(prof, override=None, model_size_mb=4700, total_layers=28)
        == 28
    )


def test_lower_headroom_offloads_more_layers():
    # Mono-flux : une marge VRAM plus basse doit offloader STRICTEMENT plus de couches
    # (le modèle ne tient pas entièrement -> offload proportionnel au budget).
    prof = HardwareProfile(True, "GPU", 6000, 12)
    conservateur = resolve_n_gpu_layers(
        prof, None, model_size_mb=5340, total_layers=35, kv_headroom_mb=1024
    )
    agressif = resolve_n_gpu_layers(
        prof, None, model_size_mb=5340, total_layers=35, kv_headroom_mb=512
    )
    assert agressif > conservateur


def test_resolve_mmproj_path_empty_returns_none():
    assert resolve_mmproj_path("", Path("/models")) is None


def test_resolve_mmproj_path_downloads_when_set(tmp_path):
    with patch("loom.serve.ensure_model") as ens:
        ens.return_value = tmp_path / "mmproj-F16.gguf"
        result = resolve_mmproj_path("mmproj-F16.gguf", tmp_path, repo="r")
    ens.assert_called_once_with("r", "mmproj-F16.gguf", tmp_path)
    assert result == str(tmp_path / "mmproj-F16.gguf")


def test_ensure_all_models_downloads_each(tmp_path):
    models = [
        ModelConfig(repo="r1", filename="a.gguf", n_layers=1, size_mb=1, id="a"),
        ModelConfig(
            repo="r2",
            filename="b.gguf",
            n_layers=1,
            size_mb=1,
            mmproj_filename="mm.gguf",
            id="b",
        ),
    ]
    with patch("loom.serve.ensure_model") as ens:
        ens.return_value = tmp_path / "x"
        ensure_all_models(models, tmp_path)
    # a.gguf, b.gguf, et le mmproj de b => 3 appels
    assert ens.call_count == 3
