from loom.hardware import HardwareProfile
from loom.config import ModelConfig
from loom.swap import build_swap_config, dump_yaml


def _models():
    return [
        ModelConfig(
            repo="r",
            filename="gemma.gguf",
            n_layers=35,
            size_mb=5000,
            mmproj_filename="mmproj.gguf",
            id="gemma",
        ),
        ModelConfig(
            repo="r", filename="qwen.gguf", n_layers=48, size_mb=21000, id="qwen"
        ),
    ]


def test_build_swap_config_structure():
    prof = HardwareProfile(True, "GPU", 6000, 12)
    cfg = build_swap_config(
        _models(), prof, llama_bin="llama-server", models_dir="/m", context=8192
    )
    assert set(cfg["models"]) == {"gemma", "qwen"}
    gemma_cmd = cfg["models"]["gemma"]["cmd"]
    assert "/m/gemma.gguf" in gemma_cmd
    assert "--mmproj" in gemma_cmd and "/m/mmproj.gguf" in gemma_cmd
    assert "${PORT}" in gemma_cmd
    assert "-c 8192" in gemma_cmd
    # qwen n'a pas de mmproj
    assert "--mmproj" not in cfg["models"]["qwen"]["cmd"]


def test_dump_yaml_simple():
    y = dump_yaml({"models": {"a": {"cmd": "x -m /p/a.gguf --port ${PORT}"}}})
    assert "models:" in y
    assert '"a":' in y
    assert "cmd:" in y
    assert "${PORT}" in y
