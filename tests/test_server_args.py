# tests/test_server_args.py
from loom.server_args import build_server_args


def test_build_server_args_gpu():
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=8192,
        n_gpu_layers=28,
        threads=12,
    )
    assert args[0] == "llama-server"
    assert "-m" in args and "/m/model.gguf" in args
    assert "--port" in args and "8080" in args
    assert "-c" in args and "8192" in args
    assert "-ngl" in args and "28" in args
    assert "-t" in args and "12" in args
    assert "--host" in args and "127.0.0.1" in args


def test_build_server_args_cpu_only():
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=9000,
        context=4096,
        n_gpu_layers=0,
        threads=16,
    )
    i = args.index("-ngl")
    assert args[i + 1] == "0"


def test_build_server_args_with_mmproj():
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=4096,
        n_gpu_layers=35,
        threads=12,
        mmproj_path="/m/mmproj.gguf",
    )
    assert "--mmproj" in args
    i = args.index("--mmproj")
    assert args[i + 1] == "/m/mmproj.gguf"
    assert "--no-mmproj-offload" in args


def test_build_server_args_without_mmproj():
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=4096,
        n_gpu_layers=35,
        threads=12,
    )
    assert "--mmproj" not in args


def test_build_server_args_gpu_tuning_adds_flash_attn_and_kv_quant():
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=8192,
        n_gpu_layers=99,
        threads=6,
        gpu_tuning=True,
    )
    # Flash-Attention + cache KV quantifié q8_0 + batch (réglages GPU prouvés)
    assert "-fa" in args
    assert "-ctk" in args and "-ctv" in args
    i = args.index("-ctk")
    assert args[i + 1] == "q8_0"
    assert "-ub" in args and "--prio" in args


def test_build_server_args_no_tuning_by_default():
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=8192,
        n_gpu_layers=0,
        threads=8,
    )
    assert "-fa" not in args and "-ctk" not in args  # CPU-only : pas de tuning GPU


def test_build_server_args_pins_parallel_slots():
    # --parallel épinglé = source de vérité du parallélisme (le harness en dérive son budget)
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=24576,
        n_gpu_layers=99,
        threads=6,
        n_parallel=4,
    )
    assert "--parallel" in args
    assert args[args.index("--parallel") + 1] == "4"


def test_build_server_args_parallel_floored_to_one():
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=8192,
        n_gpu_layers=0,
        threads=8,
        n_parallel=0,
    )
    assert args[args.index("--parallel") + 1] == "1"  # jamais 0 slot


def test_build_server_args_enables_jinja_for_tool_calls():
    # --jinja active le chat-template tool-aware (tool_calls structurés)
    args = build_server_args(
        server_bin="llama-server",
        model_path="/m/model.gguf",
        port=8080,
        context=4096,
        n_gpu_layers=35,
        threads=12,
    )
    assert "--jinja" in args
