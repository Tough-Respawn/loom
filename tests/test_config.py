# tests/test_config.py
from pathlib import Path
from loom.config import load_config, RuntimeConfig


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


BASE = """
[[models]]
id = "default"
repo = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
n_layers = 28
size_mb = 4700

[server]
context = 8192
port = 8080
bin = "llama-server"

[override]
"""


MULTI = """
[[models]]
id = "gemma"
repo = "r/gemma"
filename = "gemma.gguf"
mmproj_filename = "mmproj.gguf"
n_layers = 35
size_mb = 5340

[[models]]
id = "qwen"
repo = "r/qwen"
filename = "qwen.gguf"
n_layers = 48
size_mb = 21000

[server]
context = 8192
port = 8080
bin = "llama-server"

[chat]
default_model = "qwen"
"""


def test_models_registry_and_default(tmp_path):
    cfg = load_config(_write(tmp_path, "loom.config.toml", MULTI))
    assert [m.id for m in cfg.models] == ["gemma", "qwen"]
    assert cfg.default_model == "qwen"
    assert cfg.model_by_id("gemma").filename == "gemma.gguf"
    assert cfg.model_by_id("gemma").mmproj_filename == "mmproj.gguf"
    assert cfg.model_by_id("qwen").mmproj_filename == ""
    # propriété de compat `model` = modèle par défaut
    assert cfg.model.id == "qwen"


def test_single_model_default_id(tmp_path):
    cfg = load_config(_write(tmp_path, "loom.config.toml", BASE))
    assert len(cfg.models) == 1
    assert cfg.models[0].id == "default"
    assert cfg.default_model == "default"
    assert cfg.model.repo.endswith("Qwen2.5-Coder-7B-Instruct-GGUF")


def test_load_base_config(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.model.repo.endswith("Qwen2.5-Coder-7B-Instruct-GGUF")
    assert cfg.model.n_layers == 28
    assert cfg.model.size_mb == 4700
    assert cfg.context == 8192
    assert cfg.port == 8080
    assert cfg.server_bin == "llama-server"
    assert cfg.override_n_gpu_layers is None
    assert cfg.override_threads is None
    assert cfg.n_parallel == 4  # défaut quand absent du [server]


def test_n_parallel_read_from_server_section(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            "loom.config.toml",
            BASE.replace("port = 8080", "port = 8080\nn_parallel = 2"),
        )
    )
    assert cfg.n_parallel == 2


def test_local_override_merges(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    local = _write(
        tmp_path,
        "loom.config.local.toml",
        '[server]\nbin = "C:/tools/llama/llama-server.exe"\n'
        "[override]\nn_gpu_layers = 12\n",
    )
    cfg = load_config(cfg_path, local_path=local)
    assert cfg.server_bin == "C:/tools/llama/llama-server.exe"
    assert cfg.override_n_gpu_layers == 12


def test_chat_config_defaults_when_absent(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert cfg.chat.web_port == 8000
    assert cfg.chat.history_path.endswith("conversation.json")
    low = cfg.chat.system_prompt.lower()
    assert "agent" in low and "outil" in low  # prompt agentic par défaut


def test_chat_config_read_from_file(tmp_path):
    text = (
        BASE
        + '\n[chat]\nsystem_prompt = "Salut"\nhistory_path = "x/y.json"\nweb_port = 9001\n'
    )
    cfg_path = _write(tmp_path, "loom.config.toml", text)
    cfg = load_config(cfg_path)
    assert cfg.chat.system_prompt == "Salut"
    assert cfg.chat.history_path == "x/y.json"
    assert cfg.chat.web_port == 9001


def test_chat_skills_dir_default(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert cfg.chat.skills_dir == "loom/skills"


def test_model_has_mmproj_filename(tmp_path):
    text = BASE.replace(
        'filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"',
        'filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"\nmmproj_filename = "mmproj-F16.gguf"',
    )
    cfg_path = _write(tmp_path, "loom.config.toml", text)
    cfg = load_config(cfg_path)
    assert cfg.model.mmproj_filename == "mmproj-F16.gguf"


def test_model_mmproj_defaults_empty(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert cfg.model.mmproj_filename == ""


def test_chat_robustness_defaults(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert cfg.chat.max_tokens == 2048
    assert cfg.chat.request_timeout == 120
    assert cfg.chat.max_retries == 6
    assert cfg.chat.context_token_budget == 3000
    assert cfg.chat.keep_recent_messages == 6


def test_web_search_defaults_when_absent(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    ws = cfg.chat.web_search
    assert ws.enabled is True
    assert ws.backend == "auto"
    assert ws.searxng_url == ""
    assert ws.tavily_api_key == ""
    assert ws.max_results == 5
    assert ws.fetch_pages is True
    assert ws.http_timeout == 6
    assert ws.max_chars_per_page == 4000


def test_web_search_read_from_file(tmp_path):
    text = (
        BASE
        + '\n[web_search]\nenabled = false\nbackend = "searxng"\n'
        + 'searxng_url = "http://searx.local"\ntavily_api_key = "abc"\n'
        + "max_results = 8\nfetch_pages = false\nhttp_timeout = 10\n"
        + "max_chars_per_page = 2000\n"
    )
    cfg_path = _write(tmp_path, "loom.config.toml", text)
    cfg = load_config(cfg_path)
    ws = cfg.chat.web_search
    assert ws.enabled is False
    assert ws.backend == "searxng"
    assert ws.searxng_url == "http://searx.local"
    assert ws.tavily_api_key == "abc"
    assert ws.max_results == 8
    assert ws.fetch_pages is False
    assert ws.http_timeout == 10
    assert ws.max_chars_per_page == 2000


def test_permissions_defaults_when_absent(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert cfg.permissions.mode == "ask"
    assert cfg.permissions.workspace_root == "."


def test_permissions_read_from_file(tmp_path):
    text = (
        BASE
        + '\n[permissions]\nmode = "allowlist"\nworkspace_root = "w"\n'
        + 'allow_commands = ["git status"]\n'
    )
    cfg_path = _write(tmp_path, "loom.config.toml", text)
    cfg = load_config(cfg_path)
    assert cfg.permissions.mode == "allowlist"
    assert cfg.permissions.workspace_root == "w"
    assert cfg.permissions.allow_commands == ["git status"]
