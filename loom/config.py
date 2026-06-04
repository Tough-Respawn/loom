# loom/config.py
"""Chargement et fusion de la configuration Loom (TOML)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from loom.permissions import PermissionConfig, parse_permissions
from loom.prompts import CHAT_SYSTEM
from loom.tools.web import WebSearchConfig

# Le prompt système du chat vit dans loom/prompts/chat.system.md (source de vérité).
DEFAULT_SYSTEM_PROMPT = CHAT_SYSTEM


DEFAULT_READ_EXTENSIONS = [
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".html",
    ".js",
    ".css",
    ".yaml",
    ".yml",
    ".sh",
    ".ts",
    ".tsx",
    ".jsx",
    ".cfg",
    ".ini",
]


@dataclass
class ChatConfig:
    system_prompt: str
    history_path: str
    web_port: int
    skills_dir: str = "loom/skills"
    max_tokens: int = 2048
    request_timeout: int = 120
    max_retries: int = 6
    context_token_budget: int = 3000
    keep_recent_messages: int = 6
    # Outils (boucle tool-use). enabled vide => aucun outil exposé.
    tools_enabled: list[str] = field(default_factory=list)
    workspace_dir: str = "."
    read_file_max_bytes: int = 200_000
    read_file_extensions: list[str] = field(
        default_factory=lambda: DEFAULT_READ_EXTENSIONS
    )
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)


@dataclass
class ModelConfig:
    repo: str
    filename: str
    n_layers: int
    size_mb: int
    mmproj_filename: str = ""
    id: str = ""
    n_gpu_layers: int | None = None


@dataclass
class RuntimeConfig:
    models: list[ModelConfig]
    default_model: str
    context: int
    port: int
    server_bin: str
    swap_bin: str
    override_n_gpu_layers: int | None
    override_threads: int | None
    chat: ChatConfig
    # Slots de llama-server (--parallel). Loom est mono-flux -> 1 (cf. loom.config.toml).
    n_parallel: int = 1
    permissions: PermissionConfig = field(default_factory=PermissionConfig)

    def model_by_id(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        return self.models[0]

    @property
    def model(self) -> ModelConfig:
        return self.model_by_id(self.default_model)


def _parse_model(d: dict, default_id: str = "") -> ModelConfig:
    """Construit un ModelConfig depuis une table TOML [[models]]."""
    return ModelConfig(
        repo=d["repo"],
        filename=d["filename"],
        n_layers=int(d["n_layers"]),
        size_mb=int(d["size_mb"]),
        mmproj_filename=d.get("mmproj_filename", ""),
        id=d.get("id", "") or default_id or d["filename"],
        n_gpu_layers=d.get("n_gpu_layers"),
    )


def _parse_web_search(d: dict) -> WebSearchConfig:
    """Construit un WebSearchConfig depuis la table TOML [web_search]."""
    base = WebSearchConfig()
    return WebSearchConfig(
        enabled=bool(d.get("enabled", base.enabled)),
        backend=d.get("backend", base.backend),
        searxng_url=d.get("searxng_url", base.searxng_url),
        tavily_api_key=d.get("tavily_api_key", base.tavily_api_key),
        max_results=int(d.get("max_results", base.max_results)),
        fetch_pages=bool(d.get("fetch_pages", base.fetch_pages)),
        http_timeout=int(d.get("http_timeout", base.http_timeout)),
        max_chars_per_page=int(d.get("max_chars_per_page", base.max_chars_per_page)),
    )


def _deep_merge(base: dict, over: dict) -> dict:
    """Fusionne récursivement `over` dans une copie de `base`."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(
    path: str | Path, local_path: str | Path | None = None
) -> RuntimeConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if local_path is not None and Path(local_path).exists():
        local = tomllib.loads(Path(local_path).read_text(encoding="utf-8"))
        data = _deep_merge(data, local)

    s = data["server"]
    o = data.get("override", {})
    ch = data.get("chat", {})
    tl = data.get("tools", {})
    ws = data.get("web_search", {})
    chat = ChatConfig(
        system_prompt=ch.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        history_path=ch.get("history_path", "loom/data/conversation.json"),
        web_port=int(ch.get("web_port", 8000)),
        skills_dir=ch.get("skills_dir", "loom/skills"),
        max_tokens=int(ch.get("max_tokens", 2048)),
        request_timeout=int(ch.get("request_timeout", 120)),
        max_retries=int(ch.get("max_retries", 6)),
        context_token_budget=int(ch.get("context_token_budget", 3000)),
        keep_recent_messages=int(ch.get("keep_recent_messages", 6)),
        tools_enabled=list(tl.get("enabled", [])),
        workspace_dir=tl.get("workspace_dir", "."),
        read_file_max_bytes=int(tl.get("read_file_max_bytes", 200_000)),
        read_file_extensions=list(
            tl.get("read_file_extensions", DEFAULT_READ_EXTENSIONS)
        ),
        web_search=_parse_web_search(ws),
    )
    models = [_parse_model(rm) for rm in data["models"]]
    default_model = ch.get("default_model") or models[0].id
    return RuntimeConfig(
        models=models,
        default_model=default_model,
        context=int(s["context"]),
        port=int(s["port"]),
        server_bin=s["bin"],
        swap_bin=s.get("swap_bin", "llama-swap"),
        n_parallel=int(s.get("n_parallel", 4)),
        override_n_gpu_layers=o.get("n_gpu_layers"),
        override_threads=o.get("threads"),
        chat=chat,
        permissions=parse_permissions(data),
    )
