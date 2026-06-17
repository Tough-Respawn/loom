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


@dataclass
class ChatConfig:
    system_prompt: str
    history_path: str
    web_port: int
    skills_dir: str = "loom/skills"
    plugins_root: str = "loom/plugins"
    max_tokens: int = 2048
    request_timeout: int = 120
    max_retries: int = 6
    context_token_budget: int = 3000
    keep_recent_messages: int = 6
    # Outils (boucle tool-use). enabled vide => aucun outil exposé.
    tools_enabled: list[str] = field(default_factory=list)
    workspace_dir: str = "."
    # Cap par appel read_file (caractères). Volontairement BAS devant le contexte (24576
    # tokens) : un seul gros fichier ne doit pas le faire déborder -> on lit par tranches
    # (start_line). ~40000 car ≈ 10k tokens.
    read_file_max_bytes: int = 40_000
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    # Keep-warm : sur Windows, un llama-server inactif se fait rogner son working-set par
    # l'OS (pages des experts MoE évincées) -> 1re requête après une pause = lente (cold
    # start). Un ping minimal (1 token) toutes les keepwarm_interval secondes, À L'IDLE
    # seulement, garde le modèle chargé « chaud » sans verrouiller la RAM en dur. Pinge le
    # modèle de la session active (jamais un autre -> pas de swap llama-swap intempestif).
    keepwarm_enabled: bool = True
    keepwarm_interval: int = 150
    # Mémoire/identité : budget du bloc identité always-on injecté au prompt (SOUL/USER/MEMORY).
    identity_max_tokens: int = 400
    # Apprentissage (boucle fermée) : skills auto-appris + étape reflect post-tour.
    learned_skills_dir: str = "loom/skills_learned"
    reflect_enabled: bool = True
    reflect_min_actions: int = 1


@dataclass
class MemoryConfig:
    """Config de la mémoire (design §9). v1 : provider 'local' uniquement (offline)."""

    provider: str = "local"
    db_path: str = "loom/data/memory.db"
    soul_path: str = "loom/data/SOUL.md"
    user_path: str = "loom/data/USER.md"
    memory_md_path: str = "loom/data/MEMORY.md"
    recall_summarize: bool = True
    recall_summarize_threshold: int = 5


@dataclass
class ModelConfig:
    repo: str
    filename: str
    n_layers: int
    size_mb: int
    mmproj_filename: str = ""
    id: str = ""
    n_gpu_layers: int | None = None
    # MoE offload : si True, garde attention + FFN dense sur GPU et bascule les experts
    # ROUTÉS en RAM (--cpu-moe + -ngl 999). Rend un MoE 26-35B jouable sur 6 Go.
    cpu_moe: bool = False
    # Offload MoE PARTIEL : N = nb de couches dont les experts vont en RAM ; les (total-N)
    # restantes gardent leurs experts sur GPU (plus rapide, remplit la VRAM). Si défini,
    # émet `--n-cpu-moe N` au lieu de `--cpu-moe` (qui, lui, offloade TOUTES les couches).
    n_cpu_moe: int | None = None
    # Contexte propre au modèle (override le global). Un gros MoE = KV plus lourd -> on
    # raccourcit (ex. 16384) là où les petits tiennent 24576.
    context: int | None = None
    # Dossier du modèle (loom/models/<id>/) : porte le GGUF, le mmproj et profile.md.
    # Rempli par la découverte ; les chemins GGUF se résolvent contre lui.
    dir: str = ""


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
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    # Slots de llama-server (--parallel). Loom est mono-flux -> 1 (cf. loom.config.toml).
    n_parallel: int = 1
    # Marge VRAM (Mo) réservée hors couches offloadées : couvre le cache KV + les buffers
    # de calcul. Plus elle est BASSE, plus on offloade de couches sur GPU (perf), mais trop
    # bas -> débordement en mémoire partagée (Windows) qui écroule tout. À régler par machine.
    gpu_kv_headroom_mb: int = 1024
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
        cpu_moe=bool(d.get("cpu_moe", False)),
        n_cpu_moe=d.get("n_cpu_moe"),
        context=d.get("context"),
        dir=d.get("dir", ""),
    )


def _discover_models(models_root: Path) -> list[ModelConfig]:
    """Découvre un modèle par sous-dossier `loom/models/<id>/model.toml` (l'id = nom du
    dossier). Renvoie la liste triée par id. Chaque ModelConfig porte son `dir`. Vide si
    le dossier racine n'existe pas ou ne contient aucun model.toml."""
    if not models_root.is_dir():
        return []
    out: list[ModelConfig] = []
    for folder in sorted(p for p in models_root.iterdir() if p.is_dir()):
        # Dossiers préfixés '_' = gabarits/scaffolds (ex. _TEMPLATE), pas de vrais modèles :
        # leur model.toml porte un repo placeholder (org/mon-modele-GGUF) -> 401 au fetch.
        if folder.name.startswith("_"):
            continue
        toml_path = folder / "model.toml"
        if not toml_path.exists():
            continue
        d = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        m = _parse_model(d, default_id=folder.name)
        m.dir = str(folder)
        out.append(m)
    return out


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
        plugins_root=ch.get("plugins_root", "loom/plugins"),
        max_tokens=int(ch.get("max_tokens", 2048)),
        request_timeout=int(ch.get("request_timeout", 120)),
        max_retries=int(ch.get("max_retries", 6)),
        context_token_budget=int(ch.get("context_token_budget", 3000)),
        keep_recent_messages=int(ch.get("keep_recent_messages", 6)),
        tools_enabled=list(tl.get("enabled", [])),
        workspace_dir=tl.get("workspace_dir", "."),
        read_file_max_bytes=int(tl.get("read_file_max_bytes", 40_000)),
        web_search=_parse_web_search(ws),
        keepwarm_enabled=bool(ch.get("keepwarm_enabled", True)),
        keepwarm_interval=int(ch.get("keepwarm_interval", 150)),
        identity_max_tokens=int(ch.get("identity_max_tokens", 400)),
        learned_skills_dir=ch.get("learned_skills_dir", "loom/skills_learned"),
        reflect_enabled=bool(ch.get("reflect_enabled", True)),
        reflect_min_actions=int(ch.get("reflect_min_actions", 1)),
    )
    me = data.get("memory", {})
    memory = MemoryConfig(
        provider=me.get("provider", "local"),
        db_path=me.get("db_path", "loom/data/memory.db"),
        soul_path=me.get("soul_path", "loom/data/SOUL.md"),
        user_path=me.get("user_path", "loom/data/USER.md"),
        memory_md_path=me.get("memory_md_path", "loom/data/MEMORY.md"),
        recall_summarize=bool(me.get("recall_summarize", True)),
        recall_summarize_threshold=int(me.get("recall_summarize_threshold", 5)),
    )
    # Découverte par dossier (loom/models/<id>/model.toml) ; repli sur l'ancien bloc
    # [[models]] de la config si aucun dossier-modèle n'est présent (transition douce).
    models = _discover_models(Path(path).parent / "models")
    if not models:
        models = [_parse_model(rm) for rm in data.get("models", [])]
    if not models:
        raise ValueError(
            "aucun modèle : crée loom/models/<id>/model.toml (ou un bloc [[models]])"
        )
    default_model = ch.get("default_model") or models[0].id
    return RuntimeConfig(
        models=models,
        default_model=default_model,
        context=int(s["context"]),
        port=int(s["port"]),
        server_bin=s["bin"],
        swap_bin=s.get("swap_bin", "llama-swap"),
        n_parallel=int(s.get("n_parallel", 1)),
        gpu_kv_headroom_mb=int(s.get("gpu_kv_headroom_mb", 1024)),
        override_n_gpu_layers=o.get("n_gpu_layers"),
        override_threads=o.get("threads"),
        chat=chat,
        memory=memory,
        permissions=parse_permissions(data),
    )
