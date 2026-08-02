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

# `[storage] models_root` peut remplacer la racine embarquée des modèles.
_PACKAGE_MODELS = Path(__file__).resolve().parent / "models"


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
    # Une liste vide n'expose aucun outil.
    tools_enabled: list[str] = field(default_factory=list)
    # Schémas longue traîne chargés à la demande par tool_search. Kill-switch
    # désactivé par défaut jusqu'à validation A/B sur un modèle local.
    deferred_tools: bool = False
    # Chaîne de ROUTAGE des sous-agents (dispatch_agent), dans l'ordre d'essai
    # (ex. ["glm-flash", "glm-zai"] = gratuit puis payant) ; repli final implicite =
    # le modèle de la conversation. Vide = comportement historique (héritage).
    # Ignorée quand la session est marquée local_only (données privées/sensibles).
    dispatch_models: list[str] = field(default_factory=list)
    workspace_dir: str = "."
    # Borner chaque lecture afin qu'un fichier seul ne sature pas le contexte.
    read_file_max_bytes: int = 40_000
    # Laisser les builds longs finir, tout en conservant une borne anti-hang.
    shell_timeout: int = 180
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    # Sous Windows, un ping idle empêche l'éviction du working-set MoE. Cibler seulement
    # la session active pour ne pas provoquer de swap.
    keepwarm_enabled: bool = True
    keepwarm_interval: int = 150
    # Budget du bloc durable SOUL/USER/MEMORY injecté à chaque prompt.
    identity_max_tokens: int = 600
    # La fiche projet a son propre budget pour ne pas rogner l'identité durable.
    project_memory_max_tokens: int = 600
    # Boucle d'apprentissage et réflexion post-tour.
    learned_skills_dir: str = "var/skills_learned"
    # Garder les skills utilisateur hors du package versionné.
    user_skills_dir: str = "var/skills_user"
    reflect_enabled: bool = True
    reflect_min_actions: int = 1


@dataclass
class MemoryConfig:
    """Config de la mémoire (design §9). v1 : provider 'local' uniquement (offline)."""

    provider: str = "local"
    db_path: str = "var/memory/memory.db"
    soul_path: str = "var/identity/SOUL.md"
    user_path: str = "var/identity/USER.md"
    memory_md_path: str = "var/identity/MEMORY.md"
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
    # Garder attention/dense sur GPU et placer les experts routés en RAM.
    cpu_moe: bool = False
    # `n_cpu_moe` offloade seulement N couches, contrairement à `cpu_moe`.
    n_cpu_moe: int | None = None
    # Un contexte par modèle peut remplacer la valeur globale selon son coût KV.
    context: int | None = None
    # Des batchs de prefill plus grands amortissent le CPU mais consomment plus de VRAM.
    ubatch: int | None = None
    batch: int | None = None
    # Un second slot isole les appels annexes lorsque le cache hybride/SWA ne survit
    # pas à un autre préfixe, au prix d'un cache KV doublé.
    cache_isolation: bool = False
    # Rapprocher les checkpoints hybride/SWA borne le retraitement mais consomme plus de RAM.
    checkpoint_min_step: int | None = None
    # La découverte remplit ce dossier, base des chemins GGUF et mmproj.
    dir: str = ""
    # Description courte affichée dans le sélecteur.
    description: str = ""


@dataclass
class RemoteModelConfig:
    """Modèle servi par une API externe OpenAI-compatible (Zhipu/GLM, OpenAI, OpenRouter,
    Mistral…). Sélectionnable dans le menu au même titre qu'un modèle local : Loom route
    les requêtes de CE modèle vers son endpoint, les outils tournent toujours en local.
    Le secret se met dans config/local.toml (gitignored) ou via api_key_env (variable
    d'environnement) — jamais dans defaults.toml versionné."""

    id: str  # nom affiché dans le sélecteur (ex. "glm-distant")
    base_url: (
        str  # endpoint OpenAI-compatible (ex. https://open.bigmodel.cn/api/paas/v4)
    )
    model: str  # id du modèle CÔTÉ provider (ex. "glm-4.6")
    api_key: str = ""  # clé en clair (préfère local.toml)…
    api_key_env: str = ""  # …ou nom d'une variable d'env qui la porte
    # La plupart des providers omettent cette fenêtre dans `/models`; configurer la valeur
    # officielle, utilisée comme repli pour la jauge et le microcompact.
    context: int | None = None
    max_tokens: int | None = None  # plafond de sortie/tour (défaut = global si absent)
    vision: bool = False  # le modèle accepte-t-il les images
    # Un modèle fort reçoit un prompt allégé ; un faible garde les protections comportementales.
    strong: bool = True
    # N'activer les paramètres natifs que si l'endpoint distant les accepte.
    enable_thinking_param: bool = False
    # Prix provider en dollars par million de tokens ; zéro signifie inconnu.
    price_in: float = 0.0
    price_out: float = 0.0
    # Prix du cache d'entrée ; zéro conserve le tarif normal comme borne haute.
    price_cached: float = 0.0
    # Description courte affichée dans le sélecteur.
    description: str = ""


@dataclass
class McpServerConfig:
    """Serveur MCP tiers. La tranche courante exécute le transport stdio."""

    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout_s: float = 10.0
    # None/True = dangereux (confirmation selon le mode global). False = serveur
    # explicitement déclaré de confiance : appels autorisés sans confirmation.
    danger_override: bool | None = None


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
    # Les modèles distants rejoignent les locaux dans le sélecteur.
    remote_models: list[RemoteModelConfig] = field(default_factory=list)
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    n_parallel: int = 1
    # Le save/restore par tour est désactivé : le cache RAM natif suffit aux modèles
    # classiques et l'isolation de slot protège les hybrides.
    slot_kv: bool = False
    # La reprise à chaud restaure seulement un slot froid, jamais à chaque tour.
    hot_resume: bool = False
    # Exclure les hybrides si le binaire ne préserve pas leurs checkpoints au restore.
    restore_safe: bool = False
    # Réserver cette VRAM au cache KV et aux buffers pour éviter le spill partagé.
    gpu_kv_headroom_mb: int = 1024
    permissions: PermissionConfig = field(default_factory=PermissionConfig)
    # Plusieurs racines permettent de répartir les modèles par disque ; la première
    # occurrence d'un id gagne.
    models_root: Path = field(default_factory=lambda: _PACKAGE_MODELS)
    models_roots: list[Path] = field(default_factory=lambda: [_PACKAGE_MODELS])
    models_dir: Path = field(default_factory=lambda: _PACKAGE_MODELS)

    def model_by_id(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        if not self.models:
            # Sans modèle local, ces accesseurs doivent échouer clairement.
            raise ValueError(f"aucun modèle local (demandé : {model_id!r})")
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
        ubatch=d.get("ubatch"),
        batch=d.get("batch"),
        cache_isolation=bool(d.get("cache_isolation", False)),
        checkpoint_min_step=d.get("checkpoint_min_step"),
        dir=d.get("dir", ""),
        description=str(d.get("description", "") or ""),
    )


def _discover_models(models_root: Path) -> list[ModelConfig]:
    """Découvre un modèle par sous-dossier `loom/models/<id>/model.toml` (l'id = nom du
    dossier). Renvoie la liste triée par id. Chaque ModelConfig porte son `dir`. Vide si
    le dossier racine n'existe pas ou ne contient aucun model.toml."""
    if not models_root.is_dir():
        return []
    out: list[ModelConfig] = []
    for folder in sorted(p for p in models_root.iterdir() if p.is_dir()):
        # Ignorer les dossiers de gabarit préfixés par `_`.
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


def remote_store_path(history_path: str | Path) -> Path:
    """Chemin du store MACHINE-OWNED des modèles distants ajoutés via l'UI
    (remote_models.json, à côté de l'historique de conversation — var/ par défaut).
    Dérivation UNIQUE, partagée par load_config, build_app et le check de bootstrap."""
    return Path(history_path).resolve().parent / "remote_models.json"


def remote_model_from_dict(d: dict) -> RemoteModelConfig:
    """Public : construit un RemoteModelConfig depuis un dict (store JSON géré par l'UI ou
    table TOML). Même normalisation que le chargement config -> fusion homogène au démarrage."""
    return _parse_remote_model(d)


def _parse_remote_model(d: dict) -> RemoteModelConfig:
    """Construit un RemoteModelConfig depuis un dict : model.toml d'un dossier
    remote/<id>/ (base [remote] déjà fusionnée) ou entrée héritée en migration."""
    return RemoteModelConfig(
        id=d["id"],
        base_url=str(d["base_url"]).rstrip("/"),
        model=d["model"],
        api_key=d.get("api_key", ""),
        api_key_env=d.get("api_key_env", ""),
        context=d.get("context"),
        max_tokens=d.get("max_tokens"),
        vision=bool(d.get("vision", False)),
        strong=bool(d.get("strong", True)),
        enable_thinking_param=bool(d.get("enable_thinking_param", False)),
        price_in=float(d.get("price_in", 0.0) or 0.0),
        price_out=float(d.get("price_out", 0.0) or 0.0),
        price_cached=float(d.get("price_cached", 0.0) or 0.0),
        description=str(d.get("description", "") or ""),
    )


def _parse_mcp_server(d: dict) -> McpServerConfig:
    name = str(d.get("name", "")).strip()
    if not name:
        raise ValueError("mcp_servers: champ 'name' obligatoire")
    transport = str(d.get("transport", "stdio")).strip().lower()
    if transport not in ("stdio", "http"):
        raise ValueError(f"serveur MCP '{name}' : transport attendu 'stdio' ou 'http'")
    command = str(d.get("command", "")).strip()
    url = str(d.get("url", "")).strip()
    if transport == "stdio" and not command:
        raise ValueError(f"serveur MCP '{name}' : champ 'command' obligatoire")
    if transport == "http" and not url:
        raise ValueError(f"serveur MCP '{name}' : champ 'url' obligatoire")
    danger = d.get("danger_override")
    return McpServerConfig(
        name=name,
        transport=transport,
        command=command,
        args=[str(v) for v in d.get("args", [])],
        env={str(k): str(v) for k, v in dict(d.get("env", {})).items()},
        url=url,
        headers={str(k): str(v) for k, v in dict(d.get("headers", {})).items()},
        enabled=bool(d.get("enabled", True)),
        timeout_s=max(0.1, float(d.get("timeout_s", 10.0))),
        danger_override=(None if danger is None else bool(danger)),
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
        history_path=ch.get("history_path", "var/conversation.json"),
        web_port=int(ch.get("web_port", 8000)),
        skills_dir=ch.get("skills_dir", "loom/skills"),
        plugins_root=ch.get("plugins_root", "loom/plugins"),
        max_tokens=int(ch.get("max_tokens", 2048)),
        request_timeout=int(ch.get("request_timeout", 120)),
        max_retries=int(ch.get("max_retries", 6)),
        context_token_budget=int(ch.get("context_token_budget", 3000)),
        keep_recent_messages=int(ch.get("keep_recent_messages", 6)),
        tools_enabled=list(tl.get("enabled", [])),
        deferred_tools=bool(ch.get("deferred_tools", False)),
        dispatch_models=list(ch.get("dispatch_models", [])),
        workspace_dir=tl.get("workspace_dir", "."),
        read_file_max_bytes=int(tl.get("read_file_max_bytes", 40_000)),
        shell_timeout=int(tl.get("shell_timeout", 180)),
        web_search=_parse_web_search(ws),
        keepwarm_enabled=bool(ch.get("keepwarm_enabled", True)),
        keepwarm_interval=int(ch.get("keepwarm_interval", 150)),
        identity_max_tokens=int(ch.get("identity_max_tokens", 600)),
        project_memory_max_tokens=int(ch.get("project_memory_max_tokens", 600)),
        learned_skills_dir=ch.get("learned_skills_dir", "var/skills_learned"),
        user_skills_dir=ch.get("user_skills_dir", "var/skills_user"),
        reflect_enabled=bool(ch.get("reflect_enabled", True)),
        reflect_min_actions=int(ch.get("reflect_min_actions", 1)),
    )
    me = data.get("memory", {})
    memory = MemoryConfig(
        provider=me.get("provider", "local"),
        db_path=me.get("db_path", "var/memory/memory.db"),
        soul_path=me.get("soul_path", "var/identity/SOUL.md"),
        user_path=me.get("user_path", "var/identity/USER.md"),
        memory_md_path=me.get("memory_md_path", "var/identity/MEMORY.md"),
        recall_summarize=bool(me.get("recall_summarize", True)),
        recall_summarize_threshold=int(me.get("recall_summarize_threshold", 5)),
    )
    # Utiliser la racine configurée, sinon l'arborescence embarquée.
    st = data.get("storage", {})
    raw_roots = st.get("models_root") or _PACKAGE_MODELS
    if not isinstance(raw_roots, list):
        raw_roots = [raw_roots]
    models_roots = [Path(r).resolve() for r in raw_roots]
    models_root = models_roots[0]
    models_dir = models_root / "local" / "text"
    # Partager ces racines avec les profils sans les propager dans chaque signature.
    from loom.runtime import models_profile as _mp

    _mp.set_models_root(models_roots)
    # Découvrir chaque racine dans l'ordre, puis replier sur l'ancien bloc `[[models]]`.
    models: list[ModelConfig] = []
    seen_ids: set[str] = set()
    for root in models_roots:
        for m in _discover_models(root / "local" / "text"):
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                models.append(m)
    models.sort(key=lambda m: m.id)
    if not models:
        models = [_parse_model(rm) for rm in data.get("models", [])]
    # Les dossiers distants surchargent les réglages communs. Migrer les anciens stores
    # de façon idempotente et garder en mémoire toute entrée non migrable.
    from loom.runtime import model_store

    remote_base = {
        k: v for k, v in data.get("remote", {}).items() if k in model_store.FIELDS
    }
    leftovers = model_store.migrate_into_dirs(
        local_path, remote_store_path(chat.history_path), models_root
    )
    remote_records = model_store.discover_remote(models_roots)
    seen_remote = {md["id"] for md in remote_records}
    # Fusionner les reliquats hérités sans écraser un dossier prioritaire.
    for rm in data.get("remote_models", []):
        if isinstance(rm, dict) and rm.get("id") and rm["id"] not in seen_remote:
            seen_remote.add(rm["id"])
            remote_records.append({k: rm[k] for k in model_store.KEEP if k in rm})
    remote_records += [md for md in leftovers if md["id"] not in seen_remote]
    remote_models = [
        _parse_remote_model({**remote_base, **md}) for md in remote_records
    ]
    mcp_servers = [_parse_mcp_server(md) for md in data.get("mcp_servers", [])]
    mcp_names = [server.name for server in mcp_servers]
    if len(set(mcp_names)) != len(mcp_names):
        raise ValueError("mcp_servers: chaque 'name' doit être unique")
    # Boot « remote-only » : aucun modèle local mais au moins un distant -> models=[]
    # est toléré (loom.web sait discuter via l'API distante ; le serveur llama.cpp
    # local ne sert que les locaux et démarre à la demande). On ne lève que si NI
    # local NI distant — machine vierge, c'est maybe_bootstrap qui guide l'installeur.
    if not models and not remote_models:
        raise ValueError(
            f"aucun modèle : crée <racine>/local/text/<id>/model.toml (local) ou "
            f"<racine>/remote/<id>/model.toml (distant) sous une des racines "
            f"{[str(r) for r in models_roots]} (ou un bloc [[models]] legacy)"
        )
    default_model = ch.get("default_model") or (models[0].id if models else "")
    if not models:
        # Replier un défaut local absent sur le premier modèle distant.
        if default_model not in {rm.id for rm in remote_models}:
            default_model = remote_models[0].id
    return RuntimeConfig(
        models=models,
        default_model=default_model,
        remote_models=remote_models,
        mcp_servers=mcp_servers,
        context=int(s["context"]),
        port=int(s["port"]),
        server_bin=s["bin"],
        swap_bin=s.get("swap_bin", "llama-swap"),
        n_parallel=int(s.get("n_parallel", 1)),
        slot_kv=bool(s.get("slot_kv", False)),
        hot_resume=bool(s.get("hot_resume", False)),
        restore_safe=bool(s.get("restore_safe", False)),
        gpu_kv_headroom_mb=int(s.get("gpu_kv_headroom_mb", 1024)),
        override_n_gpu_layers=o.get("n_gpu_layers"),
        override_threads=o.get("threads"),
        chat=chat,
        memory=memory,
        permissions=parse_permissions(data),
        models_root=models_root,
        models_roots=models_roots,
        models_dir=models_dir,
    )
