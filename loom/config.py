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

# Racine des modèles par défaut : dans le package (loom/models). Surchageable par
# [storage] models_root (config/local.toml) — ex. E:/loom-models sur la machine du user.
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
    # Outils (boucle tool-use). enabled vide => aucun outil exposé.
    tools_enabled: list[str] = field(default_factory=list)
    # Chaîne de ROUTAGE des sous-agents (dispatch_agent), dans l'ordre d'essai
    # (ex. ["glm-flash", "glm-zai"] = gratuit puis payant) ; repli final implicite =
    # le modèle de la conversation. Vide = comportement historique (héritage).
    # Ignorée quand la session est marquée local_only (données privées/sensibles).
    dispatch_models: list[str] = field(default_factory=list)
    workspace_dir: str = "."
    # Cap par appel read_file (caractères). Volontairement BAS devant le contexte (24576
    # tokens) : un seul gros fichier ne doit pas le faire déborder -> on lit par tranches
    # (start_line). ~40000 car ≈ 10k tokens.
    read_file_max_bytes: int = 40_000
    # Timeout run_shell (s) : assez long pour npm install / next build (qui depassent 30s et se
    # faisaient tuer -> cascade). La commande reste tuee au-dela (anti-hang).
    shell_timeout: int = 180
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    # Keep-warm : sur Windows, un llama-server inactif se fait rogner son working-set par
    # l'OS (pages des experts MoE évincées) -> 1re requête après une pause = lente (cold
    # start). Un ping minimal (1 token) toutes les keepwarm_interval secondes, À L'IDLE
    # seulement, garde le modèle chargé « chaud » sans verrouiller la RAM en dur. Pinge le
    # modèle de la session active (jamais un autre -> pas de swap llama-swap intempestif).
    keepwarm_enabled: bool = True
    keepwarm_interval: int = 150
    # Mémoire/identité : budget du bloc identité always-on injecté au prompt (SOUL/USER/MEMORY).
    # SOUL+USER+MEMORY propres pèsent ~450 tokens : 600 laisse de la marge sans tronquer le
    # durable (négligeable sur une fenêtre de 24576). Le projet-spécifique va en épisodique.
    identity_max_tokens: int = 600
    # Fiche projet `<workspace>/loom.md` (générée par /init) auto-injectée au system prompt.
    # Budget distinct de l'identité : une fiche projet peut être plus dense (stack, arbo,
    # commandes) sans rogner SOUL/USER/MEMORY.
    project_memory_max_tokens: int = 600
    # Apprentissage (boucle fermée) : skills auto-appris + étape reflect post-tour.
    learned_skills_dir: str = "var/skills_learned"
    # Skills AJOUTÉS PAR L'UTILISATEUR depuis l'UI (« + nouveau ») : hors du package
    # (loom/skills reste versionné/officiel), machine-locale comme les appris.
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
    # Microbatch (-ub) et batch (-b) de PREFILL propres au modèle : sur un MoE offloadé,
    # un -ub plus gros amortit les passes d'experts CPU (banc 2026-07-19 : agents-a1
    # -ub 2048 -b 4096 = prefill x2,9, décode intact) — contre des buffers VRAM plus
    # gros, à valider par modèle. None = défauts serveur (512/2048).
    ubatch: int | None = None
    batch: int | None = None
    # Isolation du cache de conversation par 2e slot : True quand la sonde du bench
    # a MESURÉ que le cache de ce modèle ne survit pas à la pollution du slot par un
    # autre prompt (mémoire hybride/SWA, exclue du prompt-cache RAM natif de
    # llama-server -> re-prefill total au retour). serve monte alors --parallel à 2 :
    # les appels annexes (titre, reflect, dispatch) s'isolent dans le 2e slot par
    # similarité de préfixe et la conversation garde son cache (validé 2026-07-21 :
    # 74 s -> 4-6 s par tour sur ornith). Coût : cache KV x2 (-c est multiplié).
    cache_isolation: bool = False
    # Dossier du modèle (loom/models/<id>/) : porte le GGUF, le mmproj et profile.md.
    # Rempli par la découverte ; les chemins GGUF se résolvent contre lui.
    dir: str = ""
    # RÔLE en une ligne (model.toml `description`) : infobulle du sélecteur UI —
    # aide à choisir quand le parc grossit.
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
    # Fenêtre de contexte du modèle -> seuil de microcompact + jauge de remplissage (UI).
    # RÉFLEXE à l'ajout d'un provider : la plupart des API (Z.ai, OpenAI) ne publient PAS le
    # contexte via GET /models (schéma nu id/object/created/owned_by). La source de vérité est
    # la DOC officielle du modèle -> web fetch/search la fenêtre réelle et mets-la ici (ex.
    # glm-5.2 = 1M sur docs.z.ai, pas 200k). Certains providers l'exposent quand même via
    # /models (OpenRouter context_length, vLLM max_model_len) : là Loom la lit tout seul
    # (client.remote_context) et cette valeur ne sert que de repli.
    context: int | None = None
    max_tokens: int | None = None  # plafond de sortie/tour (défaut = global si absent)
    vision: bool = False  # le modèle accepte-t-il les images
    # Tier de harnais : True (défaut) = modèle FORT -> prompt allégé, gardes de
    # comportement coupées (décision 2026-07 : le scaffolding bride un modèle fort).
    # False = distant FAIBLE (ex. GLM-4.7-Flash free, gabarit d'un local) -> harnais
    # complet ; constaté live : sans les gardes, il annonce ses tool_calls sans agir.
    strong: bool = True
    # Une API hébergée rejette souvent un extra_body inconnu (chat_template_kwargs) ->
    # par défaut on NE l'envoie PAS pour un modèle distant. Mets True seulement si
    # l'endpoint gère ce champ (vLLM auto-hébergé…) pour piloter le thinking au template.
    enable_thinking_param: bool = False
    # Prix en $ / MILLION de tokens (input, output) chez le provider -> sert au compteur de
    # coût RÉEL de la session. 0 = inconnu (coût affiché à 0). Le tarif dépend du modèle
    # côté provider (ex. glm-5.2 chez Z.ai) : mets-le dans config/local.toml.
    price_in: float = 0.0
    price_out: float = 0.0
    # Prix $ / M des tokens d'INPUT servis par le CACHE (hit de préfixe) — bien moins cher
    # (ex. glm-5.2 : ~0.26 vs 1.40). 0 = pas de remise appliquée (coût = borne haute).
    price_cached: float = 0.0
    # RÔLE en une ligne : infobulle du sélecteur UI.
    description: str = ""


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
    # Modèles distants (API OpenAI-compatible) : s'ajoutent aux modèles locaux dans le
    # sélecteur. Vide par défaut (tout-local) ; déclarés dans config/local.toml.
    remote_models: list[RemoteModelConfig] = field(default_factory=list)
    n_parallel: int = 1
    # Cache souverain (save/restore du slot KV par tour) — OFF par défaut depuis le
    # 2026-07-22 : sur llama-server récent (b10075+) il est soit nuisible (mémoire
    # HYBRIDE : restore efface les checkpoints -> re-prefill TOTAL chaque tour, 66 s
    # mesurés), soit inutile (modèles classiques : le prompt-cache RAM natif
    # --cache-ram protège déjà, le save capture 0 token), soit cassé (save qui PEND
    # sur le parc CUDA, 502). La protection du cache de conversation passe par le
    # prompt-cache natif et, pour les modèles qui y échappent, par l'isolation des
    # appels annexes dans un 2e slot (model.toml cache_isolation, mesuré au bench).
    # True = réactiver l'ancien mécanisme (vieux llama-server sans cache RAM natif).
    slot_kv: bool = False
    # Marge VRAM (Mo) réservée hors couches offloadées : couvre le cache KV + les buffers
    # de calcul. Plus elle est BASSE, plus on offloade de couches sur GPU (perf), mais trop
    # bas -> débordement en mémoire partagée (Windows) qui écroule tout. À régler par machine.
    gpu_kv_headroom_mb: int = 1024
    permissions: PermissionConfig = field(default_factory=PermissionConfig)
    # Racines des modèles ([storage] models_root : chaîne OU liste, ex.
    # ["C:/loom-models", "E:/loom-models"] — même arbo local/{text,image,video} + remote
    # sous chacune). Plusieurs racines = répartir le parc par vitesse de disque (gros
    # modèles rechargés souvent sur NVMe, le reste sur disque externe). En cas d'id en
    # double, la PREMIÈRE racine gagne. models_root/models_dir = la racine primaire
    # (gabarit _TEMPLATE, replis legacy).
    models_root: Path = field(default_factory=lambda: _PACKAGE_MODELS)
    models_roots: list[Path] = field(default_factory=lambda: [_PACKAGE_MODELS])
    models_dir: Path = field(default_factory=lambda: _PACKAGE_MODELS)

    def model_by_id(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        if not self.models:
            # Boot remote-only (parc local vide) : cfg.model / model_by_id n'ont pas de
            # sens sans modèle local — erreur claire plutôt qu'un IndexError chez un
            # consommateur (serve.py est censé avoir vérifié avant).
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
    """Construit un RemoteModelConfig depuis une table TOML [[remote_models]]."""
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
    # Racines des modèles : [storage] models_root (chaîne OU liste) sinon le package
    # (loom/models). Arbo UNIQUE sous chaque racine : local/{text,image,video}
    # + remote + _TEMPLATE (pas de layout legacy — migration 2026-07-08).
    st = data.get("storage", {})
    raw_roots = st.get("models_root") or _PACKAGE_MODELS
    if not isinstance(raw_roots, list):
        raw_roots = [raw_roots]
    models_roots = [Path(r).resolve() for r in raw_roots]
    models_root = models_roots[0]
    models_dir = models_root / "local" / "text"
    # Les profils par modèle (models_profile) se résolvent contre ces racines partout
    # (outils, app web) sans les faire circuler dans chaque signature.
    from loom.runtime import models_profile as _mp

    _mp.set_models_root(models_roots)
    # Découverte par dossier (<racine>/local/text/<id>/model.toml) sur CHAQUE racine,
    # première racine gagnante en cas d'id en double ; repli sur l'ancien bloc
    # [[models]] de la config si aucun dossier-modèle n'est présent (transition douce).
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
    remote_models = [_parse_remote_model(rm) for rm in data.get("remote_models", [])]
    # Héritage : les distants ajoutés via l'UI vivaient dans un store JSON séparé
    # (var/remote_models.json). Unification 2026-07-21 : config/local.toml est la
    # source UNIQUE — un store résiduel est replié dans local.toml ici (puis
    # supprimé) ; ses entrées gardent la priorité par id sur ce chargement.
    from loom.runtime import model_store

    for md in model_store.migrate_to_toml(
        remote_store_path(chat.history_path), local_path
    ):
        rc = _parse_remote_model(md)
        remote_models = [rm for rm in remote_models if rm.id != rc.id]
        remote_models.append(rc)
    # Boot « remote-only » : aucun modèle local mais au moins un distant -> models=[]
    # est toléré (loom.web sait discuter via l'API distante ; le serveur llama.cpp
    # local ne sert que les locaux et démarre à la demande). On ne lève que si NI
    # local NI distant — machine vierge, c'est maybe_bootstrap qui guide l'installeur.
    if not models and not remote_models:
        raise ValueError(
            f"aucun modèle : crée <racine>/local/text/<id>/model.toml sous une des "
            f"racines {[str(r) for r in models_roots]} (ou un bloc [[models]], ou un "
            f"modèle distant [[remote_models]] dans config/local.toml)"
        )
    default_model = ch.get("default_model") or (models[0].id if models else "")
    if not models:
        # Remote-only : un default_model configuré qui pointe un local disparu
        # enverrait la session dans le vide -> retombe sur le premier distant.
        if default_model not in {rm.id for rm in remote_models}:
            default_model = remote_models[0].id
    return RuntimeConfig(
        models=models,
        default_model=default_model,
        remote_models=remote_models,
        context=int(s["context"]),
        port=int(s["port"]),
        server_bin=s["bin"],
        swap_bin=s.get("swap_bin", "llama-swap"),
        n_parallel=int(s.get("n_parallel", 1)),
        slot_kv=bool(s.get("slot_kv", False)),
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
