"""Point d'entrée : uv run python -m loom.web"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from loom.agent.client import LoomClient
from loom.agent.context import effective_context_budget
from loom.agent.conversation import Conversation
from loom.agent.session import SessionStore
from loom.config import load_config
from loom.permissions import evaluate
from loom.tools import AVAILABLE_TOOLS, build_registry
from loom.web.app import create_app

RUNTIME_DIR = Path(__file__).resolve().parent.parent  # = loom/ (le package)
CONFIG_PATH = RUNTIME_DIR.parent / "config" / "defaults.toml"
PERSONAL_CONFIG_PATH = RUNTIME_DIR.parent / "config" / "local.toml"


def build_app(cfg):
    """Construit l'app Flask depuis la config (sans la servir). Séparé de `main` pour
    être testable / lançable sur un port arbitraire (vérif UI, tests d'intégration)."""
    base_url = f"http://127.0.0.1:{cfg.port}/v1"
    conversation = Conversation.load(cfg.chat.history_path, cfg.chat.system_prompt)
    if not conversation.model:
        conversation.set_model(cfg.default_model)
    # Le store JSON reste uniquement pour la compatibilité des anciens emplacements.
    from loom.config import remote_store_path as _remote_store_path

    remote_store = _remote_store_path(cfg.chat.history_path)
    # Une clé distante manquante doit échouer à l'appel, pas au démarrage de l'app.
    routes = {}
    for rm in cfg.remote_models:
        key = rm.api_key or (
            os.environ.get(rm.api_key_env, "") if rm.api_key_env else ""
        )
        routes[rm.id] = {
            "base_url": rm.base_url,
            "api_key": key,
            "model": rm.model,
            "enable_thinking_param": rm.enable_thinking_param,
        }
    client = LoomClient(
        base_url=base_url,
        timeout=cfg.chat.request_timeout,
        max_retries=cfg.chat.max_retries,
        routes=routes,
    )
    client.slot_kv_enabled = cfg.slot_kv
    # Les modèles hybrides exigent un binaire explicitement sûr pour la restauration KV.
    client.hot_resume_enabled = cfg.hot_resume
    client.restore_safe = cfg.restore_safe
    client.hybrid_models = {m.id for m in cfg.models if m.cache_isolation}
    # Séparer le store épisodique des fichiers d'identité injectés au prompt.
    from types import SimpleNamespace

    from loom.memory import get_provider

    mem_provider = get_provider(cfg.memory.provider, db_path=cfg.memory.db_path)
    mem_paths = {
        "memory_md_path": cfg.memory.memory_md_path,
        "user_path": cfg.memory.user_path,
        "soul_path": cfg.memory.soul_path,
    }

    def _recall_summarizer(query, hits):
        # Condenser les résultats FTS évite de noyer la fenêtre d'un petit modèle.
        joined = "\n".join(f"- {h.text}" for h in hits)
        prompt = (
            f"Question : {query}\n\nSouvenirs bruts :\n{joined}\n\n"
            "Condense ces souvenirs en une synthèse dense et fidèle (3-5 lignes max), "
            "centrée sur la question. Cite les faits utiles, ignore le bruit."
        )
        out = ""
        for kind, chunk in client.stream_chat(
            [{"role": "user", "content": prompt}],
            "Tu condenses des souvenirs en une note dense.",
            max_tokens=300,
            model=cfg.default_model,
            thinking=False,
        ):
            if kind == "content":
                out += chunk
        return "Synthèse mémoire :\n" + out.strip()

    memory = SimpleNamespace(provider=mem_provider, paths=mem_paths)
    if cfg.memory.recall_summarize:
        memory.summarize = _recall_summarizer
        memory.threshold = cfg.memory.recall_summarize_threshold
    reflect_stores = SimpleNamespace(
        provider=mem_provider,
        paths=mem_paths,
        learned_dir=cfg.chat.learned_skills_dir,
    )
    budget = effective_context_budget(
        cfg.chat.context_token_budget, cfg.context, cfg.chat.max_tokens
    )

    # La lecture d'image utilise toujours le modèle sélectionné, sans coût distant implicite.
    vision_model_ids = [m.id for m in cfg.models if m.mmproj_filename] + [
        rm.id for rm in cfg.remote_models if rm.vision
    ]

    # Reconstruire le registre selon les outils, le workspace et la conversation actifs.
    def _compact_for(mid):
        # Réserver la sortie empêche le sous-agent de tronquer ses appels d'outils.
        win = model_contexts.get(mid) or cfg.context
        if mid in remote_ids:
            reserve = model_max_tokens.get(mid) or 8192
        else:
            reserve = cfg.chat.max_tokens
        return max(1024, win - reserve - 1024)

    def make_registry(active, workspace=None, conversation=None):
        _model = conversation.model if conversation else cfg.default_model
        _sub_compact = _compact_for(_model)
        # Une session privée court-circuite la chaîne; ignorer les tiers non configurés.
        _chain = [m for m in cfg.chat.dispatch_models if client.is_remote(m)]
        _priv = bool(conversation and getattr(conversation, "local_only", False))
        # Résoudre ici les rôles abstraits garde les workflows indépendants des ids machine.
        _strong_ids = {rm.id for rm in cfg.remote_models if rm.strong}
        _ordered = _chain + ([_model] if client.is_remote(_model) else [])
        _roles = {}
        for mid in _ordered:
            role = "strong" if mid in _strong_ids else "cheap"
            _roles.setdefault(role, mid)
        return build_registry(
            workspace_dir=workspace or cfg.chat.workspace_dir,
            max_bytes=cfg.chat.read_file_max_bytes,
            enabled=active,
            web_cfg=cfg.chat.web_search,
            client=client,
            conversation=conversation,
            # Propager le modèle sélectionné aux sous-agents de cette conversation.
            model=_model,
            sub_max_tokens=cfg.chat.max_tokens,
            sub_compact_after_tokens=_sub_compact,
            dispatch_models=_chain,
            dispatch_local_only=_priv,
            dispatch_model_roles=_roles,
            sub_compact_for=_compact_for,
            permission=permission,
            active_model=_model,
            skills_dir=cfg.chat.skills_dir,
            plugins_root=plugins_dir,
            memory=memory,
            learned_skills_dir=cfg.chat.learned_skills_dir,
            user_skills_dir=cfg.chat.user_skills_dir,
            shell_timeout=cfg.chat.shell_timeout,
            vision_describer=None,
            active_is_vision=(_model in vision_model_ids),
            deferred_tools=cfg.chat.deferred_tools,
            monitor_hub=monitor_hub,
            mcp_hub=mcp_hub,
        )

    if not conversation.active_tools and cfg.chat.tools_enabled:
        conversation.set_tools(cfg.chat.tools_enabled)

    # Importer l'ancienne conversation uniquement si aucune session n'existe encore.
    data_root = Path(cfg.chat.history_path).resolve().parent

    from loom.tools.monitor import MonitorHub

    monitor_hub = MonitorHub(data_root / "logs" / "monitors")

    from loom.tools.mcp import McpHub

    mcp_hub = McpHub(cfg.mcp_servers)

    from loom.extend.plugins import plugins_root as _plugins_root

    plugins_dir = str(_plugins_root(getattr(cfg.chat, "plugins_root", None)))

    from loom.runtime.image_models import discover_image_models

    image_models = discover_image_models(cfg.models_roots)

    sessions_root = data_root / "sessions"
    store = SessionStore(
        sessions_root,
        cfg.chat.system_prompt,
        default_tools=cfg.chat.tools_enabled,
        default_model=cfg.default_model,
        known_models=[m.id for m in cfg.models]
        + [rm.id for rm in cfg.remote_models]
        + [im.id for im in image_models],
    )
    if not store.list() and conversation.messages:
        seed = store.create(workspace=cfg.chat.workspace_dir, title="Session importée")
        seed.conversation = conversation
        store.save(seed)

    def permission(name, args):
        return evaluate(name, args, cfg.permissions)

    # Chaque modèle utilise sa fenêtre pour calculer son seuil de microcompaction.
    model_contexts = {m.id: (m.context or cfg.context) for m in cfg.models}
    model_contexts.update(
        {rm.id: (rm.context or cfg.context) for rm in cfg.remote_models}
    )
    model_max_tokens = {
        rm.id: rm.max_tokens for rm in cfg.remote_models if rm.max_tokens
    }
    remote_ids = {rm.id for rm in cfg.remote_models}
    remote_weak_ids = {rm.id for rm in cfg.remote_models if not rm.strong}
    local_models = [
        {
            "id": m.id,
            "dir": m.dir,
            "repo": m.repo,
            "filename": m.filename,
            "n_layers": m.n_layers,
            "size_mb": m.size_mb,
            "context": (m.context or cfg.context),
            "n_gpu_layers": m.n_gpu_layers,
            "cpu_moe": m.cpu_moe,
            "n_cpu_moe": m.n_cpu_moe,
            "vision": bool(m.mmproj_filename),
        }
        for m in cfg.models
    ]
    app = create_app(
        client,
        cfg.chat.skills_dir,
        store,
        max_tokens=cfg.chat.max_tokens,
        context_budget=budget,
        keep_recent=cfg.chat.keep_recent_messages,
        context_window=cfg.context,
        models=[m.id for m in cfg.models] + [rm.id for rm in cfg.remote_models],
        vision_models=[m.id for m in cfg.models if m.mmproj_filename]
        + [rm.id for rm in cfg.remote_models if rm.vision],
        tool_factory=make_registry,
        monitor_hub=monitor_hub,
        available_tools=AVAILABLE_TOOLS,
        permission=permission,
        permission_mode=cfg.permissions.mode,
        workspace_dir=cfg.chat.workspace_dir,
        plugins_dir=plugins_dir,
        keepwarm_enabled=cfg.chat.keepwarm_enabled,
        keepwarm_interval=cfg.chat.keepwarm_interval,
        identity_paths=mem_paths,
        identity_max_tokens=cfg.chat.identity_max_tokens,
        project_memory_max_tokens=cfg.chat.project_memory_max_tokens,
        learned_skills_dir=cfg.chat.learned_skills_dir,
        user_skills_dir=cfg.chat.user_skills_dir,
        memory_db_path=cfg.memory.db_path,
        reflect_stores=reflect_stores,
        reflect_enabled=cfg.chat.reflect_enabled,
        reflect_min_actions=cfg.chat.reflect_min_actions,
        reflect_model=cfg.default_model,
        model_contexts=model_contexts,
        model_max_tokens=model_max_tokens,
        remote_model_ids=[rm.id for rm in cfg.remote_models],
        remote_weak_ids=sorted(remote_weak_ids),
        remote_model_names={rm.id: rm.model for rm in cfg.remote_models},
        model_prices={
            rm.id: (rm.price_in, rm.price_out, rm.price_cached)
            for rm in cfg.remote_models
        },
        model_descriptions={
            **{m.id: m.description for m in cfg.models if m.description},
            **{rm.id: rm.description for rm in cfg.remote_models if rm.description},
            **{im.id: im.description for im in image_models if im.description},
        },
        remote_store_path=str(remote_store),
        config_defaults_path=str(CONFIG_PATH),
        config_local_path=str(PERSONAL_CONFIG_PATH),
        local_models=local_models,
        image_models=image_models,
        models_dir=str(cfg.models_dir),
        models_roots=[str(r) for r in cfg.models_roots],
    )
    return app


def _warn_if_fallback_context(cfg) -> None:
    """FAIL-LOUD (audit 2026-07-18, pattern « plausible-silencieux-faux ») : quand
    le contexte serveur vient du REPLI NEUTRE de defaults.toml (aucune calibration
    machine dans local.toml), on le DIT au boot au lieu de brider en silence — la
    régression du 18/07 (24576 -> 8192 après un pull) était invisible sans ça.
    N'alerte que si un modèle LOCAL sans context propre est concerné : les modèles
    dont model.toml porte `context` et les distants ne passent pas par ce repli."""
    import tomllib

    try:
        local = tomllib.loads(Path(PERSONAL_CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        local = {}
    if (local.get("server") or {}).get("context"):
        return  # calibré pour cette machine
    exposed = [m.id for m in cfg.models if not getattr(m, "context", None)]
    if not exposed:
        return
    print(
        f"[loom] ⚠️  context = {cfg.context} (REPLI NEUTRE de defaults.toml — cette "
        "machine n'est pas calibrée). Modèles locaux concernés : "
        f"{', '.join(exposed[:4])}{'…' if len(exposed) > 4 else ''}. "
        "Lance `uv run loom-setup` (étape bench) pour mesurer la vraie fenêtre."
    )


def main() -> None:
    # Un modèle distant suffit au boot; sinon guider l'installation locale manquante.
    from loom.runtime.serve import maybe_bootstrap

    code = maybe_bootstrap(remote_ok=True)
    if code is not None:
        raise SystemExit(code)
    cfg = load_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
    _warn_if_fallback_context(cfg)
    app = build_app(cfg)
    # Filtrer les polls fréquents garde le journal Werkzeug lisible.
    _POLL_PATHS = ("/sysmon", "/machine_state")
    logging.getLogger("werkzeug").addFilter(
        lambda record: not any(p in record.getMessage() for p in _POLL_PATHS)
    )
    print(
        f"[loom-chat] http://127.0.0.1:{cfg.chat.web_port}  (modèle: http://127.0.0.1:{cfg.port}/v1)"
    )
    # Le mode threaded sert la nouvelle requête pendant la fermeture de l'ancien flux.
    app.run(host="127.0.0.1", port=cfg.chat.web_port, threaded=True)


if __name__ == "__main__":
    main()
