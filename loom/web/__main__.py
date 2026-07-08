# loom/web/__main__.py
"""Point d'entrée : uv run python -m loom.web"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from loom.agent.client import LoomClient
from loom.config import load_config
from loom.agent.context import effective_context_budget
from loom.agent.conversation import Conversation
from loom.permissions import evaluate
from loom.agent.session import SessionStore
from loom.tools import AVAILABLE_TOOLS, build_registry
from loom.web.app import create_app

RUNTIME_DIR = Path(__file__).resolve().parent.parent  # = loom/ (le package)
# La config vit désormais à la racine du repo : config/defaults.toml (versionné) +
# config/local.toml (surcharge machine, gitignored).
CONFIG_PATH = RUNTIME_DIR.parent / "config" / "defaults.toml"
PERSONAL_CONFIG_PATH = RUNTIME_DIR.parent / "config" / "local.toml"


def build_app(cfg):
    """Construit l'app Flask depuis la config (sans la servir). Séparé de `main` pour
    être testable / lançable sur un port arbitraire (vérif UI, tests d'intégration)."""
    base_url = f"http://127.0.0.1:{cfg.port}/v1"
    conversation = Conversation.load(cfg.chat.history_path, cfg.chat.system_prompt)
    if not conversation.model:
        conversation.set_model(cfg.default_model)
    # Modèles distants AJOUTÉS VIA L'UI : store machine-owned (var/remote_models.json), fusionné
    # ici avec ceux de local.toml. Les gérés par l'UI l'emportent par id. Tout le reste (routes,
    # contexte, prix, sélecteur) se dérive ensuite de cfg.remote_models sans autre changement.
    from loom.config import remote_model_from_dict
    from loom.runtime import model_store

    remote_store_path = (
        Path(cfg.chat.history_path).resolve().parent / "remote_models.json"
    )
    for md in model_store.load(remote_store_path):
        rc = remote_model_from_dict(md)
        cfg.remote_models = [rm for rm in cfg.remote_models if rm.id != rc.id]
        cfg.remote_models.append(rc)
    # Routes vers les modèles distants (API OpenAI-compatible) : la clé vient du TOML en clair
    # ou d'une variable d'env (api_key_env). Un modèle sans clé résolue reste routé mais
    # échouera à l'appel (401) — message clair côté client, pas de crash au démarrage.
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
    # Mémoire persistante : provider (store épisodique) + chemins identité (SOUL/USER/MEMORY).
    # `memory` est passé à build_registry (outils recall/remember) ; `mem_paths` sert aussi
    # à injecter le bloc identité au system prompt (create_app).
    from types import SimpleNamespace

    from loom.memory import get_provider

    mem_provider = get_provider(cfg.memory.provider, db_path=cfg.memory.db_path)
    mem_paths = {
        "memory_md_path": cfg.memory.memory_md_path,
        "user_path": cfg.memory.user_path,
        "soul_path": cfg.memory.soul_path,
    }

    def _recall_summarizer(query, hits):
        # Condense les hits FTS5 en une note dense (modèle local) : un recall brut noierait
        # un petit modèle. Câblé seulement si cfg.memory.recall_summarize (design §6.6).
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

    # Modèles VISION (voient les images) : locaux avec mmproj + distants marqués vision.
    # DESCRIPTEUR d'image (approche « VLM comme outil ») : quand le modèle actif ne voit pas,
    # read_image route vers ce VLM pour obtenir une description texte. On préfère un VLM
    # DISTANT (pas besoin du serveur local allumé), sinon un local mmproj. Vide => read_image
    # renverra une erreur claire à un modèle texte-only.
    vision_model_ids = [m.id for m in cfg.models if m.mmproj_filename] + [
        rm.id for rm in cfg.remote_models if rm.vision
    ]
    _describer_order = [rm.id for rm in cfg.remote_models if rm.vision] + [
        m.id for m in cfg.models if m.mmproj_filename
    ]
    describer_model = _describer_order[0] if _describer_order else ""

    # Factory : le registre est (re)construit selon les outils cochés dans l'UI pour la
    # conversation courante. `workspace` optionnel : à défaut, celui de la config.
    # `client`/`model` arment dispatch_agent (sous-boucle tool-use) ; `conversation` arme
    # manage_todos, dont le plan vit dans `conversation.todos` (par session, persisté).
    def make_registry(active, workspace=None, conversation=None):
        _model = conversation.model if conversation else cfg.default_model
        _describer = (
            (lambda uri, q: client.describe_image(uri, q, describer_model))
            if describer_model
            else None
        )
        return build_registry(
            workspace_dir=workspace or cfg.chat.workspace_dir,
            max_bytes=cfg.chat.read_file_max_bytes,
            enabled=active,
            web_cfg=cfg.chat.web_search,
            client=client,
            conversation=conversation,
            # dispatch_agent tourne sur le modèle SÉLECTIONNÉ (celui de la conversation),
            # pas sur un défaut figé au démarrage : sélectionner un modèle dans l'UI le
            # propage donc aussi aux sous-agents. Repli sur le défaut hors session.
            model=_model,
            sub_max_tokens=cfg.chat.max_tokens,
            permission=permission,
            active_model=_model,
            skills_dir=cfg.chat.skills_dir,
            plugins_root=plugins_dir,
            memory=memory,
            learned_skills_dir=cfg.chat.learned_skills_dir,
            shell_timeout=cfg.chat.shell_timeout,
            vision_describer=_describer,
            active_is_vision=(_model in vision_model_ids),
        )

    # Amorce les outils de la conversation depuis la config au 1er lancement.
    if not conversation.active_tools and cfg.chat.tools_enabled:
        conversation.set_tools(cfg.chat.tools_enabled)

    # Sessions first-class : un fil persistant par projet (chat + runs agentic partagés).
    # Migration douce : si aucune session n'existe mais qu'une ancienne conversation est
    # là, on l'importe comme première session pour ne pas perdre l'historique.
    data_root = Path(cfg.chat.history_path).resolve().parent

    from loom.extend.plugins import plugins_root as _plugins_root

    plugins_dir = str(_plugins_root(getattr(cfg.chat, "plugins_root", None)))

    # Modèles IMAGE (ComfyUI) : découverts par dossier models/_IMAGE/<id>/, sélectionnables
    # dans l'UI comme les LLM (un message = une image).
    from loom.runtime.image_models import discover_image_models

    image_models = discover_image_models()

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

    permission = lambda name, args: evaluate(name, args, cfg.permissions)  # noqa: E731
    # Fenêtre et plafond de sortie PAR MODÈLE : un modèle local garde sa fenêtre (override
    # model.toml sinon le global) ; un modèle distant exploite SA grande fenêtre + son
    # max_tokens. Sert au seuil de microcompact (côté app) -> on profite du contexte du
    # provider sans déborder le local.
    model_contexts = {m.id: (m.context or cfg.context) for m in cfg.models}
    model_contexts.update(
        {rm.id: (rm.context or cfg.context) for rm in cfg.remote_models}
    )
    model_max_tokens = {
        rm.id: rm.max_tokens for rm in cfg.remote_models if rm.max_tokens
    }
    # Modèles LOCAUX (découverts par dossier) : détails pour l'onglet Modèles locaux de la
    # console. `dir` porte le model.toml -> édition du tuning machine (offload) via tomlkit.
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
        available_tools=AVAILABLE_TOOLS,
        permission=permission,
        permission_mode=cfg.permissions.mode,
        workspace_dir=cfg.chat.workspace_dir,
        plugins_dir=plugins_dir,
        keepwarm_enabled=cfg.chat.keepwarm_enabled,
        keepwarm_interval=cfg.chat.keepwarm_interval,
        identity_paths=mem_paths,
        identity_max_tokens=cfg.chat.identity_max_tokens,
        learned_skills_dir=cfg.chat.learned_skills_dir,
        reflect_stores=reflect_stores,
        reflect_enabled=cfg.chat.reflect_enabled,
        reflect_min_actions=cfg.chat.reflect_min_actions,
        reflect_model=cfg.default_model,
        model_contexts=model_contexts,
        model_max_tokens=model_max_tokens,
        remote_model_ids=[rm.id for rm in cfg.remote_models],
        remote_model_names={rm.id: rm.model for rm in cfg.remote_models},
        # Prix ($/M tokens) par modèle distant : (input, output, cached) -> coût réel + mesure
        # de l'effet du cache de préfixe sur la session.
        model_prices={
            rm.id: (rm.price_in, rm.price_out, rm.price_cached)
            for rm in cfg.remote_models
        },
        remote_store_path=str(remote_store_path),
        config_defaults_path=str(CONFIG_PATH),
        config_local_path=str(PERSONAL_CONFIG_PATH),
        local_models=local_models,
        image_models=image_models,
    )
    return app


def main() -> None:
    cfg = load_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
    app = build_app(cfg)
    # Le moniteur système poll GET /sysmon toutes les ~1,2 s : sans ce filtre, le log
    # d'accès werkzeug est inondé et noie les requêtes utiles au debug.
    logging.getLogger("werkzeug").addFilter(
        lambda record: "/sysmon" not in record.getMessage()
    )
    print(
        f"[loom-chat] http://127.0.0.1:{cfg.chat.web_port}  (modèle: http://127.0.0.1:{cfg.port}/v1)"
    )
    # threaded=True : permet de détecter rapidement la déconnexion client
    # (interruption d'une génération par une nouvelle soumission) et de servir
    # la requête suivante pendant que l'ancien flux se ferme.
    app.run(host="127.0.0.1", port=cfg.chat.web_port, threaded=True)


if __name__ == "__main__":
    main()
