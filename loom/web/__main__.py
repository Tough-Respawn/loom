# loom/web/__main__.py
"""Point d'entrée : uv run python -m loom.web"""

from __future__ import annotations

from pathlib import Path

from loom.client import LoomClient
from loom.config import load_config
from loom.context import effective_context_budget
from loom.conversation import Conversation
from loom.permissions import evaluate
from loom.session import SessionStore
from loom.tools import AVAILABLE_TOOLS, build_registry
from loom.tools.todo import TodoStore
from loom.web.app import create_app

RUNTIME_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = RUNTIME_DIR / "loom.config.toml"
LOCAL_CONFIG_PATH = RUNTIME_DIR / "loom.config.local.toml"


def build_app(cfg):
    """Construit l'app Flask depuis la config (sans la servir). Séparé de `main` pour
    être testable / lançable sur un port arbitraire (vérif UI, tests d'intégration)."""
    base_url = f"http://127.0.0.1:{cfg.port}/v1"
    conversation = Conversation.load(cfg.chat.history_path, cfg.chat.system_prompt)
    if not conversation.model:
        conversation.set_model(cfg.default_model)
    client = LoomClient(
        base_url=base_url,
        timeout=cfg.chat.request_timeout,
        max_retries=cfg.chat.max_retries,
    )
    budget = effective_context_budget(
        cfg.chat.context_token_budget, cfg.context, cfg.chat.max_tokens
    )

    # Mémoire de travail des todos : un SEUL store partagé entre les reconstructions
    # du registre (une par tour) -> le plan du modèle survit d'un tour au suivant.
    todo_store = TodoStore()

    # Factory : le registre est (re)construit selon les outils cochés dans l'UI pour la
    # conversation courante. `workspace` optionnel : un /run peut cibler un autre dossier
    # (champ « dossier cible ») ; à défaut, le workspace de la config. `client`/`model`
    # arment dispatch_agent (sous-boucle tool-use), `todo_store` arme manage_todos.
    def make_registry(active, workspace=None):
        return build_registry(
            workspace_dir=workspace or cfg.chat.workspace_dir,
            extensions=cfg.chat.read_file_extensions,
            max_bytes=cfg.chat.read_file_max_bytes,
            enabled=active,
            web_cfg=cfg.chat.web_search,
            client=client,
            todo_store=todo_store,
            model=cfg.default_model,
            sub_max_tokens=cfg.chat.max_tokens,
        )

    # Amorce les outils de la conversation depuis la config au 1er lancement.
    if not conversation.active_tools and cfg.chat.tools_enabled:
        conversation.set_tools(cfg.chat.tools_enabled)

    # Sessions first-class : un fil persistant par projet (chat + runs agentic partagés).
    # Migration douce : si aucune session n'existe mais qu'une ancienne conversation est
    # là, on l'importe comme première session pour ne pas perdre l'historique.
    data_root = Path(cfg.chat.history_path).resolve().parent
    sessions_root = data_root / "sessions"
    store = SessionStore(
        sessions_root, cfg.chat.system_prompt, default_tools=cfg.chat.tools_enabled
    )
    if not store.list() and conversation.messages:
        seed = store.create(workspace=cfg.chat.workspace_dir, title="Session importée")
        seed.conversation = conversation
        store.save(seed)

    permission = lambda name, args: evaluate(name, args, cfg.permissions)  # noqa: E731
    app = create_app(
        conversation,
        client,
        cfg.chat.history_path,
        cfg.chat.skills_dir,
        max_tokens=cfg.chat.max_tokens,
        context_budget=budget,
        keep_recent=cfg.chat.keep_recent_messages,
        models=[m.id for m in cfg.models],
        tool_factory=make_registry,
        available_tools=AVAILABLE_TOOLS,
        permission=permission,
        workspace_dir=cfg.chat.workspace_dir,
        session_store=store,
    )
    return app


def main() -> None:
    cfg = load_config(CONFIG_PATH, LOCAL_CONFIG_PATH)
    app = build_app(cfg)
    print(
        f"[loom-chat] http://127.0.0.1:{cfg.chat.web_port}  (modèle: http://127.0.0.1:{cfg.port}/v1)"
    )
    # threaded=True : permet de détecter rapidement la déconnexion client
    # (interruption d'une génération par une nouvelle soumission) et de servir
    # la requête suivante pendant que l'ancien flux se ferme.
    app.run(host="127.0.0.1", port=cfg.chat.web_port, threaded=True)


if __name__ == "__main__":
    main()
