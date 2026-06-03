# loom/web/__main__.py
"""Point d'entrée : uv run python -m loom.web"""

from __future__ import annotations

from pathlib import Path

from loom.client import LoomClient
from loom.config import load_config
from loom.context import effective_context_budget
from loom.conversation import Conversation
from loom.permissions import evaluate
from loom.tools import AVAILABLE_TOOLS, build_registry
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

    # Factory : le registre est (re)construit selon les outils cochés dans l'UI pour la
    # conversation courante. `workspace` optionnel : un /run peut cibler un autre dossier
    # (champ « dossier cible ») ; à défaut, le workspace de la config.
    def make_registry(active, workspace=None):
        return build_registry(
            workspace_dir=workspace or cfg.chat.workspace_dir,
            extensions=cfg.chat.read_file_extensions,
            max_bytes=cfg.chat.read_file_max_bytes,
            enabled=active,
            web_cfg=cfg.chat.web_search,
        )

    # Vérificateur déterministe (P0.4) : vérifie EXACTEMENT les fichiers écrits par le
    # développeur (syntaxe + runtime web si index.html en fait partie). Borné à ce set
    # — ne rglob jamais un dossier, donc ne peut pas étouffer sur un arbre géant.
    def make_verifier(rel_paths, workspace=None):
        from loom.verify import verify_files

        root = Path(workspace or cfg.chat.workspace_dir).resolve()
        abs_paths = [str((root / p).resolve()) for p in rel_paths]
        return verify_files(abs_paths)

    # Amorce les outils de la conversation depuis la config au 1er lancement.
    if not conversation.active_tools and cfg.chat.tools_enabled:
        conversation.set_tools(cfg.chat.tools_enabled)

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
        agents=cfg.agents,
        pipeline=cfg.default_pipeline,
        max_revisions=cfg.chat.max_revisions,
        verifier=make_verifier,
        workspace_dir=cfg.chat.workspace_dir,
        server_context=cfg.context,
        n_parallel=cfg.n_parallel,
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
