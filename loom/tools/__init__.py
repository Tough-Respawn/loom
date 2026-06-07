# loom/tools/__init__.py
"""Package outils de la boucle tool-use : API publique stable + assemblage du registre.

`from loom.tools import …` reste le point d'entrée (ToolError, ToolSpec, ToolRegistry,
AVAILABLE_TOOLS, build_registry, make_read_file). Les implémentations vivent dans des
sous-modules : base, read, fs, shell, web.
"""

from __future__ import annotations

from loom.tools.base import (
    AVAILABLE_TOOLS,
    ToolError,
    ToolRegistry,
    ToolSpec,
    _resolve_in_root,
)
from loom.tools.read import make_read_document, make_read_file, make_read_image

__all__ = [
    "AVAILABLE_TOOLS",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "_resolve_in_root",
    "build_registry",
    "build_subagent_registry",
    "make_read_document",
    "make_read_file",
    "make_read_image",
]

# Outils confiés à un SOUS-AGENT (dispatch_agent) : TOUT sauf dispatch_agent lui-même
# (anti-récursion) et manage_todos (le plan reste celui du fil principal). Le sous-agent
# peut écrire/exécuter — un ouvrier en lecture seule ne sert à rien ; la deny-list dure
# de run_shell et la politique de permission s'appliquent comme au fil principal.
_SUBAGENT_TOOLS = [
    "find_files",
    "search_text",
    "list_dir",
    "read_file",
    "read_document",
    "read_image",
    "web_search",
    "fetch_url",
    "write_file",
    "append_file",
    "edit_file",
    "replace_lines",
    "insert_lines",
    "format_code",
    "run_shell",
    "check_page",
]


def build_registry(
    workspace_dir: str,
    max_bytes: int,
    enabled: list[str],
    web_cfg=None,
    *,
    client=None,
    conversation=None,
    model: str | None = None,
    sub_max_tokens: int = 2048,
    permission=None,
    active_model: str | None = None,
    extra_specs: list[ToolSpec] | None = None,
) -> ToolRegistry:
    """Construit le registre selon la liste d'outils activés (config).

    `client`/`model` : requis pour dispatch_agent (lance une sous-boucle tool-use).
    `permission` : politique relayée à la sous-boucle (même sécurité qu'au principal).
    `conversation` : requise pour manage_todos (son plan vit dans `conversation.todos`,
    par session et persisté). Absente -> pas de manage_todos (cas du sous-agent).
    """
    # Imports locaux : les sous-modules d'écriture/shell/web importent `base`,
    # on les charge à la demande pour garder un graphe d'import simple.
    from loom.tools.fs import (
        make_append_file,
        make_edit_file,
        make_insert_lines,
        make_replace_lines,
        make_write_file,
    )
    from loom.tools.search import make_find_files, make_list_dir, make_search_text
    from loom.tools.shell import make_run_shell

    specs: list[ToolSpec] = []
    if "find_files" in enabled:
        specs.append(make_find_files(workspace_dir))
    if "search_text" in enabled:
        specs.append(make_search_text(workspace_dir))
    if "list_dir" in enabled:
        specs.append(make_list_dir(workspace_dir))
    if "read_file" in enabled:
        specs.append(make_read_file(workspace_dir, max_bytes))
    if "read_document" in enabled:
        specs.append(make_read_document(workspace_dir))
    if "read_image" in enabled:
        specs.append(make_read_image(workspace_dir))
    if "write_file" in enabled:
        specs.append(make_write_file(workspace_dir, max_bytes))
    if "append_file" in enabled:
        specs.append(make_append_file(workspace_dir, max_bytes))
    if "edit_file" in enabled:
        specs.append(make_edit_file(workspace_dir))
    if "replace_lines" in enabled:
        specs.append(make_replace_lines(workspace_dir))
    if "insert_lines" in enabled:
        specs.append(make_insert_lines(workspace_dir))
    if "format_code" in enabled:
        from loom.tools.format import make_format_code

        specs.append(make_format_code(workspace_dir))
    if "run_shell" in enabled:
        specs.append(make_run_shell(workspace_dir))
    if "check_page" in enabled:
        from loom.tools.browser import make_check_page

        specs.append(make_check_page(workspace_dir))
    if "web_search" in enabled or "fetch_url" in enabled:
        from loom.tools.web import WebSearchConfig, make_fetch_url, make_web_search

        wc = web_cfg or WebSearchConfig()
        if "web_search" in enabled:
            specs.append(make_web_search(wc))
        if "fetch_url" in enabled:
            specs.append(make_fetch_url(wc))
    if "manage_todos" in enabled and conversation is not None:
        from loom.tools.todo import make_manage_todos

        specs.append(make_manage_todos(conversation))
    if "dispatch_agent" in enabled and client is not None:
        from loom.prompts import SUBAGENT_SYSTEM
        from loom.tools.agent import make_dispatch_agent

        def _build_sub_registry() -> ToolRegistry:
            # Sous-registre complet SANS client -> pas de dispatch_agent imbriqué
            # (anti-récursion). Écriture/shell inclus : c'est un vrai ouvrier.
            return build_registry(
                workspace_dir,
                max_bytes,
                _SUBAGENT_TOOLS,
                web_cfg=web_cfg,
                active_model=active_model,
            )

        specs.append(
            make_dispatch_agent(
                client,
                _build_sub_registry,
                system_prompt=SUBAGENT_SYSTEM,
                model=model,
                max_tokens=sub_max_tokens,
                permission=permission,
            )
        )
    # Outils SUPPLÉMENTAIRES injectés par l'appelant (ex. report_verdict du harnais de
    # réflexion) : montés tels quels, hors univers AVAILABLE_TOOLS (non cochables).
    if extra_specs:
        specs.extend(extra_specs)

    from loom.models_profile import load_profile

    profile = load_profile(active_model) if active_model else None
    return ToolRegistry(specs, profile=profile)


def build_subagent_registry(
    workspace_dir: str,
    max_bytes: int,
    web_cfg=None,
    *,
    active_model: str | None = None,
    extra_specs: list[ToolSpec] | None = None,
) -> ToolRegistry:
    """Registre d'un sous-agent du HARNAIS de réflexion : les outils _SUBAGENT_TOOLS (donc
    PAS dispatch_agent ni manage_todos) + d'éventuels outils internes (report_verdict).
    Fabriqué frais à chaque tâche pour ne partager aucun état."""
    return build_registry(
        workspace_dir,
        max_bytes,
        _SUBAGENT_TOOLS,
        web_cfg=web_cfg,
        active_model=active_model,
        extra_specs=extra_specs,
    )
