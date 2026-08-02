"""Package outils de la boucle tool-use : API publique stable + assemblage du registre.

`from loom.tools import …` reste le point d'entrée (ToolError, ToolSpec, ToolRegistry,
AVAILABLE_TOOLS, build_registry, make_read_file). Les implémentations vivent dans des
sous-modules : base, read, fs, shell, web.
"""

from __future__ import annotations

from loom.tools.base import AVAILABLE_TOOLS, ToolError, ToolRegistry, ToolSpec
from loom.tools.read import make_read_file, make_read_image

__all__ = [
    "AVAILABLE_TOOLS",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
    "make_read_file",
    "make_read_image",
]

# Les sous-agents excluent récursion, plan principal et administration des plugins.
# Garder prompts/subagent.system.md synchronisé avec ce kit.
_SUBAGENT_EXCLUDED = {
    "dispatch_agent",
    "run_workflow",
    "manage_todos",
    "list_plugins",
    "add_marketplace",
    "install_plugin",
    # Un monitor appartient au fil principal qui reçoit ses événements ; un
    # sous-agent éphémère ne possède ni file persistante ni UI où les injecter.
    "monitor",
}
_SUBAGENT_TOOLS = [
    t["name"] for t in AVAILABLE_TOOLS if t["name"] not in _SUBAGENT_EXCLUDED
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
    sub_compact_after_tokens: int | None = None,
    dispatch_models: list[str] | None = None,
    dispatch_local_only: bool = False,
    dispatch_model_roles: dict[str, str] | None = None,
    sub_compact_for=None,
    permission=None,
    active_model: str | None = None,
    skills_dir: str | None = None,
    plugins_root: str | None = None,
    memory=None,
    learned_skills_dir: str | None = None,
    user_skills_dir: str | None = None,
    shell_timeout: int = 180,
    vision_describer=None,
    active_is_vision: bool = True,
    deferred_tools: bool = False,
    monitor_hub=None,
    mcp_hub=None,
) -> ToolRegistry:
    """Construit le registre selon la liste d'outils activés (config).

    `client`/`model` : requis pour dispatch_agent (lance une sous-boucle tool-use).
    `permission` : politique relayée à la sous-boucle (même sécurité qu'au principal).
    `conversation` : requise pour manage_todos (son plan vit dans `conversation.todos`,
    par session et persisté). Absente -> pas de manage_todos (cas du sous-agent).
    """
    # Les imports tardifs évitent les cycles entre le registre et ses outils.
    from loom.tools.fs import (
        make_append_file,
        make_edit_file,
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
    # Préserver l'ancien alias après la fusion de la lecture de documents.
    if "read_file" in enabled or "read_document" in enabled:
        specs.append(make_read_file(workspace_dir, max_bytes))
    # Masquer le schéma vision aux modèles texte sans changer implicitement de modèle.
    if "read_image" in enabled and active_is_vision:
        specs.append(
            make_read_image(
                workspace_dir,
                describer=vision_describer,
                active_is_vision=active_is_vision,
            )
        )
    if "write_file" in enabled:
        specs.append(
            make_write_file(workspace_dir, max_bytes, max_tokens=sub_max_tokens)
        )
    if "append_file" in enabled:
        specs.append(
            make_append_file(workspace_dir, max_bytes, max_tokens=sub_max_tokens)
        )
    if "edit_file" in enabled:
        specs.append(make_edit_file(workspace_dir))
    if "calculate" in enabled:
        from loom.tools.calc import make_calculate

        specs.append(make_calculate(workspace_dir))
    if "current_date" in enabled:
        from loom.tools.clock import make_current_date

        specs.append(make_current_date())
    if "format_code" in enabled:
        from loom.tools.format import make_format_code

        specs.append(make_format_code(workspace_dir))
    if "run_shell" in enabled:
        specs.append(make_run_shell(workspace_dir, timeout=shell_timeout))
    if "monitor" in enabled and monitor_hub is not None and conversation is not None:
        from loom.tools.monitor import make_monitor

        specs.append(
            make_monitor(
                monitor_hub,
                session_id=getattr(conversation, "runtime_session_id", ""),
                workspace_dir=workspace_dir,
            )
        )
    if "check_page" in enabled:
        from loom.tools.browser import make_check_page

        specs.append(make_check_page(workspace_dir))
    if "serve_and_check" in enabled:
        from loom.tools.browser import make_serve_and_check

        specs.append(make_serve_and_check(workspace_dir))
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
    if conversation is not None:
        from loom.tools.note import make_read_note, make_write_note

        if "write_note" in enabled:
            specs.append(make_write_note(conversation))
        if "read_note" in enabled:
            specs.append(make_read_note(conversation))
    if memory is not None:
        from loom.tools.memory import make_recall, make_remember

        if "recall" in enabled:
            specs.append(
                make_recall(
                    memory.provider,
                    summarize=getattr(memory, "summarize", None),
                    threshold=getattr(memory, "threshold", 5),
                )
            )
        if "remember" in enabled:
            specs.append(make_remember(memory.provider, memory.paths))
    if (
        "dispatch_agent" in enabled or "run_workflow" in enabled
    ) and client is not None:
        from loom.prompts import SUBAGENT_SYSTEM
        from loom.runtime.platform_info import detect as _platform_detect
        from loom.tools.agent import SubAgentRunner, make_dispatch_agent

        # Le sous-agent doit suivre les conventions de l'OS du fil principal.
        _sub_system = SUBAGENT_SYSTEM + "\n\n" + _platform_detect().prompt_block()

        def _build_sub_registry() -> ToolRegistry:
            # Omettre le client interdit les dispatchs imbriqués.
            return build_registry(
                workspace_dir,
                max_bytes,
                _SUBAGENT_TOOLS,
                web_cfg=web_cfg,
                active_model=active_model,
                active_is_vision=active_is_vision,
                deferred_tools=deferred_tools,
                mcp_hub=mcp_hub,
            )

        # Dispatch et workflows partagent routage, cache et politique de permission.
        _runner = SubAgentRunner(
            client,
            _build_sub_registry,
            system_prompt=_sub_system,
            model=model,
            max_tokens=sub_max_tokens,
            permission=permission,
            compact_after_tokens=sub_compact_after_tokens,
            model_chain=dispatch_models,
            local_only=dispatch_local_only,
            compact_for=sub_compact_for,
            model_roles=dispatch_model_roles,
        )
        if "dispatch_agent" in enabled:
            specs.append(
                make_dispatch_agent(
                    client,
                    _build_sub_registry,
                    system_prompt=_sub_system,
                    model=model,
                    runner=_runner,
                )
            )
        if "run_workflow" in enabled:
            from loom.tools.workflow import make_run_workflow

            specs.append(
                make_run_workflow(
                    _runner,
                    workspace_dir,
                    # Un slot local unique sérialise nécessairement les tâches parallèles.
                    is_remote=bool(client.is_remote(model)),
                )
            )
    if "use_skill" in enabled and skills_dir is not None:
        from loom.extend.skills import collect_skills, effective_skills
        from loom.tools.skills import make_use_skill

        def _skills_provider() -> list:
            # Une session expose seulement ses skills actifs; un sous-agent voit le disque.
            all_skills = collect_skills(
                skills_dir,
                plugins_root,
                learned_dir=learned_skills_dir,
                user_dir=user_skills_dir,
            )
            if conversation is None:
                return all_skills
            return effective_skills(
                all_skills,
                overrides=getattr(conversation, "skill_overrides", None),
                disabled=getattr(conversation, "disabled_skills", None),
            )

        specs.append(make_use_skill(_skills_provider))
    if plugins_root is not None:
        from loom.tools.plugins import (
            make_add_marketplace,
            make_install_plugin,
            make_list_plugins,
        )

        if "add_marketplace" in enabled:
            specs.append(make_add_marketplace(plugins_root))
        if "install_plugin" in enabled:
            specs.append(make_install_plugin(plugins_root))
        if "list_plugins" in enabled:
            specs.append(make_list_plugins(plugins_root))

    mcp_unavailable: dict[str, str] = {}
    if mcp_hub is not None:
        mcp_specs, mcp_unavailable, mcp_warnings = mcp_hub.build_specs()
        specs.extend(mcp_specs)
        if mcp_warnings:
            from loom.agent.debuglog import log_event

            for warning in mcp_warnings:
                log_event("mcp.unavailable", level="WARN", msg=warning)

    from loom.runtime.models_profile import load_profile

    profile = load_profile(active_model) if active_model else None
    # Cœur toujours plein ; longue traîne consultable via tool_search. Le choix
    # n'agit que si le kill-switch est actif, donc le défaut reste bit-identique.
    core = {
        "find_files",
        "search_text",
        "list_dir",
        "read_file",
        "write_file",
        "append_file",
        "edit_file",
        "run_shell",
        "manage_todos",
    }
    if deferred_tools:
        for spec in specs:
            spec.deferred = spec.deferred or spec.name not in core
    else:
        for spec in specs:
            spec.deferred = spec.always_deferred

    loaded = set(getattr(conversation, "deferred_loaded", []) or [])

    def _save_loaded(names: set[str]) -> None:
        if conversation is not None:
            conversation.deferred_loaded = sorted(names)

    registry = ToolRegistry(
        specs,
        profile=profile,
        # Les outils MCP restent différés même si le kill-switch global des
        # outils natifs est coupé : un serveur peut en annoncer des dizaines.
        deferred_enabled=deferred_tools or any(s.always_deferred for s in specs),
        deferred_loaded=loaded,
        on_deferred_loaded=_save_loaded,
    )
    # read_image gaté hors vision : le prompt système et les consignes d'images jointes
    # peuvent le mentionner — un appel doit recevoir l'explication FRANCHE de
    # make_read_image (proposer un modèle VISION), pas un « outil inconnu » trompeur
    # (vécu 2026-07-19 : glm-flash annonçait « je peux lire les images » puis échouait).
    if "read_image" in enabled and not active_is_vision:
        from loom.tools.read import VISION_UNAVAILABLE

        registry.mark_unavailable("read_image", VISION_UNAVAILABLE)
    for name, reason in mcp_unavailable.items():
        registry.mark_unavailable(name, reason)
    return registry
