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
from loom.tools.read import make_read_document, make_read_file

__all__ = [
    "AVAILABLE_TOOLS",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "_resolve_in_root",
    "build_registry",
    "make_read_document",
    "make_read_file",
]


def build_registry(
    workspace_dir: str,
    extensions: list[str],
    max_bytes: int,
    enabled: list[str],
    web_cfg=None,
) -> ToolRegistry:
    """Construit le registre selon la liste d'outils activés (config)."""
    # Imports locaux : les sous-modules d'écriture/shell/web importent `base`,
    # on les charge à la demande pour garder un graphe d'import simple.
    from loom.tools.fs import make_edit_file, make_write_file
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
        specs.append(make_read_file(workspace_dir, extensions, max_bytes))
    if "read_document" in enabled:
        specs.append(make_read_document(workspace_dir))
    if "write_file" in enabled:
        specs.append(make_write_file(workspace_dir, max_bytes))
    if "edit_file" in enabled:
        specs.append(make_edit_file(workspace_dir))
    if "run_shell" in enabled:
        specs.append(make_run_shell(workspace_dir))
    if "web_search" in enabled or "fetch_url" in enabled:
        from loom.tools.web import WebSearchConfig, make_fetch_url, make_web_search

        wc = web_cfg or WebSearchConfig()
        if "web_search" in enabled:
            specs.append(make_web_search(wc))
        if "fetch_url" in enabled:
            specs.append(make_fetch_url(wc))
    return ToolRegistry(specs)
