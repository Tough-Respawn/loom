"""Outils d'installation de plugins (appelables par le modèle, sous garde de permission) :
add_marketplace / install_plugin clonent du code tiers ; list_plugins est en lecture seule."""

from __future__ import annotations

from loom.extend import plugins as store
from loom.tools.base import ToolError, ToolSpec


def make_add_marketplace(root) -> ToolSpec:
    def run(args: dict) -> str:
        source = (args.get("source") or "").strip()
        if not source:
            raise ToolError("argument 'source' manquant (URL git ou chemin local)")
        try:
            name = store.marketplace_add(root, source)
        except store.PluginError as exc:
            raise ToolError(str(exc)) from exc
        return f"Marketplace ajoutée : {name}"

    return ToolSpec(
        name="add_marketplace",
        description=(
            "Adds a Claude Code plugin marketplace (git URL or local path) to Loom's "
            "store. Prerequisite for install_plugin."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Git URL or local path."}
            },
            "required": ["source"],
        },
        run=run,
    )


def make_install_plugin(root) -> ToolSpec:
    def run(args: dict) -> str:
        ref = (args.get("ref") or "").strip()
        if not ref:
            raise ToolError(
                "argument 'ref' manquant ('<plugin>' ou '<plugin>@<marketplace>')"
            )
        try:
            info = store.plugin_install(root, ref)
        except store.PluginError as exc:
            raise ToolError(str(exc)) from exc
        return (
            f"Plugin installé : {info['marketplace']}/{info['name']} ({info['version']}). "
            "Ses skills sont désormais au catalogue."
        )

    return ToolSpec(
        name="install_plugin",
        description=(
            "Installs a plugin from an added marketplace. ref = '<plugin>' or "
            "'<plugin>@<marketplace>'. Makes its skills available via use_skill."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "<plugin> ou <plugin>@<marketplace>",
                }
            },
            "required": ["ref"],
        },
        run=run,
    )


def make_list_plugins(root) -> ToolSpec:
    def run(args: dict) -> str:
        plugins = store.discover_plugins(root)
        if not plugins:
            return "Aucun plugin installé."
        return "\n".join(
            f"{p.marketplace}/{p.name} ({p.version}) — skills:{len(p.skills)} "
            f"agents:{len(p.agents)} hooks:{len(p.hooks)} commands:{len(p.commands)}"
            for p in plugins
        )

    return ToolSpec(
        name="list_plugins",
        description="Lists installed plugins and the inventory of their components.",
        parameters={"type": "object", "properties": {}},
        run=run,
    )
