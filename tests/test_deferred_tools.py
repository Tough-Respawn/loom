from __future__ import annotations

import json

from loom.agent.conversation import Conversation
from loom.tools.base import ToolRegistry, ToolSpec


def _spec(name: str, *, deferred: bool = False) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Description complète de {name}. Détails longs.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        run=lambda args: f"ok:{args['value']}",
        deferred=deferred,
    )


def test_deferred_schema_load_keeps_openai_prefix_stable():
    loaded: set[str] = set()
    snapshots: list[set[str]] = []
    registry = ToolRegistry(
        [_spec("read_file"), _spec("check_page", deferred=True)],
        deferred_enabled=True,
        deferred_loaded=loaded,
        on_deferred_loaded=lambda names: snapshots.append(names),
    )

    before = registry.openai_tools()
    assert [t["function"]["name"] for t in before] == ["read_file", "tool_search"]
    assert "check_page — Description complète de check_page" in json.dumps(
        before, ensure_ascii=False
    )
    assert registry.run("check_page", {"value": "x"}) == (
        "erreur: outil différé 'check_page' — charge d'abord son schéma : "
        'tool_search(names=["check_page"]).'
    )

    result = json.loads(registry.run("tool_search", {"names": ["check_page"]}))
    assert result[0]["name"] == "check_page"
    assert result[0]["parameters"]["required"] == ["value"]
    assert snapshots == [{"check_page"}]
    assert registry.run("check_page", {"value": "x"}) == "ok:x"
    assert registry.openai_tools() == before


def test_kill_switch_off_exposes_every_full_schema():
    registry = ToolRegistry(
        [_spec("read_file"), _spec("check_page", deferred=True)],
        deferred_enabled=False,
    )
    assert [t["function"]["name"] for t in registry.openai_tools()] == [
        "read_file",
        "check_page",
    ]
    assert registry.run("check_page", {"value": "x"}) == "ok:x"


def test_deferred_loaded_survives_conversation_roundtrip():
    conv = Conversation(system_prompt="p", deferred_loaded=["check_page"])
    restored = Conversation.from_dict(conv.to_dict(), "fallback")
    assert restored.deferred_loaded == ["check_page"]
    restored.reset()
    assert restored.deferred_loaded == []


def test_deferred_prefix_is_smaller_than_all_full_schemas():
    specs = [_spec("read_file")] + [
        _spec(f"long_tail_{i}", deferred=True) for i in range(8)
    ]
    full = ToolRegistry(specs).openai_tools()
    deferred = ToolRegistry(specs, deferred_enabled=True).openai_tools()
    assert len(json.dumps(deferred, ensure_ascii=False)) < len(
        json.dumps(full, ensure_ascii=False)
    )


def test_tool_search_autorise_sans_confirmation_dans_tous_les_modes():
    """Session 468415b7be1e (2026-08-12) : tool_search hors catégories tombait dans
    le « ask » par prudence -> confirmation UI pour une simple lecture de catalogue.
    Classé READ_TOOLS : allow partout, comme code_outline/code_diagnostics."""
    from loom.permissions import PermissionConfig, evaluate

    for mode in ("allow", "ask", "allowlist", "deny_all"):
        d = evaluate(
            "tool_search", {"names": ["code_outline"]}, PermissionConfig(mode=mode)
        )
        assert d.action == "allow", f"tool_search en mode {mode} -> {d.action}"
