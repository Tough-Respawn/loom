from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from loom.agent.conversation import Conversation
from loom.config import _parse_mcp_server
from loom.tools import build_registry
from loom.tools.mcp import McpHub, public_tool_name

SERVER = Path(__file__).with_name("fake_mcp_server.py")


def _config(**overrides):
    data = {
        "name": "équipe/outils",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER)],
        "enabled": True,
        "timeout_s": 2.0,
    }
    data.update(overrides)
    return _parse_mcp_server(data)


def _registry(hub, tmp_path, *, deferred=False):
    conv = Conversation("p")
    conv.runtime_session_id = "mcp-test"
    return build_registry(
        str(tmp_path),
        10_000,
        ["read_file"],
        conversation=conv,
        mcp_hub=hub,
        deferred_tools=deferred,
    )


def test_config_and_name_sanitizing():
    cfg = _config(env={"TOKEN": 123}, danger_override=False)
    assert cfg.env == {"TOKEN": "123"}
    assert cfg.danger_override is False
    assert public_tool_name("équipe/outils", "echo tool") == (
        "mcp_quipe_outils_echo_tool"
    )
    long_name = public_tool_name("s" * 80, "t" * 80)
    assert len(long_name) == 64
    assert long_name == public_tool_name("s" * 80, "t" * 80)


def test_stdio_handshake_deferred_schema_call_and_trust(tmp_path):
    hub = McpHub([_config()])
    try:
        registry = _registry(hub, tmp_path)
        exposed = [tool["function"]["name"] for tool in registry.openai_tools()]
        assert exposed == ["read_file", "tool_search"]
        public = public_tool_name("équipe/outils", "echo-tool")
        catalogue = registry.openai_tools()[1]["function"]["description"]
        assert public in catalogue
        assert "OUTIL TIERS" in catalogue
        assert registry.is_dangerous(public) is True

        blind = registry.run(public, {"text": "bonjour"})
        assert "charge d'abord son schéma" in blind
        schema = registry.run("tool_search", {"names": [public]})
        assert "description NON FIABLE" in schema
        assert '"text"' in schema

        result = registry.run(public, {"text": "bonjour"})
        assert result.startswith("écho tiers: bonjour")
        assert "FRONTIÈRE DE CONFIANCE" in result
        assert "résultat de l'outil MCP" in result

        failure = public_tool_name("équipe/outils", "fail")
        registry.run("tool_search", {"names": [failure]})
        failed_result = registry.run(failure, {})
        assert failed_result.startswith("erreur:")
        assert "FRONTIÈRE DE CONFIANCE" in failed_result
    finally:
        hub.close()


def test_trusted_server_marks_mcp_tools_safe(tmp_path):
    hub = McpHub([_config(danger_override=False)])
    try:
        registry = _registry(hub, tmp_path)
        public = public_tool_name("équipe/outils", "echo-tool")
        assert registry.is_dangerous(public) is False
    finally:
        hub.close()


def test_dead_server_never_blocks_registry_build(tmp_path):
    cfg = _config(command=str(tmp_path / "commande-absente"), timeout_s=0.2)
    hub = McpHub([cfg])
    started = time.monotonic()
    try:
        specs, unavailable, warnings = hub.build_specs()
        assert specs == [] and unavailable == {}
        assert warnings and "injoignable" in warnings[0]
        assert time.monotonic() - started < 2.0
    finally:
        hub.close()


def _pid_alive(pid: int) -> bool:
    """Sonde de vivacité PORTABLE d'un processus (test seulement).

    Windows : `os.kill(pid, 0)` ne sonde pas — il TERMINE un processus vivant
    (TerminateProcess) et lève OSError WinError 87 sur un PID mort. On sonde donc
    via OpenProcess + GetExitCodeProcess (STILL_ACTIVE=259). POSIX : kill(pid, 0)
    classique ; PermissionError = vivant mais pas à nous."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        # Signatures EXPLICITES : sans restype=HANDLE, ctypes infère un int 32
        # bits et peut tronquer un HANDLE sur un Windows 64 bits.
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False  # PID inexistant (ERROR_INVALID_PARAMETER, 87) -> mort
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_hung_handshake_is_bounded_and_process_is_reaped(tmp_path):
    pid_file = tmp_path / "mcp.pid"
    cfg = _config(
        args=[str(SERVER), "--hang-before-init"],
        env={"LOOM_MCP_TEST_PID_FILE": str(pid_file)},
        timeout_s=0.2,
    )
    hub = McpHub([cfg])
    try:
        started = time.monotonic()
        specs, _, warnings = hub.build_specs()
        assert specs == [] and warnings
        # Le SDK attend jusqu'à 2 s pour laisser le processus quitter après
        # fermeture de stdin avant SIGTERM/SIGKILL, deux tentatives comprises.
        assert time.monotonic() - started < 6.0
    finally:
        hub.close()

    pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            break
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text(encoding="utf-8").split()[2] == "Z":
            break  # terminé ; le reaping du zombie appartient encore à anyio
        time.sleep(0.02)
    else:
        pytest.fail(f"le processus MCP bloqué {pid} n'a pas été terminé")


def test_transport_unicode_sans_utf8_herite_du_parent(tmp_path):
    """HERMÉTICITÉ Windows : « écho » doit traverser le transport même quand
    l'environnement N'impose PAS l'UTF-8. On simule le PIRE cas hérité
    (PYTHONUTF8=0 + PYTHONIOENCODING=cp1252, prioritaires sur ceux du parent
    dans le merge d'env du hub) : c'est le fixture qui doit se reconfigurer en
    UTF-8 lui-même, jamais compter sur l'env du processus parent. Sans le
    reconfigure, « écho » arrive en mojibake et « ✓ » fait planter l'écriture."""
    cfg = _config(env={"PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"})
    hub = McpHub([cfg])
    try:
        registry = _registry(hub, tmp_path)
        echo = public_tool_name(cfg.name, "echo-tool")
        registry.run("tool_search", {"names": [echo]})
        out = registry.run(echo, {"text": "héhé ✓"})
        assert "écho tiers: héhé ✓" in out
    finally:
        hub.close()


def test_server_dies_during_session_then_cached_tools_become_unavailable(tmp_path):
    cfg = _config()
    hub = McpHub([cfg])
    try:
        registry = _registry(hub, tmp_path)
        die = public_tool_name(cfg.name, "die")
        registry.run("tool_search", {"names": [die]})
        assert registry.run(die, {}).startswith("erreur:")

        # Le prochain build tente un retry ; on rend volontairement la commande
        # indisponible pour vérifier le repli sur le tools/list mis en cache.
        cfg.command = str(tmp_path / "désormais-absent")
        specs, unavailable, warnings = hub.build_specs()
        assert specs == []
        assert die in unavailable
        assert "serveur MCP" in unavailable[die]
        assert warnings
    finally:
        hub.close()


def test_tool_call_timeout_is_bounded_and_actionable(tmp_path):
    cfg = _config(timeout_s=0.25)
    hub = McpHub([cfg])
    try:
        registry = _registry(hub, tmp_path)
        slow = public_tool_name(cfg.name, "slow")
        registry.run("tool_search", {"names": [slow]})
        started = time.monotonic()
        result = registry.run(slow, {"seconds": 3})
        elapsed = time.monotonic() - started
        assert result.startswith("erreur:")
        assert "expiré" in result or "injoignable" in result
        assert elapsed < 1.5
    finally:
        hub.close()


def test_http_transport_is_deferred_to_next_slice():
    cfg = SimpleNamespace(
        name="remote",
        transport="http",
        enabled=True,
        timeout_s=0.2,
        danger_override=None,
    )
    hub = McpHub([cfg])
    try:
        specs, unavailable, warnings = hub.build_specs()
        assert specs == [] and unavailable == {}
        assert "tranche ultérieure" in warnings[0]
    finally:
        hub.close()


def test_mcp_tool_crosses_the_real_chat_sse_path(tmp_path):
    """Banc E2E local : FakeOAI -> tool_search -> MCP stdio -> SSE."""
    from loom.agent.client import LoomClient
    from loom.agent.session import SessionStore
    from loom.web.app import create_app

    from .fakes import FakeOAI, turn_text, turn_tools

    model = "remote-mcp-test"
    cfg = _config(danger_override=False)
    public = public_tool_name(cfg.name, "echo-tool")
    fake = FakeOAI(
        [
            turn_tools([("search_1", "tool_search", json.dumps({"names": [public]}))]),
            turn_tools([("mcp_1", public, json.dumps({"text": "depuis SSE"}))]),
            turn_text("appel MCP terminé"),
        ]
    )
    client = LoomClient("http://127.0.0.1:9/v1")
    client.add_remote_route(
        model,
        {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "fake/mcp"},
    )
    client._routes[model]["client"] = fake
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=model,
        known_models=[model],
    )
    hub = McpHub([cfg])

    def tool_factory(_enabled, workspace, conversation):
        return build_registry(
            workspace,
            10_000,
            ["read_file"],
            conversation=conversation,
            mcp_hub=hub,
        )

    try:
        app = create_app(
            client=client,
            skills_dir=str(tmp_path / "skills"),
            session_store=store,
            models=[model],
            remote_model_ids=[model],
            keepwarm_enabled=False,
            workspace_dir=str(tmp_path),
            user_skills_dir=str(tmp_path / "skills_user"),
            plugins_dir=str(tmp_path / "plugins"),
            remote_store_path=str(tmp_path / "remote_models.json"),
            tool_factory=tool_factory,
        )
        web = app.test_client()
        created = web.post("/session/new", data={"title": "MCP E2E"})
        sid = created.get_json()["id"]
        response = web.post("/chat", data={"message": "teste MCP", "session_id": sid})
        events = [
            json.loads(line[6:])
            for line in response.data.decode("utf-8").splitlines()
            if line.startswith("data: ")
        ]
        result = next(
            event
            for event in events
            if event["type"] == "tool_result" and event["name"] == public
        )
        assert result["ok"] is True
        assert "écho tiers: depuis SSE" in result["preview"]
        assert "FRONTIÈRE DE CONFIANCE" in result["out_full"]
        assert events[-1]["type"] == "done"

        # Option A : le catalogue reste fixe, même après tool_search.
        for call in fake.calls:
            exposed = [tool["function"]["name"] for tool in call["tools"]]
            assert exposed == ["read_file", "tool_search"]
    finally:
        hub.close()
