"""Client MCP stdio : pont async/thread et fusion des outils tiers dans Loom."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import hashlib
import os
import re
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from loom.tools.base import ToolError, ToolSpec
from loom.tools.trust import untrusted, untrusted_schema

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")
_MAX_PUBLIC_NAME = 64
_CONNECT_ATTEMPTS = 2


class McpBridgeError(RuntimeError):
    """Erreur de transport MCP rendue actionnable au registre/modèle."""


def _clean_part(value: str) -> str:
    clean = _SAFE_NAME.sub("_", str(value)).strip("_-")
    return clean or "outil"


def public_tool_name(server_name: str, tool_name: str) -> str:
    """Nom OpenAI-compatible, préfixé et borné sans collision par troncature."""
    raw = f"mcp_{_clean_part(server_name)}_{_clean_part(tool_name)}"
    if len(raw) <= _MAX_PUBLIC_NAME:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{raw[: _MAX_PUBLIC_NAME - 9]}_{digest}"


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str
    input_schema: dict


class _SdkStdioAdapter:
    """Fine enveloppe du SDK officiel, importé seulement si MCP est configuré."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self._stdio_cm = None
        self._stdio_entered = False
        self._process = None
        self._terminate_process_tree = None
        self._session = None
        self._session_entered = False

    async def connect(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import _terminate_process_tree, stdio_client
        except ImportError as exc:
            raise McpBridgeError(
                "SDK MCP absent — installe l'extra avec `uv sync --extra mcp`"
            ) from exc
        env = {**os.environ, **dict(self.spec.env or {})}
        params = StdioServerParameters(
            command=self.spec.command,
            args=list(self.spec.args or []),
            env=env,
            encoding="utf-8",
            encoding_error_handler="replace",
        )
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._stdio_entered = True
        # Le SDK ne rend pas le process public, mais son context manager le garde
        # dans le frame suspendu au `yield`. Le conserver permet d'appeler SON
        # terminateur d'arbre cross-platform si le handshake lui-même est figé.
        frame = getattr(self._stdio_cm.gen, "ag_frame", None)
        self._process = frame.f_locals.get("process") if frame is not None else None
        self._terminate_process_tree = _terminate_process_tree
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        self._session_entered = True
        await self._session.initialize()

    async def list_tools(self) -> list[McpToolInfo]:
        tools: list[McpToolInfo] = []
        cursor = None
        while True:
            result = await self._session.list_tools(cursor=cursor)
            tools.extend(
                McpToolInfo(
                    name=str(tool.name),
                    description=str(tool.description or "Outil MCP tiers."),
                    input_schema=dict(tool.inputSchema or {}),
                )
                for tool in result.tools
            )
            cursor = getattr(result, "nextCursor", None)
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict, timeout_s: float):
        return await self._session.call_tool(
            name,
            arguments=arguments,
            read_timeout_seconds=timedelta(seconds=timeout_s),
        )

    async def close(self, *, abort: bool = False) -> None:
        error = asyncio.CancelledError() if abort else None
        exc = (type(error), error, error.__traceback__) if error else (None, None, None)
        if abort and self._process is not None:
            try:
                await self._terminate_process_tree(
                    self._process,
                    timeout_seconds=min(0.5, self.spec.timeout_s),
                )
            except ProcessLookupError:
                pass
        try:
            if self._session is not None and self._session_entered:
                await self._session.__aexit__(*exc)
        finally:
            self._session_entered = False
            self._session = None
            if self._stdio_cm is not None and self._stdio_entered:
                try:
                    await self._stdio_cm.__aexit__(*exc)
                finally:
                    self._stdio_entered = False
            self._stdio_cm = None
            self._process = None
            self._terminate_process_tree = None


class _ServerThread:
    """Une boucle asyncio dédiée à un serveur, avec façade synchrone bornée."""

    def __init__(self, spec, adapter_factory) -> None:
        self.spec = spec
        self._adapter_factory = adapter_factory
        self._adapter = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._main_task: asyncio.Task | None = None
        self._ready: concurrent.futures.Future[list[McpToolInfo]] = (
            concurrent.futures.Future()
        )
        self._thread = threading.Thread(
            target=self._thread_main,
            daemon=True,
            name=f"loom-mcp-{_clean_part(spec.name)}",
        )
        self.error = ""

    def start(self) -> list[McpToolInfo]:
        self._thread.start()
        try:
            return self._ready.result(timeout=self.spec.timeout_s + 0.5)
        except concurrent.futures.TimeoutError as exc:
            self.error = f"handshake expiré après {self.spec.timeout_s:g}s"
            self.close(force=True)
            raise McpBridgeError(self.error) from exc
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            self.close()
            raise McpBridgeError(self.error) from exc

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._main_task = loop.create_task(self._serve())
            loop.run_until_complete(self._main_task)
        except BaseException as exc:  # noqa: BLE001 - frontière du thread
            self.error = str(exc) or type(exc).__name__
            if not self._ready.done():
                self._ready.set_exception(McpBridgeError(self.error))
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()
            self._loop = None
            self._main_task = None

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        adapter = self._adapter_factory(self.spec)
        self._adapter = adapter

        async def heartbeat() -> None:
            # Borne aussi le select() pendant le handshake. Certaines sandboxes
            # interdisent le self-pipe d'asyncio : sans timer, une annulation
            # cross-thread ne réveillerait alors jamais la boucle bloquée en I/O.
            while True:
                await asyncio.sleep(0.05)

        heartbeat_task = asyncio.create_task(heartbeat())
        abort = False
        try:
            async with asyncio.timeout(self.spec.timeout_s):
                await adapter.connect()
                tools = await adapter.list_tools()
            if not self._ready.done():
                self._ready.set_result(tools)
            # Une échéance courte garde la boucle réveillable même dans les
            # sandboxes qui refusent le self-pipe utilisé par
            # call_soon_threadsafe(). En environnement normal, elle ne change
            # pas le comportement et les appels restent pris immédiatement.
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=0.05)
                except TimeoutError:
                    pass
        except BaseException as exc:
            abort = True
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                self.error = str(exc) or type(exc).__name__
            raise
        finally:
            try:
                await adapter.close(abort=abort)
            finally:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
                self._adapter = None

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() and not self.error and self._adapter is not None

    async def _call(self, name: str, arguments: dict):
        async with asyncio.timeout(self.spec.timeout_s):
            return await self._adapter.call_tool(
                name, arguments, timeout_s=self.spec.timeout_s
            )

    def call(self, name: str, arguments: dict):
        loop = self._loop
        if loop is None or not self.alive:
            raise McpBridgeError(self.error or "connexion fermée")
        future = asyncio.run_coroutine_threadsafe(self._call(name, arguments), loop)
        try:
            return future.result(timeout=self.spec.timeout_s + 0.5)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise McpBridgeError(
                f"appel expiré après {self.spec.timeout_s:g}s"
            ) from exc
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            raise McpBridgeError(self.error) from exc

    def close(self, *, force: bool = False) -> None:
        loop, stop, main_task = self._loop, self._stop_event, self._main_task
        if loop is not None and stop is not None and loop.is_running():
            try:
                if force and main_task is not None:
                    loop.call_soon_threadsafe(main_task.cancel)
                else:
                    loop.call_soon_threadsafe(stop.set)
            except RuntimeError:
                pass
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(1.0, self.spec.timeout_s + 0.5))


class McpHub:
    """Singleton process : connexions paresseuses, cache tools/list et appels sync."""

    def __init__(self, servers: list, *, adapter_factory=None) -> None:
        self.servers = [server for server in servers if server.enabled]
        self._by_name = {server.name: server for server in self.servers}
        self._adapter_factory = adapter_factory or _SdkStdioAdapter
        self._guard = threading.RLock()
        self._connect_guard = threading.Lock()
        self._connections: dict[str, _ServerThread] = {}
        self._tool_cache: dict[str, list[McpToolInfo]] = {}
        self._errors: dict[str, str] = {}
        atexit.register(self.close)

    def _connect(self, spec) -> tuple[list[McpToolInfo] | None, str]:
        if spec.transport != "stdio":
            return None, "transport HTTP disponible dans une tranche ultérieure"
        last_error = "connexion impossible"
        for _ in range(_CONNECT_ATTEMPTS):
            conn = _ServerThread(spec, self._adapter_factory)
            try:
                tools = conn.start()
            except McpBridgeError as exc:
                last_error = str(exc)
                conn.close()
                continue
            with self._guard:
                old = self._connections.get(spec.name)
                self._connections[spec.name] = conn
                self._tool_cache[spec.name] = tools
                self._errors.pop(spec.name, None)
            if old is not None:
                old.close()
            return tools, ""
        with self._guard:
            self._errors[spec.name] = last_error
        return None, last_error

    def _tools_for(self, spec) -> tuple[list[McpToolInfo], str]:
        with self._connect_guard:
            with self._guard:
                conn = self._connections.get(spec.name)
                cached = list(self._tool_cache.get(spec.name, ()))
            if conn is not None and conn.alive:
                return cached, ""
            tools, error = self._connect(spec)
            return (tools if tools is not None else cached), error

    def build_specs(self) -> tuple[list[ToolSpec], dict[str, str], list[str]]:
        specs: list[ToolSpec] = []
        unavailable: dict[str, str] = {}
        warnings: list[str] = []
        public_seen: set[str] = set()
        for server in self.servers:
            tools, error = self._tools_for(server)
            if error:
                warnings.append(f"serveur MCP '{server.name}' injoignable : {error}")
            for tool in tools:
                public = public_tool_name(server.name, tool.name)
                if public in public_seen:
                    digest = hashlib.sha256(
                        f"{server.name}\0{tool.name}".encode()
                    ).hexdigest()[:8]
                    public = f"{public[: _MAX_PUBLIC_NAME - 9]}_{digest}"
                public_seen.add(public)
                if error:
                    unavailable[public] = (
                        f"serveur MCP '{server.name}' injoignable : {error}"
                    )
                    continue
                parameters = tool.input_schema
                if parameters.get("type") != "object":
                    parameters = {"type": "object", "properties": {}}

                def run(args: dict, *, _server=server.name, _tool=tool.name) -> str:
                    return self.call(_server, _tool, args)

                specs.append(
                    ToolSpec(
                        name=public,
                        description=untrusted_schema(
                            tool.description,
                            f"serveur MCP '{server.name}'",
                        ),
                        parameters=parameters,
                        run=run,
                        danger=(server.danger_override is not False),
                        deferred=True,
                        always_deferred=True,
                    )
                )
        return specs, unavailable, warnings

    @staticmethod
    def _text_result(result: Any) -> str:
        parts = [
            str(block.text)
            for block in (getattr(result, "content", None) or [])
            if getattr(block, "type", None) == "text"
        ]
        text = "\n".join(parts).strip() or "(aucun contenu texte renvoyé par MCP)"
        return text

    def call(self, server_name: str, tool_name: str, arguments: dict) -> str:
        with self._guard:
            conn = self._connections.get(server_name)
        if conn is None or not conn.alive:
            error = (conn.error if conn is not None else "connexion absente") or "panne"
            raise ToolError(
                f"serveur MCP '{server_name}' injoignable : {error}. "
                "Vérifie sa commande puis réessaie dans une nouvelle conversation."
            )
        try:
            result = conn.call(tool_name, arguments)
        except McpBridgeError as exc:
            raise ToolError(f"serveur MCP '{server_name}' injoignable : {exc}") from exc
        bounded = untrusted(
            self._text_result(result),
            f"résultat de l'outil MCP '{server_name}/{tool_name}'",
        )
        if bool(getattr(result, "isError", False)):
            raise ToolError(bounded)
        return bounded

    def close(self) -> None:
        with self._guard:
            connections = list(self._connections.values())
            self._connections = {}
        for conn in connections:
            conn.close()
