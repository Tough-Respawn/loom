"""Processus de surveillance asynchrones, bornés et rattachés à une session."""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from loom.permissions import _is_hard_denied
from loom.runtime.platform_info import detect
from loom.tools.base import ToolError, ToolSpec
from loom.tools.shell import _kill_tree, _shell_argv
from loom.tools.trust import untrusted

_BATCH_GAP_S = 0.2
_MAX_BATCH_LINES = 100
_MAX_EVENTS_PER_MINUTE = 30
_MAX_PENDING_EVENTS = 100


@dataclass
class _RunningMonitor:
    id: str
    session_id: str
    command: str
    description: str
    timeout_s: int
    persistent: bool
    log_path: Path
    proc: subprocess.Popen
    started_at: float = field(default_factory=time.monotonic)
    stopped: threading.Event = field(default_factory=threading.Event)
    event_times: deque[float] = field(default_factory=deque)
    rate_stopped: bool = False


class MonitorHub:
    """Registre thread-safe des monitors et de leurs événements par session."""

    def __init__(self, logs_dir: str | Path) -> None:
        self.logs_dir = Path(logs_dir)
        self._guard = threading.RLock()
        self._monitors: dict[str, dict[str, _RunningMonitor]] = {}
        self._events: dict[str, deque[dict]] = {}
        atexit.register(self.stop_all)

    def _push(self, mon: _RunningMonitor, text: str, *, final: bool = False) -> None:
        event = {
            "id": uuid.uuid4().hex[:12],
            "monitor_id": mon.id,
            "description": mon.description,
            "text": text,
            "final": final,
        }
        with self._guard:
            q = self._events.setdefault(mon.session_id, deque())
            if len(q) >= _MAX_PENDING_EVENTS:
                q.popleft()
            q.append(event)

    def _emit_batch(self, mon: _RunningMonitor, lines: list[str]) -> bool:
        if not lines:
            return True
        if mon.stopped.is_set():
            return False
        now = time.monotonic()
        while mon.event_times and now - mon.event_times[0] >= 60:
            mon.event_times.popleft()
        if len(mon.event_times) >= _MAX_EVENTS_PER_MINUTE:
            mon.rate_stopped = True
            _kill_tree(mon.proc)
            self._push(
                mon,
                "monitor arrêté : trop bavard, resserre ton filtre",
                final=True,
            )
            return False
        mon.event_times.append(now)
        body = "\n".join(lines)
        if len(body) > 8000:
            body = body[:8000] + "\n...[lot tronqué]"
        self._push(mon, body)
        return True

    def _reader(self, mon: _RunningMonitor) -> None:
        lines: queue.Queue[tuple[float, str] | None] = queue.Queue()

        def pump() -> None:
            try:
                assert mon.proc.stdout is not None
                for raw in mon.proc.stdout:
                    lines.put((time.monotonic(), raw.rstrip("\r\n")))
            finally:
                lines.put(None)

        threading.Thread(
            target=pump,
            daemon=True,
            name=f"loom-monitor-stdout-{mon.id}",
        ).start()

        batch: list[str] = []
        last_line_at = 0.0
        ended = False
        timed_out = False
        deadline = None if mon.persistent else mon.started_at + mon.timeout_s
        while not ended:
            now = time.monotonic()
            if deadline is not None and now >= deadline and mon.proc.poll() is None:
                timed_out = True
                _kill_tree(mon.proc)
            try:
                item = lines.get(timeout=0.05)
            except queue.Empty:
                item = ...
            if item is None:
                ended = True
            elif item is not ...:
                stamp, line = item
                if batch and stamp - last_line_at >= _BATCH_GAP_S:
                    if not self._emit_batch(mon, batch):
                        batch = []
                        break
                    batch = []
                batch.append(line)
                last_line_at = stamp
                if len(batch) >= _MAX_BATCH_LINES:
                    if not self._emit_batch(mon, batch):
                        batch = []
                        break
                    batch = []
            elif batch and now - last_line_at >= _BATCH_GAP_S:
                if not self._emit_batch(mon, batch):
                    batch = []
                    break
                batch = []

        if batch and not mon.rate_stopped:
            self._emit_batch(mon, batch)
        try:
            mon.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_tree(mon.proc)
        if timed_out and not mon.rate_stopped:
            self._push(
                mon,
                f"monitor arrêté : timeout de {mon.timeout_s}s atteint",
                final=True,
            )
        elif (
            not mon.stopped.is_set()
            and not mon.rate_stopped
            and mon.proc.returncode not in (None, 0)
        ):
            self._push(
                mon,
                f"monitor terminé en erreur (exit {mon.proc.returncode}) ; "
                f"stderr disponible dans {mon.log_path}",
                final=True,
            )
        with self._guard:
            self._monitors.get(mon.session_id, {}).pop(mon.id, None)

    def start(
        self,
        session_id: str,
        command: str,
        description: str,
        workspace_dir: str,
        *,
        timeout_s: int = 300,
        persistent: bool = False,
    ) -> _RunningMonitor:
        if not session_id:
            raise ToolError("session absente : impossible de rattacher le monitor")
        if _is_hard_denied(command, []):
            raise ToolError("commande interdite par la politique de sécurité")
        monitor_id = uuid.uuid4().hex[:10]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{monitor_id}.log"
        popen_kwargs: dict = {}
        if not detect().is_windows:
            popen_kwargs["start_new_session"] = True
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            stderr_fh = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                _shell_argv(command),
                cwd=str(Path(workspace_dir)),
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                encoding="utf-8",
                errors="replace",
                env=env,
                **popen_kwargs,
            )
            stderr_fh.close()
        except OSError as exc:
            try:
                stderr_fh.close()
            except (NameError, OSError):
                pass
            raise ToolError(f"impossible de lancer le monitor : {exc}") from exc
        mon = _RunningMonitor(
            id=monitor_id,
            session_id=session_id,
            command=command,
            description=description,
            timeout_s=timeout_s,
            persistent=persistent,
            log_path=log_path,
            proc=proc,
        )
        with self._guard:
            self._monitors.setdefault(session_id, {})[monitor_id] = mon
        threading.Thread(
            target=self._reader,
            args=(mon,),
            daemon=True,
            name=f"loom-monitor-{monitor_id}",
        ).start()
        return mon

    def list(self, session_id: str) -> list[dict]:
        with self._guard:
            monitors = list(self._monitors.get(session_id, {}).values())
        return [
            {
                "id": mon.id,
                "description": mon.description,
                "command": mon.command,
                "persistent": mon.persistent,
                "running": mon.proc.poll() is None,
                "stderr_log": str(mon.log_path),
            }
            for mon in monitors
        ]

    def stop(self, session_id: str, monitor_id: str) -> bool:
        with self._guard:
            mon = self._monitors.get(session_id, {}).get(monitor_id)
        if mon is None:
            return False
        mon.stopped.set()
        if mon.proc.poll() is None:
            _kill_tree(mon.proc)
        return True

    def stop_session(self, session_id: str) -> int:
        with self._guard:
            ids = list(self._monitors.get(session_id, {}))
        for monitor_id in ids:
            self.stop(session_id, monitor_id)
        with self._guard:
            # Reset/suppression de session : aucun événement déjà en attente ne
            # doit ressusciter dans le prochain fil portant le même objet runtime.
            self._events.pop(session_id, None)
        return len(ids)

    def stop_all(self) -> None:
        with self._guard:
            session_ids = list(self._monitors)
        for session_id in session_ids:
            self.stop_session(session_id)

    def drain(self, session_id: str) -> list[dict]:
        with self._guard:
            events = list(self._events.get(session_id, ()))
            self._events[session_id] = deque()
        for event in events:
            event["model_content"] = untrusted(
                f"événement du monitor « {event['description']} » :\n{event['text']}",
                f"stdout du monitor {event['monitor_id']}",
            )
        return events


def make_monitor(hub: MonitorHub, session_id: str, workspace_dir: str) -> ToolSpec:
    """Outil start/list/stop lié au hub de la session courante."""

    def run(args: dict) -> str:
        action = args.get("action")
        if action == "list":
            return json.dumps(hub.list(session_id), ensure_ascii=False, indent=2)
        if action == "stop":
            monitor_id = (args.get("monitor_id") or "").strip()
            if not monitor_id:
                raise ToolError("'monitor_id' manquant pour action='stop'")
            if not hub.stop(session_id, monitor_id):
                raise ToolError(f"monitor inconnu ou déjà terminé : {monitor_id}")
            return f"monitor {monitor_id} arrêté (process et descendance tués)"
        if action != "start":
            raise ToolError("action attendue : start, list ou stop")
        command = (args.get("command") or "").strip()
        description = (args.get("description") or "").strip()
        if not command:
            raise ToolError("'command' manquant pour action='start'")
        if not description:
            raise ToolError("'description' manquante pour action='start'")
        timeout_s = max(1, min(int(args.get("timeout_s", 300)), 3600))
        persistent = bool(args.get("persistent", False))
        mon = hub.start(
            session_id,
            command,
            description,
            workspace_dir,
            timeout_s=timeout_s,
            persistent=persistent,
        )
        duration = "durée de la session" if persistent else f"{timeout_s}s max"
        return (
            f"monitor démarré : id={mon.id}, {duration}. "
            f"stderr: {mon.log_path}. Les lignes stdout seront injectées en événements."
        )

    return ToolSpec(
        name="monitor",
        description=(
            "Surveille une commande en arrière-plan : chaque ligne stdout devient un "
            "événement dans la conversation. Actions: start, list, stop. Pour une seule "
            "notification attendue, utilise run_shell, pas monitor. Chaque étage d'un pipe "
            "doit flusher (ex. grep --line-buffered). Le filtre doit couvrir les états "
            "d'échec, pas seulement le succès : le silence ressemble à « toujours en cours ». "
            "Un monitor trop bavard est arrêté automatiquement. stderr va dans un fichier "
            "de log et n'est jamais injecté."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "list", "stop"],
                },
                "command": {"type": "string", "description": "Commande à surveiller."},
                "description": {
                    "type": "string",
                    "description": "Libellé affiché dans chaque événement.",
                },
                "monitor_id": {
                    "type": "string",
                    "description": "Identifiant requis pour stop.",
                },
                "timeout_s": {
                    "type": "integer",
                    "default": 300,
                    "minimum": 1,
                    "maximum": 3600,
                },
                "persistent": {
                    "type": "boolean",
                    "default": False,
                    "description": "Si vrai, reste actif jusqu'à la fin de la session.",
                },
            },
            "required": ["action"],
        },
        run=run,
        danger=True,
        deferred=True,
    )
