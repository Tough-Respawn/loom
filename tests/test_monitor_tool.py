from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from loom.tools.monitor import MonitorHub, _RunningMonitor, make_monitor


class _Proc:
    pid = 123
    returncode = None

    def poll(self):
        return self.returncode


def _running(tmp_path):
    return _RunningMonitor(
        id="mon1",
        session_id="session1",
        command="watch",
        description="build",
        timeout_s=300,
        persistent=False,
        log_path=tmp_path / "mon1.log",
        proc=_Proc(),
    )


def test_batches_are_queued_and_marked_untrusted(tmp_path):
    hub = MonitorHub(tmp_path)
    mon = _running(tmp_path)
    assert hub._emit_batch(mon, ["build ok", "tests ok"])

    events = hub.drain("session1")
    assert events[0]["text"] == "build ok\ntests ok"
    assert "événement du monitor « build »" in events[0]["model_content"]
    assert "FRONTIÈRE DE CONFIANCE" in events[0]["model_content"]
    assert hub.drain("session1") == []


def test_rate_limit_stops_a_chatty_monitor(tmp_path, monkeypatch):
    hub = MonitorHub(tmp_path)
    mon = _running(tmp_path)
    mon.event_times = deque([mon.started_at] * 30)
    killed = []
    monkeypatch.setattr("loom.tools.monitor._kill_tree", lambda proc: killed.append(proc))

    assert hub._emit_batch(mon, ["encore"]) is False
    assert killed == [mon.proc]
    event = hub.drain("session1")[0]
    assert event["final"] is True
    assert "trop bavard" in event["text"]


def test_monitor_tool_start_list_stop_contract(tmp_path):
    calls = []

    class FakeHub:
        def start(self, sid, command, description, workspace, **kwargs):
            calls.append((sid, command, description, workspace, kwargs))
            return SimpleNamespace(id="abc", log_path=tmp_path / "abc.log")

        def list(self, sid):
            return [{"id": "abc", "running": True, "session": sid}]

        def stop(self, sid, monitor_id):
            calls.append(("stop", sid, monitor_id))
            return True

    tool = make_monitor(FakeHub(), "sid", str(tmp_path))
    started = tool.run(
        {
            "action": "start",
            "command": "tail -f app.log",
            "description": "erreurs app",
            "timeout_s": 12,
            "persistent": False,
        }
    )
    assert "id=abc" in started
    assert calls[0][-1] == {"timeout_s": 12, "persistent": False}
    assert '"running": true' in tool.run({"action": "list"})
    assert "arrêté" in tool.run({"action": "stop", "monitor_id": "abc"})
