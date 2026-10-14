# Le vrai llama-swap (var/runtime) avec des serveurs factices : modifier le catalogue
# pendant que A tourne ne doit PAS recharger llama-swap (A garde son processus, donc
# son cache) ; la config en attente s'applique à la transition vers un autre modèle.
# Ignoré si le binaire n'est pas installé.
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

from loom.runtime import swap_sync

EXE = (
    Path(__file__).resolve().parents[1]
    / "var"
    / "runtime"
    / "llama-swap"
    / "llama-swap.exe"
)
psutil = pytest.importorskip("psutil")
pytestmark = pytest.mark.skipif(not EXE.exists(), reason="llama-swap non installé")

WORKER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok", "pid": os.getpid()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, raw: bool = False):
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read().decode("utf-8", "replace")
    return body if raw else json.loads(body)


def _alive(port: int) -> bool:
    try:
        _get(f"http://127.0.0.1:{port}/health")
        return True
    except OSError:
        return False


@pytest.fixture()
def swap(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    ports = {m: _port() for m in ("A", "B", "C")}

    def entry(m):
        cmd = subprocess.list2cmdline([sys.executable, str(worker), str(ports[m])])
        return {"cmd": cmd, "proxy": f"http://127.0.0.1:{ports[m]}"}

    def text(models):
        return yaml.safe_dump(
            {"healthCheckTimeout": 15, "models": {m: entry(m) for m in models}}
        )

    y = tmp_path / "llama-swap.yaml"
    y.write_text(text(["A", "B"]), encoding="utf-8")
    sport = _port()
    proc = subprocess.Popen(
        [
            str(EXE),
            "--config",
            str(y),
            "--listen",
            f"127.0.0.1:{sport}",
            "--watch-config",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    root = f"http://127.0.0.1:{sport}"
    deadline = time.monotonic() + 15
    while True:
        try:
            _get(root + "/running")
            break
        except OSError:
            assert time.monotonic() < deadline and proc.poll() is None, (
                "llama-swap muet"
            )
            time.sleep(0.1)
    try:
        yield type(
            "Swap",
            (),
            {"root": root, "yaml": y, "text": staticmethod(text), "ports": ports},
        )
    finally:
        for child in psutil.Process(proc.pid).children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        proc.terminate()
        proc.wait(timeout=10)


def test_modifier_le_catalogue_ne_coupe_pas_le_modele_actif(swap):
    pid_a = _get(swap.root + "/upstream/A/health")["pid"]  # A démarre (la conversation)
    running = swap_sync.running_models(_get(swap.root + "/running", raw=True))
    assert running == {"A"}

    # Suppression de B puis ajout de C pendant que A tourne : tout est mis en attente.
    assert swap_sync.write_or_defer(swap.text(["A"]), swap.yaml, running) == "deferred"
    assert (
        swap_sync.write_or_defer(swap.text(["A", "C"]), swap.yaml, running)
        == "deferred"
    )
    time.sleep(3.0)  # plus que l'intervalle de surveillance de llama-swap (2 s)
    assert "reloading configuration" not in _get(swap.root + "/logs", raw=True)
    assert (
        _get(f"http://127.0.0.1:{swap.ports['A']}/health")["pid"] == pid_a
    )  # même processus

    # Transition vers C : la config en attente est appliquée, rechargement attendu.
    assert swap_sync.apply_pending(swap.yaml, root=swap.root, timeout=15) == "applied"
    ids = {m["id"] for m in _get(swap.root + "/v1/models")["data"]}
    assert ids == {"A", "C"}
    assert _get(swap.root + "/upstream/C/health")["status"] == "ok"
    assert not _alive(
        swap.ports["A"]
    )  # A arrêté par la transition, comme un swap normal
