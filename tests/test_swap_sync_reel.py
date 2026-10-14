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


@pytest.fixture()
def loom_side(swap, monkeypatch):
    """Les vraies fonctions de Loom (régénération, préalable) branchées sur ce
    llama-swap : la config suit la liste `ids` comme load_config suit les dossiers."""
    from types import SimpleNamespace

    from loom.agent.client import LoomClient
    from loom.runtime import serve

    ids = ["A", "B"]

    def cfg(*a, **k):
        return SimpleNamespace(
            models=[SimpleNamespace(id=m, cache_isolation=False) for m in ids],
            server_bin="llama-server",
            n_parallel=1,
        )

    monkeypatch.setattr(serve, "SWAP_YAML", swap.yaml)
    monkeypatch.setattr(serve, "load_config", cfg)
    monkeypatch.setattr("loom.config.load_config", cfg)
    monkeypatch.setattr(serve, "detect_hardware", lambda *a, **k: object())
    monkeypatch.setattr(
        serve,
        "_swap_config",
        lambda c, profile: yaml.safe_load(swap.text([m.id for m in c.models])),
    )
    S = SimpleNamespace(
        client=LoomClient(base_url=swap.root + "/v1"),
        config_defaults_path="defaults.toml",
        config_local_path="local.toml",
    )
    return SimpleNamespace(S=S, ids=ids)


def _status(url: str) -> int:
    import urllib.error

    try:
        _get(url)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def test_modifier_le_catalogue_ne_coupe_pas_le_modele_actif(swap, loom_side):
    from loom.web.routes.config import _regen_swap_yaml, _swap_preflight

    pid_a = _get(swap.root + "/upstream/A/health")["pid"]  # A démarre (la conversation)
    loom_side.ids[:] = ["A", "C"]  # B supprimé, C ajouté, pendant que A tourne
    assert _regen_swap_yaml(loom_side.S) == "deferred"
    time.sleep(3.0)  # plus que l'intervalle de surveillance de llama-swap (2 s)
    assert "reloading configuration" not in _get(swap.root + "/logs", raw=True)
    assert _get(f"http://127.0.0.1:{swap.ports['A']}/health")["pid"] == pid_a

    # Transition vers C : la config en attente est posée ET son chargement attendu.
    _swap_preflight(loom_side.S, "C")
    assert {m["id"] for m in _get(swap.root + "/v1/models")["data"]} == {"A", "C"}
    assert _status(swap.root + "/upstream/C/health") == 200
    assert not _alive(swap.ports["A"])  # arrêté par la transition, comme un swap normal


def test_sonde_en_echec_ne_coupe_pas_le_modele_actif(swap, loom_side, monkeypatch):
    # [P1 revue] /running en échec pendant que A tourne : rien n'est écrit.
    from loom.web.routes.config import _regen_swap_yaml

    pid_a = _get(swap.root + "/upstream/A/health")["pid"]
    monkeypatch.setattr(
        loom_side.S.client, "running_local", lambda timeout=5.0: (False, "")
    )
    loom_side.ids[:] = ["A"]
    assert _regen_swap_yaml(loom_side.S) == "deferred"
    time.sleep(3.0)
    assert _get(f"http://127.0.0.1:{swap.ports['A']}/health")["pid"] == pid_a


def test_ajout_ecrit_tout_de_suite_le_premier_appel_attend(swap, loom_side):
    # [P2 revue] Rien ne tourne : écriture immédiate. Sans attente, le premier appel
    # vers C partait vers l'ancien catalogue -> HTTP 404.
    from loom.web.routes.config import _regen_swap_yaml, _swap_preflight

    loom_side.ids[:] = ["A", "B", "C"]
    assert _regen_swap_yaml(loom_side.S) == "written"
    _swap_preflight(loom_side.S, "C")
    assert _status(swap.root + "/upstream/C/health") == 200


def test_sans_journal_le_catalogue_confirme_le_rechargement(
    swap, loom_side, monkeypatch
):
    # [P3 revue] /logs indisponible : la confirmation vient du catalogue /v1/models.
    from loom.web.routes.config import _regen_swap_yaml, _swap_preflight

    _get(swap.root + "/upstream/A/health")
    loom_side.ids[:] = ["A", "C"]
    assert _regen_swap_yaml(loom_side.S) == "deferred"
    monkeypatch.setattr(swap_sync, "fetch_logs", lambda root: None)
    _swap_preflight(loom_side.S, "C")
    assert _status(swap.root + "/upstream/C/health") == 200
