# fetch_url lisait tout le corps en mémoire, sans délai total : un fichier de
# plusieurs Go ou un serveur qui distille un octet par seconde bloquaient le tour.
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from loom.tools import web


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        try:
            if self.path == "/enorme":
                chunk = b"a" * 65536
                for _ in range(2000):  # ~130 Mo si personne ne s'arrête
                    self.wfile.write(chunk)
            elif self.path == "/goutte":
                for _ in range(600):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.05)
            else:
                self.wfile.write("été".encode())
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_corps_normal_inchange(server):
    resp = web._httpx_get(server + "/ok", timeout=5)
    assert resp.status_code == 200 and resp.text == "été"


def test_corps_enorme_plafonne(server, monkeypatch):
    monkeypatch.setattr(web, "_FETCH_MAX_BYTES", 1_000_000)
    resp = web._httpx_get(server + "/enorme", timeout=5)
    assert resp.status_code == 200
    assert len(resp.content) <= 1_000_000 + 65536


def test_serveur_goutte_a_goutte_coupe_au_delai_total(server, monkeypatch):
    monkeypatch.setattr(web, "_FETCH_TOTAL_S", 1.0)
    t0 = time.monotonic()
    with pytest.raises(httpx.TimeoutException):
        web._httpx_get(server + "/goutte", timeout=5)
    assert time.monotonic() - t0 < 3


def test_impersonate_corps_normal(server):
    resp = web._impersonate_get(server + "/ok", None, {}, 5, "chrome")
    assert resp.status_code == 200 and resp.text == "été"


def test_impersonate_corps_enorme_plafonne(server, monkeypatch):
    monkeypatch.setattr(web, "_FETCH_MAX_BYTES", 1_000_000)
    resp = web._impersonate_get(server + "/enorme", None, {}, 5, "chrome")
    assert len(resp.content) <= 1_000_000


def test_impersonate_goutte_a_goutte_coupe(server, monkeypatch):
    monkeypatch.setattr(web, "_FETCH_TOTAL_S", 1.0)
    t0 = time.monotonic()
    with pytest.raises(httpx.TimeoutException):
        web._impersonate_get(server + "/goutte", None, {}, 5, "chrome")
    assert time.monotonic() - t0 < 3


def _status_error(code):
    req = httpx.Request("GET", "http://searx.local/search")
    return httpx.HTTPStatusError(
        f"HTTP {code}", request=req, response=httpx.Response(code, request=req)
    )


def test_searxng_en_erreur_http_bascule_sur_ddgs_en_auto(monkeypatch):
    cfg = web.WebSearchConfig(searxng_url="http://searx.local")

    def boom(q, c):
        raise _status_error(429)

    monkeypatch.setattr(web, "_search_searxng", boom)
    monkeypatch.setattr(
        web, "_search_ddgs", lambda q, c: [{"title": "t", "url": "u", "snippet": "s"}]
    )
    assert web.web_search("q", cfg) == [{"title": "t", "url": "u", "snippet": "s"}]


def test_searxng_force_en_erreur_http_reste_explicite(monkeypatch):
    cfg = web.WebSearchConfig(backend="searxng", searxng_url="http://searx.local")

    def boom(q, c):
        raise _status_error(503)

    monkeypatch.setattr(web, "_search_searxng", boom)
    with pytest.raises(httpx.HTTPStatusError):
        web.web_search("q", cfg)
