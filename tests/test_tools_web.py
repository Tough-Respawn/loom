# tests/test_tools_web.py
import httpx
import pytest

import loom.tools.web as tools_web
from loom.tools import ToolError, build_registry
from loom.tools.web import (
    WebSearchConfig,
    available,
    fetch_page,
    make_web_search,
    web_search,
)

# --- SearXNG : json figé -> liste attendue ------------------------------


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _searxng_payload(n=8):
    return {
        "results": [
            {
                "title": f"Titre {i}",
                "url": f"https://ex.test/{i}",
                "content": f"extrait {i}",
            }
            for i in range(n)
        ]
    }


def test_searxng_returns_expected_list(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResp(_searxng_payload(8))

    monkeypatch.setattr(tools_web, "_http_get", fake_get)
    cfg = WebSearchConfig(
        backend="searxng", searxng_url="http://searx.test", max_results=3
    )
    out = web_search("python", cfg)
    assert len(out) == 3
    assert out[0]["title"] == "Titre 0"
    assert out[0]["url"] == "https://ex.test/0"
    assert out[0]["snippet"] == "extrait 0"


def test_searxng_respects_max_results(monkeypatch):
    monkeypatch.setattr(
        tools_web, "_http_get", lambda *a, **k: _FakeResp(_searxng_payload(20))
    )
    cfg = WebSearchConfig(
        backend="searxng", searxng_url="http://searx.test", max_results=5
    )
    out = web_search("x", cfg)
    assert len(out) == 5


# --- réseau absent -> dégradé structuré, available() sans exception -----


def test_web_search_offline_returns_structured(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(tools_web, "_http_get", boom)
    cfg = WebSearchConfig(backend="searxng", searxng_url="http://searx.test")
    out = web_search("x", cfg)
    assert out == []


def test_make_web_search_offline_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(tools_web, "_http_get", boom)
    cfg = WebSearchConfig(backend="searxng", searxng_url="http://searx.test")
    spec = make_web_search(cfg)
    out = spec.run({"query": "x"})
    assert "indisponible" in out.lower()
    assert "hors-ligne" in out.lower()


def test_make_web_search_timeout_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr(tools_web, "_http_get", boom)
    cfg = WebSearchConfig(backend="searxng", searxng_url="http://searx.test")
    spec = make_web_search(cfg)
    out = spec.run({"query": "x"})
    assert "indisponible" in out.lower()


def test_available_no_backend_configured():
    # auto sans url ni clé : ddgs reste un fallback best-effort -> True
    cfg = WebSearchConfig(backend="auto", searxng_url="", tavily_api_key="")
    assert available(cfg) is True


def test_available_searxng_without_url_is_false():
    # backend searxng explicite mais aucune url : indisponible, sans exception
    cfg = WebSearchConfig(backend="searxng", searxng_url="")
    assert available(cfg) is False


def test_available_tavily_without_key_is_false():
    cfg = WebSearchConfig(backend="tavily", tavily_api_key="")
    assert available(cfg) is False


def test_available_disabled():
    cfg = WebSearchConfig(enabled=False)
    assert available(cfg) is False


def test_available_searxng_when_url_set():
    cfg = WebSearchConfig(backend="searxng", searxng_url="http://searx.test")
    assert available(cfg) is True


# --- fetch_page : tronque + fallback snippet ----------------------------


def test_fetch_page_truncates(monkeypatch):
    monkeypatch.setattr(tools_web, "_http_get", lambda *a, **k: _FakeResp({}))

    class _FakeResp2:
        text = "page"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tools_web, "_http_get", lambda *a, **k: _FakeResp2())
    monkeypatch.setattr(tools_web, "_extract", lambda html: "B" * 9000)
    cfg = WebSearchConfig(max_chars_per_page=100)
    out = fetch_page("https://ex.test/p", cfg, snippet="fb")
    assert len(out) <= 100 + len("\n...[tronqué]")
    assert "[tronqué]" in out


def test_fetch_page_fallback_snippet(monkeypatch):
    class _FakeResp2:
        text = "page"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tools_web, "_http_get", lambda *a, **k: _FakeResp2())
    monkeypatch.setattr(tools_web, "_extract", lambda html: None)
    cfg = WebSearchConfig(max_chars_per_page=100)
    out = fetch_page("https://ex.test/p", cfg, snippet="mon extrait")
    assert out == "mon extrait"


def test_fetch_page_offline_fallback_snippet(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(tools_web, "_http_get", boom)
    cfg = WebSearchConfig(max_chars_per_page=100)
    out = fetch_page("https://ex.test/p", cfg, snippet="repli")
    assert out == "repli"


# --- auto-sélection backend selon config --------------------------------


def test_auto_selects_searxng_when_url(monkeypatch):
    chosen = {}

    def fake_searx(query, cfg):
        chosen["b"] = "searxng"
        return []

    monkeypatch.setattr(tools_web, "_search_searxng", fake_searx)
    cfg = WebSearchConfig(
        backend="auto", searxng_url="http://searx.test", tavily_api_key="k"
    )
    web_search("x", cfg)
    assert chosen["b"] == "searxng"


def test_auto_selects_tavily_when_key_no_url(monkeypatch):
    chosen = {}

    def fake_tav(query, cfg):
        chosen["b"] = "tavily"
        return []

    monkeypatch.setattr(tools_web, "_search_tavily", fake_tav)
    cfg = WebSearchConfig(backend="auto", searxng_url="", tavily_api_key="k")
    web_search("x", cfg)
    assert chosen["b"] == "tavily"


def test_auto_selects_ddgs_when_nothing(monkeypatch):
    chosen = {}

    def fake_ddgs(query, cfg):
        chosen["b"] = "ddgs"
        return []

    monkeypatch.setattr(tools_web, "_search_ddgs", fake_ddgs)
    cfg = WebSearchConfig(backend="auto", searxng_url="", tavily_api_key="")
    web_search("x", cfg)
    assert chosen["b"] == "ddgs"


# --- tavily backend : json figé -----------------------------------------


def test_tavily_returns_expected_list(monkeypatch):
    payload = {
        "results": [
            {"title": "T0", "url": "https://t.test/0", "content": "c0"},
            {"title": "T1", "url": "https://t.test/1", "content": "c1"},
        ]
    }

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(payload)

    monkeypatch.setattr(tools_web, "_http_post", fake_post)
    cfg = WebSearchConfig(backend="tavily", tavily_api_key="k", max_results=5)
    out = web_search("x", cfg)
    assert [r["title"] for r in out] == ["T0", "T1"]
    assert out[0]["snippet"] == "c0"


# --- make_web_search : résultats formatés en texte ----------------------


def test_make_web_search_formats_results(monkeypatch):
    monkeypatch.setattr(
        tools_web, "_http_get", lambda *a, **k: _FakeResp(_searxng_payload(2))
    )
    cfg = WebSearchConfig(
        backend="searxng", searxng_url="http://searx.test", fetch_pages=False
    )
    spec = make_web_search(cfg)
    out = spec.run({"query": "python"})
    assert "Titre 0" in out
    assert "https://ex.test/0" in out


def test_make_web_search_missing_query():
    cfg = WebSearchConfig(backend="searxng", searxng_url="http://searx.test")
    spec = make_web_search(cfg)
    with pytest.raises(ToolError):
        spec.run({})


# --- registre + schéma ---------------------------------------------------


def test_build_registry_registers_web_search(tmp_path):
    reg = build_registry(
        workspace_dir=str(tmp_path),
        extensions=[".py"],
        max_bytes=1000,
        enabled=["web_search"],
    )
    assert "web_search" in reg
    assert len(reg) == 1


def test_web_search_schema():
    cfg = WebSearchConfig(backend="searxng", searxng_url="http://searx.test")
    spec = make_web_search(cfg)
    props = spec.to_openai()["function"]["parameters"]["properties"]
    assert "query" in props
    assert spec.name == "web_search"


def test_module_exposes_httpx_for_monkeypatch():
    assert tools_web.httpx is httpx
