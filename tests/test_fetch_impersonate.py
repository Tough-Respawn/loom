# Impersonation TLS (curl_cffi) + multi-URL concurrent sur fetch_url (2026-07-15).
# Spike réel : SeLoger/Datadome renvoie 403 à httpx ET au Chromium headless, mais 200
# (~1 Mo d'annonces) à curl_cffi impersonate=chrome ; 15 villes en parallèle = 3,8 s.
# Le branchement est au primitif partagé (_http_get/fetch_page) -> fetch_url ET
# web_search en profitent. L'anti-SSRF (validation d'hôte + pin IP) est PRÉSERVÉ.
from __future__ import annotations

import pytest

import loom.tools.web as web
from loom.tools.base import ToolError


class _Resp:
    def __init__(self, status=200, text="OK", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {"content-type": "text/html"}


def test_impersonate_route_vers_curl_cffi(monkeypatch):
    seen = {}

    def fake_imp(url, pin_ip, headers, timeout, impersonate):
        seen["url"] = url
        seen["impersonate"] = impersonate
        seen["pin"] = pin_ip
        return _Resp(200, "<html>annonce</html>")

    monkeypatch.setattr(web, "_resolve_validated", lambda u: (None, "1.2.3.4"))
    monkeypatch.setattr(web, "_impersonate_get", fake_imp)
    # httpx ne doit PAS être utilisé quand impersonate est demandé
    monkeypatch.setattr(
        web,
        "_httpx_get",
        lambda *a, **k: pytest.fail("httpx utilisé au lieu de curl_cffi"),
    )
    fu = web.make_fetch_url(web.WebSearchConfig())
    out = fu.run({"url": "https://www.seloger.com/x", "impersonate": "chrome"})
    assert "annonce" in out
    assert seen["impersonate"] == "chrome"
    assert seen["pin"] == "1.2.3.4"  # pin IP transmis -> anti-rebinding préservé


def test_impersonate_preserve_anti_ssrf(monkeypatch):
    # Hôte interne : REJETÉ (ToolError) AVANT tout fetch, même avec impersonate.
    monkeypatch.setattr(
        web, "_resolve_validated", lambda u: ("hôte interdit (127.0.0.1)", None)
    )
    monkeypatch.setattr(
        web,
        "_impersonate_get",
        lambda *a, **k: pytest.fail("fetch effectué sur un hôte interne"),
    )
    fu = web.make_fetch_url(web.WebSearchConfig())
    with pytest.raises(ToolError) as e:
        fu.run({"url": "http://169.254.169.254/latest", "impersonate": "chrome"})
    assert "interdit" in str(e.value).lower()


def test_multi_urls_concurrent(monkeypatch):
    # `urls` (liste) : chaque page récupérée, résultats recollés avec un en-tête par URL.
    calls = []

    def fake_page(url, cfg, snippet="", raise_status=False, impersonate=None):
        calls.append((url, impersonate))
        return f"contenu de {url}"

    monkeypatch.setattr(web, "fetch_page", fake_page)
    monkeypatch.setattr(
        web, "_blocked_host_reason", lambda u: None
    )  # SSRF ok (testé ailleurs)
    fu = web.make_fetch_url(web.WebSearchConfig())
    out = fu.run(
        {
            "urls": [
                "https://a.example/1",
                "https://a.example/2",
                "https://a.example/3",
            ],
            "impersonate": "chrome",
        }
    )
    assert "contenu de https://a.example/1" in out
    assert "contenu de https://a.example/3" in out
    assert out.count("http") >= 3  # un en-tête par URL
    assert all(imp == "chrome" for _, imp in calls)  # impersonate propagé à toutes


def test_http_get_impersonate_delegue(monkeypatch):
    # Le primitif partagé route vers curl_cffi quand impersonate est fourni,
    # vers httpx sinon -> web_search (qui passe par _http_get) en bénéficie aussi.
    monkeypatch.setattr(web, "_impersonate_get", lambda *a, **k: _Resp(200, "imp"))
    monkeypatch.setattr(web, "_httpx_get", lambda *a, **k: _Resp(200, "httpx"))
    assert web._http_get("https://x/y", impersonate="chrome").text == "imp"
    assert web._http_get("https://x/y").text == "httpx"


def test_categorize_ip_nat64_depaquette_ipv4_embarquee():
    # RFC 6052 : sur réseau IPv6-only/DNS64, tout site IPv4-only se résout en
    # 64:ff9b::<ipv4>. La catégorie est celle de l'IPv4 EMBARQUÉE : un site
    # public reste public (Python marque le préfixe is_reserved -> il était
    # bloqué à tort), une cible interne embarquée reste bloquée (anti-SSRF).
    import ipaddress

    from loom.tools._net import categorize_ip

    ip = lambda s: ipaddress.ip_address(s)  # noqa: E731
    assert categorize_ip(ip("64:ff9b::8c52:7904")) == "public"  # 140.82.121.4
    assert categorize_ip(ip("64:ff9b::7f00:1")) == "loopback"  # 127.0.0.1
    assert categorize_ip(ip("64:ff9b::c0a8:101")) == "private"  # 192.168.1.1
    assert categorize_ip(ip("64:ff9b::a9fe:a9fe")) == "link-local"  # 169.254.169.254
