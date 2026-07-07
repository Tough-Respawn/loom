# loom/tools/web.py
"""Outil web_search : recherche en ligne opportuniste, dégradée hors-ligne.

Online-only : si le réseau est absent, l'outil renvoie un texte explicite
(« recherche indisponible (hors-ligne) ») SANS jamais lever — la boucle
tool-use continue normalement.

Dispatch de backend (`backend='auto'`) :
1. SearXNG si une URL d'instance est configurée ;
2. sinon Tavily si une clé API est fournie ;
3. sinon DuckDuckGo via la lib `ddgs` (fallback best-effort).

Les accès HTTP passent par `_http_get` / `_http_post` (httpx) et l'extraction
de page par `_extract` (trafilatura) : ces indirections rendent les tests
monkeypatchables sans aucun appel réseau réel.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from loom.tools._net import categorize_ip
from loom.tools.base import ToolError, ToolSpec
from loom.tools.trust import untrusted


def _resolve_validated(url: str) -> tuple[str | None, str | None]:
    """Anti-SSRF + anti DNS-rebinding. Résout l'hôte UNE seule fois et valide TOUTES ses
    IP (loopback, privée, link-local 169.254.x cloud-metadata, réservée, multicast,
    unspecified). Renvoie `(raison_blocage, ip_épinglée)` :
    - raison non-None -> bloqué (au moins une IP interne, ou hôte introuvable) ;
    - sinon -> ip_épinglée = l'adresse VALIDÉE à laquelle se connecter directement.

    Pinner l'IP ferme le TOCTOU du rebinding : sans ça, on valide le nom puis httpx le
    RE-résout au fetch, et un DNS hostile peut renvoyer une IP publique au contrôle puis
    une IP interne à la connexion. Ici l'IP validée EST celle utilisée."""
    host = urlparse(url).hostname
    if not host:
        return "url sans hôte valide", None
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return f"hôte introuvable : {host}", None
    pinned: str | None = None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        cat = categorize_ip(ip)
        if cat != "public":
            return f"hôte interdit (adresse interne/privée : {ip})", None
        if pinned is None:
            pinned = str(ip)
    return None, pinned


def _blocked_host_reason(url: str) -> str | None:
    """Raison de blocage anti-SSRF d'une URL, ou None si l'hôte est public. Conservé pour
    le refus EXPLICITE côté outils (fetch_url, check_page) ; le fetch réel pinne l'IP via
    `_resolve_validated`."""
    return _resolve_validated(url)[0]


@dataclass
class WebSearchConfig:
    enabled: bool = True
    backend: str = "auto"
    searxng_url: str = ""
    tavily_api_key: str = ""
    max_results: int = 5
    fetch_pages: bool = True
    http_timeout: int = 6
    max_chars_per_page: int = 4000


# --- indirections HTTP / extraction (points de monkeypatch) -------------


def _http_get(url, params=None, headers=None, timeout=None, pin_ip=None):
    """GET via httpx (isolé pour faciliter le monkeypatch en test).

    follow_redirects=False : un redirect 30x pourrait renvoyer vers une adresse interne
    (contournement du garde anti-SSRF). On ne suit aucun saut automatiquement.

    Si `pin_ip` est fourni (anti DNS-rebinding), on se connecte à CETTE IP déjà validée
    en préservant le Host et le SNI d'origine : httpx ne re-résout pas le nom d'hôte."""
    if pin_ip is None:
        return httpx.get(
            url, params=params, headers=headers, timeout=timeout, follow_redirects=False
        )
    parsed = urlparse(url)
    host = parsed.hostname or ""
    host_hdr = host if parsed.port is None else f"{host}:{parsed.port}"
    ip_lit = f"[{pin_ip}]" if ":" in pin_ip else pin_ip  # crochets pour l'IPv6
    ip_netloc = ip_lit if parsed.port is None else f"{ip_lit}:{parsed.port}"
    pinned_url = parsed._replace(netloc=ip_netloc).geturl()
    hdrs = {**(headers or {}), "Host": host_hdr}
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        req = httpx.Request(
            "GET",
            pinned_url,
            params=params,
            headers=hdrs,
            extensions={"sni_hostname": host},
        )
        return client.send(req)


def _http_post(url, json=None, headers=None, timeout=None):
    """POST via httpx (isolé pour faciliter le monkeypatch en test)."""
    return httpx.post(url, json=json, headers=headers, timeout=timeout)


def _extract(html: str) -> str | None:
    """Extrait le texte principal d'une page HTML via trafilatura."""
    import trafilatura

    return trafilatura.extract(html)


def _truncate(text: str, max_chars: int) -> str:
    """Tronque `text` à `max_chars` caractères avec un marqueur explicite."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[tronqué]"
    return text


# --- backends -----------------------------------------------------------


def _search_searxng(query: str, cfg: WebSearchConfig) -> list[dict]:
    """Interroge une instance SearXNG (format JSON)."""
    base = cfg.searxng_url.rstrip("/")
    resp = _http_get(
        f"{base}/search",
        params={"q": query, "format": "json"},
        headers={"Accept": "application/json"},
        timeout=cfg.http_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    out = []
    for r in data.get("results", [])[: cfg.max_results]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
        )
    return out


def _search_tavily(query: str, cfg: WebSearchConfig) -> list[dict]:
    """Interroge l'API Tavily (POST JSON)."""
    resp = _http_post(
        "https://api.tavily.com/search",
        json={
            "api_key": cfg.tavily_api_key,
            "query": query,
            "max_results": cfg.max_results,
        },
        headers={"Content-Type": "application/json"},
        timeout=cfg.http_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    out = []
    for r in data.get("results", [])[: cfg.max_results]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
        )
    return out


def _search_ddgs(query: str, cfg: WebSearchConfig) -> list[dict]:
    """Interroge DuckDuckGo via la lib `ddgs` (fallback best-effort)."""
    from ddgs import DDGS

    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=cfg.max_results):
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", "") or r.get("url", ""),
                    "snippet": r.get("body", "") or r.get("snippet", ""),
                }
            )
    return out


def _pick_backend(cfg: WebSearchConfig) -> str:
    """Sélectionne le backend effectif selon la config (mode 'auto')."""
    if cfg.backend != "auto":
        return cfg.backend
    if cfg.searxng_url:
        return "searxng"
    if cfg.tavily_api_key:
        return "tavily"
    return "ddgs"


# --- API publique -------------------------------------------------------


def web_search(query: str, cfg: WebSearchConfig) -> list[dict]:
    """Recherche en ligne ; renvoie une liste {title,url,snippet}.

    Dégrade en liste vide si le réseau est absent (jamais d'exception réseau).
    """
    backend = _pick_backend(cfg)
    try:
        if backend == "searxng":
            return _search_searxng(query, cfg)
        if backend == "tavily":
            return _search_tavily(query, cfg)
        return _search_ddgs(query, cfg)
    except (httpx.ConnectError, httpx.TimeoutException):
        return []


def fetch_page(url: str, cfg: WebSearchConfig, snippet: str = "") -> str:
    """Récupère et extrait le texte principal d'une page ; replie sur snippet."""
    reason, pin_ip = _resolve_validated(url)  # anti-SSRF + IP épinglée (anti-rebinding)
    if reason:  # hôte interne/introuvable : on ne fetch jamais
        return snippet
    try:
        resp = _http_get(url, timeout=cfg.http_timeout, pin_ip=pin_ip)
        resp.raise_for_status()
        text = _extract(resp.text)
    except (httpx.ConnectError, httpx.TimeoutException):
        return snippet
    if not text:
        return snippet
    return _truncate(text, cfg.max_chars_per_page)


def _format_results(query: str, results: list[dict], cfg: WebSearchConfig) -> str:
    """Formate les résultats en texte lisible pour le modèle."""
    lines = [f"Résultats pour : {query}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(r["url"])
        body = r.get("snippet", "")
        if cfg.fetch_pages and r.get("url"):
            body = fetch_page(r["url"], cfg, snippet=r.get("snippet", ""))
        if body:
            lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip()


def make_web_search(cfg: WebSearchConfig) -> ToolSpec:
    """Outil web_search : recherche en ligne, dégradé proprement hors-ligne."""

    def run(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            raise ToolError("argument 'query' manquant")
        results = web_search(query, cfg)
        if not results:
            return "recherche indisponible (hors-ligne) ou aucun résultat"
        return untrusted(
            _format_results(query, results, cfg), f"recherche web : {query}"
        )

    return ToolSpec(
        name="web_search",
        description=(
            "Searches the web for up-to-date information and returns the top "
            "results (title, URL, snippet). Returns an explicit message if the "
            "network is unavailable."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                }
            },
            "required": ["query"],
        },
        run=run,
    )


def make_fetch_url(cfg: WebSearchConfig) -> ToolSpec:
    """Outil fetch_url : récupère le TEXTE d'une URL précise (page web / doc en ligne)."""

    def run(args: dict) -> str:
        url = (args.get("url") or "").strip()
        if not url:
            raise ToolError("argument 'url' manquant")
        if not re.match(r"^https?://", url, re.IGNORECASE):
            raise ToolError("l'url doit commencer par http:// ou https://")
        blocked = _blocked_host_reason(url)  # anti-SSRF (refus explicite et clair)
        if blocked:
            raise ToolError(blocked)
        text = fetch_page(url, cfg)
        if not text:
            return "page indisponible (hors-ligne) ou sans contenu extractible"
        return untrusted(text, f"page web {url}")

    return ToolSpec(
        name="fetch_url",
        description=(
            "Fetches and returns the TEXT content of a specific URL (web page, "
            "online doc). Use it when you ALREADY have the URL. If you don't have "
            "a URL, run web_search first."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to read (http:// or https://).",
                }
            },
            "required": ["url"],
        },
        run=run,
    )
