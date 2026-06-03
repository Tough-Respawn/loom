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

from dataclasses import dataclass

import httpx

from loom.tools.base import ToolError, ToolSpec


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


def _http_get(url, params=None, headers=None, timeout=None):
    """GET via httpx (isolé pour faciliter le monkeypatch en test)."""
    return httpx.get(url, params=params, headers=headers, timeout=timeout)


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
    try:
        resp = _http_get(url, timeout=cfg.http_timeout)
        resp.raise_for_status()
        text = _extract(resp.text)
    except (httpx.ConnectError, httpx.TimeoutException):
        return snippet
    if not text:
        return snippet
    return _truncate(text, cfg.max_chars_per_page)


def available(cfg: WebSearchConfig) -> bool:
    """Indique si la recherche est utilisable (sans jamais lever)."""
    if not cfg.enabled:
        return False
    backend = _pick_backend(cfg)
    if backend == "searxng":
        return bool(cfg.searxng_url)
    if backend == "tavily":
        return bool(cfg.tavily_api_key)
    return True


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
        return _format_results(query, results, cfg)

    return ToolSpec(
        name="web_search",
        description=(
            "Recherche des informations à jour sur le web et renvoie les "
            "principaux résultats (titre, URL, extrait). Renvoie un message "
            "explicite si le réseau est indisponible."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Requête de recherche en langage naturel.",
                }
            },
            "required": ["query"],
        },
        run=run,
    )
