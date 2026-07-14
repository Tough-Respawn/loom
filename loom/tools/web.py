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
from urllib.parse import urljoin, urlparse

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


# User-Agent navigateur par défaut : beaucoup de sites renvoient 403 au UA httpx nu.
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


def _with_ua(headers):
    """Ajoute un User-Agent navigateur si l'appelant n'en a pas fourni."""
    h = dict(headers or {})
    h.setdefault("User-Agent", _DEFAULT_UA)
    return h


def _http_get(url, params=None, headers=None, timeout=None, pin_ip=None):
    """GET via httpx (isolé pour faciliter le monkeypatch en test).

    follow_redirects=False : un redirect 30x pourrait renvoyer vers une adresse interne
    (contournement du garde anti-SSRF). On ne suit aucun saut automatiquement ; le suivi
    est fait par `fetch_page` en RE-VALIDANT chaque saut (anti-SSRF préservé).

    Si `pin_ip` est fourni (anti DNS-rebinding), on se connecte à CETTE IP déjà validée
    en préservant le Host et le SNI d'origine : httpx ne re-résout pas le nom d'hôte."""
    if pin_ip is None:
        return httpx.get(
            url,
            params=params,
            headers=_with_ua(headers),
            timeout=timeout,
            follow_redirects=False,
        )
    parsed = urlparse(url)
    host = parsed.hostname or ""
    host_hdr = host if parsed.port is None else f"{host}:{parsed.port}"
    ip_lit = f"[{pin_ip}]" if ":" in pin_ip else pin_ip  # crochets pour l'IPv6
    ip_netloc = ip_lit if parsed.port is None else f"{ip_lit}:{parsed.port}"
    pinned_url = parsed._replace(netloc=ip_netloc).geturl()
    hdrs = {**_with_ua(headers), "Host": host_hdr}
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
    try:
        from ddgs import DDGS
    except ImportError as exc:
        # `ddgs` est un EXTRA (pyproject [web-search]) : un venv synchronisé sans lui
        # cassait web_search avec un « No module named 'ddgs' » cryptique en pleine
        # session (vécu 2026-07-10). Message ACTIONNABLE pour l'humain ET le modèle.
        raise RuntimeError(
            "web_search indisponible : la lib `ddgs` n'est pas installée. "
            "Installe l'extra (`uv sync --extra web-search`) ou configure un backend "
            "searxng_url / tavily_api_key dans [web_search]."
        ) from exc

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
            try:
                return _search_searxng(query, cfg)
            except (httpx.ConnectError, httpx.TimeoutException):
                # Instance gérée arrêtée ? On la RELANCE (docker start, jamais de
                # pull) et on retente UNE fois ; sinon repli ddgs en mode auto —
                # SearXNG absent ne doit jamais priver l'utilisateur de recherche.
                from loom.runtime.searxng import ensure_running

                if ensure_running(cfg.searxng_url):
                    return _search_searxng(query, cfg)
                if cfg.backend == "auto":
                    return _search_ddgs(query, cfg)
                raise
        if backend == "tavily":
            return _search_tavily(query, cfg)
        return _search_ddgs(query, cfg)
    except (httpx.ConnectError, httpx.TimeoutException):
        return []


def fetch_page(
    url: str, cfg: WebSearchConfig, snippet: str = "", raise_status: bool = False
) -> str:
    """Récupère et extrait le texte principal d'une page ; replie sur snippet.

    `raise_status=True` (fetch_url) : une erreur HTTP devient une ToolError EXPLICITE
    (statut + URL d'ORIGINE — jamais l'URL à IP épinglée, illisible pour le modèle).
    Défaut False (web_search) : repli silencieux sur snippet, un résultat qui 403
    ne casse pas la recherche."""
    # Suivi des redirections MANUEL et RE-VALIDÉ : http->https, / final, apex->www sont
    # ultra-fréquents. Chaque saut repasse par _resolve_validated (anti-SSRF préservé :
    # un 30x vers une IP interne est bloqué). Borné à 5 sauts.
    try:
        for _hop in range(5):
            reason, pin_ip = _resolve_validated(url)
            if reason:  # hôte interne/introuvable : on ne fetch jamais
                return snippet
            resp = _http_get(url, timeout=cfg.http_timeout, pin_ip=pin_ip)
            if resp.is_redirect and resp.headers.get("location"):
                url = urljoin(url, resp.headers["location"])
                continue
            resp.raise_for_status()
            # Réponse JSON (API) : trafilatura n'extrait que de l'HTML et renverrait
            # vide -> on renvoie le corps BRUT (tronqué). Détection par content-type
            # ET par le premier caractère (des API servent du JSON en text/plain).
            ctype = (resp.headers.get("content-type") or "").lower()
            body = resp.text or ""
            if "json" in ctype or body.lstrip()[:1] in ("{", "["):
                return _truncate(body, cfg.max_chars_per_page)
            text = _extract(body)
            if not text:
                return snippet
            return _truncate(text, cfg.max_chars_per_page)
        return snippet  # trop de redirections
    except httpx.HTTPStatusError as e:
        if raise_status:
            # `url` = le saut courant sous sa forme NOM D'HÔTE (seul _http_get pinne
            # l'IP) : c'est elle qu'on montre, pas e.request.url (IP épinglée).
            code = e.response.status_code
            phrase = e.response.reason_phrase or ""
            extra = (
                " — accès refusé (anti-bot probable) : essaye une autre source"
                if code == 403
                else ""
            )
            raise ToolError(f"erreur HTTP {code} ({phrase}) sur {url}{extra}") from e
        return snippet
    except (httpx.ConnectError, httpx.TimeoutException):
        return snippet


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
        # Domaine nu (example.com, www.x.org/foo) : un modèle omet souvent le schéma.
        # On préfixe https:// au lieu de rejeter (le suivi de redirections gère le repli
        # http et apex/www). Un schéma non-http (file:, ftp:) reste refusé clairement.
        m = re.match(r"^([a-z][a-z0-9+.-]*)://", url, re.IGNORECASE)
        if not m:
            url = "https://" + url
        elif m.group(1).lower() not in ("http", "https"):
            raise ToolError("seuls http:// et https:// sont supportés")
        blocked = _blocked_host_reason(url)  # anti-SSRF (refus explicite et clair)
        if blocked:
            raise ToolError(blocked)
        text = fetch_page(url, cfg, raise_status=True)
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
