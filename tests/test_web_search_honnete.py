# Recherche impossible ≠ aucun résultat. Vécu 2026-10-17 : SearXNG répondait HTTP 200 avec
# zéro résultat (moteurs en CAPTCHA ou 429) ; l'outil disait « hors-ligne ou aucun
# résultat » sans essayer DDGS, et le modèle relançait en boucle.
from types import SimpleNamespace

import httpx
import pytest

from loom.tools import web
from loom.tools.base import ToolError

CFG = web.WebSearchConfig(searxng_url="http://searx.local")
HIT = [{"title": "t", "url": "https://u.test", "snippet": "s"}]
BLOCKED = {
    "results": [],
    "unresponsive_engines": [["brave", "too many requests"], ["duckduckgo", "CAPTCHA"]],
}


def _searxng_answers(monkeypatch, data):
    resp = SimpleNamespace(
        status_code=200, raise_for_status=lambda: None, json=lambda: data
    )
    monkeypatch.setattr(web, "_http_get", lambda *a, **k: resp)


def test_moteurs_bloques_repli_ddgs_signale_au_modele(monkeypatch):
    _searxng_answers(monkeypatch, BLOCKED)
    monkeypatch.setattr(web, "_search_ddgs", lambda q, c: HIT)
    notes = []
    assert web.web_search("q", CFG, notes) == HIT
    assert "brave : too many requests" in notes[0] and "DDGS" in notes[0]


def test_tout_en_echec_erreur_qui_nomme_la_cause(monkeypatch):
    _searxng_answers(monkeypatch, BLOCKED)
    monkeypatch.setattr(web, "_search_ddgs", lambda q, c: [])
    with pytest.raises(ToolError) as err:
        web.make_web_search(CFG).run({"query": "q"})
    assert "duckduckgo : CAPTCHA" in str(err.value)
    assert "pas « aucun résultat »" in str(err.value)


def test_vraie_absence_de_resultat_sans_repli(monkeypatch):
    _searxng_answers(monkeypatch, {"results": [], "unresponsive_engines": []})
    monkeypatch.setattr(web, "_search_ddgs", lambda q, c: pytest.fail("repli inutile"))
    assert web.make_web_search(CFG).run({"query": "zzz"}) == "aucun résultat pour : zzz"


def test_searxng_injoignable_apres_relance_repli_ddgs(monkeypatch):
    from loom.runtime import searxng

    def down(q, c):
        raise httpx.ConnectError("connexion refusée")

    monkeypatch.setattr(web, "_search_searxng", down)
    monkeypatch.setattr(
        searxng, "ensure_running", lambda url: True
    )  # relancé, toujours muet
    monkeypatch.setattr(web, "_search_ddgs", lambda q, c: HIT)
    assert web.web_search("q", CFG) == HIT


def test_hors_ligne_erreur_explicite(monkeypatch):
    def down(q, c):
        raise httpx.ConnectError("pas de réseau")

    monkeypatch.setattr(web, "_search_ddgs", down)
    with pytest.raises(ToolError, match="réseau injoignable"):
        web.make_web_search(web.WebSearchConfig()).run({"query": "q"})


def test_lot_d_urls_tout_en_echec_est_un_echec(monkeypatch):
    def page(url, cfg, snippet="", raise_status=False, impersonate=None):
        if "ok" in url:
            return "contenu lu"
        raise ToolError(f"erreur HTTP 403 () sur {url}")

    monkeypatch.setattr(web, "fetch_page", page)
    monkeypatch.setattr(web, "_blocked_host_reason", lambda u: None)
    fu = web.make_fetch_url(web.WebSearchConfig())
    with pytest.raises(ToolError, match="aucune des 2 pages"):
        fu.run({"urls": ["https://a.test/1", "https://b.test/2"]})
    # Une page lue suffit : le lot reste un succès, l'échec de l'autre y est écrit.
    out = fu.run({"urls": ["https://ok.test/1", "https://b.test/2"]})
    assert "contenu lu" in out and "erreur" in out


def _page(monkeypatch, text):
    monkeypatch.setattr(web, "_resolve_validated", lambda u: (None, "1.2.3.4"))
    resp = SimpleNamespace(
        status_code=200, headers={"content-type": "text/html"}, text="<html/>"
    )
    monkeypatch.setattr(web, "_http_get", lambda *a, **k: resp)
    monkeypatch.setattr(web, "_extract", lambda html: text)


def test_page_de_blocage_courte_devient_une_erreur(monkeypatch):
    # Texte réel renvoyé par last.fm le 2026-10-17, avec un HTTP 200.
    _page(
        monkeypatch,
        "JavaScript is disabled in your browser.\nPlease enable JavaScript to proceed.",
    )
    with pytest.raises(ToolError, match="page de blocage"):
        web.fetch_page(
            "https://www.last.fm/x", web.WebSearchConfig(), raise_status=True
        )
    # Côté web_search (pages des résultats), l'extrait du moteur reste le repli.
    assert (
        web.fetch_page(
            "https://www.last.fm/x", web.WebSearchConfig(), snippet="EXTRAIT"
        )
        == "EXTRAIT"
    )


def test_article_long_qui_cite_ces_mots_n_est_pas_un_blocage(monkeypatch):
    article = (
        "Pour enable JavaScript dans Firefox, ouvrez about:config. "
        + "Détail utile. " * 200
    )
    _page(monkeypatch, article)
    out = web.fetch_page(
        "https://doc.test/js", web.WebSearchConfig(), raise_status=True
    )
    assert out.startswith("Pour enable JavaScript")
