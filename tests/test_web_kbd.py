from __future__ import annotations


def _text(response) -> str:
    return response.data.decode("utf-8")


def test_kbd_module_est_servi_et_decrit_les_raccourcis_reels(web):
    response = web.get("/static/kbd.js")
    assert response.status_code == 200
    source = _text(response)
    assert "export function initKbdCheatsheet" in source
    assert "export function ensureKbdCheatsheetButton" in source
    for label in (
        "Scinder la vue",
        "Revenir à une vue",
        "Focaliser un panneau",
        "Parcourir l’historique",
        "Valider la palette",
        "Menu, palette ou fenêtre",
    ):
        assert label in source


def test_app_initialise_la_cheatsheet(web):
    source = _text(web.get("/static/app.js"))
    assert 'import { initKbdCheatsheet } from "./kbd.js";' in source
    assert source.rstrip().endswith("initKbdCheatsheet();")


def test_render_tabs_reinsere_le_bouton_a_chaque_rendu(web):
    source = _text(web.get("/static/tabs.js"))
    assert 'import { ensureKbdCheatsheetButton } from "./kbd.js";' in source
    clear = source.index('tabbarEl.innerHTML = "";')
    append_plus = source.index("tabbarEl.append(plus);")
    ensure = source.index("ensureKbdCheatsheetButton();", append_plus)
    assert clear < append_plus < ensure


def test_css_cheatsheet_utilise_les_variables_reelles_de_loom(web):
    page = _text(web.get("/"))
    css = page[page.index("/* ---- Cheatsheet clavier"):page.index("* { box-sizing")]
    assert ".kb-overlay" in css and ".kb-modal" in css and ".kb-row" in css
    assert "var(--bg-elev)" in css
    assert "var(--text-dim)" in css
    assert "var(--panel)" not in css
    assert "var(--muted)" not in css
