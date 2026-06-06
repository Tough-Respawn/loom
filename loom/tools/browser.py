# loom/tools/browser.py
"""Outil check_page : donne des YEUX à Loom sur une page web rendue (Playwright headless).

Sans ça, le modèle édite du HTML/JS à l'aveugle et confabule « ça marche » : il ne voit
ni l'erreur console qui plante le jeu, ni que la grille ne s'affiche pas. check_page charge
la page dans Chromium headless, EXÉCUTE le JS, et renvoie les ERREURS CONSOLE, le compte
d'éléments (count_selectors) et un extrait du texte visible — le modèle VOIT le crash / la
grille manquante et peut itérer jusqu'à « 0 erreur ». Lazy-import de playwright : message
clair et actionnable si la lib (ou le navigateur) n'est pas installée.

Le contenu d'une page est EXTERNE/non fiable -> renvoyé via untrusted() : donnée à
analyser, jamais des ordres (une page peut contenir « ignore tes consignes »).
"""

from __future__ import annotations

from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.trust import untrusted

_INSTALL_HINT = (
    "playwright non installé. Lance une fois : `uv add playwright` puis "
    "`uv run playwright install chromium`."
)


def make_check_page(workspace_dir: str) -> ToolSpec:
    """Outil check_page borné au workspace pour les chemins relatifs (absolus acceptés)."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        target = (args.get("url") or "").strip()
        if not target:
            raise ToolError(
                "argument 'url' manquant (URL http(s):// OU chemin d'un fichier .html)"
            )
        # URL telle quelle, sinon chemin local -> file:// (Path.as_uri()).
        if target.startswith(("http://", "https://", "file://")):
            url = target
        else:
            path = _resolve_in_root(root, target)
            if not path.exists():
                raise ToolError(f"fichier introuvable : {target}")
            url = path.as_uri()

        wait_selector = (args.get("wait_selector") or "").strip() or None
        count_selectors = [
            s.strip()
            for s in (args.get("count_selectors") or "").split(",")
            if s.strip()
        ]

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ToolError(_INSTALL_HINT) from exc

        console: list[tuple[str, str]] = []
        page_errors: list[str] = []
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                page.on("console", lambda m: console.append((m.type, m.text)))
                page.on("pageerror", lambda e: page_errors.append(str(e)))
                page.goto(url, wait_until="load", timeout=15000)
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=5000)
                    except Exception:  # noqa: BLE001 - absence = info, pas un crash
                        pass
                page.wait_for_timeout(1200)  # laisse le JS d'init s'exécuter
                title = page.title()
                body = page.query_selector("body")
                body_text = body.inner_text()[:2000] if body else ""
                counts = {
                    sel: len(page.query_selector_all(sel)) for sel in count_selectors
                }
                browser.close()
        except Exception as exc:  # noqa: BLE001 - navigateur absent / page injoignable
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                raise ToolError(_INSTALL_HINT) from exc
            raise ToolError(f"échec du chargement de {url} : {msg[:200]}") from exc

        errors = [t for (k, t) in console if k == "error"] + page_errors
        warnings = [t for (k, t) in console if k == "warning"]
        lines = [
            f"page : {title!r} ({url})",
            f"console : {len(errors)} erreur(s), {len(warnings)} warning(s)",
        ]
        for e in errors[:8]:
            lines.append(f"  [erreur] {e[:200]}")
        if counts:
            lines.append(
                "éléments : " + " · ".join(f"{s} ×{n}" for s, n in counts.items())
            )
        visible = " ".join(body_text.split())
        if visible:
            lines.append(f"texte visible : {visible[:400]}")
        if not errors and not page_errors:
            lines.append("(aucune erreur console — la page s'est chargée et exécutée)")
        return untrusted("\n".join(lines), f"page {url}")

    return ToolSpec(
        name="check_page",
        description=(
            "Charge une page web (URL http(s):// OU chemin d'un fichier .html local) dans "
            "un navigateur headless, EXÉCUTE son JavaScript, et renvoie : les ERREURS de "
            "la console, le nombre d'éléments correspondant à count_selectors (ex "
            "'.cell,#board'), et un extrait du texte visible. SERS-T'EN pour VÉRIFIER "
            "qu'une page HTML que tu viens d'écrire s'affiche et fonctionne (0 erreur "
            "console, éléments attendus présents) AU LIEU de supposer que ça marche. Si des "
            "erreurs apparaissent, corrige puis relance check_page jusqu'à 0 erreur."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "URL http(s):// ou chemin d'un fichier .html (relatif au dossier "
                        "de travail ou absolu)."
                    ),
                },
                "wait_selector": {
                    "type": "string",
                    "description": (
                        "Sélecteur CSS à attendre avant de lire la page (optionnel)."
                    ),
                },
                "count_selectors": {
                    "type": "string",
                    "description": (
                        "Sélecteurs CSS à compter, séparés par des virgules (ex "
                        "'.cell,.flag') — pour vérifier que des éléments sont bien rendus."
                    ),
                },
            },
            "required": ["url"],
        },
        run=run,
    )
