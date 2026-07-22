# loom/setup/tooling.py
"""Inventaire de l'OUTILLAGE externe des outils de l'agent (hors moteur/modèle).

Vécu 2026-07-22 (llama-swap jamais provisionné -> crash au 2e modèle) : tout ce
dont Loom a besoin pour fonctionner doit être constaté — et si possible installé
— par loom-setup, pas découvert par une panne. Ici : les dépendances DÉGRADABLES
(l'agent perd une capacité, Loom ne plante pas), vérifiées et conseillées.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def playwright_browser_present() -> bool:
    """Vrai si un chromium Playwright est déjà téléchargé (check_page /
    check_interactive en ont besoin ; le paquet Python seul ne suffit pas)."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "ms-playwright"
    else:
        root = Path.home() / ".cache" / "ms-playwright"
    if not root.is_dir():
        return False
    return any(p.name.startswith("chromium") for p in root.iterdir() if p.is_dir())


def install_playwright_browser(timeout: int = 600) -> tuple[bool, str]:
    """`playwright install chromium` (~130 Mo). (ok, détail) — jamais d'exception."""
    try:
        res = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, str(exc)
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-1:]
        return False, " | ".join(tail) or f"code {res.returncode}"
    return True, "chromium installé"


def tool_checks() -> list[dict]:
    """État des outils externes DÉGRADABLES : [{name, present, role, hint}].
    `hint` = quoi faire s'il manque (affiché en [attention], jamais bloquant)."""
    return [
        {
            "name": "navigateur Playwright (chromium)",
            "present": playwright_browser_present(),
            "role": "check_page / check_interactive (vérification de rendu web)",
            "hint": "installe-le via loom-setup, ou : uv run playwright install chromium",
            "autofix": "playwright",
        },
        {
            "name": "rg (ripgrep)",
            "present": shutil.which("rg") is not None,
            "role": "search_text (grep rapide du workspace)",
            "hint": "winget install BurntSushi.ripgrep.MSVC (ou apt/brew install ripgrep)",
            "autofix": None,
        },
        {
            "name": "npx (Node.js)",
            "present": shutil.which("npx") is not None,
            "role": "format_code / lint web (prettier, oxlint)",
            "hint": "installe Node.js (winget install OpenJS.NodeJS.LTS) — optionnel",
            "autofix": None,
        },
        {
            "name": "docker",
            "present": shutil.which("docker") is not None,
            "role": "web_search auto-hébergé (SearXNG) — repli ddgs sinon",
            "hint": "optionnel : Docker Desktop pour un web_search sans quota",
            "autofix": None,
        },
    ]
