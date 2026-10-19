"""Fenêtre d'application de Loom, LIÉE au processus loom.web.

Le lanceur ouvrait Loom en `--app` dans le navigateur déjà ouvert : la fenêtre n'était
qu'un onglet déguisé d'un processus qu'on ne possède pas, donc impossible de savoir
quand l'utilisateur la ferme, et la console restait à fermer à la main (et inversement).

Ici loom.web ouvre LUI-MÊME la fenêtre dans un profil navigateur dédié (`var/browser-
profile`) : c'est alors un arbre de processus à nous, qui s'éteint quand sa dernière
fenêtre se ferme (vérifié sur Brave le 2026-10-19). Deux liens :
- fenêtre fermée -> `on_close()` : arrêt propre de Loom (serveur modèle déchargé, sortie) ;
- console fermée -> loom.web meurt -> le Job Object « kill-on-close » auquel la fenêtre est
  attachée (`attach`) la ferme aussi, comme le serveur modèle.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

#: Navigateurs Chromium connus, dans l'ordre de préférence (tous supportent `--app`).
_CANDIDATES = (
    r"%ProgramFiles%\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
    r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    "/usr/bin/brave-browser",
    "/usr/bin/microsoft-edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)


def find_browser(preferred: str = "", candidates=_CANDIDATES) -> str | None:
    """Exécutable d'un navigateur Chromium : `preferred` (config) s'il existe, sinon le
    premier candidat connu présent sur la machine. None si aucun."""
    if preferred and Path(os.path.expandvars(preferred)).is_file():
        return os.path.expandvars(preferred)
    for cand in candidates:
        p = os.path.expandvars(cand)
        if "%" not in p and Path(p).is_file():
            return p
    return None


def browser_command(exe: str, url: str, profile_dir: str) -> list[str]:
    """Ligne de commande d'une fenêtre d'application dans un profil DÉDIÉ : séparée du
    navigateur personnel de l'utilisateur (c'est ce qui la rend surveillable), sans
    premier lancement ni mode arrière-plan qui survivrait à la fenêtre."""
    return [
        exe,
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-mode",
        "--disable-session-crashed-bubble",
    ]


def wait_ready(url: str, timeout_s: float = 90.0) -> bool:
    """Attend que l'UI réponde (HTTP 200) avant d'ouvrir la fenêtre dessus."""
    import urllib.request

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - pas encore prêt
            pass
        time.sleep(0.4)
    return False


def watch(proc: subprocess.Popen, on_close) -> threading.Thread:
    """Fil qui attend la fin du processus navigateur puis appelle `on_close()`."""

    def _run():
        try:
            proc.wait()
        finally:
            on_close()

    t = threading.Thread(target=_run, daemon=True, name="loom-app-window")
    t.start()
    return t


def open_app_window(
    url: str,
    *,
    profile_dir: str,
    on_close,
    exe: str = "",
    attach=None,
    log=print,
) -> threading.Thread:
    """Ouvre la fenêtre d'application en fond dès que l'UI répond, l'attache au Job
    (`attach(pid)`), puis la surveille : sa fermeture déclenche `on_close()`.
    Sans navigateur Chromium trouvé : le dit, et Loom continue sans fenêtre liée."""

    def _run():
        browser = find_browser(exe)
        if browser is None:
            log(
                "[loom] fenêtre d'application : aucun navigateur Chromium trouvé (Brave, Edge, Chrome) — ouvre "
                + url
            )
            return
        if not wait_ready(url):
            log(
                "[loom] fenêtre d'application : l'UI n'a pas répondu, fenêtre non ouverte"
            )
            return
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.Popen(
                browser_command(browser, url, profile_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log(
                f"[loom] fenêtre d'application : lancement impossible ({exc}) — ouvre {url}"
            )
            return
        if attach is not None and not attach(proc.pid):
            log(
                "[loom] fenêtre d'application : non attachée au Job (fermer la console ne la fermera pas)"
            )
        log(
            f"[loom] fenêtre d'application ouverte ({Path(browser).name}, pid {proc.pid}) : la fermer arrête Loom"
        )
        watch(proc, on_close)

    t = threading.Thread(target=_run, daemon=True, name="loom-app-window-open")
    t.start()
    return t


def console_title(title: str) -> None:
    """Titre de la console (Windows) : dit ce que ferme la croix."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except Exception:  # noqa: BLE001 - cosmétique
            pass
