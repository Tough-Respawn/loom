# Fenêtre d'application liée à loom.web : détection du navigateur, ligne de commande en
# profil dédié, surveillance (fermeture -> on_close). Aucun navigateur réel lancé.
from __future__ import annotations

import subprocess
import sys
import threading

from loom.web import appwindow


def test_find_browser_prefere_la_config_puis_les_candidats(tmp_path):
    good = tmp_path / "brave.exe"
    good.write_bytes(b"")
    other = tmp_path / "edge.exe"
    other.write_bytes(b"")
    assert appwindow.find_browser(str(good), candidates=[str(other)]) == str(good)
    assert appwindow.find_browser(
        "", candidates=[str(tmp_path / "absent.exe"), str(other)]
    ) == str(other)
    assert appwindow.find_browser(str(tmp_path / "nope.exe"), candidates=[]) is None
    # Une variable d'environnement non résolue n'est jamais prise pour un chemin.
    assert appwindow.find_browser("", candidates=["%INCONNUE_LOOM%/x.exe"]) is None


def test_browser_command_profil_dedie_et_app():
    cmd = appwindow.browser_command("C:/b/brave.exe", "http://127.0.0.1:8000/", "C:/p")
    assert cmd[0] == "C:/b/brave.exe"
    assert "--app=http://127.0.0.1:8000/" in cmd
    assert "--user-data-dir=C:/p" in cmd
    assert "--disable-background-mode" in cmd and "--no-first-run" in cmd


def test_watch_appelle_on_close_a_la_fin_du_processus():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    done = threading.Event()
    appwindow.watch(proc, done.set)
    assert done.wait(10)


def test_open_app_window_sans_navigateur_ne_bloque_pas(tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(appwindow, "find_browser", lambda *a, **k: None)
    t = appwindow.open_app_window(
        "http://127.0.0.1:1/",
        profile_dir=str(tmp_path / "p"),
        on_close=lambda: messages.append("close"),
        log=messages.append,
    )
    t.join(5)
    assert any("aucun navigateur" in m for m in messages) and "close" not in messages
