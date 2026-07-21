# Routes /soul/* : export/import chiffré de l'état portable (spec 2026-07-21).
from __future__ import annotations

from pathlib import Path

from loom.web import soul


def test_soul_sessions_liste(web_sess):
    data = web_sess.get("/soul/sessions").get_json()
    assert len(data["sessions"]) == 1
    s = data["sessions"][0]
    assert set(s) >= {"id", "title", "updated_at"}


def test_passphrase_check_et_generate(web):
    r = web.post("/soul/passphrase/check", data={"passphrase": "azerty123"}).get_json()
    assert r["ok"] is False
    g = web.post("/soul/passphrase/generate", data={}).get_json()
    assert len(g["passphrase"].split("-")) == 5
    r2 = web.post(
        "/soul/passphrase/check", data={"passphrase": g["passphrase"]}
    ).get_json()
    assert r2["ok"] is True and r2["crack_display"]


def test_export_refuse_passphrase_faible(web_sess, tmp_env):
    dest = tmp_env / "usb"
    dest.mkdir()
    r = web_sess.post(
        "/soul/export",
        data={"dest_dir": str(dest), "passphrase": "azerty123", "session_ids": ""},
    )
    assert r.status_code == 400
    assert "faible" in r.get_json()["error"]


def test_export_puis_import_aller_retour(web_sess, tmp_env):
    sid = web_sess.get("/sessions").get_json()["active"]
    dest = tmp_env / "usb"
    dest.mkdir()
    phrase = soul.generate_passphrase()
    r = web_sess.post(
        "/soul/export",
        data={"dest_dir": str(dest), "passphrase": phrase, "session_ids": sid},
    )
    assert r.status_code == 200, r.data
    path = r.get_json()["path"]
    assert Path(path).exists()
    # Mauvaise passphrase à l'import -> 400 avec message clair, rien touché.
    bad = web_sess.post("/soul/import", data={"file": path, "passphrase": "xxx"})
    assert bad.status_code == 400
    assert "passphrase" in bad.get_json()["error"].lower()
    # Bonne passphrase -> rapport de fusion (la session existe déjà : ignorée).
    ok = web_sess.post("/soul/import", data={"file": path, "passphrase": phrase})
    assert ok.status_code == 200
    rep = ok.get_json()["report"]
    assert rep["sessions"]["ignorees"] == 1
