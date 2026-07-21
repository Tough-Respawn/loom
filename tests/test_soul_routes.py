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


# ---------- correctifs revue 2026-07-21 ----------


def _soul_pour_session(tmp_env, sid, title, phrase):
    # Archive contenant la MÊME session que la cible, plus récente, titre différent.
    import io
    import json
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:

        def _add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        _add(
            "manifest.json",
            json.dumps(
                {
                    "version": 1,
                    "date": "2099-01-01T00:00:00+00:00",
                    "machine": "autre-machine",
                    "sessions": [sid],
                    "counts": {},
                }
            ).encode("utf-8"),
        )
        _add(
            f"sessions/{sid}/session.json",
            json.dumps(
                {
                    "id": sid,
                    "title": title,
                    "workspace": ".",
                    "updated_at": "2099-01-01T00:00:00+00:00",
                    "conversation": {},
                }
            ).encode("utf-8"),
        )
    f = tmp_env / "session-importee.soul"
    f.write_bytes(soul.encrypt(buf.getvalue(), phrase))
    return f


def test_import_recharge_la_session_active(web_sess, tmp_env):
    # BLOQUANT (revue) : sans rechargement, l'objet session en mémoire ré-écrirait
    # son ancien état par-dessus les fichiers importés au prochain save.
    S = web_sess.application.S
    sid = web_sess.get("/sessions").get_json()["active"]
    f = _soul_pour_session(tmp_env, sid, "titre importe", "p")
    r = web_sess.post("/soul/import", data={"file": str(f), "passphrase": "p"})
    assert r.status_code == 200, r.data
    assert r.get_json()["report"]["sessions"]["remplacees"] == 1
    # L'objet EN MÉMOIRE reflète la version importée, pas seulement le disque.
    assert S.cur["session"].title == "titre importe"
    assert S.sessions_cache[sid].title == "titre importe"


def test_import_refuse_pendant_generation(web_sess, tmp_env):
    import threading

    S = web_sess.application.S
    lk = S.sess_locks.setdefault("session-en-vol", threading.Lock())
    assert lk.acquire(blocking=False)
    try:
        r = web_sess.post(
            "/soul/import", data={"file": "peu-importe", "passphrase": "p"}
        )
        assert r.status_code == 409
        assert "génération en cours" in r.get_json()["error"]
    finally:
        lk.release()


def test_export_oserror_message_clair(web_sess, tmp_env, monkeypatch):
    # USB pleine/retirée : 400 avec la vraie cause, pas un 500 « échec réseau ».
    from loom.web import soul as soul_mod

    def _boom(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(soul_mod, "export_soul", _boom)
    dest = tmp_env / "usb"
    dest.mkdir()
    r = web_sess.post(
        "/soul/export",
        data={
            "dest_dir": str(dest),
            "passphrase": soul.generate_passphrase(),
            "session_ids": "",
        },
    )
    assert r.status_code == 400
    assert "écriture impossible" in r.get_json()["error"]
