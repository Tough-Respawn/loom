# Caractérisation de la file de notes en vol (/note) — comportement 2026-07-13,
# bornes incluses (fix audit : plafond 5000 chars, cap de file à 10).
from __future__ import annotations


def test_note_vide_400(web_sess):
    r = web_sess.post("/note", data={"text": "   "})
    assert r.status_code == 400
    assert r.get_json()["error"] == "note vide"


def test_note_trop_longue_413(web_sess):
    r = web_sess.post("/note", data={"text": "x" * 5001})
    assert r.status_code == 413


def test_note_5000_passe_encore(web_sess):
    r = web_sess.post("/note", data={"text": "x" * 5000})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "queued": 1}


def test_note_session_inconnue_404(web_sess):
    r = web_sess.post("/note", data={"text": "hello", "session_id": "n-existe-pas"})
    assert r.status_code == 404


def test_note_file_pleine_429(web_sess):
    for i in range(10):
        r = web_sess.post("/note", data={"text": f"note {i}"})
        assert r.status_code == 200
        assert r.get_json()["queued"] == i + 1
    r = web_sess.post("/note", data={"text": "la onzième"})
    assert r.status_code == 429
    assert "pleine" in r.get_json()["error"]
