# tests/test_soul.py
# L'Âme : export/import chiffré (spec 2026-07-21). Tests purs, sans Flask.
from __future__ import annotations

import pytest

from loom.web import soul


def test_chiffrement_aller_retour():
    blob = soul.encrypt(b"donnees privees", "phrase-de-test-longue")
    assert blob.startswith(soul.MAGIC)
    assert b"donnees privees" not in blob
    assert soul.decrypt(blob, "phrase-de-test-longue") == b"donnees privees"


def test_mauvaise_passphrase_erreur_propre():
    blob = soul.encrypt(b"x", "bonne-phrase")
    with pytest.raises(soul.SoulError, match="[Pp]assphrase"):
        soul.decrypt(blob, "mauvaise-phrase")


def test_fichier_tronque_ou_etranger():
    blob = soul.encrypt(b"x", "p")
    with pytest.raises(soul.SoulError):
        soul.decrypt(blob[:20], "p")  # tronqué
    with pytest.raises(soul.SoulError):
        soul.decrypt(b"PASLOOMSOUL" + blob, "p")  # mauvais magic


def test_generate_passphrase_diceware():
    p = soul.generate_passphrase()
    words = p.split("-")
    assert len(words) == 5
    assert all(w.isalpha() or "'" in w for w in words)
    assert soul.generate_passphrase() != p  # aléatoire


def test_check_passphrase_faible_vs_forte():
    faible = soul.check_passphrase("azerty123")
    assert faible["ok"] is False and faible["score"] < 3
    forte = soul.check_passphrase(soul.generate_passphrase())
    assert forte["ok"] is True and forte["score"] >= 3
    assert forte["crack_display"]
    vide = soul.check_passphrase("")
    assert vide == {"score": 0, "ok": False, "crack_display": ""}
