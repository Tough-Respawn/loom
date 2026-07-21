# tests/test_soul.py
# L'Âme : export/import chiffré (spec 2026-07-21). Tests purs, sans Flask.
from __future__ import annotations

import io as _io
import json as _json
import tarfile as _tarfile
from pathlib import Path

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


@pytest.fixture()
def var(tmp_path):
    """Un var/ réaliste : 2 sessions (+logs à exclure), mémoire, identité, skills,
    et un remote_models.json qui ne doit JAMAIS partir."""
    import sqlite3

    root = tmp_path / "var"
    for sid, title in (("aaa111", "Session A"), ("bbb222", "Session B")):
        d = root / "sessions" / sid
        d.mkdir(parents=True)
        (d / "session.json").write_text(
            _json.dumps(
                {"id": sid, "title": title, "updated_at": "2026-07-21T10:00:00+00:00"}
            ),
            encoding="utf-8",
        )
        (d / "timeline.jsonl").write_text('{"type":"text"}\n', encoding="utf-8")
        (d / "serve.log").write_text("bruit machine", encoding="utf-8")
    (root / "memory").mkdir()
    con = sqlite3.connect(root / "memory" / "memory.db")
    con.execute(
        "CREATE TABLE episodes (id INTEGER PRIMARY KEY, ts TEXT NOT NULL,"
        " kind TEXT NOT NULL DEFAULT 'episodic', source TEXT NOT NULL DEFAULT '',"
        " text TEXT NOT NULL)"
    )
    con.execute(
        "INSERT INTO episodes (ts, kind, source, text) VALUES"
        " ('2026-07-20T00:00:00+00:00', 'episodic', 'chat', 'souvenir un')"
    )
    con.commit()
    con.close()
    (root / "identity").mkdir()
    (root / "identity" / "SOUL.md").write_text("ame de la machine A", encoding="utf-8")
    (root / "identity" / "USER.md").write_text("amine", encoding="utf-8")
    sk = root / "skills_learned" / "detect-boucle"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("skill appris", encoding="utf-8")
    (root / "skills_user").mkdir()
    (root / "remote_models.json").write_text(
        '[{"api_key": "sk-SECRET"}]', encoding="utf-8"
    )
    return root


def _paths(root):
    return soul.SoulPaths(
        sessions_root=root / "sessions",
        memory_db=root / "memory" / "memory.db",
        identity_dir=root / "identity",
        learned_skills_dir=root / "skills_learned",
        user_skills_dir=root / "skills_user",
    )


def test_archive_contenu_et_anti_fuite(var):
    data = soul.build_archive(_paths(var), ["aaa111"])
    with _tarfile.open(fileobj=_io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
        manifest = _json.load(tar.extractfile("manifest.json"))
    assert "sessions/aaa111/session.json" in names
    assert "sessions/aaa111/timeline.jsonl" in names
    assert "sessions/bbb222/session.json" not in names  # sélection respectée
    assert "memory/memory.db" in names
    assert "identity/SOUL.md" in names
    assert "skills_learned/detect-boucle/SKILL.md" in names
    # ANTI-FUITE (décision ferme) : ni logs machine, ni le moindre secret.
    assert not any("serve.log" in n or "debug.log" in n for n in names)
    assert not any("remote_models" in n for n in names)
    assert manifest["version"] == 1 and manifest["sessions"] == ["aaa111"]


def test_export_soul_ecrit_le_fichier(var, tmp_path):
    dest = tmp_path / "usb"
    dest.mkdir()
    recap = soul.export_soul(
        _paths(var), ["aaa111", "bbb222"], dest, "phrase-forte-de-test"
    )
    p = Path(recap["path"])
    assert p.exists() and p.suffix == ".soul" and recap["sessions"] == 2
    assert p.read_bytes().startswith(soul.MAGIC)
    # Deuxième export le même jour : pas d'écrasement, suffixe.
    recap2 = soul.export_soul(_paths(var), [], dest, "phrase-forte-de-test")
    assert recap2["path"] != recap["path"]


def test_export_dest_introuvable(var, tmp_path):
    with pytest.raises(soul.SoulError, match="introuvable"):
        soul.export_soul(_paths(var), [], tmp_path / "nexiste-pas", "p")
