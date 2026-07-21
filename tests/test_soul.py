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


@pytest.fixture()
def soul_file(var, tmp_path):
    dest = tmp_path / "transport"
    dest.mkdir()
    recap = soul.export_soul(
        _paths(var), ["aaa111", "bbb222"], dest, "phrase-transport"
    )
    return Path(recap["path"])


def test_import_dans_var_vierge(soul_file, tmp_path):
    cible = tmp_path / "cible"
    for d in ("sessions", "skills_learned", "skills_user"):
        (cible / d).mkdir(parents=True)
    rep = soul.import_soul(_paths(cible), soul_file, "phrase-transport")
    assert rep["sessions"]["ajoutees"] == 2
    assert (cible / "sessions" / "aaa111" / "session.json").exists()
    assert (cible / "identity" / "SOUL.md").read_text(
        encoding="utf-8"
    ) == "ame de la machine A"
    assert (cible / "skills_learned" / "detect-boucle" / "SKILL.md").exists()
    assert rep["memoire"]["ajoutes"] == 1
    assert rep["manifest"]["machine"]


def test_import_fusion_conflits(soul_file, var, tmp_path):
    import sqlite3

    cible = tmp_path / "cible"
    # Session aaa111 déjà là, PLUS RÉCENTE que l'archive -> conservée.
    d = cible / "sessions" / "aaa111"
    d.mkdir(parents=True)
    (d / "session.json").write_text(
        _json.dumps(
            {
                "id": "aaa111",
                "title": "plus recent ici",
                "updated_at": "2026-07-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    # Skill homonyme au contenu DIFFÉRENT -> import sous suffixe.
    sk = cible / "skills_learned" / "detect-boucle"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("version locale differente", encoding="utf-8")
    (cible / "skills_user").mkdir()
    # Identité DIVERGENTE -> posée à côté, jamais écrasée.
    (cible / "identity").mkdir()
    (cible / "identity" / "SOUL.md").write_text("ame de la machine B", encoding="utf-8")
    # Mémoire cible avec le MÊME épisode -> dédup.
    (cible / "memory").mkdir()
    src = sqlite3.connect(var / "memory" / "memory.db")
    src.backup(dst := sqlite3.connect(cible / "memory" / "memory.db"))
    dst.close()
    src.close()

    rep = soul.import_soul(_paths(cible), soul_file, "phrase-transport")
    meta = _json.loads(
        (cible / "sessions" / "aaa111" / "session.json").read_text(encoding="utf-8")
    )
    assert meta["title"] == "plus recent ici"
    assert rep["sessions"]["ignorees"] == 1 and rep["sessions"]["ajoutees"] == 1
    assert (cible / "skills_learned" / "detect-boucle-importe" / "SKILL.md").exists()
    assert (cible / "identity" / "SOUL.md").read_text(
        encoding="utf-8"
    ) == "ame de la machine B"
    a_cote = list((cible / "identity").glob("SOUL.imported-*.md"))
    assert len(a_cote) == 1
    assert rep["memoire"]["ajoutes"] == 0 and rep["memoire"]["ignores"] == 1
    # Ré-import du même fichier : idempotent (rien ne bouge, pas de -importe-2).
    rep2 = soul.import_soul(_paths(cible), soul_file, "phrase-transport")
    assert rep2["skills_learned"]["renommes"] == 0
    assert len(list((cible / "skills_learned").glob("detect-boucle-importe*"))) == 1


def test_import_erreurs_sans_degats(soul_file, tmp_path):
    cible = tmp_path / "cible"
    (cible / "sessions").mkdir(parents=True)
    with pytest.raises(soul.SoulError):
        soul.import_soul(_paths(cible), soul_file, "mauvaise-phrase")
    assert list((cible / "sessions").iterdir()) == []  # rien touché
    with pytest.raises(soul.SoulError, match="introuvable"):
        soul.import_soul(_paths(cible), tmp_path / "absent.soul", "p")


# ---------- correctifs revue 2026-07-21 ----------


def test_export_session_inexistante_ni_comptee_ni_au_manifest(var, tmp_path):
    # Session supprimée entre l'affichage de la liste et le clic Exporter : le récap
    # et le manifest ne doivent pas annoncer une sauvegarde qui n'existe pas.
    dest = tmp_path / "usb"
    dest.mkdir()
    recap = soul.export_soul(_paths(var), ["zzz999", "aaa111"], dest, "p")
    assert recap["sessions"] == 1
    data = soul.decrypt(Path(recap["path"]).read_bytes(), "p")
    with _tarfile.open(fileobj=_io.BytesIO(data), mode="r:gz") as tar:
        manifest = _json.load(tar.extractfile("manifest.json"))
    assert manifest["sessions"] == ["aaa111"]


def test_export_memoire_corrompue_erreur_propre(var, tmp_path):
    (var / "memory" / "memory.db").write_bytes(b"PAS UNE BASE SQLITE")
    dest = tmp_path / "usb"
    dest.mkdir()
    with pytest.raises(soul.SoulError, match="[Mm]émoire"):
        soul.export_soul(_paths(var), [], dest, "p")


def test_remplacement_purge_la_timeline_perimee(tmp_path):
    # Source plus récente SANS timeline (reset côté source) : le timeline local
    # périmé ne doit pas survivre au session.json remplacé.
    src = tmp_path / "src" / "aaa"
    src.mkdir(parents=True)
    (src / "session.json").write_text(
        _json.dumps({"id": "aaa", "updated_at": "2026-07-22T00:00:00+00:00"}),
        encoding="utf-8",
    )
    dst_root = tmp_path / "dst"
    d = dst_root / "aaa"
    d.mkdir(parents=True)
    (d / "session.json").write_text(
        _json.dumps({"id": "aaa", "updated_at": "2026-07-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (d / "timeline.jsonl").write_text("VIEUX EVENEMENT LOCAL", encoding="utf-8")
    rep = soul._merge_sessions(tmp_path / "src", dst_root)
    assert rep["remplacees"] == 1
    assert not (d / "timeline.jsonl").exists()


def _soul_avec_memoire_corrompue(tmp_path, phrase):
    buf = _io.BytesIO()
    with _tarfile.open(fileobj=buf, mode="w:gz") as tar:

        def _add(name, data):
            info = _tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, _io.BytesIO(data))

        _add(
            "manifest.json",
            _json.dumps(
                {
                    "version": 1,
                    "date": "2026-07-21T00:00:00+00:00",
                    "machine": "x",
                    "sessions": ["s1"],
                    "counts": {},
                }
            ).encode("utf-8"),
        )
        _add(
            "sessions/s1/session.json",
            _json.dumps(
                {"id": "s1", "updated_at": "2026-07-21T00:00:00+00:00"}
            ).encode("utf-8"),
        )
        _add("memory/memory.db", b"PAS UNE BASE SQLITE")
    p = tmp_path / "corrompu.soul"
    p.write_bytes(soul.encrypt(buf.getvalue(), phrase))
    return p


def test_import_memoire_corrompue_annule_TOUT(tmp_path):
    # Pré-validation : la base corrompue est détectée AVANT toute fusion — même les
    # sessions (fusionnées en premier) ne doivent pas avoir été écrites.
    f = _soul_avec_memoire_corrompue(tmp_path, "p")
    cible = tmp_path / "cible"
    for d in ("sessions", "skills_learned", "skills_user"):
        (cible / d).mkdir(parents=True)
    with pytest.raises(soul.SoulError, match="import annulé"):
        soul.import_soul(_paths(cible), f, "p")
    assert list((cible / "sessions").iterdir()) == []
