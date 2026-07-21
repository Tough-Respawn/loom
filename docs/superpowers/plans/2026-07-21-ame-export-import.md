# L'Âme (export/import chiffré) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exporter l'état portable de Loom (sessions sélectionnables, mémoire, identité, skills) en un fichier `.soul` chiffré, et l'importer par fusion sur une autre machine.

**Architecture:** Un module pur `loom/web/soul.py` (archive tar.gz + AES-256-GCM/scrypt + règles de fusion, zéro Flask), des routes fines dans `routes.py` (`_register_soul_routes`), un onglet « Âme » dans la console de configuration (index.html + app.js). Spec : `docs/superpowers/specs/2026-07-21-ame-export-import-design.md`.

**Tech Stack:** Python (tarfile, sqlite3, cryptography AESGCM+Scrypt, zxcvbn), Flask, JS vanilla (patterns existants du panneau engrenage).

## Global Constraints

- AUCUN secret dans l'archive : `remote_models.json` exclu par construction (test anti-fuite obligatoire).
- scrypt N=2^17, r=8, p=1, clé 32 octets ; AES-256-GCM ; sel 16 o + nonce 12 o en clair en tête, magic `LOOMSOUL1`.
- Passphrase : score zxcvbn >= 3 requis à l'export (bloquant, vérifié CÔTÉ SERVEUR aussi), aucun contrôle à l'import.
- Fusion : rien n'est jamais supprimé côté cible par un import ; erreur (passphrase/corruption/version) → état local intact.
- Sessions exportées : `session.json` + `timeline.jsonl` seulement (jamais `serve.log`/`debug.log`).
- Messages UI en français, ton sobre, pas d'emojis criards (préférence user).
- Commits courts (subject conventionnel < 72 chars, pas de trailers d'attribution).
- Lancer `ruff check loom tests` avant chaque commit ; suite complète : `python -m pytest tests -q` (394 tests verts avant ce chantier).

---

### Task 1: Dépendances + listes diceware

**Files:**
- Modify: `pyproject.toml` (bloc `dependencies`)
- Create: `loom/web/data/diceware_fr.txt`, `loom/web/data/diceware_en.txt`

**Interfaces:**
- Produces: deux fichiers texte UTF-8, un mot par ligne (~7776 chacun), lus par `soul._wordlist()` (Task 3).

- [ ] **Step 1: Ajouter les dépendances**

Dans `pyproject.toml`, ajouter à `dependencies` (avant la ligne `ruff`) :

```toml
    # L'Âme (export/import chiffré) : AES-GCM + scrypt, et jauge de passphrase offline.
    "cryptography>=43",
    "zxcvbn>=4.4",
```

Puis : `uv sync` — vérifier que `python -c "from cryptography.hazmat.primitives.ciphers.aead import AESGCM; from zxcvbn import zxcvbn; print('ok')"` affiche `ok`.

- [ ] **Step 2: Télécharger et normaliser les listes de mots**

```bash
mkdir -p loom/web/data
curl -sL https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt | awk '{print $2}' > loom/web/data/diceware_en.txt
curl -sL https://raw.githubusercontent.com/mbelivo/diceware-wordlists-fr/master/wordlist_fr_5d.txt | awk '{print $NF}' > loom/web/data/diceware_fr.txt
wc -l loom/web/data/diceware_en.txt loom/web/data/diceware_fr.txt
```

Attendu : ~7776 lignes chacun. Vérifier `head -3` de chaque fichier : un mot nu par ligne (pas d'indice de dés). Si le miroir FR est indisponible, repli : `https://raw.githubusercontent.com/chmduquesne/diceware-fr/master/diceware-fr-5-jets.txt` (même normalisation `awk '{print $NF}'`).

- [ ] **Step 3: Vérifier l'empaquetage**

Les données doivent partir avec le paquet : vérifier que `loom/web/data/` est inclus (hatchling embarque tout le paquet `loom` par défaut ; s'il existe une liste d'exclusions dans `pyproject.toml`, ne pas y toucher).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock loom/web/data
git commit -m "feat(ame): deps cryptography+zxcvbn et listes diceware fr/en"
```

---

### Task 2: soul.py — chiffrement

**Files:**
- Create: `loom/web/soul.py`
- Test: `tests/test_soul.py`

**Interfaces:**
- Produces: `soul.encrypt(data: bytes, passphrase: str) -> bytes`, `soul.decrypt(blob: bytes, passphrase: str) -> bytes` (lève `soul.SoulError`), `soul.SoulPaths` (dataclass), `soul.MAGIC`.

- [ ] **Step 1: Écrire les tests qui échouent**

```python
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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `python -m pytest tests/test_soul.py -x -q`
Expected: FAIL (`ModuleNotFoundError: loom.web.soul`).

- [ ] **Step 3: Implémenter le cœur crypto**

```python
# loom/web/soul.py
"""L'Âme : export/import chiffré de l'état portable de Loom.

Un fichier .soul = tar.gz (manifest + sessions + mémoire + identité + skills)
chiffré AES-256-GCM, clé dérivée de la passphrase par scrypt (memory-hard :
~100 ms/essai, brute force offline neutralisé). AUCUN secret dans l'archive :
remote_models.json est exclu par construction. Logique pure (chemins explicites),
testable sans Flask. Spec : docs/superpowers/specs/2026-07-21-ame-export-import-design.md
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import secrets
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"LOOMSOUL1"
_SALT_LEN, _NONCE_LEN = 16, 12
# N=2^17 : ~128 Mo de RAM et ~100 ms par dérivation — c'est le prix d'UN essai
# de passphrase pour un attaquant qui a volé le fichier. Ne pas baisser.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**17, 8, 1

IDENTITY_FILES = ("SOUL.md", "USER.md", "MEMORY.md")
SESSION_FILES = ("session.json", "timeline.jsonl")  # jamais serve.log/debug.log


class SoulError(Exception):
    """Erreur montrable au user : passphrase fausse, fichier corrompu, version inconnue."""


@dataclass
class SoulPaths:
    """Racines de l'état portable — injectées par les routes, jamais devinées ici."""

    sessions_root: Path
    memory_db: Path
    identity_dir: Path
    learned_skills_dir: Path
    user_skills_dir: Path


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt(data: bytes, passphrase: str) -> bytes:
    salt = secrets.token_bytes(_SALT_LEN)
    nonce = secrets.token_bytes(_NONCE_LEN)
    ct = AESGCM(_derive_key(passphrase, salt)).encrypt(nonce, data, MAGIC)
    return MAGIC + salt + nonce + ct


def decrypt(blob: bytes, passphrase: str) -> bytes:
    head = len(MAGIC) + _SALT_LEN + _NONCE_LEN
    if len(blob) < head + 16 or not blob.startswith(MAGIC):
        raise SoulError("fichier .soul invalide ou tronqué")
    salt = blob[len(MAGIC) : len(MAGIC) + _SALT_LEN]
    nonce = blob[len(MAGIC) + _SALT_LEN : head]
    try:
        return AESGCM(_derive_key(passphrase, salt)).decrypt(nonce, blob[head:], MAGIC)
    except InvalidTag as e:
        raise SoulError("passphrase incorrecte ou fichier corrompu") from e
```

- [ ] **Step 4: Vérifier le vert**

Run: `python -m pytest tests/test_soul.py -x -q`
Expected: 3 passed. (Note : chaque scrypt prend ~100 ms, c'est normal.)

- [ ] **Step 5: Commit**

```bash
git add loom/web/soul.py tests/test_soul.py
git commit -m "feat(ame): chiffrement AES-256-GCM + scrypt du fichier .soul"
```

---

### Task 3: soul.py — passphrase (générer + challenger)

**Files:**
- Modify: `loom/web/soul.py`
- Test: `tests/test_soul.py`

**Interfaces:**
- Consumes: `loom/web/data/diceware_{fr,en}.txt` (Task 1).
- Produces: `soul.generate_passphrase(n_words: int = 5) -> str`, `soul.check_passphrase(p: str) -> dict` (`{"score": int, "ok": bool, "crack_display": str}`).

- [ ] **Step 1: Tests qui échouent**

Ajouter à `tests/test_soul.py` :

```python
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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `python -m pytest tests/test_soul.py -x -q` — Expected: FAIL (`AttributeError: generate_passphrase`).

- [ ] **Step 3: Implémenter**

Ajouter à `loom/web/soul.py` :

```python
_WORDS: list[str] | None = None


def _wordlist() -> list[str]:
    # Fusion FR+EN (~15.5k mots) : ~13,9 bits/mot -> 5 mots ≈ 69 bits d'entropie,
    # et un dictionnaire d'attaque mono-langue ne couvre que la moitié des mots.
    global _WORDS
    if _WORDS is None:
        data = Path(__file__).parent / "data"
        _WORDS = [
            w
            for name in ("diceware_fr.txt", "diceware_en.txt")
            for w in (data / name).read_text(encoding="utf-8").split()
        ]
    return _WORDS


def generate_passphrase(n_words: int = 5) -> str:
    words = _wordlist()
    return "-".join(secrets.choice(words) for _ in range(n_words))


def check_passphrase(passphrase: str) -> dict:
    if not passphrase:
        return {"score": 0, "ok": False, "crack_display": ""}
    from zxcvbn import zxcvbn  # import local : ~1 chargement de dictionnaires

    r = zxcvbn(passphrase)
    return {
        "score": int(r["score"]),
        "ok": int(r["score"]) >= 3,
        "crack_display": str(
            r["crack_times_display"]["offline_slow_hashing_1e4_per_second"]
        ),
    }
```

- [ ] **Step 4: Vérifier le vert**

Run: `python -m pytest tests/test_soul.py -x -q` — Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add loom/web/soul.py tests/test_soul.py
git commit -m "feat(ame): diceware bilingue + jauge zxcvbn (seuil 3/4)"
```

---

### Task 4: soul.py — export (archive + manifest, anti-fuite)

**Files:**
- Modify: `loom/web/soul.py`
- Test: `tests/test_soul.py`

**Interfaces:**
- Consumes: `SoulPaths`, `encrypt` (Task 2).
- Produces: `soul.build_archive(paths: SoulPaths, session_ids: list[str]) -> bytes` (tar.gz en mémoire), `soul.export_soul(paths, session_ids, dest_dir, passphrase) -> dict` (`{"path", "size", "sessions"}`).

- [ ] **Step 1: Fixture d'état + tests qui échouent**

Ajouter à `tests/test_soul.py` :

```python
import tarfile as _tarfile
import io as _io
import json as _json


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
            _json.dumps({"id": sid, "title": title, "updated_at": "2026-07-21T10:00:00+00:00"}),
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
    (root / "remote_models.json").write_text('[{"api_key": "sk-SECRET"}]', encoding="utf-8")
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
    recap = soul.export_soul(_paths(var), ["aaa111", "bbb222"], dest, "phrase-forte-de-test")
    p = Path(recap["path"])
    assert p.exists() and p.suffix == ".soul" and recap["sessions"] == 2
    assert p.read_bytes().startswith(soul.MAGIC)
    # Deuxième export le même jour : pas d'écrasement, suffixe.
    recap2 = soul.export_soul(_paths(var), [], dest, "phrase-forte-de-test")
    assert recap2["path"] != recap["path"]


def test_export_dest_introuvable(var, tmp_path):
    with pytest.raises(soul.SoulError, match="introuvable"):
        soul.export_soul(_paths(var), [], tmp_path / "nexiste-pas", "p")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `python -m pytest tests/test_soul.py -x -q` — Expected: FAIL (`AttributeError: build_archive`).

- [ ] **Step 3: Implémenter**

Ajouter à `loom/web/soul.py` (note : `from loom.utils import now_iso` en tête de fichier, avec les autres imports) :

```python
def build_archive(paths: SoulPaths, session_ids: list[str]) -> bytes:
    """tar.gz en mémoire : manifest + sessions choisies + mémoire + identité + skills.
    Liste d'inclusion STRICTE — remote_models.json et les logs n'ont aucun chemin
    d'entrée possible ici."""
    buf = io.BytesIO()
    counts = {"sessions": 0, "skills": 0, "identite": 0, "memoire": 0}
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sid in session_ids:
            for fname in SESSION_FILES:
                f = Path(paths.sessions_root) / sid / fname
                if f.exists():
                    tar.add(str(f), arcname=f"sessions/{sid}/{fname}", recursive=False)
            counts["sessions"] += 1
        db = Path(paths.memory_db)
        if db.exists():
            tar.add(str(db), arcname="memory/memory.db", recursive=False)
            counts["memoire"] = 1
        for fname in IDENTITY_FILES:
            f = Path(paths.identity_dir) / fname
            if f.exists():
                tar.add(str(f), arcname=f"identity/{fname}", recursive=False)
                counts["identite"] += 1
        for kind, root in (
            ("skills_learned", paths.learned_skills_dir),
            ("skills_user", paths.user_skills_dir),
        ):
            root = Path(root)
            if not root.is_dir():
                continue
            for skill in sorted(p for p in root.iterdir() if p.is_dir()):
                for f in sorted(skill.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(skill).as_posix()
                        tar.add(str(f), arcname=f"{kind}/{skill.name}/{rel}", recursive=False)
                counts["skills"] += 1
        manifest = {
            "version": 1,
            "date": now_iso(),
            "machine": platform.node(),
            "sessions": list(session_ids),
            "counts": counts,
        }
        mdata = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(mdata)
        tar.addfile(info, io.BytesIO(mdata))
    return buf.getvalue()


def export_soul(paths: SoulPaths, session_ids: list[str], dest_dir, passphrase: str) -> dict:
    dest = Path(dest_dir)
    if not dest.is_dir():
        raise SoulError(f"dossier de destination introuvable : {dest}")
    day = now_iso()[:10]
    out, n = dest / f"ame-loom-{day}.soul", 2
    while out.exists():
        out = dest / f"ame-loom-{day}-{n}.soul"
        n += 1
    blob = encrypt(build_archive(paths, session_ids), passphrase)
    out.write_bytes(blob)
    return {"path": str(out), "size": len(blob), "sessions": len(session_ids)}
```

- [ ] **Step 4: Vérifier le vert**

Run: `python -m pytest tests/test_soul.py -x -q` — Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add loom/web/soul.py tests/test_soul.py
git commit -m "feat(ame): export .soul (archive+manifest, logs et secrets exclus)"
```

---

### Task 5: soul.py — import par fusion

**Files:**
- Modify: `loom/web/soul.py`
- Test: `tests/test_soul.py`

**Interfaces:**
- Consumes: `decrypt`, `build_archive`, `export_soul`, fixture `var` (Task 4).
- Produces: `soul.import_soul(paths: SoulPaths, soul_file, passphrase: str) -> dict` — rapport `{"manifest": {...}, "sessions": {"ajoutees", "ignorees", "remplacees"}, "memoire": {"ajoutes", "ignores"}, "identite": {"poses", "copies_a_cote"}, "skills_learned"/"skills_user": {"ajoutes", "ignores", "renommes"}}`.

- [ ] **Step 1: Tests qui échouent**

Ajouter à `tests/test_soul.py` :

```python
@pytest.fixture()
def soul_file(var, tmp_path):
    dest = tmp_path / "transport"
    dest.mkdir()
    recap = soul.export_soul(_paths(var), ["aaa111", "bbb222"], dest, "phrase-transport")
    return Path(recap["path"])


def test_import_dans_var_vierge(soul_file, tmp_path):
    cible = tmp_path / "cible"
    for d in ("sessions", "skills_learned", "skills_user"):
        (cible / d).mkdir(parents=True)
    rep = soul.import_soul(_paths(cible), soul_file, "phrase-transport")
    assert rep["sessions"]["ajoutees"] == 2
    assert (cible / "sessions" / "aaa111" / "session.json").exists()
    assert (cible / "identity" / "SOUL.md").read_text(encoding="utf-8") == "ame de la machine A"
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
        _json.dumps({"id": "aaa111", "title": "plus recent ici", "updated_at": "2026-07-22T00:00:00+00:00"}),
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
    dst.close(); src.close()

    rep = soul.import_soul(_paths(cible), soul_file, "phrase-transport")
    meta = _json.loads((cible / "sessions" / "aaa111" / "session.json").read_text(encoding="utf-8"))
    assert meta["title"] == "plus recent ici"
    assert rep["sessions"]["ignorees"] == 1 and rep["sessions"]["ajoutees"] == 1
    assert (cible / "skills_learned" / "detect-boucle-importe" / "SKILL.md").exists()
    assert (cible / "identity" / "SOUL.md").read_text(encoding="utf-8") == "ame de la machine B"
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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `python -m pytest tests/test_soul.py -x -q` — Expected: FAIL (`AttributeError: import_soul`).

- [ ] **Step 3: Implémenter la fusion**

Ajouter à `loom/web/soul.py` :

```python
def _updated_at(session_json: Path) -> str:
    # ISO 8601 en +00:00 partout (loom.utils.now_iso) : comparaison lexicale valide.
    try:
        return json.loads(session_json.read_text(encoding="utf-8")).get("updated_at", "")
    except Exception:
        return ""


def _copy_session(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in SESSION_FILES:
        if (src_dir / fname).exists():
            shutil.copy2(src_dir / fname, dst_dir / fname)


def _merge_sessions(src: Path, dst_root: Path) -> dict:
    rep = {"ajoutees": 0, "ignorees": 0, "remplacees": 0}
    if not src.is_dir():
        return rep
    dst_root.mkdir(parents=True, exist_ok=True)
    for sdir in sorted(p for p in src.iterdir() if p.is_dir()):
        dst = dst_root / sdir.name
        if not dst.exists():
            _copy_session(sdir, dst)
            rep["ajoutees"] += 1
        elif _updated_at(sdir / "session.json") > _updated_at(dst / "session.json"):
            _copy_session(sdir, dst)  # même session déplacée deux fois : la plus récente gagne
            rep["remplacees"] += 1
        else:
            rep["ignorees"] += 1
    return rep


def _dir_digest(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(root).as_posix().encode("utf-8"))
            h.update(f.read_bytes())
    return h.hexdigest()


def _merge_skills(src: Path, dst_root: Path) -> dict:
    rep = {"ajoutes": 0, "ignores": 0, "renommes": 0}
    if not src.is_dir():
        return rep
    dst_root.mkdir(parents=True, exist_ok=True)
    for skill in sorted(p for p in src.iterdir() if p.is_dir()):
        dst = dst_root / skill.name
        if not dst.exists():
            shutil.copytree(skill, dst)
            rep["ajoutes"] += 1
            continue
        digest = _dir_digest(skill)
        if digest == _dir_digest(dst):
            rep["ignores"] += 1
            continue
        # Homonyme divergent : import sous suffixe — sauf si un import précédent
        # IDENTIQUE existe déjà (ré-import idempotent, pas de -importe-2 en rafale).
        alt, n = dst_root / f"{skill.name}-importe", 2
        while alt.exists() and _dir_digest(alt) != digest:
            alt = dst_root / f"{skill.name}-importe-{n}"
            n += 1
        if alt.exists():
            rep["ignores"] += 1
        else:
            shutil.copytree(skill, alt)
            rep["renommes"] += 1
    return rep


def _merge_memory(src_db: Path, dst_db: Path) -> dict:
    rep = {"ajoutes": 0, "ignores": 0}
    if not src_db.exists():
        return rep
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    if not dst_db.exists():
        shutil.copy2(src_db, dst_db)  # cible vierge : la base arrive entière (FTS inclus)
        con = sqlite3.connect(dst_db)
        rep["ajoutes"] = con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        con.close()
        return rep
    src, dst = sqlite3.connect(src_db), sqlite3.connect(dst_db)
    try:
        for ts, kind, source, text in src.execute(
            "SELECT ts, kind, source, text FROM episodes ORDER BY id"
        ):
            dup = dst.execute(
                "SELECT 1 FROM episodes WHERE ts=? AND kind=? AND source=? AND text=? LIMIT 1",
                (ts, kind, source, text),
            ).fetchone()
            if dup:
                rep["ignores"] += 1
            else:
                # Les triggers episodes_ai maintiennent l'index FTS à l'insert.
                dst.execute(
                    "INSERT INTO episodes (ts, kind, source, text) VALUES (?, ?, ?, ?)",
                    (ts, kind, source, text),
                )
                rep["ajoutes"] += 1
        dst.commit()
    finally:
        src.close()
        dst.close()
    return rep


def _merge_identity(src: Path, dst_dir: Path, day: str) -> dict:
    rep = {"poses": 0, "copies_a_cote": 0}
    if not src.is_dir():
        return rep
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in IDENTITY_FILES:
        f = src / fname
        if not f.exists():
            continue
        data = f.read_bytes()
        dst = dst_dir / fname
        if not dst.exists():
            dst.write_bytes(data)
            rep["poses"] += 1
        elif dst.read_bytes() != data:
            # Jamais d'écrasement silencieux d'une âme par une autre : la version
            # importée est posée à côté, le user arbitre.
            stem = fname.rsplit(".", 1)[0]
            side = dst_dir / f"{stem}.imported-{day}.md"
            if not side.exists() or side.read_bytes() != data:
                side.write_bytes(data)
                rep["copies_a_cote"] += 1
    return rep


def import_soul(paths: SoulPaths, soul_file, passphrase: str) -> dict:
    src = Path(soul_file)
    if not src.is_file():
        raise SoulError(f"fichier introuvable : {src}")
    data = decrypt(src.read_bytes(), passphrase)  # échec = SoulError AVANT toute écriture
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(tdp, filter="data")  # refuse chemins hors racine, liens, etc.
        except (tarfile.TarError, OSError) as e:
            raise SoulError(f"archive illisible : {e}") from e
        man = tdp / "manifest.json"
        if not man.exists():
            raise SoulError("archive sans manifest — pas un fichier .soul")
        manifest = json.loads(man.read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise SoulError(f"version d'archive inconnue : {manifest.get('version')!r}")
        day = str(manifest.get("date", ""))[:10] or "inconnu"
        return {
            "manifest": {
                "date": manifest.get("date", ""),
                "machine": manifest.get("machine", ""),
            },
            "sessions": _merge_sessions(tdp / "sessions", Path(paths.sessions_root)),
            "memoire": _merge_memory(tdp / "memory" / "memory.db", Path(paths.memory_db)),
            "identite": _merge_identity(tdp / "identity", Path(paths.identity_dir), day),
            "skills_learned": _merge_skills(
                tdp / "skills_learned", Path(paths.learned_skills_dir)
            ),
            "skills_user": _merge_skills(tdp / "skills_user", Path(paths.user_skills_dir)),
        }
```

- [ ] **Step 4: Vérifier le vert**

Run: `python -m pytest tests/test_soul.py -q` — Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add loom/web/soul.py tests/test_soul.py
git commit -m "feat(ame): import par fusion (sessions/skills/memoire/identite), rien detruit"
```

---

### Task 6: routes /soul/* + câblage des chemins

**Files:**
- Modify: `loom/web/app.py` (signature `create_app` + namespace `S`, ~lignes 377-430 et 565-640)
- Modify: `loom/web/routes.py` (nouvelle fonction `_register_soul_routes(app, S)`)
- Modify: `loom/web/__main__.py` (passage de `memory_db_path`)
- Modify: `tests/conftest.py` (isolation des nouveaux chemins)
- Test: `tests/test_soul_routes.py`

**Interfaces:**
- Consumes: `soul.SoulPaths`, `export_soul`, `import_soul`, `check_passphrase`, `generate_passphrase` (Tasks 2-5) ; `S.session_store` (`.list() -> [SessionMeta(id, title, workspace, updated_at)]`, `.root`), `S.identity_paths` (dict `{memory_md_path, user_path, soul_path}` ou None), `S.learned_skills_dir`, `S.user_skills_dir`.
- Produces: `GET /soul/sessions`, `POST /soul/passphrase/check`, `POST /soul/passphrase/generate`, `POST /soul/export`, `POST /soul/import` (formulaires `application/x-www-form-urlencoded`, comme les autres routes POST du projet).

- [ ] **Step 1: Câbler le chemin mémoire**

Dans `loom/web/app.py` : ajouter le kwarg `memory_db_path=None` à `create_app` (près de `identity_paths`), et `memory_db_path=memory_db_path,` dans le `SimpleNamespace S` (près de `identity_paths=identity_paths`).

Dans `loom/web/__main__.py`, à l'appel de `create_app` (~ligne 272, près de `reflect_stores=reflect_stores`), ajouter :

```python
        memory_db_path=cfg.memory.db_path,
```

Dans `tests/conftest.py`, fixture `app`, ajouter aux kwargs de `create_app` (isolation : rien ne doit toucher le vrai `var/`) :

```python
        learned_skills_dir=str(tmp_env / "skills_learned"),
        memory_db_path=str(tmp_env / "memory" / "memory.db"),
        identity_paths={
            "soul_path": str(tmp_env / "identity" / "SOUL.md"),
            "user_path": str(tmp_env / "identity" / "USER.md"),
            "memory_md_path": str(tmp_env / "identity" / "MEMORY.md"),
        },
```

Run: `python -m pytest tests -q` — Expected: tout vert (le câblage seul ne casse rien).

- [ ] **Step 2: Tests de routes qui échouent**

```python
# tests/test_soul_routes.py
# Routes /soul/* : export/import chiffré de l'état portable (spec 2026-07-21).
from __future__ import annotations

import json
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
    r2 = web.post("/soul/passphrase/check", data={"passphrase": g["passphrase"]}).get_json()
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
```

Run: `python -m pytest tests/test_soul_routes.py -q` — Expected: FAIL (404 sur /soul/*).

- [ ] **Step 3: Implémenter les routes**

Dans `loom/web/routes.py`, ajouter en fin de fichier (même style que `_register_model_routes`) :

```python
def _register_soul_routes(app, S):
    """L'Âme : export/import chiffré de l'état portable (sessions, mémoire,
    identité, skills). AUCUN secret ne part (remote_models.json exclu par
    construction dans soul.build_archive). Spec : specs/2026-07-21-ame-*."""
    from loom.web import soul as soul_mod

    def _soul_paths():
        idp = S.identity_paths or {}
        identity_dir = (
            Path(idp["soul_path"]).parent if idp.get("soul_path") else Path("var/identity")
        )
        return soul_mod.SoulPaths(
            sessions_root=Path(S.session_store.root),
            memory_db=Path(S.memory_db_path or "var/memory/memory.db"),
            identity_dir=identity_dir,
            learned_skills_dir=Path(S.learned_skills_dir or "var/skills_learned"),
            user_skills_dir=Path(S.user_skills_dir),
        )

    @app.get("/soul/sessions")
    def soul_sessions():
        return jsonify(
            sessions=[
                {"id": m.id, "title": m.title, "updated_at": m.updated_at}
                for m in S.session_store.list()
            ]
        )

    @app.post("/soul/passphrase/check")
    def soul_passphrase_check():
        return jsonify(soul_mod.check_passphrase(request.form.get("passphrase", "")))

    @app.post("/soul/passphrase/generate")
    def soul_passphrase_generate():
        return jsonify(passphrase=soul_mod.generate_passphrase())

    @app.post("/soul/export")
    def soul_export():
        passphrase = request.form.get("passphrase", "")
        if not soul_mod.check_passphrase(passphrase)["ok"]:
            return jsonify(error="passphrase trop faible (score zxcvbn < 3)"), 400
        ids = [s for s in request.form.get("session_ids", "").split(",") if s.strip()]
        try:
            recap = soul_mod.export_soul(
                _soul_paths(), ids, request.form.get("dest_dir", ""), passphrase
            )
        except soul_mod.SoulError as e:
            return jsonify(error=str(e)), 400
        log_event("soul.export", sessions=len(ids), path=recap["path"])
        return jsonify(recap)

    @app.post("/soul/import")
    def soul_import():
        try:
            report = soul_mod.import_soul(
                _soul_paths(),
                request.form.get("file", ""),
                request.form.get("passphrase", ""),
            )
        except soul_mod.SoulError as e:
            return jsonify(error=str(e)), 400
        # Les sessions fusionnées doivent apparaître sans redémarrage : purge du
        # cache d'objets session (rechargés depuis le disque au prochain accès).
        S.sessions_cache.clear()
        log_event(
            "soul.import",
            ajoutees=report["sessions"]["ajoutees"],
            remplacees=report["sessions"]["remplacees"],
        )
        return jsonify(report=report)
```

Notes d'implémentation : reprendre les imports déjà présents en tête de `routes.py` (`jsonify`, `request`, `Path`, `log_event` — vérifier les noms exacts utilisés par les autres routes du fichier et s'y conformer). Enregistrer la fonction dans `loom/web/app.py`, à l'endroit où les autres `_register_*_routes(app, S)` sont appelées (chercher `_register_model_routes(app, S)`), en ajoutant `routes._register_soul_routes(app, S)` juste après.

ATTENTION (leçon session `4f8599e0a8b2`) : si `S.cur["session"]` référence une session remplacée par l'import, ne PAS la recharger d'office — le cache purgé suffit, la session focus se recharge à la prochaine activation. Ne rien faire de plus.

- [ ] **Step 4: Vérifier le vert + suite complète**

Run: `python -m pytest tests/test_soul_routes.py -q` — Expected: 4 passed.
Run: `python -m pytest tests -q && ruff check loom tests` — Expected: tout vert.

- [ ] **Step 5: Commit**

```bash
git add loom/web/app.py loom/web/routes.py loom/web/__main__.py tests/conftest.py tests/test_soul_routes.py
git commit -m "feat(ame): routes /soul/* (export/import/passphrase) + cablage memory_db_path"
```

---

### Task 7: UI — onglet « Âme » dans la console de configuration

**Files:**
- Modify: `loom/web/templates/index.html` (onglets `#cfg-tabs` ~ligne 1176, panneaux `.cfg-panel` ~ligne 1183, CSS près du bloc « Gestionnaire de modèles distants » ~ligne 486)
- Modify: `loom/web/static/app.js` (nouvelle IIFE après le « Gestionnaire de modèles distants », ~ligne 3327)

**Interfaces:**
- Consumes: routes `/soul/*` (Task 6) ; mécanique d'onglets existante (`.cfg-tab`/`.cfg-panel` par `data-tab`, générique — un nouvel onglet marche sans JS d'onglet supplémentaire).
- Produces: onglet « Âme » avec deux blocs Export / Import.

- [ ] **Step 1: HTML — onglet + panneau**

Dans `index.html`, ajouter le bouton d'onglet à la fin de `#cfg-tabs` :

```html
        <button type="button" class="cfg-tab" data-tab="ame">Âme</button>
```

Puis, après le panneau `#cfg-models` (`</div>` de `data-tab="modeles"`), le panneau :

```html
        <!-- Onglet Âme : export/import chiffré de l'état portable (sessions, mémoire,
             identité, skills). Aucun secret ne part : les clés API restent ici. -->
        <div id="cfg-ame" class="cfg-panel" data-tab="ame" hidden>
          <div class="cfg-group-sub">Emporte tes sessions, ta mémoire, ton identité et tes skills
            dans un fichier chiffré (.soul) — clé USB, disque ou dossier cloud synchronisé.
            Les clés API ne partent jamais. Sur l'autre machine : Importer, même passphrase, fusion sans perte.</div>

          <div class="ame-block">
            <div class="ame-head">Exporter</div>
            <div class="ame-sub">Socle toujours inclus : mémoire, identité, skills. Sessions au choix :</div>
            <label class="ame-all"><input type="checkbox" id="ame-all" checked> toutes les sessions</label>
            <div id="ame-sessions" class="ame-sessions"></div>
            <input id="ame-dest" class="rm-in" placeholder="dossier de destination (E:\ , D:\sauvegardes, dossier Drive...)">
            <div class="ame-pass-row">
              <input id="ame-pass" class="rm-in" type="password" placeholder="passphrase (jamais stockée)" autocomplete="new-password">
              <button type="button" id="ame-eye" class="rm-add-btn" title="afficher/masquer">&#128065;</button>
              <button type="button" id="ame-gen" class="rm-add-btn">générer</button>
            </div>
            <div id="ame-gauge" class="ame-gauge"></div>
            <button type="button" id="ame-export" class="rm-add-btn" disabled>Exporter</button>
          </div>

          <div class="ame-block">
            <div class="ame-head">Importer</div>
            <input id="ame-file" class="rm-in" placeholder="chemin du fichier .soul">
            <div class="ame-pass-row">
              <input id="ame-ipass" class="rm-in" type="password" placeholder="passphrase" autocomplete="off">
              <button type="button" id="ame-ieye" class="rm-add-btn" title="afficher/masquer">&#128065;</button>
            </div>
            <button type="button" id="ame-import" class="rm-add-btn">Importer</button>
          </div>
          <div id="ame-msg" class="rm-msg"></div>
        </div>
```

- [ ] **Step 2: CSS**

Dans le `<style>` d'`index.html`, près du bloc « Gestionnaire de modèles distants » :

```css
    /* --- Âme (export/import chiffré) --- */
    .ame-block { margin: 12px 0; padding: 10px; border: 1px solid var(--border, #333); border-radius: 8px; }
    .ame-head { font-weight: 600; margin-bottom: 6px; }
    .ame-sub { opacity: .75; font-size: .85em; margin-bottom: 6px; }
    .ame-sessions { max-height: 160px; overflow-y: auto; margin: 6px 0; }
    .ame-sessions label { display: block; font-size: .9em; padding: 2px 0; }
    .ame-sessions .ame-date { opacity: .6; font-size: .85em; margin-left: 6px; }
    .ame-pass-row { display: flex; gap: 6px; align-items: center; }
    .ame-pass-row .rm-in { flex: 1; }
    .ame-gauge { font-size: .85em; min-height: 1.2em; margin: 4px 0 8px; opacity: .85; }
    .ame-gauge.ok { color: #7dba7d; }
    .ame-gauge.ko { color: #c98a8a; }
```

(Couleurs mutées, cohérentes avec le thème sombre — pas de rouge/vert criards.)

- [ ] **Step 3: JS — IIFE Âme**

Dans `app.js`, après l'IIFE du gestionnaire de modèles distants, ajouter :

```javascript
// --- Âme (panneau engrenage) : export/import chiffré de l'état portable.
// Jauge zxcvbn côté serveur (source unique), passphrase masquée PAR DÉFAUT
// (anti-screenshot), génération diceware bilingue. ---
(function () {
  const $ = (id) => document.getElementById(id);
  const panel = $("cfg-ame");
  if (!panel) return;
  const esc = (s) =>
    String(s == null ? "" : s).replace(
      /[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
    );
  const msg = (txt, kind) => {
    const m = $("ame-msg");
    m.textContent = txt || "";
    m.className = "rm-msg" + (kind ? " " + kind : "");
  };

  // Liste des sessions (cases cochées par défaut = tout), rechargée à chaque
  // ouverture de l'onglet (les sessions bougent).
  function loadSessions() {
    fetch("/soul/sessions")
      .then((r) => r.json())
      .then((d) => {
        $("ame-sessions").innerHTML = (d.sessions || [])
          .map(
            (s) =>
              '<label><input type="checkbox" class="ame-sess" value="' + esc(s.id) +
              '" checked> ' + esc(s.title || s.id) +
              '<span class="ame-date">' + esc((s.updated_at || "").slice(0, 10)) + "</span></label>",
          )
          .join("");
      });
  }
  document.querySelectorAll('#cfg-tabs [data-tab="ame"]').forEach((b) =>
    b.addEventListener("click", loadSessions),
  );
  $("ame-all").addEventListener("change", (e) => {
    panel.querySelectorAll(".ame-sess").forEach((c) => (c.checked = e.target.checked));
  });

  // Jauge de force : POST débouncé vers le serveur (zxcvbn Python = source unique
  // du verdict ; le bouton Exporter suit `ok`, et le serveur re-vérifie de toute façon).
  let debTimer = null;
  function gauge() {
    clearTimeout(debTimer);
    debTimer = setTimeout(() => {
      const p = $("ame-pass").value;
      if (!p) {
        $("ame-gauge").textContent = "";
        $("ame-export").disabled = true;
        return;
      }
      fetch("/soul/passphrase/check", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "passphrase=" + encodeURIComponent(p),
      })
        .then((r) => r.json())
        .then((d) => {
          const g = $("ame-gauge");
          g.className = "ame-gauge " + (d.ok ? "ok" : "ko");
          g.textContent = d.ok
            ? "force " + d.score + "/4 — crack estimé : " + d.crack_display
            : "trop faible (" + d.score + "/4) — allonge ou clique générer";
          $("ame-export").disabled = !d.ok;
        });
    }, 250);
  }
  $("ame-pass").addEventListener("input", gauge);

  // Générer : remplit le champ SANS le révéler (le user décide via l'œil).
  $("ame-gen").addEventListener("click", () => {
    fetch("/soul/passphrase/generate", { method: "POST" })
      .then((r) => r.json())
      .then((d) => {
        $("ame-pass").value = d.passphrase;
        gauge();
        msg("phrase générée — clique l'œil pour la lire et la mémoriser", "");
      });
  });
  const eye = (inputId, btnId) =>
    $(btnId).addEventListener("click", () => {
      const i = $(inputId);
      i.type = i.type === "password" ? "text" : "password";
    });
  eye("ame-pass", "ame-eye");
  eye("ame-ipass", "ame-ieye");

  $("ame-export").addEventListener("click", () => {
    const ids = Array.from(panel.querySelectorAll(".ame-sess:checked")).map((c) => c.value);
    msg("export en cours…");
    fetch("/soul/export", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body:
        "dest_dir=" + encodeURIComponent($("ame-dest").value) +
        "&passphrase=" + encodeURIComponent($("ame-pass").value) +
        "&session_ids=" + encodeURIComponent(ids.join(",")),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok) return msg(d.error || "échec de l'export", "err");
        msg(
          "exporté : " + d.path + " (" + Math.round(d.size / 1024) + " Ko, " +
          d.sessions + " session(s))",
          "ok",
        );
      })
      .catch(() => msg("échec de l'export (réseau)", "err"));
  });

  $("ame-import").addEventListener("click", () => {
    msg("import en cours…");
    fetch("/soul/import", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body:
        "file=" + encodeURIComponent($("ame-file").value) +
        "&passphrase=" + encodeURIComponent($("ame-ipass").value),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok) return msg(d.error || "échec de l'import", "err");
        const s = d.report.sessions;
        msg(
          "importé : " + s.ajoutees + " session(s) ajoutée(s), " + s.remplacees +
          " remplacée(s), " + s.ignorees + " ignorée(s) ; skills +" +
          (d.report.skills_learned.ajoutes + d.report.skills_user.ajoutes) +
          " ; mémoire +" + d.report.memoire.ajoutes,
          "ok",
        );
        if (typeof loadSessionList === "function") loadSessionList();
        loadSessions();
      })
      .catch(() => msg("échec de l'import (réseau)", "err"));
  });
})();
```

Note : si `loadSessionList` n'existe pas sous ce nom dans `app.js`, chercher la fonction qui recharge la barre des sessions (celle appelée après `/session/new`) et l'appeler à la place ; à défaut, omettre l'appel (l'onglet sessions se recharge au prochain focus).

- [ ] **Step 4: Vérification manuelle rapide**

Lancer une instance éphémère (port 8001, la tuer après) : `python -m loom.web` avec l'env de test habituel, ouvrir la console de configuration → onglet Âme : la liste des sessions se charge, la jauge réagit (faible/forte), « générer » remplit le champ masqué, l'œil bascule. Pas encore d'E2E complet ici (Task 8).

- [ ] **Step 5: Commit**

```bash
git add loom/web/templates/index.html loom/web/static/app.js
git commit -m "feat(ame): onglet Ame (export selectif, jauge passphrase, import fusion)"
```

---

### Task 8: E2E réel + finitions

**Files:**
- Test: manuel/Playwright sur instance éphémère (port 8001)
- Modify: mémoire projet (`MEMORY.md` + fiche)

- [ ] **Step 1: E2E aller-retour complet sur DEUX instances**

Scénario (Playwright MCP ou manuel, instances éphémères port 8001/8002 avec des `var/` jetables — JAMAIS le var/ réel du user en cible d'import) :
1. Instance A : créer une session avec 2-3 messages (modèle stub/distant), ouvrir l'onglet Âme, générer une passphrase (noter sa valeur via l'œil), exporter vers un dossier temporaire. Vérifier le récap.
2. Vérifier sur disque : le `.soul` existe, commence par `LOOMSOUL1`, et `strings`/lecture brute ne montre AUCUN contenu en clair.
3. Instance B (var/ vierge) : onglet Âme → Importer avec une MAUVAISE passphrase → message d'erreur clair, rien créé. Puis bonne passphrase → rapport de fusion, la session apparaît dans la barre de sessions, sa timeline s'ouvre et montre l'historique.
4. Tuer les deux instances (kill TOUS les PIDs sur les ports — leçon des zombies du 2026-07-19).

- [ ] **Step 2: Suite complète + lint**

Run: `python -m pytest tests -q && ruff check loom tests` — Expected: tout vert (394 + ~15 nouveaux).

- [ ] **Step 3: Push + mémoire**

```bash
source ~/.bashrc
git push "https://oauth2:${GITLAB_TOKEN}@gitlab.com/Aharrak/loom.git" master
```

Mettre à jour la mémoire projet : nouvelle fiche `loom-ame-export-import.md` (feature livrée, format .soul, décisions : pas de clés API, fusion, zxcvbn bloquant) + ligne dans `MEMORY.md`.

- [ ] **Step 4: Rapport final au user**

Dire ce qui est vérifié E2E et ce qui ne l'est pas (règle absolue : pas de « ça marche » sans runtime observé). L'E2E du Step 1 EST le test runtime — le citer avec ses résultats concrets.
