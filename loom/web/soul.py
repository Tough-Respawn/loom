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

from loom.utils import now_iso


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


def build_archive(paths: SoulPaths, session_ids: list[str]) -> bytes:
    """tar.gz en mémoire : manifest + sessions choisies + mémoire + identité + skills.
    Liste d'inclusion STRICTE — remote_models.json et les logs n'ont aucun chemin
    d'entrée possible ici."""
    buf = io.BytesIO()
    counts = {"sessions": 0, "skills": 0, "identite": 0, "memoire": 0}
    exported_ids: list[str] = []
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sid in session_ids:
            sdir = Path(paths.sessions_root) / sid
            if not (sdir / "session.json").exists():
                # Session disparue entre l'affichage de la liste et le clic Exporter :
                # ne pas la compter ni la lister — un récap/manifest qui annonce une
                # session absente ferait croire à une sauvegarde qui n'existe pas.
                continue
            for fname in SESSION_FILES:
                f = sdir / fname
                if f.exists():
                    tar.add(str(f), arcname=f"sessions/{sid}/{fname}", recursive=False)
            exported_ids.append(sid)
            counts["sessions"] += 1
        db = Path(paths.memory_db)
        if db.exists():
            # Snapshot via l'API backup : copie COHÉRENTE même si Loom écrit la base
            # au même moment (une copie de fichier à chaud peut embarquer un état
            # transactionnel incohérent qui explosera à l'import sur l'autre machine).
            with tempfile.TemporaryDirectory() as td:
                snap = Path(td) / "memory.db"
                try:
                    src_con = sqlite3.connect(db)
                    dst_con = sqlite3.connect(snap)
                    try:
                        src_con.backup(dst_con)
                    finally:
                        dst_con.close()
                        src_con.close()
                except sqlite3.Error as e:
                    raise SoulError(f"mémoire illisible ({e}) — export annulé") from e
                tar.add(str(snap), arcname="memory/memory.db", recursive=False)
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
                        tar.add(
                            str(f),
                            arcname=f"{kind}/{skill.name}/{rel}",
                            recursive=False,
                        )
                counts["skills"] += 1
        manifest = {
            "version": 1,
            "date": now_iso(),
            "machine": platform.node(),
            "sessions": exported_ids,
            "counts": counts,
        }
        mdata = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(mdata)
        tar.addfile(info, io.BytesIO(mdata))
    return buf.getvalue()


def export_soul(
    paths: SoulPaths, session_ids: list[str], dest_dir, passphrase: str
) -> dict:
    dest = Path(dest_dir)
    if not dest.is_dir():
        raise SoulError(f"dossier de destination introuvable : {dest}")
    day = now_iso()[:10]
    out, n = dest / f"ame-loom-{day}.soul", 2
    while out.exists():
        out = dest / f"ame-loom-{day}-{n}.soul"
        n += 1
    ids = [
        s
        for s in session_ids
        if (Path(paths.sessions_root) / s / "session.json").exists()
    ]
    blob = encrypt(build_archive(paths, ids), passphrase)
    out.write_bytes(blob)
    return {"path": str(out), "size": len(blob), "sessions": len(ids)}


def _updated_at(session_json: Path) -> str:
    # ISO 8601 en +00:00 partout (loom.utils.now_iso) : comparaison lexicale valide.
    try:
        return json.loads(session_json.read_text(encoding="utf-8")).get(
            "updated_at", ""
        )
    except Exception:
        return ""


def _copy_session(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in SESSION_FILES:
        if (src_dir / fname).exists():
            shutil.copy2(src_dir / fname, dst_dir / fname)
        elif (dst_dir / fname).exists():
            # Remplacement intégral : un timeline.jsonl local périmé ne doit pas
            # survivre à un session.json plus récent qui arrive sans timeline
            # (l'UI rejouerait un historique qui ne correspond plus au contexte).
            (dst_dir / fname).unlink()


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
            _copy_session(
                sdir, dst
            )  # même session déplacée deux fois : la plus récente gagne
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
        shutil.copy2(
            src_db, dst_db
        )  # cible vierge : la base arrive entière (FTS inclus)
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
    data = decrypt(
        src.read_bytes(), passphrase
    )  # échec = SoulError AVANT toute écriture
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(
                    tdp, filter="data"
                )  # refuse chemins hors racine, liens, etc.
        except (tarfile.TarError, OSError) as e:
            raise SoulError(f"archive illisible : {e}") from e
        man = tdp / "manifest.json"
        if not man.exists():
            raise SoulError("archive sans manifest — pas un fichier .soul")
        manifest = json.loads(man.read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise SoulError(f"version d'archive inconnue : {manifest.get('version')!r}")
        day = str(manifest.get("date", ""))[:10] or "inconnu"
        # Pré-validation AVANT toute écriture : une base mémoire corrompue ou d'un
        # schéma inconnu doit annuler l'import ENTIER (« état local intact »), pas
        # exploser en plein milieu après que les sessions ont déjà été fusionnées.
        arc_db = tdp / "memory" / "memory.db"
        if arc_db.exists():
            try:
                con = sqlite3.connect(f"file:{arc_db.as_posix()}?mode=ro", uri=True)
                try:
                    con.execute("SELECT ts, kind, source, text FROM episodes LIMIT 1")
                finally:
                    con.close()
            except sqlite3.Error as e:
                raise SoulError(
                    f"mémoire de l'archive illisible ({e}) — import annulé, "
                    "rien n'a été touché"
                ) from e
        try:
            return {
                "manifest": {
                    "date": manifest.get("date", ""),
                    "machine": manifest.get("machine", ""),
                },
                "sessions": _merge_sessions(
                    tdp / "sessions", Path(paths.sessions_root)
                ),
                "memoire": _merge_memory(arc_db, Path(paths.memory_db)),
                "identite": _merge_identity(
                    tdp / "identity", Path(paths.identity_dir), day
                ),
                "skills_learned": _merge_skills(
                    tdp / "skills_learned", Path(paths.learned_skills_dir)
                ),
                "skills_user": _merge_skills(
                    tdp / "skills_user", Path(paths.user_skills_dir)
                ),
            }
        except (sqlite3.Error, OSError, shutil.Error) as e:
            # Filet résiduel (disque plein, permission...) : message honnête —
            # la fusion a pu être partielle, on le dit au lieu d'un 500 muet.
            raise SoulError(
                f"fusion interrompue ({e}) — l'import peut être partiel, "
                "relance-le une fois la cause corrigée (un ré-import est sans danger)"
            ) from e
