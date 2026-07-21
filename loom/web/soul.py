# loom/web/soul.py
"""L'Âme : export/import chiffré de l'état portable de Loom.

Un fichier .soul = tar.gz (manifest + sessions + mémoire + identité + skills)
chiffré AES-256-GCM, clé dérivée de la passphrase par scrypt (memory-hard :
~100 ms/essai, brute force offline neutralisé). AUCUN secret dans l'archive :
remote_models.json est exclu par construction. Logique pure (chemins explicites),
testable sans Flask. Spec : docs/superpowers/specs/2026-07-21-ame-export-import-design.md
"""

from __future__ import annotations

import secrets
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
