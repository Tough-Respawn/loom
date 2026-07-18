# loom/runtime/gguf_meta.py
"""Lecture MINIMALE du header GGUF, sans dépendance externe.

On ne lit QUE le header (magic/version/counts + paires clé/valeur) — jamais les
tenseurs — pour compléter model.toml après téléchargement : n_layers (block_count),
contexte max (context_length), MoE (expert_count). Spécification du format :
https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from pathlib import Path

# Types scalaires GGUF -> format struct (little-endian). 8=string et 9=array sont
# traités à part. Le type 7 (bool) est lu comme u8.
_SCALAR = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "B",
    10: "Q",
    11: "q",
    12: "d",
}


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    if vtype == 8:
        return _read_string(f)
    if vtype == 9:
        (itype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        # Les arrays (ex. vocab tokenizer) doivent être TRAVERSÉS pour atteindre les
        # clés suivantes ; leur contenu ne nous sert pas, on le jette.
        for _ in range(count):
            _read_value(f, itype)
        return None
    fmt = _SCALAR.get(vtype)
    if fmt is None:
        raise ValueError(f"type GGUF inconnu : {vtype}")
    (v,) = struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))
    return v


def read_gguf_meta(path: str | Path) -> dict:
    """{'architecture','n_layers','context_length','expert_count', + champs
    d'attention pour le calcul du cache KV : 'head_count','head_count_kv',
    'embedding_length','key_length'} (None si absents).

    Lève ValueError si le fichier n'est pas un GGUF lisible — l'appelant traite ça
    en best-effort (un GGUF exotique n'empêche pas l'installation)."""
    kv: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("pas un fichier GGUF")
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                raise ValueError(f"GGUF v{version} non géré")
            _tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            for _ in range(kv_count):
                key = _read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                kv[key] = _read_value(f, vtype)
    except struct.error as exc:  # header tronqué/corrompu = pas un GGUF valide
        raise ValueError(f"header GGUF tronqué ou corrompu ({exc})") from exc

    arch = kv.get("general.architecture")

    def _int(suffix: str) -> int | None:
        v = kv.get(f"{arch}.{suffix}") if arch else None
        return int(v) if isinstance(v, int) else None

    return {
        "architecture": arch,
        "n_layers": _int("block_count"),
        "context_length": _int("context_length"),
        "expert_count": _int("expert_count"),
        # Attention (pour estimer le cache KV par token) : head_dim = key_length
        # si présent, sinon embedding_length / head_count.
        "head_count": _int("attention.head_count"),
        "head_count_kv": _int("attention.head_count_kv"),
        "embedding_length": _int("embedding_length"),
        "key_length": _int("attention.key_length"),
    }
