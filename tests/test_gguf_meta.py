# Lecture du header GGUF sur un fichier MINIMAL fabriqué à la main (format v3 :
# magic, version, tensor_count, kv_count, puis paires clé/valeur typées).
import struct

import pytest

from loom.runtime.gguf_meta import read_gguf_meta


def _s(txt: bytes) -> bytes:  # string GGUF = u64 longueur + octets
    return struct.pack("<Q", len(txt)) + txt


def _kv_str(key: bytes, val: bytes) -> bytes:  # type 8 = string
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_u32(key: bytes, val: int) -> bytes:  # type 4 = uint32
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_arr_str(key: bytes, items: list[bytes]) -> bytes:  # type 9 = array (de strings)
    body = struct.pack("<I", 8) + struct.pack("<Q", len(items))
    for it in items:
        body += _s(it)
    return _s(key) + struct.pack("<I", 9) + body


def _write(path, kvs: list[bytes]):
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kvs))
    path.write_bytes(blob + b"".join(kvs))


def test_meta_complete(tmp_path):
    p = tmp_path / "m.gguf"
    _write(
        p,
        [
            _kv_str(b"general.architecture", b"qwen3moe"),
            _kv_u32(b"qwen3moe.block_count", 48),
            _kv_u32(b"qwen3moe.context_length", 262144),
            _kv_u32(b"qwen3moe.expert_count", 128),
            # un array (façon vocab tokenizer) : doit être TRAVERSÉ sans casser
            _kv_arr_str(b"tokenizer.ggml.tokens", [b"a", b"b", b"c"]),
        ],
    )
    meta = read_gguf_meta(p)
    assert meta["architecture"] == "qwen3moe"
    assert meta["n_layers"] == 48
    assert meta["context_length"] == 262144
    assert meta["expert_count"] == 128


def test_meta_dense_sans_experts(tmp_path):
    p = tmp_path / "m.gguf"
    _write(
        p,
        [_kv_str(b"general.architecture", b"llama"), _kv_u32(b"llama.block_count", 32)],
    )
    meta = read_gguf_meta(p)
    assert meta["n_layers"] == 32
    assert meta["expert_count"] is None
    assert meta["recurrent"] is False


def test_meta_memoire_recurrente(tmp_path):
    # Bonsai 2 / Ornith (qwen35, Gated DeltaNet) portent des clés `<arch>.ssm.*`
    p = tmp_path / "m.gguf"
    _write(
        p,
        [
            _kv_str(b"general.architecture", b"qwen35"),
            _kv_u32(b"qwen35.block_count", 64),
            _kv_u32(b"qwen35.ssm.state_size", 128),
        ],
    )
    assert read_gguf_meta(p)["recurrent"] is True


def test_pas_un_gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"PASGGUF!")
    with pytest.raises(ValueError):
        read_gguf_meta(p)


def test_gguf_tronque_leve_valueerror(tmp_path):
    # Magic OK mais header coupé net : struct.error doit devenir ValueError
    # (contrat best-effort de finalize_model_toml).
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUFxxxx")
    with pytest.raises(ValueError):
        read_gguf_meta(p)


def test_longueurs_aberrantes_levent_valueerror(tmp_path, monkeypatch):
    """Vérification adverse : une longueur de chaîne de 2^62 (en-tête corrompu) faisait
    lever MemoryError/OverflowError, qu'aucun appelant n'attrape — loom-setup mourait
    sans verdict. Comme gguf.cpp (chaînes et tableaux bornés à 1 Gi, n_dims à 4) :
    ValueError, donc « GGUF illisible ». Chaque borne avec SON message (une seule
    retirée, le test casse), n_dims sur un fichier par ailleurs COMPLET."""
    import struct

    from loom.runtime import gguf_meta

    entete = b"GGUF" + struct.pack("<I", 3)
    chaine = tmp_path / "chaine.gguf"
    chaine.write_bytes(entete + struct.pack("<QQ", 0, 1) + struct.pack("<Q", 1 << 62))
    with pytest.raises(ValueError, match="chaîne GGUF"):
        read_gguf_meta(chaine)
    p = tmp_path / "tableau.gguf"
    kv = struct.pack("<Q", 3) + b"cle" + struct.pack("<I", 9)  # tableau
    kv += struct.pack("<I", 4) + struct.pack("<Q", 1 << 40)  # 2^40 éléments u32
    p.write_bytes(entete + struct.pack("<QQ", 0, 1) + kv)
    with pytest.raises(ValueError, match="tableau GGUF"):
        read_gguf_meta(p)
    arch = b"general.architecture"
    kv = struct.pack("<Q", len(arch)) + arch + struct.pack("<I", 8)
    kv += struct.pack("<Q", 5) + b"llama"
    tenseur = struct.pack("<Q", 1) + b"t" + struct.pack("<I", 5)  # 5 dimensions
    tenseur += struct.pack("<5Q", 4, 4, 4, 4, 4) + struct.pack("<IQ", 0, 0)
    blob = entete + struct.pack("<QQ", 1, 1) + kv + tenseur
    p = tmp_path / "dims.gguf"
    p.write_bytes(blob + b"\0" * ((-len(blob)) % 32) + b"\0" * 4096)
    with pytest.raises(ValueError, match="5 dimensions"):
        read_gguf_meta(p)
    # Repli : une allocation démesurée SOUS les bornes devient ValueError aussi.
    monkeypatch.setattr(gguf_meta, "_MAX_CHAINE", 1 << 63)
    with pytest.raises(ValueError, match="tronqué ou corrompu"):
        read_gguf_meta(chaine)


def test_alignement_invalide_rejete_comme_gguf_cpp(tmp_path):
    """gguf.cpp exige general.alignment en uint32, puissance de 2 (gguf.cpp:622-636).
    Un tableau faisait lever TypeError (int(list)), qu'aucun appelant n'attrape."""
    import struct

    entete = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1)
    cle = struct.pack("<Q", 17) + b"general.alignment"
    p = tmp_path / "tableau.gguf"
    tableau = struct.pack("<I", 9) + struct.pack("<IQ", 4, 1) + struct.pack("<I", 32)
    p.write_bytes(entete + cle + tableau)
    with pytest.raises(ValueError, match="alignment"):
        read_gguf_meta(p)
    p = tmp_path / "trois.gguf"
    p.write_bytes(entete + cle + struct.pack("<I", 4) + struct.pack("<I", 3))
    with pytest.raises(ValueError, match="alignment"):
        read_gguf_meta(p)
