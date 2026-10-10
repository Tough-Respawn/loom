# tests/test_gguf_profile.py
"""Profil GGUF EXPLOITABLE (lot 2 du bench de placement).

Le lecteur ne gardait que quelques scalaires et jetait les tableaux ; le KV se
calculait en f16 sur toutes les couches alors que l'exécutant tourne en q8_0 et que
les hybrides (Ornith, Bonsai : 1 couche d'attention sur 4) n'ont pas de KV sur les
couches GDN. Ici : tableaux courts conservés, catalogue des tenseurs lu par OFFSETS
(agnostique au type de quant — Bonsai PQ2_0 porte le type 142), poids par famille et
par couche, et un profil qui dit d'où vient chaque donnée : déclaré, déduit, inconnu.
"""

from __future__ import annotations

import struct

from loom.runtime.gguf_meta import read_gguf_meta
from loom.runtime.model_profile import ModelProfile

_ALIGN = 32


def _s(txt: str) -> bytes:
    b = txt.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, val) -> bytes:
    if isinstance(val, bool):
        return _s(key) + struct.pack("<I", 7) + struct.pack("<B", int(val))
    if isinstance(val, int):
        return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)
    if isinstance(val, float):
        return _s(key) + struct.pack("<I", 6) + struct.pack("<f", val)
    if isinstance(val, str):
        return _s(key) + struct.pack("<I", 8) + _s(val)
    if isinstance(val, list):
        if val and isinstance(val[0], str):
            body = struct.pack("<I", 8) + struct.pack("<Q", len(val))
            body += b"".join(_s(x) for x in val)
        else:
            body = struct.pack("<I", 5) + struct.pack("<Q", len(val))
            body += b"".join(struct.pack("<i", x) for x in val)
        return _s(key) + struct.pack("<I", 9) + body
    raise TypeError(type(val))


def _gguf(path, kvs: dict, tensors: list[tuple[str, int]] | None = None, types=None):
    """GGUF v3 minimal : header, KV, infos de tenseurs (1 dim, taille = octets /4 en
    F32 sauf `types`), section de données remplie de zéros. Les tailles passées sont
    des multiples de 32 pour que la lecture par offsets soit exacte."""
    tensors = tensors or []
    types = types or {}
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(kvs))
    blob += b"".join(_kv(k, v) for k, v in kvs.items())
    offset = 0
    for name, size in tensors:
        ty = types.get(name, 0)
        n_elems = size // 4 if ty == 0 else size
        blob += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", n_elems)
        blob += struct.pack("<I", ty) + struct.pack("<Q", offset)
        offset += size
    pad = (-len(blob)) % _ALIGN
    blob += b"\0" * pad + b"\0" * offset
    path.write_bytes(blob)
    return path


# ── lecteur : tableaux, champs, catalogue ────────────────────────────────────────


def test_tableaux_courts_conserves_tokenizer_traverse(tmp_path):
    p = _gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "qwen35",
            "qwen35.rope.dimension_sections": [11, 11, 10, 0],
            "tokenizer.ggml.tokens": ["a"] * 5000,  # traversé, jamais stocké
            "qwen35.block_count": 64,  # lu APRÈS le gros tableau : curseur aligné
        },
    )
    meta = read_gguf_meta(p)
    assert meta["arrays"]["qwen35.rope.dimension_sections"] == [11, 11, 10, 0]
    assert "tokenizer.ggml.tokens" not in meta["arrays"]
    assert meta["n_layers"] == 64


def test_champs_architecture_supplementaires(tmp_path):
    p = _gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "qwen35moe",
            "qwen35moe.block_count": 41,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_used_count": 8,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.attention.value_length": 128,
            "qwen35moe.attention.sliding_window": 4096,
            "qwen35moe.attention.sliding_window_pattern": 6,
            "qwen35moe.full_attention_interval": 4,
        },
    )
    meta = read_gguf_meta(p)
    assert meta["expert_used_count"] == 8
    assert meta["value_length"] == 128
    assert meta["sliding_window"] == 4096
    assert meta["sliding_window_pattern"] == 6
    assert meta["full_attention_interval"] == 4


def test_catalogue_tenseurs_par_famille_et_par_couche(tmp_path):
    tensors = [
        ("token_embd.weight", 1024),
        ("blk.0.attn_q.weight", 320),
        ("blk.0.attn_k.weight", 64),
        ("blk.0.ffn_up_exps.weight", 4096),
        ("blk.0.ffn_gate_inp.weight", 32),
        ("blk.1.ssm_out.weight", 512),
        ("blk.1.ssm_a", 32),
        ("blk.1.ffn_up.weight", 640),
        ("blk.1.post_attention_norm.weight", 32),
        ("output_norm.weight", 32),
        ("output.weight", 800),
    ]
    p = _gguf(
        tmp_path / "m.gguf",
        {"general.architecture": "qwen35moe", "qwen35moe.block_count": 2},
        tensors,
        types={"blk.0.ffn_up_exps.weight": 142},  # type de quant inconnu : sans effet
    )
    w = read_gguf_meta(p)["weights"]
    assert w["total"] == sum(s for _, s in tensors)
    fam = w["familles"]
    assert fam["embeddings"] == 1024
    assert fam["attention"] == 384
    assert fam["experts"] == 4096
    assert fam["ffn"] == 32 + 640  # routeur + FFN dense
    assert fam["recurrent"] == 544
    assert fam["output"] == 832
    assert fam["autres"] == 32
    assert w["par_couche"] == [384 + 4096 + 32, 544 + 640 + 32]
    assert w["experts_par_couche"] == [4096, 0]
    # Une couche a un cache KV si elle porte une projection K ; récurrente si ssm/wkv…
    assert w["couches_attention"] == [0]
    assert w["couches_recurrentes"] == [1]
    assert "offsets" in w["provenance"]


def test_tete_nextn_exclue_des_couches_a_cache_kv(tmp_path):
    """Ornith 1.5 : block_count = 41 dont le bloc 40 est la tête MTP `nextn` (inutilisée
    à l'inférence) qui porte pourtant une projection K — elle n'a pas de cache KV."""
    tensors = [
        ("blk.0.attn_k.weight", 64),
        ("blk.1.ssm_a", 32),
        ("blk.1.attn_qkv.weight", 64),
        ("blk.2.attn_k.weight", 64),
        ("blk.2.nextn.eh_proj.weight", 128),
    ]
    p = _gguf(
        tmp_path / "m.gguf",
        {"general.architecture": "qwen35moe", "qwen35moe.block_count": 3},
        tensors,
    )
    w = read_gguf_meta(p)["weights"]
    assert w["couches_attention"] == [0]
    assert w["couches_recurrentes"] == [1]
    assert w["couches_nextn"] == [2]


def test_sans_tenseur_pas_de_catalogue(tmp_path):
    p = _gguf(
        tmp_path / "m.gguf", {"general.architecture": "llama", "llama.block_count": 2}
    )
    assert read_gguf_meta(p)["weights"] is None


# ── profil : couches, KV, octets GPU, provenance ─────────────────────────────────


def _meta(**over) -> dict:
    base = {
        "architecture": "qwen35",
        "n_layers": 8,
        "context_length": 262144,
        "expert_count": None,
        "expert_used_count": None,
        "head_count": 16,
        "head_count_kv": 2,
        "embedding_length": 2048,
        "key_length": 256,
        "value_length": 256,
        "sliding_window": None,
        "sliding_window_pattern": None,
        "full_attention_interval": None,
        "recurrent": False,
        "arrays": {},
        "weights": None,
    }
    base.update(over)
    return base


def test_profil_couches_attention_deduites_du_catalogue():
    w = {
        "total": 100,
        "familles": {},
        "par_couche": [50, 50],
        "experts_par_couche": [0, 0],
        "couches_attention": [1],
        "couches_recurrentes": [0],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    prof = ModelProfile.from_meta(_meta(n_layers=2, recurrent=True, weights=w))
    assert prof.attention_layers == [1]
    assert prof.recurrent_layers == [0]
    assert prof.provenance["couches_attention"].startswith("déduit")


def test_profil_couches_attention_par_intervalle_declare_sans_catalogue():
    # Ornith / Bonsai (qwen35*) : full_attention_interval = 4 -> couches 3, 7, 11…
    prof = ModelProfile.from_meta(
        _meta(n_layers=8, recurrent=True, full_attention_interval=4)
    )
    assert prof.attention_layers == [3, 7]
    assert prof.recurrent_layers == [0, 1, 2, 4, 5, 6]
    assert "full_attention_interval" in prof.provenance["couches_attention"]


def test_profil_recurrent_sans_motif_suppose_toutes_les_couches_attention():
    # Borne haute honnête : on ne sait pas quelles couches portent un KV.
    prof = ModelProfile.from_meta(_meta(n_layers=4, recurrent=True))
    assert prof.attention_layers == [0, 1, 2, 3]
    assert prof.provenance["couches_attention"].startswith("inconnu")


def test_profil_kv_q8_0_et_f16():
    prof = ModelProfile.from_meta(_meta(n_layers=8, full_attention_interval=4))
    # 2 couches d'attention x 1024 tokens x 2 têtes KV x (256 K + 256 V) x octets/élément
    assert prof.kv_bytes(1024, "f16") == 2 * 1024 * 2 * 512 * 2
    # q8_0 : bloc de 32 éléments = 34 octets (32 q8 + 1 échelle f16) -> 1,0625 o/élément
    assert prof.kv_bytes(1024, "q8_0") == int(2 * 1024 * 2 * 512 * 34 / 32)
    # Le KV est PAR SLOT : 2 slots = 2 fois.
    assert prof.kv_bytes(1024, "f16", slots=2) == 2 * prof.kv_bytes(1024, "f16")


def test_profil_kv_fenetre_glissante_bornee_par_la_fenetre():
    # Motif gemma : 1 couche pleine toutes les `pattern`, les autres bornées à la fenêtre.
    prof = ModelProfile.from_meta(
        _meta(n_layers=4, sliding_window=512, sliding_window_pattern=2)
    )
    assert prof.swa_layers == [0, 2]
    par_tok = 2 * 512 * 2  # têtes KV x (K+V) x f16
    assert prof.kv_bytes(4096, "f16") == 2 * 4096 * par_tok + 2 * 512 * par_tok


def test_profil_fenetre_sans_motif_reste_conservateur():
    prof = ModelProfile.from_meta(_meta(n_layers=4, sliding_window=512))
    assert prof.swa_layers == []
    assert prof.kv_bytes(4096, "f16") == 4 * 4096 * 2 * 512 * 2
    assert prof.provenance["couches_swa"].startswith("inconnu")


def test_profil_kv_repli_sans_dimensions_d_attention():
    prof = ModelProfile.from_meta(
        _meta(head_count_kv=None, key_length=None, embedding_length=None)
    )
    assert prof.kv_bytes(1000, "q8_0") == 150_000 * 1000
    assert prof.provenance["kv"].startswith("inconnu")


def test_profil_octets_gpu_par_placement():
    w = {
        "total": 1000 + 300 + 4000 + 500 + 600 + 800,
        "familles": {
            "embeddings": 1000,
            "attention": 300,
            "experts": 4000,
            "recurrent": 500,
            "ffn": 600,
            "output": 800,
            "autres": 0,
        },
        "par_couche": [300 + 4000, 500 + 600],
        "experts_par_couche": [4000, 0],
        "couches_attention": [0],
        "couches_recurrentes": [1],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    prof = ModelProfile.from_meta(_meta(n_layers=2, expert_count=8, weights=w))
    # Les embeddings d'entrée restent sur CPU ; la sortie suit seulement l'offload total.
    assert prof.gpu_bytes(ngl=999) == 4300 + 1100 + 800
    assert prof.gpu_bytes(ngl=2) == 4300 + 1100
    # -ngl k : les k DERNIÈRES couches (llama.cpp : i_gpu_start = n_layer - ngl).
    assert prof.gpu_bytes(ngl=1) == 1100
    assert prof.gpu_bytes(ngl=0) == 0
    assert prof.gpu_bytes(ngl=999, cpu_moe=True) == 300 + 1100 + 800
    assert prof.gpu_bytes(ngl=999, n_cpu_moe=1) == 300 + 1100 + 800
    assert prof.gpu_bytes(ngl=999, n_cpu_moe=0) == 4300 + 1100 + 800


def test_profil_sans_catalogue_octets_gpu_inconnus():
    prof = ModelProfile.from_meta(_meta(expert_count=8))
    assert prof.gpu_bytes(ngl=999) is None
    assert prof.gpu_bytes(ngl=999, cpu_moe=True) is None
    assert prof.provenance["poids"].startswith("inconnu")


def test_profil_provenance_declare_deduit_inconnu():
    prof = ModelProfile.from_meta(_meta(value_length=None, expert_count=128))
    assert prof.value_length == 256  # = key_length
    assert prof.provenance["value_length"].startswith("déduit")
    assert prof.provenance["expert_count"] == "déclaré"
    assert prof.moe is True
    lignes = prof.describe()
    assert any("déduit" in ln for ln in lignes) and any(
        "déclaré" in ln for ln in lignes
    )
