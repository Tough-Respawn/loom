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


def test_cles_de_completude_exposees(tmp_path):
    """Lot L2 (précontrôle) : les clés qui rendent l'estimation du KV ou du catalogue
    incertaine sont exposées — GGUF en plusieurs parties, MLA, KV partagé, têtes SWA
    dédiées, head_count_kv en tableau."""
    p = _gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "deepseek2",
            "deepseek2.block_count": 4,
            "split.count": 3,
            "deepseek2.attention.key_length_mla": 192,
            "deepseek2.attention.kv_lora_rank": 512,
            "deepseek2.attention.shared_kv_layers": 2,
            "deepseek2.attention.key_length_swa": 128,
            "deepseek2.attention.head_count_kv": [1, 2, 1, 2],
        },
    )
    meta = read_gguf_meta(p)
    assert meta["split_count"] == 3
    assert meta["key_length_mla"] == 192 and meta["kv_lora_rank"] == 512
    assert meta["shared_kv_layers"] == 2 and meta["key_length_swa"] == 128
    assert meta["head_count_kv_array"] is True and meta["head_count_kv"] is None
    simple = read_gguf_meta(
        _gguf(
            tmp_path / "s.gguf",
            {"general.architecture": "llama", "llama.block_count": 2},
        )
    )
    assert simple["split_count"] is None and simple["head_count_kv_array"] is False


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
    # Règle de llama.cpp (src/llama-model.cpp:1619-1644) : i_gpu_start =
    # max(n_layer + 1 - ngl, 0), la couche de SORTIE (indice n_layer) passe EN PREMIER,
    # la couche 0 en dernier ; les embeddings d'entrée restent toujours sur CPU.
    assert prof.gpu_bytes(ngl=999) == 4300 + 1100 + 800
    assert prof.gpu_bytes(ngl=3) == 4300 + 1100 + 800  # n + 1 : tout
    assert prof.gpu_bytes(ngl=2) == 1100 + 800  # ngl = n : la couche 0 reste sur CPU
    assert prof.gpu_bytes(ngl=1) == 800  # la sortie seule
    assert prof.gpu_bytes(ngl=0) == 0
    assert prof.gpu_bytes(ngl=999, cpu_moe=True) == 300 + 1100 + 800
    assert prof.gpu_bytes(ngl=999, n_cpu_moe=1) == 300 + 1100 + 800
    assert prof.gpu_bytes(ngl=999, n_cpu_moe=0) == 4300 + 1100 + 800
    # Côté hôte : tout ce qui n'est pas sur le device (embeddings + couche 0 à ngl 2).
    assert prof.host_bytes(ngl=2) == 1000 + 4300
    assert prof.host_bytes(ngl=999) == 1000
    assert prof.host_bytes(ngl=0) == w["total"]


def test_profil_bloc_nextn_non_charge_mais_compte_dans_la_regle():
    """Revue de conception (2026-10-10) : sans MTP, llama.cpp crée le bloc nextn en
    TENSOR_SKIP (src/llama-model.cpp:3833-3851) : ses octets ne sont jamais alloués,
    mais son indice compte dans n_layer_all pour -ngl. Ornith 1.5 : « -ngl 2 » = la
    sortie seule (le bloc 40 est le premier offloadé après elle)."""
    w = {
        "total": 1000 + 300 + 300 + 900 + 500,
        "familles": {"embeddings": 1000, "output": 500},
        "par_couche": [300, 300, 900],
        "experts_par_couche": [0, 0, 800],
        "couches_attention": [0, 1],
        "couches_recurrentes": [],
        "couches_nextn": [2],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    prof = ModelProfile.from_meta(_meta(n_layers=3, weights=w))
    assert prof.loaded_bytes() == w["total"] - 900
    assert prof.gpu_bytes(ngl=999) == 300 + 300 + 500
    assert prof.gpu_bytes(ngl=2) == 500  # sortie + bloc nextn (non chargé)
    assert prof.gpu_bytes(ngl=3) == 300 + 500
    assert prof.host_bytes(ngl=999) == 1000
    assert prof.host_bytes(ngl=0) == w["total"] - 900


def test_profil_sortie_liee_dupliquee_sur_le_device():
    """Sans output.weight, llama.cpp duplique token_embd comme sortie (TENSOR_DUPLICATED)
    sur le device de la sortie : une vraie allocation device dès ngl >= 1."""
    w = {
        "total": 1000 + 400 + 400,
        "familles": {"embeddings": 1000, "output": 0},
        "par_couche": [400, 400],
        "experts_par_couche": [0, 0],
        "couches_attention": [0, 1],
        "couches_recurrentes": [],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    prof = ModelProfile.from_meta(_meta(n_layers=2, weights=w))
    assert prof.gpu_bytes(ngl=999) == 800 + 1000
    assert prof.gpu_bytes(ngl=1) == 1000
    assert prof.gpu_bytes(ngl=0) == 0
    # L'hôte garde token_embd (la copie device s'ajoute, elle ne le remplace pas).
    assert prof.host_bytes(ngl=999) == 1000


def test_sortie_liee_detectee_sur_un_catalogue_lu(tmp_path):
    """Revue adverse : le lecteur range `output_norm` dans la famille « output » — une
    sortie liée (aucun `output.weight`) n'était donc jamais vue, et la copie de
    token_embd sur le device manquait. Le catalogue dit si la sortie est liée."""
    tensors = [
        ("token_embd.weight", 4096),
        ("blk.0.attn_k.weight", 2048),
        ("blk.1.attn_k.weight", 2048),
        ("output_norm.weight", 32),
    ]
    kvs = {"general.architecture": "llama", "llama.block_count": 2}
    meta = read_gguf_meta(_gguf(tmp_path / "liee.gguf", kvs, tensors))
    assert meta["weights"]["sortie_liee"] is True
    assert meta["weights"]["familles"]["output"] == 32  # output_norm seul
    prof = ModelProfile.from_meta(meta)
    assert prof.gpu_bytes(ngl=999) == 2 * 2048 + 32 + 4096
    assert prof.gpu_bytes(ngl=1) == 32 + 4096
    assert prof.host_bytes(ngl=999) == 4096
    non_liee = read_gguf_meta(
        _gguf(tmp_path / "non.gguf", kvs, tensors + [("output.weight", 1024)])
    )
    assert non_liee["weights"]["sortie_liee"] is False
    assert ModelProfile.from_meta(non_liee).gpu_bytes(ngl=1) == 32 + 1024


def test_profil_couches_sur_le_device_selon_ngl():
    prof = ModelProfile.from_meta(_meta(n_layers=4, full_attention_interval=2))
    assert prof.device_layers(ngl=999) == {0, 1, 2, 3}
    assert prof.device_layers(ngl=4) == {1, 2, 3}
    assert prof.device_layers(ngl=1) == set()
    assert prof.device_layers(ngl=0) == set()
    # Aucun device dans le build : rien n'est offloadé, quel que soit -ngl.
    assert prof.device_layers(ngl=999, has_device=False) == set()


def test_profil_sans_catalogue_octets_gpu_inconnus():
    prof = ModelProfile.from_meta(_meta(expert_count=8))
    assert prof.gpu_bytes(ngl=999) is None
    assert prof.gpu_bytes(ngl=999, cpu_moe=True) is None
    assert prof.provenance["poids"].startswith("inconnu")


def test_champs_ssm_lus_dans_le_header(tmp_path):
    p = _gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "qwen35",
            "qwen35.block_count": 64,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.inner_size": 6144,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.group_count": 16,
        },
    )
    meta = read_gguf_meta(p)
    assert meta["ssm_conv_kernel"] == 4 and meta["ssm_inner_size"] == 6144
    assert meta["ssm_state_size"] == 128 and meta["ssm_group_count"] == 16


def test_profil_etat_recurrent_bonsai_environ_150_mio_par_checkpoint():
    """Bonsai 2 (qwen35, 64 couches dont 48 GDN) : chaque checkpoint pèse l'état
    récurrent complet, observé ~150 Mio (2026-09-30). Formule llama.cpp : par couche
    récurrente, conv (d_conv-1) x (d_inner + 2 x n_group x d_state) + état d_state x
    d_inner, en f32."""
    prof = ModelProfile.from_meta(
        _meta(
            architecture="qwen35",
            n_layers=64,
            recurrent=True,
            full_attention_interval=4,
            ssm_conv_kernel=4,
            ssm_inner_size=6144,
            ssm_state_size=128,
            ssm_group_count=16,
        )
    )
    assert len(prof.recurrent_layers) == 48
    par_couche = (3 * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    assert prof.recurrent_state_bytes == 48 * par_couche
    assert 148 * 1024 * 1024 < prof.recurrent_state_bytes < 152 * 1024 * 1024
    assert prof.provenance["etat_recurrent"].startswith("déduit")
    # Par slot : l'état vivant + les checkpoints ; x slots.
    assert (
        prof.recurrent_bytes(slots=2, checkpoints=32)
        == 2 * 33 * prof.recurrent_state_bytes
    )


def test_profil_etat_recurrent_inconnu_sans_dimensions():
    prof = ModelProfile.from_meta(
        _meta(n_layers=8, recurrent=True, full_attention_interval=4)
    )
    assert prof.recurrent_state_bytes is None
    assert prof.provenance["etat_recurrent"].startswith("inconnu")
    assert prof.recurrent_bytes(slots=2, checkpoints=32) == 0


def test_profil_purement_recurrent_kv_nul_pas_un_forfait():
    """Revue P2 (2026-10-10) : un Mamba sans aucune couche d'attention recevait 18,3 Gio
    de KV « de secours » à 65 536 / 2 slots. « Aucun KV » (couches connues, aucune
    d'attention) n'est pas « KV inconnu »."""
    w = {
        "total": 1000,
        "familles": {},
        "par_couche": [10] * 48,
        "experts_par_couche": [0] * 48,
        "couches_attention": [],
        "couches_recurrentes": list(range(48)),
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    prof = ModelProfile.from_meta(
        _meta(
            architecture="mamba",
            n_layers=48,
            recurrent=True,
            head_count_kv=None,
            key_length=None,
            embedding_length=None,
            weights=w,
        )
    )
    assert prof.attention_layers == [] and prof.kv_bytes(65_536, "q8_0", 2) == 0
    assert prof.provenance["kv"].startswith("déduit (aucune couche")
    # Couches INCONNUES et dimensions absentes : le forfait de secours reste (borne haute).
    flou = ModelProfile.from_meta(
        _meta(
            n_layers=4,
            recurrent=True,
            head_count_kv=None,
            key_length=None,
            embedding_length=None,
        )
    )
    assert flou.kv_bytes(1000, "q8_0") == 150_000 * 1000


def test_metadonnees_inconnues_gardent_le_forfait_kv():
    """Revue #14 P2 (2026-10-10) : `from_meta({})` (repli de loom-setup quand le GGUF est
    illisible) rendait un KV NUL — « aucune couche connue » lu comme « aucune couche
    d'attention établie ». Zéro est réservé à une absence d'attention ÉTABLIE (catalogue
    des tenseurs) ; sinon forfait de secours et provenance « inconnu »."""
    from loom.runtime.model_profile import KV_FALLBACK_BYTES_PER_TOKEN

    vide = ModelProfile.from_meta({}, model_size_mb=5_600)
    assert vide.kv_bytes(8192, "q8_0", 2) == KV_FALLBACK_BYTES_PER_TOKEN * 8192 * 2
    assert vide.provenance["couches_attention"].startswith("inconnu")
    assert vide.provenance["kv"].startswith("inconnu")
    # Récurrent déclaré mais sans nombre de couches : inconnu aussi, pas zéro.
    rec = ModelProfile.from_meta({"recurrent": True})
    assert rec.kv_bytes(1000, "q8_0") == KV_FALLBACK_BYTES_PER_TOKEN * 1000
    # Un catalogue VIDE de couches (poids sans couche) ne prouve rien non plus.
    sans_couche = ModelProfile.from_meta(
        {
            "weights": {
                "total": 0,
                "familles": {},
                "par_couche": [],
                "experts_par_couche": [],
                "couches_attention": [],
                "couches_recurrentes": [],
                "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
            }
        }
    )
    assert sans_couche.kv_bytes(1000, "q8_0") == KV_FALLBACK_BYTES_PER_TOKEN * 1000


def test_profil_sans_recurrence_etat_nul():
    prof = ModelProfile.from_meta(_meta(n_layers=8))
    assert (
        prof.recurrent_state_bytes == 0
        and prof.recurrent_bytes(slots=2, checkpoints=8) == 0
    )


def test_estimation_memoire_par_slot_kv_plus_etat_recurrent():
    from loom.setup.placement import memory_estimate_mb

    prof = ModelProfile.from_meta(
        _meta(
            n_layers=8,
            recurrent=True,
            full_attention_interval=4,
            ssm_conv_kernel=4,
            ssm_inner_size=4096,
            ssm_state_size=128,
            ssm_group_count=16,
        )
    )
    kv = prof.kv_bytes(8192, "q8_0", 2)
    rec = prof.recurrent_bytes(slots=2, checkpoints=8)
    est = memory_estimate_mb(prof, 8192, gpu_tuning=True, slots=2, checkpoints=8)
    assert est["kv_mb"] == kv // (1024 * 1024)
    assert est["recurrent_mb"] == rec // (1024 * 1024)
    assert est["total_mb"] == est["kv_mb"] + est["recurrent_mb"]
    assert est["checkpoints"] == 8
    # Revue P1 : les checkpoints sont en mémoire HÔTE (tableaux du serveur), l'état
    # vivant suit le KV sur le device. La répartition est explicite.
    vivant = prof.recurrent_bytes(slots=2, checkpoints=0) // (1024 * 1024)
    assert est["recurrent_live_mb"] == vivant
    assert est["checkpoints_mb"] == est["recurrent_mb"] - vivant
    assert est["device_mb"] == est["kv_mb"] + est["recurrent_live_mb"]
    assert est["host_mb"] == est["checkpoints_mb"]
    assert est["total_mb"] == est["device_mb"] + est["host_mb"]


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
