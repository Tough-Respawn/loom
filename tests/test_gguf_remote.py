# Recommandation de quant à partir de l'en-tête GGUF lu à distance (sans réseau ici).
import struct

from loom.runtime import gguf_remote as gr
from loom.runtime.model_install import (
    RESERVE_CTX,
    VRAM_OVERHEAD_MB,
    family_stem,
    recommend_quant,
    suggest_alternatives,
    third_party_origin,
    vram_reserve_mb,
)


def _gguf(tensors, arch="qwen35", experts=(0, 0), alignment=32, n_embd=0, extra=()):
    """GGUF v3 minimal : métadonnées + infos de tenseurs (nom, type, éléments, octets)."""

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kv = [("general.architecture", 8, s(arch)), ("general.alignment", 4, struct.pack("<I", alignment)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s("a") + s("b"))]  # fmt: skip
    if n_embd:
        kv.append((f"{arch}.embedding_length", 4, struct.pack("<I", n_embd)))
    kv += list(extra)
    if experts[0]:
        kv += [(f"{arch}.expert_count", 4, struct.pack("<I", experts[0])),
               (f"{arch}.expert_used_count", 4, struct.pack("<I", experts[1]))]  # fmt: skip
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        head += s(k) + struct.pack("<I", t) + v
    off, infos = 0, b""
    for name, ttype, n_elem, nbytes in tensors:
        infos += s(name) + struct.pack("<I", 1) + struct.pack("<Q", n_elem)
        infos += struct.pack("<IQ", ttype, off)
        off += nbytes
    body = head + infos
    data_start = -(-len(body) // alignment) * alignment
    blob = body + b"\0" * (data_start - len(body)) + b"\0" * off

    def fetch(start, end):
        return blob[start : end + 1], len(blob)

    return fetch


def test_en_tete_dense_octets_exacts_et_types():
    fetch = _gguf([("token_embd.weight", 42, 100, 1000), ("blk.0.attn_q.weight", 42, 400, 4000),
                   ("output.weight", 30, 100, 2000)])  # fmt: skip
    s = gr.summarize(gr.read_header(fetch))
    assert s["moe"] is False and s["n_params"] == 600 and s["unknown_types"] == []
    assert abs(s["f_emb"] - 1000 / 7000) < 1e-9


def test_en_tete_moe_part_des_experts_et_type_du_fork():
    fetch = _gguf([("blk.0.ffn_up_exps.weight", 142, 800, 8000), ("blk.0.attn_q.weight", 8, 100, 2000)],
                  experts=(256, 8))  # fmt: skip
    s = gr.summarize(gr.read_header(fetch))
    assert s["moe"] is True and s["expert_used"] == 8 and abs(s["f_exp"] - 0.8) < 1e-9
    assert s["unknown_types"] == [142]  # PQ2_0 du fork Prism


def test_libelles_de_quant():
    assert gr.quant_label("Ternary-Bonsai-2-27B-PQ2_0.gguf") == "PQ2_0"
    assert gr.quant_label("Qwen3.8-27B-UD-Q4_K_XL.gguf") == "Q4_K_XL"
    assert gr.quant_label("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf") == "BF16"
    assert gr.quant_label("Ornith-1.5-35B-Q8_0.gguf") == "Q8_0"
    sup = {"Q4_K", "Q2_0", "Q8_0"}
    assert gr.label_supported("Q4_K_XL", sup) and not gr.label_supported("PQ2_0", sup)
    assert gr.label_supported("Q6_K_M", sup | {"Q6_K"})  # nom unsloth, types standard


_DENSE = {"moe": False, "expert_count": 0, "expert_used": 0, "f_exp": 0.0,
          "f_emb": 0.05, "n_params": 27_000_000_000, "unknown_types": [], "file": "x", "split": False}  # fmt: skip
_SUP = {"Q2_0", "Q4_K", "Q8_0", "IQ2_XXS", "IQ1_S", "F16", "BF16"}


def _f(name, gb):
    return {"filename": name, "size_mb": int(gb * 1024)}


def test_dense_27b_sur_6_go_rien_ne_fait_le_taff():
    # Vécu 2026-10-17 : les règles successives ont recommandé le F16 de 50 Go, puis
    # l'IQ2_XXS qui déborde de la VRAM. Un dense qui n'y tient pas entier ne tourne pas.
    files = [
        _f("Q-IQ1_S.gguf", 5.8),
        _f("Q-UD-IQ1_M.gguf", 6.9),  # vécu : unsloth, ≈ 2,0 bits effectifs
        _f("Q-IQ2_XXS.gguf", 7.0),
        _f("Q-Q4_K_M.gguf", 15.3),
        _f("Q-PQ2_0.gguf", 6.7),
        _f("Q-F16.gguf", 50.1),
    ]
    out = recommend_quant(
        files, 6143, 65297, analysis=_DENSE, supported=_SUP | {"Q4_K_M", "IQ1_M"}
    )
    by = {f["filename"]: f for f in out}
    assert not any(f["recommended"] for f in out)
    assert by["Q-PQ2_0.gguf"]["status"] == "unsupported"
    assert by["Q-UD-IQ1_M.gguf"]["status"] == by["Q-IQ1_S.gguf"]["status"] == "degraded"
    assert by["Q-F16.gguf"]["status"] == "ok"  # la pleine précision n'est plus bannie
    assert by["Q-Q4_K_M.gguf"]["tps"] < by["Q-IQ2_XXS.gguf"]["tps"] < 8


def test_dense_27b_sur_12_go_le_plus_gros_qui_tient_entier():
    # Rien d'au moins 4 bits n'y tient : le plus gros qui y tient entier, même sous 4 bits.
    files = [_f("Q-IQ2_XXS.gguf", 7.0), _f("Q-IQ3_XXS.gguf", 10.5),
             _f("Q-Q4_K_M.gguf", 15.3)]  # fmt: skip
    sup = _SUP | {"Q4_K_M", "IQ3_XXS"}
    out = recommend_quant(files, 12288, 65297, analysis=_DENSE, supported=sup)
    by = {f["filename"]: f for f in out}
    assert by["Q-IQ3_XXS.gguf"]["recommended"]
    assert by["Q-Q4_K_M.gguf"]["tps"] < 8


def test_la_pleine_precision_quand_elle_tient_en_vram():
    small = dict(_DENSE, n_params=4_000_000_000)
    files = [_f("S-Q4_K_M.gguf", 2.5), _f("S-Q8_0.gguf", 4.3), _f("S-F16.gguf", 8.0)]
    out = recommend_quant(
        files, 24576, 65297, analysis=small, supported=_SUP | {"Q4_K_M"}
    )
    by = {f["filename"]: f for f in out}
    assert by["S-F16.gguf"]["recommended"] and by["S-F16.gguf"]["ram_share"] == 0
    # Sur 6 Go, le F16 déborde de moitié : le Q8 tient entier.
    out = recommend_quant(
        files, 6143, 65297, analysis=small, supported=_SUP | {"Q4_K_M"}
    )
    assert [f["filename"] for f in out if f["recommended"]] == ["S-Q8_0.gguf"]


def test_la_vram_d_abord_plutot_qu_un_quant_a_peine_plus_gros_qui_deborde():
    mid = dict(_DENSE, n_params=8_000_000_000)
    files = [_f("M-Q5_K_M.gguf", 5.7), _f("M-Q6_K.gguf", 7.0)]
    out = recommend_quant(
        files, 8192, 65297, analysis=mid, supported=_SUP | {"Q5_K_M", "Q6_K"}
    )
    by = {f["filename"]: f for f in out}
    assert by["M-Q6_K.gguf"]["tps"] > 8  # assez rapide, mais il déborde
    assert by["M-Q5_K_M.gguf"]["recommended"]


def test_dense_qui_deborde_jamais_recommande_meme_estime_rapide():
    mid = dict(_DENSE, n_params=12_000_000_000)
    files = [_f("G-Q2_K.gguf", 4.0), _f("G-Q3_K_M.gguf", 5.5)]
    out = recommend_quant(
        files, 6143, 65297, analysis=mid, supported=_SUP | {"Q2_K", "Q3_K_M"}
    )
    by = {f["filename"]: f for f in out}
    assert by["G-Q3_K_M.gguf"]["tps"] >= 12 and not by["G-Q3_K_M.gguf"]["recommended"]
    assert by["G-Q2_K.gguf"]["recommended"] and by["G-Q2_K.gguf"]["ram_share"] == 0


def test_moe_le_plus_gros_a_12_t_s_sinon_a_8():
    moe = dict(_DENSE, moe=True, expert_count=256, expert_used=8, f_exp=0.9)
    files = [_f("M-A.gguf", 40.0), _f("M-B.gguf", 60.0)]
    by = {f["filename"]: f for f in recommend_quant(files, 6143, 65297, analysis=moe)}
    assert by["M-A.gguf"]["tps"] >= 12 > by["M-B.gguf"]["tps"] >= 8
    assert by["M-A.gguf"]["recommended"]  # pas le plus gros, à la limite des 8 t/s
    out = recommend_quant(files[1:], 6143, 65297, analysis=moe)
    assert out[0]["recommended"]  # seul au-dessus de 8 t/s : il reste recommandé


def test_moe_le_plus_gros_assez_rapide_experts_en_ram():
    moe = dict(_DENSE, moe=True, expert_count=256, expert_used=8, f_exp=0.9)
    files = [
        _f("M-Q4_K_M.gguf", 19.5),
        _f("M-Q8_0.gguf", 34.4),
        _f("M-BF16.gguf", 64.6),  # ne tient pas en VRAM + RAM
    ]
    out = recommend_quant(files, 6143, 65297, analysis=moe, supported=_SUP | {"Q4_K_M"})
    by = {f["filename"]: f for f in out}
    assert by["M-Q8_0.gguf"]["recommended"] and not by["M-BF16.gguf"]["fits"]
    assert 0 < by["M-Q8_0.gguf"]["ram_share"] < 1  # experts actifs lus en RAM
    # Calage : ornith Q8 (mêmes proportions) mesuré à 15,1 t/s sur la RTX 2060.
    assert 12 < by["M-Q8_0.gguf"]["tps"] < 20


def test_cache_kv_lu_dans_l_en_tete():
    fetch = _gguf([("blk.0.attn_k.weight", 8, 100 * 8, 800), ("blk.0.attn_v.weight", 8, 100 * 8, 800),
                   ("blk.1.ssm_a", 0, 100, 400)], n_embd=100)  # fmt: skip
    s = gr.summarize(gr.read_header(fetch))
    # Une seule couche d'attention, 8 lignes de K et de V.
    assert s["kv_per_token"] == 16


def test_cache_kv_des_couches_a_fenetre_glissante():
    # Vécu 2026-10-17, Gemma 4 12B : 40 couches sur 48 à fenêtre de 1024 tokens, comptées
    # pleines elles donnaient 2,9 Go de KV à 16k et un Q2_K « lu à 48 % en RAM ».
    tensors = [
        (f"blk.{i}.attn_{kv}.weight", 8, 10 * 4, 40) for i in range(3) for kv in "kv"
    ]
    fetch = _gguf(tensors, arch="gemma4", n_embd=10, extra=[
        ("gemma4.attention.sliding_window", 4, struct.pack("<I", 1024)),
        ("gemma4.attention.sliding_window_pattern", 9, struct.pack("<IQ", 7, 3) + bytes([1, 1, 0])),
    ])  # fmt: skip
    a = gr.summarize(gr.read_header(fetch))
    assert (a["kv_per_token"], a["kv_swa_per_token"], a["swa_window"]) == (8, 16, 1024)
    kv_mb = (8 * RESERVE_CTX + 16 * 1024) * 1.0625 / 2**20
    assert vram_reserve_mb(a) == VRAM_OVERHEAD_MB + kv_mb


def test_famille_d_un_modele():
    assert family_stem("unsloth/Qwen3.8-27B-GGUF") == "Qwen3.8"
    assert family_stem("prism-ml/Ternary-Bonsai-2-27B-gguf-dev") == "Ternary-Bonsai-2"
    assert family_stem("google/gemma-4-E4B-it-qat-q4_0-gguf") == "gemma-4"
    assert family_stem("Qwen/Qwen3.6-35B-A3B") == "Qwen3.6"
    assert family_stem("org/Flash-Next-GGUF") is None and family_stem(None) is None


def test_alternatives_non_mesurees_conservent_heuristique_capacite():
    hits = [
        {"repo_id": "u/Qwen3.8-27B-GGUF", "params": 27_000_000_000, "arch": "qwen35"},
        {"repo_id": "x/Qwen3.8-27B-Abliterated-GGUF", "params": 27_000_000_000, "arch": "qwen35"},
        {"repo_id": "e/Qwen3.8-35B-A3B-Distill-GGUF", "params": 35_500_000_000, "arch": "qwen35moe"},
        {"repo_id": "e/Qwen3.8-9B-Distill-GGUF", "params": 9_200_000_000, "arch": "qwen35"},
        {"repo_id": "p/Qwen3.8-4B-gguf", "params": 1_500_000_000, "arch": "dflash"},  # le nom fait foi
        {"repo_id": "z/Qwen3.8-2B-GGUF", "params": 1_900_000_000, "arch": "qwen35"},
        # HF a lu le drafter MTP du repo : le nom (27B) fait foi, même modèle, rien n'y tient.
        {"repo_id": "h/Qwen3.8-27B-MTP-GGUF", "params": 1_900_000_000, "arch": "qwen35"},
        {"repo_id": "k/Qwen3.8-14B-GGUF", "params": 14_000_000_000, "arch": "qwen35"},
    ]  # fmt: skip
    files = {
        "x/Qwen3.8-27B-Abliterated-GGUF": [_f("A-IQ2_XXS.gguf", 7.0)],
        "e/Qwen3.8-35B-A3B-Distill-GGUF": [_f("D-Q4_K_M.gguf", 20.0)],
        "e/Qwen3.8-9B-Distill-GGUF": [
            _f("N-Q3_K_M.gguf", 4.2),
            _f("N-Q4_K_M.gguf", 5.6),
        ],
        "p/Qwen3.8-4B-gguf": [_f("F-Q8_0.gguf", 4.3)],
        "h/Qwen3.8-27B-MTP-GGUF": [
            _f("H-FastMTP.gguf", 0.8),
            _f("H-Q4_K_M.gguf", 16.0),
        ],
        "z/Qwen3.8-2B-GGUF": [_f("T-Q8_0.gguf", 2.0)],
    }

    def list_files(repo):
        if repo not in files:
            raise RuntimeError("repo illisible")
        return files[repo]

    alts = suggest_alternatives("u/Qwen3.8-27B-GGUF", _DENSE, hits, list_files, 6143,
                                65297, _SUP | {"Q3_K_M", "Q4_K_M"})  # fmt: skip
    assert [a["repo_id"] for a in alts] == [
        "e/Qwen3.8-35B-A3B-Distill-GGUF",  # heuristique de capacité, pas un score mesuré
        "e/Qwen3.8-9B-Distill-GGUF",
        "p/Qwen3.8-4B-gguf",
    ]
    # Dense : le quant qui tient entier en VRAM, pas le Q4_K_M qui déborde.
    assert alts[0]["active"] == 3e9 and alts[1]["pick"]["filename"] == "N-Q3_K_M.gguf"


# Filiation réelle sur Hugging Face (2026-10-17) : balises base_model des dépôts.
_PARENTS = {
    "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF": ("quantized", "empero-ai/Qwen3.8-35B-A3B-Distill"),
    "empero-ai/Qwen3.8-35B-A3B-Distill": ("finetune", "Qwen/Qwen3.6-35B-A3B"),
    "unsloth/Qwen3.6-35B-A3B-GGUF": ("quantized", "Qwen/Qwen3.6-35B-A3B"),
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": ("quantized", "Qwen/Qwen3.6-35B-A3B"),
    "bartowski/Qwen_Qwen3.8-27B-GGUF": ("quantized", "Qwen/Qwen3.8-27B"),
    "mradermacher/Qwen3.8-27B-i1-GGUF": ("quantized", "Qwen/Qwen3.8-27B"),
    "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF": ("quantized", "Qwen/Qwen3.8-27B"),
    "prism-ml/Ternary-Bonsai-2-9B-gguf": ("quantized", "Qwen/Qwen3.5-9B"),
    "Qwen/Qwen3.5-9B": ("finetune", "Qwen/Qwen3.5-9B-Base"),
    "google/gemma-4-12B-it-qat-q4_0-gguf": ("quantized", "google/gemma-4-12B-it"),
    "google/gemma-4-12B-it": ("finetune", "google/gemma-4-12B"),
}  # fmt: skip


def test_variante_tierce_signalee_quantification_officielle_non():
    origin = lambda repo: third_party_origin(repo, _PARENTS.get)  # noqa: E731
    # Affinage déclaré par un tiers : le « Qwen3.8-35B-A3B » qui n'existe pas chez Qwen.
    assert origin("empero-ai/Qwen3.8-35B-A3B-Distill-GGUF") == "Qwen/Qwen3.6-35B-A3B"
    # Variantes qui se déclarent « quantized » du modèle officiel : leur nom les trahit.
    assert origin("huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF") == "Qwen/Qwen3.8-27B"
    assert origin("prism-ml/Ternary-Bonsai-2-9B-gguf") == "Qwen/Qwen3.5-9B"
    # Simples quantifications par un tiers, et GGUF de l'éditeur lui-même.
    for repo in ("unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
                 "bartowski/Qwen_Qwen3.8-27B-GGUF", "mradermacher/Qwen3.8-27B-i1-GGUF",
                 "google/gemma-4-12B-it-qat-q4_0-gguf", "x/sans-filiation-GGUF"):  # fmt: skip
        assert origin(repo) is None, repo
