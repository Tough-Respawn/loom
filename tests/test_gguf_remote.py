# Recommandation de quant à partir de l'en-tête GGUF lu à distance (sans réseau ici).
import struct

from loom.runtime import gguf_remote as gr
from loom.runtime.model_install import recommend_quant


def _gguf(tensors, arch="qwen35", experts=(0, 0), alignment=32):
    """GGUF v3 minimal : métadonnées + infos de tenseurs (nom, type, éléments, octets)."""

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kv = [("general.architecture", 8, s(arch)), ("general.alignment", 4, struct.pack("<I", alignment)),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s("a") + s("b"))]  # fmt: skip
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


def test_dense_jamais_le_f16_ni_le_format_du_fork():
    # Vécu 2026-10-17 : l'ancienne règle recommandait le F16 de 50 Go d'un dense 27B.
    files = [_f("B-PTQ1_0.gguf", 5.5), _f("B-PQ2_0.gguf", 6.7), _f("B-F16.gguf", 50.1)]
    out = recommend_quant(files, 6143, 65297, analysis=_DENSE, supported=_SUP)
    by = {f["filename"]: f for f in out}
    assert by["B-PQ2_0.gguf"]["status"] == "unsupported"
    assert by["B-F16.gguf"]["status"] == "full"
    assert not any(f["recommended"] for f in out)


def test_dense_qui_ne_tient_pas_en_vram_le_moins_lu_en_ram_sans_quant_poubelle():
    files = [
        _f("Q-IQ1_S.gguf", 5.8),
        _f("Q-UD-IQ1_M.gguf", 6.9),  # vécu : unsloth, ≈ 2,0 bits effectifs
        _f("Q-IQ2_XXS.gguf", 7.0),
        _f("Q-Q4_K_M.gguf", 15.3),
    ]
    out = recommend_quant(
        files, 6143, 65297, analysis=_DENSE, supported=_SUP | {"Q4_K_M", "IQ1_M"}
    )
    by = {f["filename"]: f for f in out}
    assert by["Q-IQ2_XXS.gguf"]["recommended"]  # IQ1_S et UD-IQ1_M écartés
    assert by["Q-Q4_K_M.gguf"]["ram_share"] > by["Q-IQ2_XXS.gguf"]["ram_share"] > 0


def test_dense_qui_tient_en_vram_le_plus_gros_qui_y_tient():
    files = [_f("S-Q2_0.gguf", 2.4), _f("S-Q8_0.gguf", 4.4), _f("S-F16.gguf", 9.0)]
    out = recommend_quant(files, 6143, 65297, analysis=_DENSE, supported=_SUP)
    by = {f["filename"]: f for f in out}
    assert by["S-Q8_0.gguf"]["recommended"] and by["S-Q8_0.gguf"]["ram_share"] == 0


def test_moe_le_plus_gros_hors_pleine_precision():
    moe = dict(_DENSE, moe=True, expert_count=256, expert_used=8, f_exp=0.9)
    files = [
        _f("M-Q4_K_M.gguf", 19.5),
        _f("M-Q8_0.gguf", 34.4),
        _f("M-BF16.gguf", 64.6),
    ]
    out = recommend_quant(files, 6143, 65297, analysis=moe, supported=_SUP | {"Q4_K_M"})
    by = {f["filename"]: f for f in out}
    assert by["M-Q8_0.gguf"]["recommended"] and not by["M-BF16.gguf"]["recommended"]
    assert 0 < by["M-Q8_0.gguf"]["ram_share"] < 1  # experts actifs lus en RAM
