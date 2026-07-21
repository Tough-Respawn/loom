# ubatch/batch PAR MODÈLE : le microbatch de prefill est le levier mesuré au banc
# 2026-07-19 (agents-a1 : -ub 2048 -b 4096 = prefill x2,9, décode intact).
from loom.config import ModelConfig, _parse_model
from loom.runtime.server_args import build_server_args


def _args(**kw):
    return build_server_args(
        server_bin="llama-server",
        model_path="m.gguf",
        port=8080,
        context=8192,
        n_gpu_layers=99,
        threads=6,
        gpu_tuning=True,
        **kw,
    )


def test_defauts_inchanges_sans_override():
    a = _args()
    assert a[a.index("-b") + 1] == "2048"
    assert a[a.index("-ub") + 1] == "512"


def test_ubatch_batch_par_modele_emis():
    a = _args(ubatch=2048, batch=4096)
    assert a[a.index("-b") + 1] == "4096"
    assert a[a.index("-ub") + 1] == "2048"


def test_model_toml_porte_ubatch_batch():
    d = {
        "repo": "r/x",
        "filename": "x.gguf",
        "n_layers": 40,
        "size_mb": 100,
        "ubatch": 2048,
        "batch": 4096,
    }
    m = _parse_model(d, "x")
    assert m.ubatch == 2048 and m.batch == 4096
    # absent -> None (défauts du serveur)
    m2 = _parse_model({k: d[k] for k in ("repo", "filename", "n_layers", "size_mb")})
    assert m2.ubatch is None and m2.batch is None
    assert isinstance(m, ModelConfig)


def test_no_mmap_reserve_aux_gpu_discrets():
    # --no-mmap = tuning DMA mesuré sur les dGPU CUDA du parc. Sur Vulkan/iGPU
    # c'est un bug UPSTREAM connu (ggml-org/llama.cpp #18317 : « Cannot Run
    # Model with mmap = 0 », #14999 : erreurs mémoire --no-mmap + MoE) et vécu
    # ici (Ornith 35B / Radeon 860M : ErrorOutOfDeviceMemory au chargement).
    # mmap (défaut officiel llama.cpp) conservé, le reste du profil GPU intact.
    a = _args(unified_memory=True)
    assert "--no-mmap" not in a
    assert "-fa" in a and "-ctk" in a
    assert "--no-mmap" in _args()  # défaut (dGPU discret) inchangé
