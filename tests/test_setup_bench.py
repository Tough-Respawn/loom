# Installeur loom-setup : bench matériel (parse llama-bench, choix des réglages,
# calcul du contexte qui tient en RAM) — SANS lancer le vrai binaire.
import json
from types import SimpleNamespace

import pytest

from loom.setup.bench import (
    compute_context,
    find_llama_bench,
    kv_bytes_per_token,
    pick_best,
    run_llama_bench,
    thread_candidates,
)


def test_thread_candidates_hybride():
    # i5-1235U : 10 physiques (2P+8E), 12 logiques -> on mesure 5, 10 et 12
    assert thread_candidates(12, 10) == [5, 10, 12]
    # physique inconnu -> logique partout, dédupliqué
    assert thread_candidates(8, None) == [4, 8]


def test_kv_bytes_per_token():
    # Qwen3-8B : 36 couches, 8 têtes KV, head_dim 128 -> 2*36*8*128*2 = 147456
    meta = {"n_layers": 36, "head_count_kv": 8, "key_length": 128}
    assert kv_bytes_per_token(meta) == 147_456
    # head_dim dérivé de embedding/head_count si key_length absent
    meta = {"n_layers": 36, "head_count_kv": 8, "embedding_length": 4096, "head_count": 32}
    assert kv_bytes_per_token(meta) == 147_456
    # champs manquants -> repli conservateur
    assert kv_bytes_per_token({}) == 150_000


def test_compute_context():
    # 10 Go dispo, modèle 5,6 Go, marge 2 Go -> 2592 Mo de KV / 147456 o
    # = 18432 tokens pile (multiple de 2048), sous le cap 24576
    assert compute_context(10_240, 5_600, 147_456) == 18_432
    # grosse machine -> cap du parc
    assert compute_context(60_000, 5_600, 147_456) == 24_576
    # machine étouffée -> plancher 4096, jamais moins
    assert compute_context(6_000, 5_600, 147_456) == 4_096


def test_pick_best_tg_tranche_pp_departage():
    rows = [
        {"threads": 10, "ngl": 0, "kind": "tg", "ts": 3.2},
        {"threads": 10, "ngl": 0, "kind": "pp", "ts": 20.0},
        {"threads": 12, "ngl": 0, "kind": "tg", "ts": 2.1},
        {"threads": 12, "ngl": 0, "kind": "pp", "ts": 25.0},
        {"threads": 10, "ngl": 99, "kind": "tg", "ts": 3.2},
        {"threads": 10, "ngl": 99, "kind": "pp", "ts": 31.0},  # pp départage
    ]
    best = pick_best(rows)
    assert (best["threads"], best["ngl"]) == (10, 99)
    assert pick_best([{"threads": 1, "ngl": 0, "kind": "pp", "ts": 5}]) is None


def test_run_llama_bench_parse_et_erreurs(tmp_path):
    payload = json.dumps(
        [
            {"n_threads": 10, "n_gpu_layers": 0, "n_prompt": 128, "n_gen": 0, "avg_ts": 21.5},
            {"n_threads": 10, "n_gpu_layers": 0, "n_prompt": 0, "n_gen": 16, "avg_ts": 3.4},
        ]
    )

    def fake_run(cmd, **kw):
        assert "-o" in cmd and "json" in cmd
        return SimpleNamespace(returncode=0, stdout=payload, stderr="")

    rows = run_llama_bench("bench.exe", "m.gguf", [10], [0], runner=fake_run)
    assert rows == [
        {"threads": 10, "ngl": 0, "kind": "pp", "ts": 21.5},
        {"threads": 10, "ngl": 0, "kind": "tg", "ts": 3.4},
    ]

    def boom(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="vulkan: out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_llama_bench("bench.exe", "m.gguf", [10], [0], runner=boom)


def test_find_llama_bench(tmp_path):
    (tmp_path / "llama-server.exe").write_bytes(b"")
    assert find_llama_bench(tmp_path / "llama-server.exe") is None
    (tmp_path / "llama-bench.exe").write_bytes(b"")
    assert find_llama_bench(tmp_path / "llama-server.exe").name == "llama-bench.exe"
