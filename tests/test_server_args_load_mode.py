"""`--no-mmap` a été retiré de llama.cpp le 2026-09-09 (#28334), remplacé par
`--load-mode none`. Un binaire récent REFUSE `--no-mmap` (argument inconnu, le
serveur ne démarre pas) ; un ancien ignore `--load-mode`. Loom sonde donc l'aide
du binaire une fois (par chemin) et émet le bon flag."""

from __future__ import annotations

import subprocess

import pytest

from loom.runtime import server_args as sa


@pytest.fixture(autouse=True)
def _cache_vierge():
    # Le cache est global au process : ne jamais le laisser fuiter vers un autre test.
    sa.no_mmap_args.cache_clear()
    yield
    sa.no_mmap_args.cache_clear()


def _args(bin_path="X:/llama-server.exe"):
    return sa.build_server_args(
        server_bin=bin_path,
        model_path="m.gguf",
        port=1,
        context=1024,
        n_gpu_layers=99,
        threads=4,
        gpu_tuning=True,
    )


def _fake_help(monkeypatch, text, calls):
    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    sa.no_mmap_args.cache_clear()
    monkeypatch.setattr(sa.subprocess, "run", run)


def test_binaire_recent_recoit_load_mode_none(monkeypatch):
    calls = []
    _fake_help(monkeypatch, "-lm,   --load-mode MODE   model loading mode", calls)
    a = _args()
    assert "--no-mmap" not in a
    i = a.index("--load-mode")
    assert a[i + 1] == "none"


def test_binaire_ancien_garde_no_mmap(monkeypatch):
    calls = []
    _fake_help(monkeypatch, "--mmap, --no-mmap   whether to memory-map model", calls)
    a = _args()
    assert "--no-mmap" in a and "--load-mode" not in a


def test_sonde_une_seule_fois_par_binaire(monkeypatch):
    calls = []
    _fake_help(monkeypatch, "--load-mode MODE", calls)
    _args("X:/a.exe")
    _args("X:/a.exe")
    _args("X:/b.exe")
    assert len(calls) == 2


def test_binaire_injoignable_retombe_sur_no_mmap(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    sa.no_mmap_args.cache_clear()
    monkeypatch.setattr(sa.subprocess, "run", boom)
    assert "--no-mmap" in _args()


def test_placeholder_swap_non_sonde(monkeypatch):
    # swap.py passe le binaire réel, mais un chemin contenant ${...} (macro
    # llama-swap) ne doit jamais être exécuté.
    calls = []
    _fake_help(monkeypatch, "--load-mode MODE", calls)
    assert sa.no_mmap_args("${LLAMA_BIN}") == ["--no-mmap"]
    assert calls == []
