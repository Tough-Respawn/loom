# tests/test_precontrole_rebench.py
"""Précontrôle mémoire dans /rebench (revue n°16), au niveau de _run_calibration :
le TEST DÉCISIF (impossibilité établie → aucune sonde construite ni lancée) et le
contre-test (métadonnées illisibles → incertain, flux inchangé, isolation sur une
copie à 1 slot). Toutes les dépendances sont remplacées : aucun processus."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from loom.runtime.hardware import HardwareProfile
from loom.setup.placement import DemarrageImpossible

MIB = 1024 * 1024


def _meta_complete(n=40, par_mb=300, sortie_mb=300, emb_mb=300):
    return {
        "architecture": "llama",
        "n_layers": n,
        "context_length": 32768,
        "expert_count": None,
        "head_count": 32,
        "head_count_kv": 8,
        "embedding_length": 4096,
        "key_length": 128,
        "value_length": 128,
        "sliding_window": None,
        "sliding_window_pattern": None,
        "full_attention_interval": None,
        "recurrent": False,
        "split_count": None,
        "head_count_kv_array": False,
        "arrays": {},
        "weights": {
            "total": (n * par_mb + sortie_mb + emb_mb) * MIB,
            "familles": {"embeddings": emb_mb * MIB, "output": sortie_mb * MIB},
            "par_couche": [par_mb * MIB] * n,
            "experts_par_couche": [0] * n,
            "couches_attention": list(range(n)),
            "couches_recurrentes": [],
            "couches_nextn": [],
            "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
        },
    }


GPU_6G = HardwareProfile(
    True,
    "RTX 6G",
    6000,
    16,
    vram_total_mb=6144,
    backend="CUDA",
    vram_is_discrete=True,
    gpu_count=1,
)


class _Stop(Exception):
    """Arrête le déroulé juste après l'isolation (contre-test)."""


def _environnement(monkeypatch, tmp_path, *, meta, hw, ram_mb, journal):
    import psutil

    from loom.runtime import gguf_meta, hardware
    from loom.setup import steps, topology

    mdir = tmp_path / "models" / "m1"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'filename = "m.gguf"\nsize_mb = 12600\nn_gpu_layers = 999\n', encoding="utf-8"
    )
    (mdir / "m.gguf").write_bytes(b"GGUF")
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(
        steps, "read_raw_config", lambda *a, **k: {"server": {"bin": str(exe)}}
    )
    monkeypatch.setattr(steps, "server_bin_status", lambda raw: (True, str(exe)))
    monkeypatch.setattr(steps, "resolve_bin", lambda name: exe)
    if isinstance(meta, Exception):

        def _lecture(p):
            raise meta

        monkeypatch.setattr(gguf_meta, "read_gguf_meta", _lecture)
    else:
        monkeypatch.setattr(gguf_meta, "read_gguf_meta", lambda p: meta)
    monkeypatch.setattr(hardware, "detect_hardware", lambda server_bin=None: hw)
    monkeypatch.setattr(topology, "gpu_vram_total_mb", lambda: 0)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(total=ram_mb * MIB)
    )

    @dataclass
    class Sonde:
        server_bin: str = ""
        model_path: str = ""
        threads: int = 0
        ngl: int = 0
        topology: str = ""
        mmproj_path: object = None
        cpu_moe: bool = False
        n_cpu_moe: object = None
        n_parallel: int = 1
        ubatch: object = None
        batch: object = None
        checkpoint_min_step: object = None
        ctx_checkpoints: object = None
        profile: object = None

        def __post_init__(self):
            journal.append(("construite", self.ngl, self.n_parallel))

        def probe_isolation(self, ctx=4096):
            journal.append(("isolation", self.ngl, self.n_parallel))
            return 600, 4

        def run(self, ctx, depth):
            journal.append(("run", ctx, depth))
            raise AssertionError("aucune mesure attendue ici")

        def verify_cache(self, ctx=4096):
            journal.append(("verify_cache", ctx))
            raise AssertionError("aucune vérification attendue ici")

    monkeypatch.setattr(topology, "ServerProbe", Sonde)
    return {"id": "m1", "dir": str(mdir), "size_mb": 12600}


def test_run_calibration_impossibilite_etablie_aucune_sonde(monkeypatch, tmp_path):
    """TEST DÉCISIF /rebench : 12 600 Mo résidents pour 6 144 Mo de VRAM + 4 000 de RAM
    → DemarrageImpossible avant même la construction de la sonde."""
    from loom.web.routes import rebench

    journal: list = []
    statuts: list = []
    trace: dict = {}
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=_meta_complete(),
        hw=GPU_6G,
        ram_mb=4000,
        journal=journal,
    )
    with pytest.raises(DemarrageImpossible) as exc:
        rebench._run_calibration(None, spec, statuts.append, trace_out=trace)
    assert exc.value.etabli is True and "mémoire physique" in str(exc.value)
    assert journal == []  # ni construction, ni lancement
    assert trace["etape"] == "précontrôle"
    assert trace["precontrole"]["verdict"] == "impossible"
    assert not any("isolation" in s for s in statuts)


def test_run_calibration_metadonnees_illisibles_incertain_isolation_a_un_slot(
    monkeypatch, tmp_path
):
    """Contre-test : GGUF illisible (ValueError, non rattrapée avant) → « incertain »,
    flux inchangé ; la sonde d'isolation tourne sur une copie à 1 slot."""
    from loom.web.routes import rebench

    journal: list = []
    trace: dict = {}
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=ValueError("pas un fichier GGUF"),
        hw=GPU_6G,
        ram_mb=4000,
        journal=journal,
    )

    def _stop(*a, **k):
        raise _Stop()

    monkeypatch.setattr(rebench, "_measure_placement", _stop)
    with pytest.raises(_Stop):
        rebench._run_calibration(None, spec, lambda m: None, trace_out=trace)
    assert trace["precontrole"]["verdict"] == "incertain"
    isolations = [e for e in journal if e[0] == "isolation"]
    assert len(isolations) == 1 and isolations[0][2] == 1  # copie à 1 slot
    assert trace["isolation"]["slots_mesure"] == 1


def test_measure_placement_voit_la_meme_vram_que_la_topologie():
    """Une seule machine pour l'étape 1 et l'étape 2 : la VRAM de repli (nvidia-smi)
    qui a fixé la topologie et le -ngl de la sonde est aussi celle du plan."""
    from loom.web.routes.rebench import _measure_placement

    @dataclass
    class P:
        ngl: int = 999
        cpu_moe: bool = False
        n_cpu_moe: object = None

    sans_total = HardwareProfile(True, "RTX", 6000, 16, vram_is_discrete=True)
    trace: dict = {}
    _measure_placement(
        P(),
        {"n_layers": 32},
        model_size_mb=5000,
        hw=sans_total,
        ram_total_mb=64_000,
        headroom_mb=640,
        gpu_backend=True,
        progress=lambda m: None,
        trace=trace,
        vram_total_mb=24_000,
    )
    assert [c.key for c in trace["plan"].candidates] == ["gpu_total"]
