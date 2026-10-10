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
from loom.setup.placement import (
    AucunPlacementFaisable,
    DemarrageImpossible,
    PlacementNonValide,
)

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


def _gpu(vram, nom):
    return HardwareProfile(
        True,
        nom,
        vram,
        16,
        vram_total_mb=vram,
        backend="CUDA",
        vram_is_discrete=True,
        gpu_count=1,
    )


GPU_6G = _gpu(6144, "RTX 6G")
GPU_8G = _gpu(8192, "RTX 8G")
GPU_24G = _gpu(24_576, "RTX 24G")


class _Stop(Exception):
    """Arrête le déroulé juste après l'isolation (contre-test)."""


def _refuse_les_mesures(ctx, depth):
    raise AssertionError("aucune mesure attendue ici")


def _environnement(
    monkeypatch,
    tmp_path,
    *,
    meta,
    hw,
    ram_mb,
    journal,
    toml_extra="",
    run_impl=_refuse_les_mesures,
):
    import psutil

    from loom.runtime import gguf_meta, hardware
    from loom.setup import steps, topology

    mdir = tmp_path / "models" / "m1"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'filename = "m.gguf"\nsize_mb = 12600\nn_gpu_layers = 999\n' + toml_extra,
        encoding="utf-8",
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
            journal.append(("run", self.ngl, ctx, depth))
            return run_impl(ctx, depth)

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


def test_run_calibration_gguf_illisible_impossibilite_etablie_aucune_sonde(
    monkeypatch, tmp_path
):
    """Revue adverse (régression du lot L5) : un en-tête rejeté (pas un GGUF, version
    < 2, tronqué) devenait des métadonnées {} → « incertain » → sondes lancées sur un
    fichier que llama-server refuserait aussi. Impossibilité établie, aucune sonde ; la
    trace porte de quoi l'archiver (GGUF, matériel)."""
    from loom.web.routes import rebench

    journal: list = []
    statuts: list = []
    trace: dict = {}
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=ValueError("pas un fichier GGUF"),
        hw=GPU_6G,
        ram_mb=4000,
        journal=journal,
    )
    with pytest.raises(DemarrageImpossible) as exc:
        rebench._run_calibration(None, spec, statuts.append, trace_out=trace)
    assert exc.value.etabli is True and "GGUF illisible" in str(exc.value)
    assert "pas un fichier GGUF" in str(exc.value)
    assert journal == [] and not any("isolation" in s for s in statuts)
    assert trace["gguf"].endswith("m.gguf") and trace["materiel"] is GPU_6G


def test_run_calibration_metadonnees_inconnues_incertain_isolation_a_un_slot(
    monkeypatch, tmp_path
):
    """Contre-test : type de valeur GGUF inconnu du lecteur (fichier plus récent que
    lui, pas invalide) → métadonnées inconnues, « incertain », flux inchangé. Modèle
    en cache_isolation = true : la sonde principale est à 2 slots, la copie d'isolation
    à 1 (à 2 slots, B partirait sur le slot libre)."""
    from loom.runtime.gguf_meta import TypeGGUFInconnu
    from loom.web.routes import rebench

    journal: list = []
    trace: dict = {}
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=TypeGGUFInconnu("type GGUF inconnu : 99"),
        hw=GPU_6G,
        ram_mb=4000,
        journal=journal,
        toml_extra="cache_isolation = true\n",
    )

    def _stop(*a, **k):
        raise _Stop()

    monkeypatch.setattr(rebench, "_measure_placement", _stop)
    with pytest.raises(_Stop):
        rebench._run_calibration(None, spec, lambda m: None, trace_out=trace)
    assert trace["precontrole"]["verdict"] == "incertain"
    construites = [e for e in journal if e[0] == "construite"]
    isolations = [e for e in journal if e[0] == "isolation"]
    assert construites[0][2] == 2  # sonde principale : isolation actuelle, 2 slots
    assert len(isolations) == 1 and isolations[0][2] == 1  # copie à 1 slot
    assert trace["isolation"]["slots_mesure"] == 1


def test_run_calibration_hors_budget_au_plancher_aucune_sonde(monkeypatch, tmp_path):
    """Mémoire unifiée seulement présumée (Vulkan) : pas d'« établi », mais rien ne
    tient même à 4096 x 1 → DemarrageImpossible non établi, aucune sonde."""
    from loom.web.routes import rebench

    journal: list = []
    vulkan = HardwareProfile(
        True, "Radeon", 46_350, 16, vram_total_mb=48_789, backend="Vulkan", gpu_count=1
    )
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=_meta_complete(),
        hw=vulkan,
        ram_mb=4000,
        journal=journal,
    )
    with pytest.raises(DemarrageImpossible) as exc:
        rebench._run_calibration(None, spec, lambda m: None, trace_out={})
    assert exc.value.etabli is False and exc.value.details["verdict"] == "hors_budget"
    assert "hors budget même au contexte plancher" in str(exc.value)
    assert journal == []


def test_run_calibration_demarrage_modeste_pour_la_sonde_d_isolation(
    monkeypatch, tmp_path
):
    """Le démarrage prévu (tout GPU) ne tient pas dans 8 Go à 4096 x 1 : la copie
    d'isolation part sur un offload partiel qui tient, et c'est annoncé."""
    from loom.web.routes import rebench

    journal: list = []
    statuts: list = []
    trace: dict = {}
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=_meta_complete(),
        hw=GPU_8G,
        ram_mb=32_000,
        journal=journal,
    )

    def _stop(*a, **k):
        raise _Stop()

    monkeypatch.setattr(rebench, "_measure_placement", _stop)
    with pytest.raises(_Stop):
        rebench._run_calibration(None, spec, statuts.append, trace_out=trace)
    isolations = [e for e in journal if e[0] == "isolation"]
    assert len(isolations) == 1
    assert 0 < isolations[0][1] < 40 and isolations[0][2] == 1
    assert any(s.startswith("sonde d'isolation : le démarrage prévu") for s in statuts)
    assert "sonde d'isolation du cache (A -> pollution -> A)…" in statuts
    assert trace["isolation"]["demarrage"]["modeste"] is True


def test_run_calibration_isolation_imposee_par_la_recurrence_sans_sonde(
    monkeypatch, tmp_path
):
    """Mémoire récurrente, démarrage prévu qui ne tient pas : sonde non lancée, verdict
    imposé, la suite à 2 slots. Le statut n'annonce pas une sonde qui ne tourne pas
    (revue adverse)."""
    from loom.web.routes import rebench

    journal: list = []
    statuts: list = []
    trace: dict = {}
    vu: dict = {}
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=dict(_meta_complete(), recurrent=True),
        hw=GPU_8G,
        ram_mb=32_000,
        journal=journal,
    )

    def _note(*a, **k):
        vu["slots"] = k.get("slots")
        raise _Stop()

    monkeypatch.setattr(rebench, "_measure_placement", _note)
    with pytest.raises(_Stop):
        rebench._run_calibration(None, spec, statuts.append, trace_out=trace)
    assert not any(e[0] == "isolation" for e in journal)
    assert "sonde d'isolation du cache (A -> pollution -> A)…" not in statuts
    assert any(s.startswith("sonde d'isolation non lancée") for s in statuts)
    iso = trace["isolation"]
    assert iso["necessaire"] is True and iso["slots_mesure"] == 0
    assert iso["slots_retenus"] == 2 and vu["slots"] == 2


def test_run_calibration_demarrage_prevu_condamne_jamais_repris_en_repli(
    monkeypatch, tmp_path
):
    """Revue adverse : aucun placement validé et le démarrage prévu ne tient pas à
    4096 x 1 → la calibration chargeait les flags actuels, condamnés. PlacementNonValide
    à l'étape placement ; aucun processus lancé avec le démarrage prévu."""
    from loom.web.routes import rebench

    journal: list = []
    trace: dict = {}

    def _oom(ctx, depth):
        raise RuntimeError("ErrorOutOfDeviceMemory")

    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=_meta_complete(),
        hw=GPU_8G,
        ram_mb=32_000,
        journal=journal,
        run_impl=_oom,
    )
    with pytest.raises(PlacementNonValide) as exc:
        rebench._run_calibration(None, spec, lambda m: None, trace_out=trace)
    assert "aucun placement validé" in str(exc.value)
    assert "le démarrage prévu ne tient pas" in str(exc.value)
    lances = [e for e in journal if e[0] in ("run", "isolation", "verify_cache")]
    assert lances and not any(e[1] == 999 for e in lances)
    assert trace["etape"] == "placement"


def test_run_calibration_refus_d_etape_2_trace_le_contexte_utile(monkeypatch, tmp_path):
    """L'archive d'un refus d'étape 2 dit quel contexte ne tenait pas : contexte utile
    posé AVANT le contrôle, étape « placement »."""
    from loom.web.routes import rebench

    trace: dict = {}
    spec = _environnement(
        monkeypatch,
        tmp_path,
        meta=_meta_complete(),
        hw=GPU_24G,
        ram_mb=64_000,
        journal=[],
    )

    def _refus(*a, **k):
        raise AucunPlacementFaisable("le contexte utile 8192 (1 slot) ne tient pas")

    monkeypatch.setattr(rebench, "_measure_placement", _refus)
    with pytest.raises(AucunPlacementFaisable):
        rebench._run_calibration(None, spec, lambda m: None, trace_out=trace)
    assert trace["contexte_utile"] == 8192 and trace["etape"] == "placement"


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
