# tests/test_fidelite_sonde.py
"""Fidélité sonde / exécutant (lot 1 du bench de placement).

La sonde (loom-setup, /rebench) doit lancer llama-server avec EXACTEMENT les flags
que llama-swap lancera ensuite, sinon elle mesure une autre configuration. Vécu le
2026-10-09 sur la Radeon 860M (Vulkan, mémoire unifiée, pas de nvidia-smi) :
- la topologie venait de nvidia-smi -> « ram » -> sonde sans -fa on / q8_0 / --prio 2
  alors que l'exécutant (profil `--list-devices`) les pose ;
- llama-swap ignorait [override] threads que le bench venait d'écrire ;
- le warmup ne précédait que le PREMIER démarrage d'une sonde, les suivants
  mesuraient des caches froids ;
- la mémoire d'une topologie GPU se lisait via nvidia-smi : absent sur AMD.
"""

from __future__ import annotations

import shlex
from dataclasses import replace

import pytest

from loom.config import ModelConfig
from loom.runtime.hardware import HardwareProfile
from loom.runtime.swap import _model_cmd, build_swap_config
from loom.setup import topology as topo
from loom.setup.topology import TOPO_GPU_DENSE, TOPO_MOE_HYBRIDE, TOPO_RAM, ServerProbe

# Radeon 860M vue par `llama-server --list-devices` (Vulkan0, 48 Go partagés) ;
# nvidia-smi absent -> rien de discret.
_UMA = HardwareProfile(
    True, "Radeon 860M", 46_350, 16, vram_total_mb=48_789, backend="Vulkan"
)
_DGPU = HardwareProfile(
    True,
    "RTX 2060",
    5_500,
    16,
    vram_total_mb=6_144,
    backend="CUDA",
    vram_is_discrete=True,
)
_CPU = HardwareProfile(False, None, 0, 16)
_MODEL = ModelConfig(
    id="m", repo="o/r", filename="m.gguf", n_layers=40, size_mb=35_193, n_gpu_layers=999
)


# ── résolveur commun des flags machine ──────────────────────────────────────────


def test_launch_flags_threads_override_puis_physiques_puis_tous():
    from loom.runtime.effective import launch_flags

    assert launch_flags(_UMA, override_threads=6).threads == 6
    # Avec GPU : ~cœurs physiques (logiques / 2) ; sans : tous les threads.
    assert launch_flags(_UMA, override_threads=None).threads == 8
    assert launch_flags(_CPU, override_threads=None).threads == 16
    assert launch_flags(_CPU, override_threads=0).threads == 16  # 0 = non renseigné


def test_launch_flags_gpu_tuning_et_memoire_unifiee_depuis_le_profil():
    from loom.runtime.effective import launch_flags

    uma = launch_flags(_UMA, None)
    assert uma.gpu_tuning is True and uma.unified_memory is True
    dgpu = launch_flags(_DGPU, None)
    assert dgpu.gpu_tuning is True and dgpu.unified_memory is False
    cpu = launch_flags(_CPU, None)
    assert cpu.gpu_tuning is False


# ── llama-swap honore [override] threads comme serve.py ─────────────────────────


def test_swap_cmd_honore_override_threads():
    cmd = _model_cmd(
        _MODEL, _UMA, "llama-server", "/models", 8192, n_parallel=1, override_threads=6
    )
    args = shlex.split(cmd)
    assert args[args.index("-t") + 1] == "6"


def test_swap_cmd_sans_override_garde_les_coeurs_physiques():
    cmd = _model_cmd(_MODEL, _UMA, "llama-server", "/models", 8192, n_parallel=1)
    args = shlex.split(cmd)
    assert args[args.index("-t") + 1] == "8"


def test_build_swap_config_propage_override_threads():
    cfg = build_swap_config(
        [_MODEL],
        _UMA,
        llama_bin="llama-server",
        models_dir="/models",
        context=8192,
        override_threads=6,
    )
    assert "-t 6 " in cfg["models"]["m"]["cmd"] + " "


# ── parité sonde / exécutant ────────────────────────────────────────────────────

# Flags sans effet sur la mesure (adressage, journal, sauvegarde de slot).
_HORS_MESURE = {"--host", "--port", "--slot-save-path", "--log-file", "-lv"}


def _flags_mesures(args: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in _HORS_MESURE:
            skip = True
            continue
        out.append(a)
    return out


def _args_sonde(probe: ServerProbe, ctx: int) -> list[str]:
    captured: dict = {}

    def _popen(args, **kw):
        captured["args"] = [str(a) for a in args]
        raise RuntimeError("stop avant le vrai lancement")

    probe = replace(probe, popen=_popen, kill=lambda p: None)
    with pytest.raises(RuntimeError):
        probe._start(ctx)
    return captured["args"]


def test_sonde_lance_les_flags_exacts_de_llama_swap_sur_amd_uma():
    """Le cas du 2026-10-09 : même modèle, même machine, la sonde doit produire la
    ligne de commande de l'exécutant (à l'adressage près)."""
    attendu = shlex.split(
        _model_cmd(
            _MODEL,
            _UMA,
            "llama-server",
            "/models",
            8192,
            n_parallel=1,
            override_threads=8,
        )
    )
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="/models/m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=_UMA,
    )
    obtenu = _args_sonde(probe, 8192)
    assert _flags_mesures(obtenu) == _flags_mesures(attendu)
    # Ce que la sonde « ram » perdait : le profil GPU benchmarké…
    assert "-fa" in obtenu and "q8_0" in obtenu and "--prio" in obtenu
    # …et sans no-mmap : la mémoire unifiée Vulkan ne le supporte pas (exécutant idem).
    assert "--no-mmap" not in obtenu and "--load-mode" not in obtenu


def test_sonde_gpu_discret_garde_no_mmap_comme_l_executant():
    attendu = shlex.split(
        _model_cmd(
            _MODEL,
            _DGPU,
            "llama-server",
            "/models",
            8192,
            n_parallel=1,
            override_threads=8,
        )
    )
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="/models/m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_GPU_DENSE,
        profile=_DGPU,
    )
    obtenu = _args_sonde(probe, 8192)
    assert _flags_mesures(obtenu) == _flags_mesures(attendu)
    assert "--no-mmap" in obtenu or "--load-mode" in obtenu


def test_sonde_passe_les_checkpoints_comme_l_executant():
    """Un hybride (Bonsai 2) démarre avec --checkpoint-min-step / --ctx-checkpoints :
    chaque checkpoint pèse l'état récurrent complet (150 Mio), la mémoire mesurée par
    la sonde doit être celle de l'exécutant."""
    hybride = replace(_MODEL, checkpoint_min_step=2048, ctx_checkpoints=8)
    attendu = shlex.split(
        _model_cmd(
            hybride,
            _UMA,
            "llama-server",
            "/models",
            8192,
            n_parallel=2,
            override_threads=8,
        )
    )
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="/models/m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=_UMA,
        n_parallel=2,
        checkpoint_min_step=2048,
        ctx_checkpoints=8,
    )
    obtenu = _args_sonde(probe, 8192)
    assert _flags_mesures(obtenu) == _flags_mesures(attendu)
    assert obtenu[obtenu.index("--checkpoint-min-step") + 1] == "2048"
    assert obtenu[obtenu.index("--ctx-checkpoints") + 1] == "8"


def test_sonde_prend_les_batchs_machine_comme_l_executant():
    """Vu sur la vraie ligne de commande (2026-10-10) : la sonde tournait en -ub 512 /
    -b 2048 alors que l'exécutant prend ubatch = 2048 / batch = 4096 du [server] machine
    quand model.toml ne dit rien. Même précédence : modèle, sinon machine."""
    from loom.setup.topology import probe_batches

    assert probe_batches({}, {"ubatch": 2048, "batch": 4096}) == (2048, 4096)
    assert probe_batches({"ubatch": 1024}, {"ubatch": 2048, "batch": 4096}) == (
        1024,
        4096,
    )
    assert probe_batches({}, {}) == (None, None)
    attendu = shlex.split(
        _model_cmd(
            _MODEL,
            _UMA,
            "llama-server",
            "/models",
            8192,
            n_parallel=1,
            override_threads=8,
            default_ubatch=2048,
            default_batch=4096,
        )
    )
    ub, b = probe_batches({}, {"ubatch": 2048, "batch": 4096})
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="/models/m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=_UMA,
        ubatch=ub,
        batch=b,
    )
    obtenu = _args_sonde(probe, 8192)
    assert _flags_mesures(obtenu) == _flags_mesures(attendu)
    assert obtenu[obtenu.index("-ub") + 1] == "2048"


def test_binaire_propre_au_modele_et_slots_globaux():
    """Revue 2026-10-10 : l'exécutant lance `model.server_bin or [server].bin` avec
    resolve_parallel([server] n_parallel, cache_isolation) slots ; la sonde prenait le
    binaire global et partait de 1 slot. Mêmes résolveurs des deux côtés."""
    from loom.setup.topology import model_server_bin, probe_slots

    assert model_server_bin(
        {"server_bin": "C:/autre/llama-server.exe"}, "C:/g/llama-server.exe"
    ) == ("C:/autre/llama-server.exe")
    assert (
        model_server_bin({"server_bin": ""}, "C:/g/llama-server.exe")
        == "C:/g/llama-server.exe"
    )
    assert model_server_bin({}, "C:/g/llama-server.exe") == "C:/g/llama-server.exe"
    # Slots : le global [server] n_parallel, monté à 2 au minimum si isolation.
    assert probe_slots({}, isolation=False) == 1
    assert probe_slots({}, isolation=True) == 2
    assert probe_slots({"n_parallel": 3}, isolation=False) == 3
    assert probe_slots({"n_parallel": 3}, isolation=True) == 3
    assert probe_slots({"n_parallel": 0}, isolation=None) == 1


def test_sonde_cpu_seul_sans_profil_gpu():
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="/models/m.gguf",
        threads=16,
        ngl=0,
        topology=TOPO_RAM,
        profile=_CPU,
    )
    obtenu = _args_sonde(probe, 8192)
    assert "-fa" not in obtenu and "--prio" not in obtenu


# ── mémoire mesurée : une définition par type de machine ─────────────────────────


class _FauxProc:
    pid = 4242


def _sonde_demarree(monkeypatch, probe: ServerProbe) -> ServerProbe:
    """Démarrage simulé : popen renvoie un faux process, /health répond 200."""

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(topo.urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(topo.time, "sleep", lambda s: None)
    return replace(probe, popen=lambda *a, **k: _FauxProc(), kill=lambda p: None)


def test_memoire_unifiee_mesuree_en_delta_de_ram_disponible(monkeypatch):
    """UMA : la VRAM EST la RAM. On lit la RAM disponible du système avant le lancement
    et après /health : la différence compte UNE fois poids, KV et buffers Vulkan, sans
    nvidia-smi (absent sur AMD) ni RSS (qui ignore les allocations du pilote)."""
    dispo = iter([60_000, 24_800])
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=_UMA,
        ram_avail=lambda: next(dispo),
    )
    probe = _sonde_demarree(monkeypatch, probe)
    assert probe.run(8192, None).mem_mb == 35_200
    assert probe.memory_mode == "ram_delta"


def test_gpu_discret_lit_la_memoire_du_device(monkeypatch):
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_GPU_DENSE,
        profile=_DGPU,
        vram_mb=lambda: 3_268,
        ram_avail=lambda: pytest.fail("la RAM n'est pas le device d'un GPU discret"),
    )
    probe = _sonde_demarree(monkeypatch, probe)
    assert probe.run(8192, None).mem_mb == 3_268
    assert probe.memory_mode == "device"


def test_cpu_seul_lit_le_rss_du_process(monkeypatch):
    class _PsProc:
        def __init__(self, pid):
            assert pid == 4242

        def memory_info(self):
            return type("mi", (), {"rss": 1_500 * 1024 * 1024})()

    import psutil

    monkeypatch.setattr(psutil, "Process", _PsProc)
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=16,
        ngl=0,
        topology=TOPO_RAM,
        profile=_CPU,
    )
    probe = _sonde_demarree(monkeypatch, probe)
    assert probe.run(8192, None).mem_mb == 1_500
    assert probe.memory_mode == "rss"


# ── warmup à CHAQUE démarrage ────────────────────────────────────────────────────


def test_warmup_precede_chaque_mesure_de_debit(monkeypatch):
    """Chaque run relance un serveur neuf (caches froids, allocation pinnée) : le
    warmup jetable doit précéder CHAQUE mesure, pas seulement la première de l'objet."""
    monkeypatch.setattr(topo.time, "sleep", lambda s: None)
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=_UMA,
        kill=lambda p: None,
    )
    predits: list[int] = []

    def fake_completion(prompt, n_predict, cache_prompt=False):
        predits.append(n_predict)
        return {"timings": {"predicted_per_second": 14.4, "prompt_per_second": 262.0}}

    monkeypatch.setattr(probe, "_start", lambda ctx: _FauxProc())
    monkeypatch.setattr(probe, "_measure_mem", lambda proc: 100)
    monkeypatch.setattr(probe, "_completion", fake_completion)
    monkeypatch.setattr(probe, "_tokens_of", lambda text: 12)
    probe.run(8192, 4096)
    probe.run(8192, 4096)
    # [warmup, mesure, warmup, mesure] — le warmup génère 16 tokens, la mesure 96.
    assert predits == [16, 96, 16, 96]


# ── /rebench : topologie et threads depuis le PROFIL, pas nvidia-smi ─────────────


def test_rebench_reglages_de_sonde_depuis_le_profil_amd():
    from loom.web.routes.rebench import _probe_settings

    meta = {"expert_count": 128, "n_layers": 40}
    topo_, vram, threads, ngl = _probe_settings(
        meta,
        mt={"cpu_moe": False},
        over={},
        hw=_UMA,
        gpu_backend=True,
        vram_fallback_mb=0,  # nvidia-smi absent
        size_mb=35_193,
    )
    assert topo_ == TOPO_MOE_HYBRIDE and vram == 48_789
    # ngl par le résolveur du runtime (resolve_ngl) : 36 Go dans 46 Go libres -> 999.
    assert threads == 8 and ngl == 999


def test_rebench_reglages_de_sonde_petit_gpu_resout_un_partiel():
    """Le même model.toml (cpu_moe = false) sur un GPU de 8 Go : l'exécutant résout 8
    couches ; la sonde doit partir de là, pas de 99."""
    from loom.web.routes.rebench import _probe_settings

    petite = HardwareProfile(
        True,
        "GPU 8 Go",
        8_000,
        16,
        vram_total_mb=8_192,
        backend="CUDA",
        vram_is_discrete=True,
    )
    _topo, _vram, _threads, ngl = _probe_settings(
        {"expert_count": 128, "n_layers": 41},
        mt={"cpu_moe": False},
        over={},
        hw=petite,
        gpu_backend=True,
        vram_fallback_mb=8_192,
        size_mb=36_050,
        headroom=640,
    )
    assert ngl == 8


def test_rebench_reglages_de_sonde_honorent_override_et_borne_modele():
    from loom.web.routes.rebench import _probe_settings

    topo_, _vram, threads, ngl = _probe_settings(
        {"n_layers": 42},
        mt={"n_gpu_layers": 36},
        over={"threads": 6, "n_gpu_layers": 99},
        hw=_DGPU,
        gpu_backend=True,
        vram_fallback_mb=6_144,
        size_mb=8_000,
    )
    assert topo_ == TOPO_GPU_DENSE and threads == 6 and ngl == 36


def test_rebench_sans_gpu_tout_en_ram():
    from loom.web.routes.rebench import _probe_settings

    topo_, vram, threads, ngl = _probe_settings(
        {"n_layers": 42},
        mt={},
        over={},
        hw=_CPU,
        gpu_backend=False,
        vram_fallback_mb=0,
        size_mb=8_000,
    )
    assert topo_ == TOPO_RAM and vram == 0 and threads == 16 and ngl == 0
