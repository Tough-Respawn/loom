# Placement MESURÉ d'un modèle (où vivent les poids : GPU total, dense sur GPU + experts
# sur CPU, partiel, CPU seul). Jusqu'ici une règle décidait (MoE -> experts en RAM) :
# Ornith 35B-A3B perdait 17-21 % (mesuré 2026-10-09, bench-data/ecart-tg-2026-10-09)
# parce que la seule mesure existante (21/07) ne portait que sur le prefill, à une
# répétition. Ici : candidats par FAISABILITÉ, une sonde par candidat avec les flags
# exacts de l'exécutant, les DEUX axes au point de fonctionnement, et une marge de bruit.
from __future__ import annotations

from loom.setup.placement import (
    PLACEMENT_MARGIN_PCT,
    PLACEMENT_PROBE_CTX,
    PLACEMENT_PROBE_PROMPT,
    PLACEMENT_REPS,
    Placement,
    device_budget_mb,
    pick_placement,
    placement_candidates,
    probe_placement,
)
from loom.setup.topology import ProbeResult


def _cands(**kw):
    base = dict(
        moe=False,
        n_layers=40,
        model_size_mb=8_000,
        kv_mb=2_000,
        gpu_backend=True,
        vram_total_mb=48_000,
        ram_total_mb=64_000,
        uma=False,
        headroom_mb=640,
    )
    base.update(kw)
    return placement_candidates(**base)


def test_sans_gpu_un_seul_candidat_cpu():
    cands = _cands(gpu_backend=False)
    assert [c.label for c in cands] == ["cpu"]
    assert cands[0].ngl == 0 and not cands[0].cpu_moe


def test_dense_qui_tient_tout_gpu_seul():
    cands = _cands()
    assert [c.label for c in cands] == ["gpu_total"]
    assert cands[0].ngl == 999


def test_dense_trop_gros_offload_partiel_estime():
    # Deux partiels estimés : serré, puis prudent (moins de couches), cf. lot 3.
    cands = _cands(model_size_mb=80_000, vram_total_mb=24_000)
    assert [c.label for c in cands] == ["gpu_partiel", "gpu_partiel"]
    assert 0 < cands[1].ngl < cands[0].ngl < 40 and cands[0].estime


def test_moe_qui_tient_experts_cpu_en_base_puis_tout_gpu():
    # Ornith 1.0 Q8_0 sur la 860M : 35 Go dans 48 Go vus par Vulkan.
    cands = _cands(moe=True, model_size_mb=35_193, uma=True)
    assert [c.label for c in cands] == ["experts_cpu", "gpu_total"]
    assert cands[0].cpu_moe and cands[0].ngl == 999
    assert not cands[1].cpu_moe and cands[1].n_cpu_moe is None and cands[1].ngl == 999


def test_moe_trop_gros_experts_cpu_puis_partiel_estime():
    cands = _cands(moe=True, model_size_mb=35_193, vram_total_mb=16_000, uma=False)
    # Deux partiels estimés : serré, puis prudent (plus de couches sur CPU), cf. lot 3.
    assert [c.label for c in cands] == [
        "experts_cpu",
        "experts_partiel",
        "experts_partiel",
    ]
    partiel = cands[1]
    assert partiel.n_cpu_moe is not None and 0 < partiel.n_cpu_moe < 40
    assert partiel.estime and partiel.ngl == 999
    assert cands[2].n_cpu_moe > partiel.n_cpu_moe


def test_budget_device_uma_borne_par_la_ram():
    # Mémoire unifiée : le « device » est la RAM, on ne la compte qu'une fois et on
    # laisse la marge OS ; discret : la VRAM seule moins la marge KV.
    assert (
        device_budget_mb(48_789, 64_000, uma=True, headroom_mb=640)
        == min(48_789, 64_000 - 3_072) - 640
    )
    assert device_budget_mb(24_000, 64_000, uma=False, headroom_mb=640) == 24_000 - 640


def _mes(**par_label):
    return {k: {"tg_ts": v[0], "pp_ts": v[1]} for k, v in par_label.items()}


CPU = Placement("experts_cpu", 999, cpu_moe=True)
GPU = Placement("gpu_total", 999)


def test_pick_adopte_l_alternative_au_dela_de_la_marge():
    best, mecanisme = pick_placement(
        _mes(experts_cpu=(12.1, 217.0), gpu_total=(14.4, 262.0)), [CPU, GPU]
    )
    assert best is GPU
    assert "+19" in mecanisme and "experts_cpu" in mecanisme


def test_pick_garde_la_base_sous_la_marge():
    # +3 % de tg : dans le bruit (marge 5 %), la base, plus simple, est conservée et
    # le mécanisme le dit.
    best, mecanisme = pick_placement(
        _mes(experts_cpu=(12.0, 200.0), gpu_total=(12.36, 240.0)), [CPU, GPU]
    )
    assert best is CPU
    assert f"{PLACEMENT_MARGIN_PCT:g}" in mecanisme and "conservé" in mecanisme


def test_pick_depart_au_prefill_deux_alternatives_qui_battent_la_base():
    # Deux alternatives au-dessus de la marge et équivalentes entre elles au tg :
    # le prefill tranche.
    partiel = Placement("experts_partiel", 999, n_cpu_moe=10, estime=True)
    best, _ = pick_placement(
        _mes(
            experts_cpu=(12.0, 200.0),
            gpu_total=(14.3, 250.0),
            experts_partiel_n10=(14.4, 200.0),  # les mesures sont indexées par CLÉ
        ),
        [CPU, GPU, partiel],
    )
    assert best is GPU


# ---- orchestration de la sonde ---------------------------------------------------------


class _Sonde:
    """Sonde factice : rejoue (tg, pp) selon le placement demandé ; `echec` lève."""

    def __init__(self, placement, table, journal):
        self.placement = placement
        self.table = table
        self.journal = journal

    def run(self, ctx, depth):
        self.journal.append((self.placement.label, ctx, depth))
        tg, pp = self.table[self.placement.label]
        if tg == "boom":
            raise RuntimeError("ErrorOutOfDeviceMemory")
        return ProbeResult(ctx=ctx, mem_mb=1234, tg_ts=tg, pp_ts=pp)


def _usine(table):
    journal = []

    def make(placement):
        return _Sonde(placement, table, journal)

    make.journal = journal
    return make


def test_un_seul_candidat_est_valide_une_fois_sans_comparaison():
    # Rien à comparer, mais une validation minimale : le candidat charge et génère.
    make = _usine({"gpu_total": (8.0, 60.0)})
    r = probe_placement(make, [GPU])
    assert r["placement"] is GPU and "seul candidat faisable" in r["mecanisme"]
    assert r["compare"] is False and len(make.journal) == 1


def test_chaque_candidat_est_sonde_reps_fois_au_point_de_fonctionnement():
    make = _usine({"experts_cpu": (12.1, 217.0), "gpu_total": (14.4, 262.0)})
    r = probe_placement(make, [CPU, GPU])
    assert r["placement"] is GPU
    assert r["gain_pct"] == 19.0
    m = r["mesures"]["gpu_total"]
    assert (m["tg_ts"], m["pp_ts"], m["mem_mb"]) == (14.4, 262.0, 1234)
    # L'ordre ALTERNE d'une répétition à l'autre (lot 4) : A B A B, pas A A B B.
    attendu = [
        (c.label, PLACEMENT_PROBE_CTX, PLACEMENT_PROBE_PROMPT)
        for _ in range(PLACEMENT_REPS)
        for c in (CPU, GPU)
    ]
    assert make.journal == attendu


def test_candidat_en_echec_ecarte_et_nomme():
    make = _usine({"experts_cpu": (12.1, 217.0), "gpu_total": ("boom", 0)})
    r = probe_placement(make, [CPU, GPU])
    assert r["placement"] is CPU
    assert r["mesures"]["gpu_total"]["echec"].startswith("RuntimeError")
    assert "gpu_total" in r["mecanisme"]


def test_sonde_muette_si_rien_n_est_mesurable():
    def _boom(placement):
        raise OSError("serveur KO")

    assert probe_placement(_boom, [CPU, GPU]) is None


# ---- écriture dans model.toml ----------------------------------------------------------


def _toml(tmp_path, texte):
    (tmp_path / "model.toml").write_text(texte, encoding="utf-8")
    return tmp_path / "m.gguf"


def _lire(tmp_path):
    import tomllib

    return tomllib.loads((tmp_path / "model.toml").read_text(encoding="utf-8"))


def test_set_model_placement_tout_gpu_remplace_cpu_moe(tmp_path):
    from loom.setup.cli import _set_model_placement

    gguf = _toml(
        tmp_path,
        '# en-tete conserve\nfilename = "m.gguf"\nn_layers = 40\ncpu_moe = true\n',
    )
    _set_model_placement(gguf, GPU, "tg 14,4 vs 12,1 (+19 %), build c8f208b1c")
    d = _lire(tmp_path)
    assert d["cpu_moe"] is False and d["n_gpu_layers"] == 999 and "n_cpu_moe" not in d
    txt = (tmp_path / "model.toml").read_text(encoding="utf-8")
    assert "# en-tete conserve" in txt and "build c8f208b1c" in txt
    assert txt.count("placement élu par la sonde") == 1


def test_set_model_placement_partiel_puis_retour_experts_cpu(tmp_path):
    from loom.setup.cli import _set_model_placement

    gguf = _toml(tmp_path, 'filename = "m.gguf"\ncpu_moe = true\nn_gpu_layers = 999\n')
    partiel = Placement("experts_partiel", 999, n_cpu_moe=25, estime=True)
    _set_model_placement(gguf, partiel, "mesure 1")
    d = _lire(tmp_path)
    assert d["cpu_moe"] is False and d["n_cpu_moe"] == 25 and "n_gpu_layers" not in d
    # Idempotent et réversible : retour à la base, une seule ligne par clé, un tampon.
    _set_model_placement(gguf, CPU, "mesure 2")
    d = _lire(tmp_path)
    assert d["cpu_moe"] is True and "n_cpu_moe" not in d and "n_gpu_layers" not in d
    txt = (tmp_path / "model.toml").read_text(encoding="utf-8")
    assert txt.count("cpu_moe =") == 1 and txt.count("placement élu par la sonde") == 1
    assert "mesure 2" in txt and "mesure 1" not in txt


def test_set_model_placement_cpu_seul(tmp_path):
    from loom.setup.cli import _set_model_placement

    gguf = _toml(tmp_path, 'filename = "m.gguf"\n')
    _set_model_placement(gguf, Placement("cpu", 0), "sans GPU")
    d = _lire(tmp_path)
    assert d["n_gpu_layers"] == 0 and d.get("cpu_moe") is False


# ---- câblage loom-setup (step_bench) ----------------------------------------------------


def test_step_bench_mesure_le_placement_et_l_ecrit(monkeypatch, tmp_path):
    """Le cas Ornith sur la 860M : MoE qui tient en mémoire unifiée. La sonde doit
    comparer experts-CPU et tout-GPU avec la vraie fabrique de sondes, élire tout-GPU
    (+19 %), l'écrire dans model.toml et tracer la mesure dans [bench]."""
    import tomllib
    from dataclasses import dataclass as _dc

    from loom.runtime.hardware import HardwareProfile
    from loom.setup import cli
    from tests.test_setup_cli import _console, _deps, _patch_paths

    _patch_paths(monkeypatch, tmp_path)
    exe = tmp_path / "rt" / "llama-server.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\n', encoding="utf-8"
    )
    mdir = tmp_path / "models" / "local" / "text" / "ornith"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'repo = "o/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 35193\ncpu_moe = true\n',
        encoding="utf-8",
    )
    (mdir / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(
        cli,
        "read_gguf_meta",
        lambda p: {
            "n_layers": 40,
            "expert_count": 128,
            "context_length": 32768,
            "head_count_kv": 4,
            "key_length": 256,
        },
    )
    rows = [
        {"threads": 8, "ngl": 999, "kind": "tg", "ts": 12.0},
        {"threads": 8, "ngl": 999, "kind": "pp", "ts": 200.0},
    ]

    @_dc
    class FakeProbe:
        server_bin: str
        model_path: str
        threads: int
        ngl: int
        topology: str
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
            sondes.append(self)

        def probe_isolation(self, ctx=4096):
            return 600, 4

        def run(self, ctx, depth):
            r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
            if depth:
                r.tg_ts, r.pp_ts = (12.1, 217.0) if self.cpu_moe else (14.4, 262.0)
            return r

    sondes: list = []
    con, printed = _console(assume_yes=True)
    deps = _deps(
        tmp_path,
        run_bench=lambda b, m, t, g, n_cpu_moe=0, progress=None: rows,
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: True,
        cpu_physical=lambda: 8,
        gpu_vram_total_mb=lambda: 0,  # pas de nvidia-smi : la VRAM vient du profil
        ram_total_mb=lambda: 64_000,
        make_probe=FakeProbe,
        detect_hardware=lambda server_bin=None: HardwareProfile(
            True,
            "Radeon 860M",
            46_350,
            16,
            vram_total_mb=48_789,
            vram_is_discrete=False,
        ),
    )
    # La sonde ubatch séparée a disparu : les batchs viennent du 2x2 des finalistes.
    monkeypatch.setattr(
        cli.bench_mod,
        "probe_ubatch",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe_ubatch obsolète")),
    )
    assert cli.run(con, deps) == 0
    mt = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
    assert mt["cpu_moe"] is False and mt["n_gpu_layers"] == 999
    assert mt["ubatch"] in (512, 2048) and mt["batch"] in (2048, 4096)
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert local["bench"]["placement"] == "gpu_total"
    assert local["bench"]["placement_gain_pct"] == 19.0
    assert "tout sur GPU" in "\n".join(printed)
    # Fidélité : sans nvidia-smi, la VRAM vient du profil -> topologie GPU (pas « ram »),
    # et la sonde reçoit le profil de l'exécutant pour en dériver ses flags machine.
    assert sondes and sondes[0].topology == "moe_hybride"
    assert sondes[0].profile is not None and sondes[0].profile.vram_total_mb == 48_789


def test_step_bench_build_statique_sans_dll_garde_le_gpu(monkeypatch, tmp_path):
    """Le cas réel du 2026-10-10 : build maison Vulkan sans ggml-vulkan.dll. Le profil
    `--list-devices` (backend Vulkan) fait foi : candidats GPU, topologie GPU, et la
    sonde reçoit les batchs machine ([server] ubatch/batch) comme l'exécutant."""
    import tomllib
    from dataclasses import dataclass as _dc

    from loom.runtime.hardware import HardwareProfile
    from loom.setup import cli
    from tests.test_setup_cli import _console, _deps, _patch_paths

    _patch_paths(monkeypatch, tmp_path)
    exe = tmp_path / "rt" / "llama-server.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\nubatch = 2048\nbatch = 4096\n',
        encoding="utf-8",
    )
    mdir = tmp_path / "models" / "local" / "text" / "ornith"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'repo = "o/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 35193\ncpu_moe = false\n',
        encoding="utf-8",
    )
    (mdir / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(
        cli,
        "read_gguf_meta",
        lambda p: {
            "n_layers": 40,
            "expert_count": 128,
            "head_count_kv": 2,
            "key_length": 256,
        },
    )
    rows = [
        {"threads": 8, "ngl": 999, "kind": "tg", "ts": 14.0},
        {"threads": 8, "ngl": 999, "kind": "pp", "ts": 250.0},
    ]
    sondes: list = []

    @_dc
    class FakeProbe:
        server_bin: str
        model_path: str
        threads: int
        ngl: int
        topology: str
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
            sondes.append(self)

        def probe_isolation(self, ctx=4096):
            return 600, 4

        def run(self, ctx, depth):
            r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
            if depth:
                r.tg_ts, r.pp_ts = (12.1, 217.0) if self.cpu_moe else (14.4, 262.0)
            return r

    captured: dict = {}

    def fake_bench(b, m, threads, ngl, n_cpu_moe=0, progress=None):
        captured["ngl"] = list(ngl)
        return rows

    con, printed = _console(assume_yes=True)
    deps = _deps(
        tmp_path,
        run_bench=fake_bench,
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: False,  # AUCUNE DLL : build statique
        cpu_physical=lambda: 8,
        gpu_vram_total_mb=lambda: 0,
        ram_total_mb=lambda: 64_000,
        make_probe=FakeProbe,
        detect_hardware=lambda server_bin=None: HardwareProfile(
            True, "Radeon 860M", 46_350, 16, vram_total_mb=48_789, backend="Vulkan"
        ),
    )
    assert cli.run(con, deps) == 0
    assert captured["ngl"] == [999]  # llama-bench mesure la config GPU, pas ngl 0
    assert sondes and sondes[0].topology == "moe_hybride"
    assert (sondes[0].ubatch, sondes[0].batch) == (2048, 4096)
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert local["bench"]["context_mode"] == "moe_hybride"
    assert local["bench"]["placement"] in ("gpu_total", "experts_cpu")


# ---- câblage /rebench : helper de mesure ------------------------------------------------


def test_measure_placement_rebench_renvoie_un_verdict_serialisable_et_la_sonde_elue():
    from dataclasses import dataclass as _dc

    from loom.runtime.hardware import HardwareProfile
    from loom.web.routes.rebench import _measure_placement

    @_dc
    class FakeProbe:
        ngl: int = 999
        cpu_moe: bool = True
        n_cpu_moe: object = None

        def run(self, ctx, depth):
            tg, pp = (12.1, 217.0) if self.cpu_moe else (14.4, 262.0)
            return ProbeResult(ctx=ctx, mem_mb=100, tg_ts=tg, pp_ts=pp)

    hw = HardwareProfile(
        True, "Radeon 860M", 46_350, 16, vram_total_mb=48_789, vram_is_discrete=False
    )
    meta = {"n_layers": 40, "expert_count": 128, "head_count_kv": 4, "key_length": 256}
    verdict, probe = _measure_placement(
        FakeProbe(),
        meta,
        model_size_mb=35_193,
        hw=hw,
        ram_total_mb=64_000,
        headroom_mb=640,
        gpu_backend=True,
        progress=lambda m: None,
        mt={"cpu_moe": True},  # configuration actuelle = experts sur CPU (la base)
    )
    assert verdict["label"] == "gpu_total" and verdict["gain_pct"] == 19.0
    assert verdict["cpu_moe"] is False and verdict["n_cpu_moe"] is None
    assert probe.cpu_moe is False  # la suite de la calibration mesure l'élu
    import json

    json.dumps(verdict)  # l'état b_apply est persisté en JSON


def test_measure_placement_rebench_sans_comparaison_renvoie_none():
    from dataclasses import dataclass as _dc

    from loom.runtime.hardware import HardwareProfile
    from loom.web.routes.rebench import _measure_placement

    @_dc
    class FakeProbe:
        ngl: int = 999
        cpu_moe: bool = False
        n_cpu_moe: object = None

    # Dense qui tient : un seul candidat, rien à comparer, la sonde reste telle quelle.
    hw = HardwareProfile(
        True, "GPU", 20_000, 16, vram_total_mb=24_000, vram_is_discrete=True
    )
    verdict, probe = _measure_placement(
        FakeProbe(),
        {"n_layers": 32},
        model_size_mb=5_000,
        hw=hw,
        ram_total_mb=32_000,
        headroom_mb=640,
        gpu_backend=True,
        progress=lambda m: None,
    )
    assert verdict is None and probe.cpu_moe is False


def test_pick_ignore_les_candidats_sans_mesure():
    best, mecanisme = pick_placement(
        {"experts_cpu": {"tg_ts": 12.0, "pp_ts": 200.0}, "gpu_total": {"echec": "OOM"}},
        [CPU, GPU],
    )
    assert best is CPU and "gpu_total" in mecanisme and "OOM" in mecanisme
