# tests/test_placement_candidats.py
"""Candidats élargis et finalistes au contexte utile (lot 3 du bench de placement).

Revue du 2026-10-10 : un dense ne produisait qu'un candidat (donc aucune mesure), un
MoE ne proposait jamais CPU seul, le partiel n'apparaissait que si tout-GPU ne tenait
pas, un candidat unique n'était pas sondé, la config ACTUELLE n'était pas la référence
du /rebench, et les mesures étaient indexées par un label qui ne distinguait pas deux
valeurs de n_cpu_moe. Le gagnant à 4 096 tokens n'était jamais réévalué plus loin.

Objectif retenu : maximiser la génération au contexte utile, sous contraintes de
mémoire, de conservation du cache et, si souhaité, d'un délai maximal de prefill
(« N nouveaux tokens en moins de T secondes »). CPU seul n'est pas démontré dominé :
quand on ne le mesure pas, la trace dit « non exploré », pas « moins performant ».
"""

from __future__ import annotations

from loom.runtime.model_profile import ModelProfile
from loom.setup.placement import (
    PLACEMENT_FINAL_DEPTH_MAX,
    PLACEMENT_PROBE_CTX,
    PLACEMENT_PROBE_PROMPT,
    Placement,
    PrefillConstraint,
    final_depth,
    pick_placement,
    placement_from_config,
    plan_placements,
    probe_placement,
)
from loom.setup.topology import ProbeResult

# ── identité et configuration actuelle ───────────────────────────────────────────


def test_la_cle_d_un_placement_porte_ses_parametres():
    assert Placement("gpu_total", 999).key == "gpu_total"
    assert Placement("experts_cpu", 999, cpu_moe=True).key == "experts_cpu"
    assert Placement("experts_partiel", 999, n_cpu_moe=20).key == "experts_partiel_n20"
    assert Placement("gpu_partiel", 36).key == "gpu_partiel_ngl36"
    assert Placement("cpu", 0).key == "cpu"


def test_placement_depuis_le_model_toml():
    assert placement_from_config({"cpu_moe": True}, n_layers=40).key == "experts_cpu"
    assert placement_from_config({"n_cpu_moe": 25}, n_layers=40).key == (
        "experts_partiel_n25"
    )
    assert placement_from_config({"n_gpu_layers": 0}, n_layers=40).key == "cpu"
    assert placement_from_config({"n_gpu_layers": 36}, n_layers=42).key == (
        "gpu_partiel_ngl36"
    )
    assert placement_from_config({"n_gpu_layers": 999}, n_layers=42).key == "gpu_total"
    # Frontière : -ngl 42 sur 42 couches n'est PAS 999 pour llama.cpp (la couche de sortie
    # reste sur CPU, « 42/43 »). Le réglage exact est conservé tel quel.
    assert (
        placement_from_config({"n_gpu_layers": 42}, n_layers=42).key
        == "gpu_partiel_ngl42"
    )
    assert placement_from_config({"n_gpu_layers": 42}, n_layers=42).estime is False
    # Sans réglage EXPLICITE, le fichier seul ne dit pas où tourne le modèle : c'est le
    # résolveur du runtime qui le sait (current_placement), pas une règle sur cpu_moe.
    assert placement_from_config({"cpu_moe": False}, n_layers=42) is None
    assert placement_from_config({}, n_layers=42) is None


def test_configuration_actuelle_resolue_comme_l_executant():
    """Revue 2026-10-10 : `cpu_moe = false` ne vaut pas `ngl 999`. Le runtime tient compte
    de l'override machine et de la VRAM libre (resolve_ngl) : la référence du bench doit
    être construite par le MÊME résolveur, sinon elle annonce 999 quand l'exécutant
    tourne en 8, 0 ou 20."""
    from loom.runtime.hardware import HardwareProfile
    from loom.setup.placement import current_placement

    large = HardwareProfile(
        True, "Radeon 860M", 46_350, 16, vram_total_mb=48_789, backend="Vulkan"
    )
    petite = HardwareProfile(
        True,
        "GPU 8 Go",
        8_000,
        16,
        vram_total_mb=8_192,
        backend="CUDA",
        vram_is_discrete=True,
    )
    cpu = HardwareProfile(False, None, 0, 16)
    kw = dict(n_layers=41, size_mb=36_050, headroom=640)
    # MoE 36 Go, cpu_moe = false écrit : la VRAM libre décide, comme l'exécutant.
    actuel = current_placement({"cpu_moe": False}, profile=large, **kw)
    assert actuel.key == "gpu_total" and actuel.actuel is True
    # Même fichier, petit GPU : recommend_gpu_layers(8000, 36050, 41, 640) -> 8 couches.
    assert (
        current_placement({"cpu_moe": False}, profile=petite, **kw).key
        == "gpu_partiel_ngl8"
    )
    # L'override machine ([override] n_gpu_layers) prime sur la recommandation.
    assert (
        current_placement({}, profile=large, override_ngl=20, **kw).key
        == "gpu_partiel_ngl20"
    )
    assert current_placement({}, profile=cpu, **kw).key == "cpu"
    # Les réglages explicites du model.toml restent prioritaires.
    assert (
        current_placement({"cpu_moe": True}, profile=large, **kw).key == "experts_cpu"
    )
    assert (
        current_placement({"n_cpu_moe": 25}, profile=large, **kw).key
        == "experts_partiel_n25"
    )
    assert (
        current_placement({"n_gpu_layers": 36}, profile=petite, **kw).key
        == "gpu_partiel_ngl36"
    )
    # Frontière n_layers : le runtime conserve 41, la référence aussi (pas 999).
    assert (
        current_placement({"n_gpu_layers": 41}, profile=large, **kw).key
        == "gpu_partiel_ngl41"
    )
    # Sans profil matériel, impossible de résoudre : seuls les explicites comptent.
    assert current_placement({"cpu_moe": False}, profile=None, **kw) is None
    assert placement_from_config({"cpu_moe": True}, n_layers=40).actuel is True


# ── plan : candidats et non explorés ─────────────────────────────────────────────


def _plan(**kw):
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
    return plan_placements(**base)


def _keys(plan):
    return [c.key for c in plan.candidates]


def test_plan_sans_gpu_cpu_seul_rien_de_non_explore():
    plan = _plan(gpu_backend=False)
    assert _keys(plan) == ["cpu"] and plan.non_explores == []


def test_plan_l_actuel_devient_la_base():
    # Ornith après le 2026-10-09 : cpu_moe = false (tout GPU) dans model.toml.
    plan = _plan(
        moe=True, model_size_mb=35_193, uma=True, current=Placement("gpu_total", 999)
    )
    assert _keys(plan) == ["gpu_total", "experts_cpu"]
    assert plan.candidates[0].actuel is True
    # Une config actuelle hors des candidats générés (n_cpu_moe = 25) entre en base.
    plan = _plan(
        moe=True,
        model_size_mb=35_193,
        uma=True,
        current=Placement("experts_partiel", 999, n_cpu_moe=25, actuel=True),
    )
    assert _keys(plan) == ["experts_partiel_n25", "experts_cpu", "gpu_total"]


def test_plan_moe_qui_tient_nomme_les_non_explores():
    plan = _plan(moe=True, model_size_mb=35_193, uma=True)
    assert _keys(plan) == ["experts_cpu", "gpu_total"]
    non = {n["key"]: n["raison"] for n in plan.non_explores}
    assert "cpu" in non and "non exploré" in non["cpu"] and "GPU" in non["cpu"]
    assert any(k.startswith("experts_partiel") for k in non)


def test_plan_dense_trop_gros_deux_partiels_serre_et_prudent():
    # 80 Go dense, 24 Go de VRAM : il faut assez de RAM pour les couches restées sur CPU.
    plan = _plan(model_size_mb=80_000, vram_total_mb=24_000, ram_total_mb=96_000)
    keys = _keys(plan)
    assert len(keys) == 2 and all(k.startswith("gpu_partiel_ngl") for k in keys)
    serre, prudent = plan.candidates
    assert 0 < prudent.ngl < serre.ngl < 40 and serre.estime and prudent.estime
    assert any(n["key"] == "cpu" for n in plan.non_explores)


def test_plan_dense_le_partiel_prudent_tient_aussi_en_ram():
    """Revue P1 : le côté hôte est vérifié lui aussi. 80 Go dense, 24 Go de VRAM, 64 Go de
    RAM : le serré (ngl 11) laisse ~59,5 Go en RAM (tient dans 60,9), le prudent (ngl 7)
    en laisserait ~67,7 Go — infaisable, dit tel quel plutôt que proposé."""
    plan = _plan(model_size_mb=80_000, vram_total_mb=24_000, ram_total_mb=64_000)
    assert _keys(plan) == ["gpu_partiel_ngl11"]
    non = {n["key"]: n["raison"] for n in plan.non_explores}
    assert "gpu_partiel_ngl7" in non and "RAM" in non["gpu_partiel_ngl7"]


def test_plan_moe_trop_gros_deux_partiels():
    plan = _plan(moe=True, model_size_mb=35_193, vram_total_mb=16_000)
    keys = _keys(plan)
    assert keys[0] == "experts_cpu" and len(keys) == 3
    serre, prudent = plan.candidates[1], plan.candidates[2]
    assert serre.n_cpu_moe < prudent.n_cpu_moe < 40


def _profil_moe(n_layers, attention, experts, output=0):
    w = {
        "total": n_layers * (attention + experts) + output,
        "familles": {
            "attention": n_layers * attention,
            "experts": n_layers * experts,
            "output": output,
        },
        "par_couche": [attention + experts] * n_layers,
        "experts_par_couche": [experts] * n_layers,
        "couches_attention": list(range(n_layers)),
        "couches_recurrentes": [],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    return ModelProfile.from_meta(
        {"n_layers": n_layers, "expert_count": 8, "weights": w}
    )


def test_plan_les_checkpoints_ram_ne_sont_pas_imputes_a_la_vram():
    """Revue P1 (2026-10-10) : 16 Gio de poids sur un GPU de 24 Gio perdaient leur
    candidat tout-GPU parce que ~9,35 Gio de checkpoints RAM étaient comptés en VRAM."""
    plan = plan_placements(
        moe=False,
        n_layers=40,
        model_size_mb=16 * 1024,
        kv_mb=500,  # côté device : KV + état vivant
        host_extra_mb=9_575,  # côté hôte : checkpoints
        gpu_backend=True,
        vram_total_mb=24 * 1024,
        ram_total_mb=64_000,
        uma=False,
        headroom_mb=640,
    )
    assert _keys(plan) == ["gpu_total"]
    assert (
        "RAM" in plan.candidates[0].faisabilite
        or "hôte" in plan.candidates[0].faisabilite
    )


def test_plan_en_uma_la_memoire_est_comptee_une_seule_fois():
    """Mémoire unifiée : le plafond device (heap Vulkan, 48 789 - 640 = 48 149) borne
    le device SEUL ; la RAM physique (64 000 - 3 072 = 60 928) borne la somme device +
    hôte, comptée une fois. Les checkpoints RAM ne sont pas imputés au plafond device."""
    kw = dict(
        moe=True,
        n_layers=40,
        model_size_mb=35_193,
        kv_mb=1_000,
        gpu_backend=True,
        vram_total_mb=48_789,
        ram_total_mb=64_000,
        uma=True,
        headroom_mb=640,
    )
    # Device 36 193 <= 48 149 ; somme 49 193 <= 60 928 : tout-GPU tient. (Avant : les
    # 13 000 Mo de checkpoints comptés en device -> 49 193 > 48 149, candidat perdu.)
    assert "gpu_total" in _keys(plan_placements(**kw, host_extra_mb=13_000))
    # Somme 61 193 > 60 928 : ne tient plus, bien que le device seul (36 193) tienne.
    assert "gpu_total" not in _keys(plan_placements(**kw, host_extra_mb=25_000))


def test_plan_experts_cpu_infaisable_quand_la_ram_manque():
    """GPU discret, experts en RAM : poids CPU + checkpoints contre la RAM moins la marge
    OS. 34 Gio d'experts + 9 Gio de checkpoints ne tiennent pas dans 32 Go."""
    mib = 1024 * 1024
    prof = _profil_moe(4, attention=500 * mib, experts=8_500 * mib)
    plan = plan_placements(
        moe=True,
        n_layers=4,
        model_size_mb=36_000,
        kv_mb=500,
        host_extra_mb=9_000,
        gpu_backend=True,
        vram_total_mb=24_000,
        ram_total_mb=32_000,
        uma=False,
        headroom_mb=640,
        profile=prof,
    )
    keys = _keys(plan)
    assert "experts_cpu" not in keys and "gpu_total" not in keys
    # Le partiel retenu tient des DEUX côtés : device 36 000 - n x 8 500 + 500 <= 23 360
    # (n >= 2) et hôte n x 8 500 + 9 000 <= 28 928 (n <= 2) -> n = 2, pas de prudent n = 3.
    assert keys == ["experts_partiel_n2"]
    non = {n["key"]: n["raison"] for n in plan.non_explores}
    assert "experts_cpu" in non and "RAM" in non["experts_cpu"]


def test_plan_utilise_le_profil_pour_la_faisabilite():
    """Le catalogue dit combien pèsent VRAIMENT les experts : le nombre de couches à
    laisser sur CPU en découle, au lieu d'une proportion aveugle de la taille."""
    mib = 1024 * 1024
    prof = _profil_moe(4, attention=500 * mib, experts=8_500 * mib)
    kw = dict(
        moe=True,
        n_layers=4,
        model_size_mb=36_000,
        kv_mb=1_000,
        gpu_backend=True,
        vram_total_mb=28_000 + 640,
        ram_total_mb=64_000,
        uma=False,
        headroom_mb=640,
    )
    aveugle = plan_placements(**kw)
    informe = plan_placements(**kw, profile=prof)
    # Proportionnel : déficit 9 000 x 4 / 36 000 -> 1 couche. Catalogue : 1 couche
    # laisse 28 500 > 28 000 sur le device, il en faut 2.
    assert aveugle.candidates[1].n_cpu_moe == 1
    assert informe.candidates[1].n_cpu_moe == 2
    assert "catalogue" in informe.candidates[1].faisabilite


# ── sonde : validation, présélection, finalistes ─────────────────────────────────


class _Sonde:
    """Rejoue (tg, pp) par (clé, ctx) ; "boom" lève. Journalise (clé, ctx, depth)."""

    def __init__(self, placement, table, journal):
        self.placement, self.table, self.journal = placement, table, journal

    def run(self, ctx, depth):
        self.journal.append((self.placement.key, ctx, depth))
        val = (
            self.table.get((self.placement.key, ctx)) or self.table[self.placement.key]
        )
        if val[0] == "boom":
            raise RuntimeError("ErrorOutOfDeviceMemory")
        return ProbeResult(ctx=ctx, mem_mb=1234, tg_ts=val[0], pp_ts=val[1])


def _usine(table):
    journal = []

    def make(placement):
        return _Sonde(placement, table, journal)

    make.journal = journal
    return make


GPU = Placement("gpu_total", 999)
CPU = Placement("experts_cpu", 999, cpu_moe=True)
P10 = Placement("experts_partiel", 999, n_cpu_moe=10, estime=True)
P20 = Placement("experts_partiel", 999, n_cpu_moe=20, estime=True)


def test_un_seul_candidat_est_valide_au_contexte_utile():
    make = _usine({"gpu_total": (8.0, 60.0)})
    r = probe_placement(make, [GPU], useful_ctx=32_768, reps=1)
    assert r["placement"] is GPU and r["compare"] is False
    assert "validé" in r["mecanisme"] and "non comparé" in r["mecanisme"]
    assert r["mesures"]["gpu_total"]["tg_ts"] == 8.0
    assert make.journal == [("gpu_total", 32_768, final_depth(32_768))]


def test_un_seul_candidat_en_echec_est_dit_en_echec():
    make = _usine({"gpu_total": ("boom", 0)})
    r = probe_placement(make, [GPU], useful_ctx=8_192, reps=1)
    assert r["placement"] is None and "ÉCHEC" in r["mecanisme"]
    assert r["mesures"]["gpu_total"]["echec"].startswith("RuntimeError")


def test_profondeur_finale_bornee():
    assert final_depth(32_768) == 16_384
    assert final_depth(65_536) == PLACEMENT_FINAL_DEPTH_MAX
    assert final_depth(8_192) == 4_096


def test_finalistes_compares_au_contexte_utile_et_decision_a_cette_profondeur():
    """Présélection à 8 192 : B et C devant, D loin derrière. Au contexte utile
    (32 768, profondeur 16 384) le classement s'inverse entre B et C : la décision
    suit la mesure en profondeur, D n'y est pas remesuré."""
    table = {
        "experts_cpu": (10.0, 200.0),
        "gpu_total": (14.0, 260.0),
        "experts_partiel_n10": (14.2, 240.0),
        "experts_partiel_n20": (9.0, 150.0),
        ("experts_cpu", 32_768): (8.0, 180.0),
        ("gpu_total", 32_768): (12.0, 230.0),
        ("experts_partiel_n10", 32_768): (11.0, 220.0),
    }
    make = _usine(table)
    r = probe_placement(make, [CPU, GPU, P10, P20], useful_ctx=32_768, reps=1)
    assert r["placement"] is GPU and r["compare"] is True
    assert r["ctx_final"] == 32_768 and r["depth_final"] == 16_384
    assert sorted(r["finalistes"]) == [
        "experts_cpu",
        "experts_partiel_n10",
        "gpu_total",
    ]
    assert r["tg_ts"] == 12.0  # la mesure au contexte utile, pas celle de présélection
    assert r["preselection"]["experts_partiel_n10"]["tg_ts"] == 14.2
    profond = [j for j in make.journal if j[1] == 32_768]
    assert ("experts_partiel_n20", 32_768, 16_384) not in profond
    assert len(profond) == 3


def test_contexte_utile_court_une_seule_phase():
    make = _usine({"experts_cpu": (12.1, 217.0), "gpu_total": (14.4, 262.0)})
    r = probe_placement(make, [CPU, GPU], useful_ctx=PLACEMENT_PROBE_CTX, reps=1)
    assert r["placement"] is GPU
    assert all(
        j[1:] == (PLACEMENT_PROBE_CTX, PLACEMENT_PROBE_PROMPT) for j in make.journal
    )
    assert r["ctx_final"] == PLACEMENT_PROBE_CTX


def test_trace_distingue_non_explore_et_mesure_moins_performant():
    make = _usine({"experts_cpu": (12.1, 217.0), "gpu_total": (14.4, 262.0)})
    non = [{"key": "cpu", "raison": "non exploré : déprioritisé, GPU disponible"}]
    r = probe_placement(make, [CPU, GPU], reps=1, non_explores=non)
    assert r["non_explores"] == non
    assert "non exploré" in r["mecanisme"] and "cpu" in r["mecanisme"]
    # experts_cpu a été MESURÉ moins performant : le mécanisme donne l'écart, pas « non exploré ».
    assert "experts_cpu" in r["mecanisme"] and "12.1" in r["mecanisme"]


# ── contraintes de prefill ───────────────────────────────────────────────────────


def _mes(**par_cle):
    return {k: {"tg_ts": v[0], "pp_ts": v[1]} for k, v in par_cle.items()}


def test_contrainte_prefill_explicite_ecarte_et_le_dit():
    # « 2 000 nouveaux tokens en moins de 10 s » : 150 t/s = 13,3 s -> écarté.
    best, mecanisme = pick_placement(
        _mes(experts_cpu=(12.0, 400.0), gpu_total=(14.4, 150.0)),
        [CPU, GPU],
        prefill=PrefillConstraint(new_tokens=2_000, max_seconds=10.0),
    )
    assert best is CPU
    assert "contrainte prefill" in mecanisme and "gpu_total" in mecanisme
    assert "13" in mecanisme and "écarté" in mecanisme


def test_contrainte_prefill_insatisfiable_decide_a_la_generation():
    best, mecanisme = pick_placement(
        _mes(experts_cpu=(12.0, 100.0), gpu_total=(14.4, 120.0)),
        [CPU, GPU],
        prefill=PrefillConstraint(new_tokens=2_000, max_seconds=10.0),
    )
    assert best is GPU
    assert "aucun candidat ne satisfait" in mecanisme


def test_plancher_relatif_de_prefill_est_un_choix_de_confort():
    best, mecanisme = pick_placement(
        _mes(experts_cpu=(12.0, 400.0), gpu_total=(14.4, 150.0)),
        [CPU, GPU],
        pp_floor_ratio=0.5,
    )
    assert best is CPU
    assert "choix de confort" in mecanisme


def test_sans_contrainte_la_generation_tranche():
    best, _ = pick_placement(
        _mes(experts_cpu=(12.0, 400.0), gpu_total=(14.4, 150.0)), [CPU, GPU]
    )
    assert best is GPU


def test_indecis_conserve_l_actuel():
    actuel = Placement("gpu_total", 999, actuel=True)
    best, mecanisme = pick_placement(
        _mes(gpu_total=(14.0, 250.0), experts_cpu=(14.4, 260.0)), [actuel, CPU]
    )
    assert best is actuel and "conservé" in mecanisme


# ── persistance d'un partiel dense ───────────────────────────────────────────────


def test_set_model_placement_partiel_dense_ecrit_le_ngl_exact(tmp_path):
    import tomllib

    from loom.setup.cli import _set_model_placement

    (tmp_path / "model.toml").write_text('filename = "m.gguf"\n', encoding="utf-8")
    _set_model_placement(
        tmp_path / "m.gguf", Placement("gpu_partiel", 36, estime=True), "d"
    )
    d = tomllib.loads((tmp_path / "model.toml").read_text(encoding="utf-8"))
    assert d["n_gpu_layers"] == 36 and d["cpu_moe"] is False
