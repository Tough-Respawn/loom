# tests/test_topology.py
"""Calibration agnostique (topology.py) : pente mesurée, capacité, échelle de
vitesse, décision tracée. Fixture « MACHINE DORÉE » = des sondes RÉELLES figées
(GPU 6 Go + modèle 35B MoE, relevé du 2026-07-18) :
le recommandeur doit retrouver la zone validée à la main PAR LE BON MÉCANISME —
un chiffre juste au mauvais mécanisme est un échec (leçon du cap qui maquillait).
"""

from __future__ import annotations

import pytest

from loom.setup.topology import (
    TOPO_GPU_DENSE,
    TOPO_MOE_HYBRIDE,
    TOPO_RAM,
    ProbeResult,
    calibrate,
    capacity_ctx,
    discover_topology,
    kv_slope,
    memory_budget_mb,
)

# ── Mesures réelles du 2026-07-18 (VRAM au chargement, Ornith Q8, n_cpu_moe=40) ──
GOLDEN_RUNGS = [(24576, 3268), (32768, 3307), (49152, 3451), (65536, 3637)]
GOLDEN_META = {"context_length": 262144, "expert_count": 256, "n_layers": 40}
GOLDEN_BUDGET = 6144 - 640  # VRAM totale − gpu_kv_headroom_mb
GOLDEN_TG = {16384: 14.4, 32768: 13.5, 65536: 12.5, 131072: 11.8, 262144: 11.0}


class FakeProbe:
    """Rejoue des mesures : mem par interpolation des barreaux dorés, tg par table.
    Compte les runs pour vérifier le protocole (2 chargements de pente d'abord)."""

    def __init__(self, tg_table=None, fail_at=None, slope_rungs=GOLDEN_RUNGS):
        self.tg = tg_table or GOLDEN_TG
        self.fail_at = fail_at or set()
        self.calls: list[tuple[int, int | None]] = []
        (x1, y1), (x2, y2) = slope_rungs[0], slope_rungs[-1]
        self._a = (y2 - y1) / (x2 - x1)
        self._b = y1 - self._a * x1

    def run(self, ctx, depth):
        self.calls.append((ctx, depth))
        if ctx in self.fail_at:
            raise RuntimeError("health timeout")
        r = ProbeResult(ctx=ctx, mem_mb=int(self._a * ctx + self._b))
        if depth:
            r.tg_ts = self.tg.get(ctx, 10.0)
            r.pp_ts = 195.0
        return r


# ── briques pures ────────────────────────────────────────────────────────────


def test_kv_slope_sur_les_mesures_reelles():
    slope, base = kv_slope(GOLDEN_RUNGS)
    # Pente réelle mesurée ~9-12 Ko/token (sliding-window) — la formule header
    # donnait 43,5 Ko (q8_0) et 81,9 (f16) : c'est LE bug que ce module corrige.
    assert 8 * 1024 < slope < 13 * 1024, (
        f"pente {slope / 1024:.1f} Ko/tok hors zone mesurée"
    )
    assert 2800 < base < 3200  # ~poids attention + buffers sur GPU


def test_kv_slope_refuse_un_seul_barreau():
    with pytest.raises(ValueError):
        kv_slope([(8192, 3000)])


def test_capacity_bornee_par_le_modele():
    slope, base = kv_slope(GOLDEN_RUNGS)
    cap = capacity_ctx(slope, base, GOLDEN_BUDGET, model_limit=32768)
    assert cap == 32768
    cap2 = capacity_ctx(slope, base, GOLDEN_BUDGET, model_limit=262144)
    assert cap2 % 2048 == 0 and cap2 > 65536  # la 2060 porte bien plus que l'ex-mur


def test_capacity_budget_trop_petit_retombe_au_plancher():
    slope, base = kv_slope(GOLDEN_RUNGS)
    assert capacity_ctx(slope, base, budget_mb=1000, model_limit=262144) == 4096


def test_discover_topology():
    assert discover_topology(GOLDEN_META, True, 6144) == TOPO_MOE_HYBRIDE
    assert discover_topology({"expert_count": None}, True, 6144) == TOPO_GPU_DENSE
    assert discover_topology(GOLDEN_META, False, 0) == TOPO_RAM
    assert discover_topology(GOLDEN_META, True, 0) == TOPO_RAM  # backend sans VRAM vue


def test_budget_deterministe():
    # Sur les TOTAUX moins marges fixes — jamais la mémoire disponible du moment (P3).
    assert memory_budget_mb(TOPO_MOE_HYBRIDE, 6144, 65536, 640) == 5504
    assert memory_budget_mb(TOPO_RAM, 0, 65536, 640) == 65536 - 3072


# ── MACHINE DORÉE : le recommandeur rejoue le 2026-07-18 ────────────────────────


def test_machine_doree_retrouve_la_zone_validee_par_le_bon_mecanisme():
    probe = FakeProbe()
    out = calibrate(
        probe, GOLDEN_META, topology=TOPO_MOE_HYBRIDE, budget_mb=GOLDEN_BUDGET
    )
    # Protocole : 2 chargements de pente À VIDE d'abord, puis l'échelle de vitesse.
    assert probe.calls[0] == (8192, None) and probe.calls[1] == (16384, None)
    assert all(d is not None for _, d in probe.calls[2:])
    # Zone validée à la main le 18/07 : ≥ 49152 (l'ex-mur 24576 est aboli), et le
    # chiffre vient de barreaux de VITESSE mesurés, pas d'un cap ni d'une formule.
    assert out["mode"] == TOPO_MOE_HYBRIDE
    assert out["context"] >= 49152
    assert out["context"] == out["valide_jusqua"]  # jamais l'extrapolation seule
    assert 8 < out["slope_kb_tok"] < 13
    assert out["vitesses"], "aucun barreau de vitesse mesuré"


def test_effondrement_de_vitesse_conserve_le_barreau_precedent():
    # Décode qui s'écroule à 131072 (< 70 % de la référence) -> on garde 65536,
    # et le mécanisme le DIT (spill/dégradation).
    tg = {**GOLDEN_TG, 131072: 8.0}
    out = calibrate(
        FakeProbe(tg_table=tg),
        GOLDEN_META,
        topology=TOPO_MOE_HYBRIDE,
        budget_mb=GOLDEN_BUDGET,
    )
    assert out["context"] == 65536
    assert "vitesse" in out["mecanisme"] and "spill" in out["mecanisme"]


def test_exception_quelconque_d_un_barreau_conserve_le_dernier_sain():
    """Régression du test de vérité live (18/07) : un HTTPError (400) au barreau de
    vitesse tuait TOUTE la calibration au lieu d'être un mécanisme d'arrêt."""

    class Boom(FakeProbe):
        def run(self, ctx, depth):
            if ctx == 65536:
                raise OSError("HTTP Error 400: Bad Request")
            return super().run(ctx, depth)

    out = calibrate(
        Boom(), GOLDEN_META, topology=TOPO_MOE_HYBRIDE, budget_mb=GOLDEN_BUDGET
    )
    assert out["context"] == 32768
    assert "échec du barreau" in out["mecanisme"]


def test_echec_de_chargement_conserve_le_dernier_barreau_sain():
    out = calibrate(
        FakeProbe(fail_at={131072}),
        GOLDEN_META,
        topology=TOPO_MOE_HYBRIDE,
        budget_mb=GOLDEN_BUDGET,
    )
    assert out["context"] == 65536
    assert "échec du barreau ctx=131072" in out["mecanisme"]
    assert "health timeout" in out["mecanisme"]


def test_budget_temps_arrete_l_echelle_et_le_dit():
    out = calibrate(
        FakeProbe(),
        GOLDEN_META,
        topology=TOPO_MOE_HYBRIDE,
        budget_mb=GOLDEN_BUDGET,
        time_budget_s=0,  # tout de suite épuisé : aucun barreau de vitesse
    )
    assert out["context"] == 4096  # rien de VALIDÉ en vitesse -> plancher prudent
    assert "budget temps" in out["mecanisme"]


def test_petit_modele_borne_par_sa_limite():
    meta = dict(GOLDEN_META, context_length=16384)
    out = calibrate(
        FakeProbe(),
        meta,
        topology=TOPO_MOE_HYBRIDE,
        budget_mb=GOLDEN_BUDGET,
    )
    assert out["context"] <= 16384
