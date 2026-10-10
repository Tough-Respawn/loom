# tests/test_placement_decision.py
"""Décision FIABLE sans multiplier les mesures (lot 4 du bench de placement).

Revue du 2026-10-10 : deux mesures moyennées ne disaient rien du bruit ; les
répétitions d'un candidat s'enchaînaient (dérive thermique, caches) ; la marge de 5 %
est une politique de changement, pas une mesure du bruit ; un résultat indécis doit
conserver la configuration actuelle quand elle reste valide.

Ici : chaque échantillon est conservé (tg, pp, mémoire, tokens réellement traités),
la dispersion est calculée, l'ordre des candidats ALTERNE d'une répétition à l'autre,
on ne répète davantage que si le classement reste incertain, et un gain sous la
dispersion mesurée est « indécis » : la base est conservée et le mécanisme le dit.
"""

from __future__ import annotations

from loom.setup.placement import (
    PLACEMENT_MAX_REPS,
    Placement,
    pick_placement,
    probe_placement,
)
from loom.setup.topology import ProbeResult

GPU = Placement("gpu_total", 999)
CPU = Placement("experts_cpu", 999, cpu_moe=True)


class _Sonde:
    """Rejoue, par clé, une SUITE de (tg, pp) : un échantillon par run."""

    def __init__(self, placement, suites, journal):
        self.placement, self.suites, self.journal = placement, suites, journal
        self.i = 0

    def run(self, ctx, depth):
        self.journal.append(self.placement.key)
        suite = self.suites[self.placement.key]
        tg, pp = suite[self.i % len(suite)]
        self.i += 1
        return ProbeResult(
            ctx=ctx,
            mem_mb=1000 + self.i,
            tg_ts=tg,
            pp_ts=pp,
            prompt_n=4000,
            predicted_n=96,
        )


def _usine(suites):
    journal: list[str] = []

    def make(placement):
        return _Sonde(placement, suites, journal)

    make.journal = journal
    return make


def test_echantillons_et_dispersion_conserves():
    make = _usine(
        {"experts_cpu": [(12.0, 200.0)], "gpu_total": [(14.0, 260.0), (14.8, 264.0)]}
    )
    r = probe_placement(make, [CPU, GPU], reps=2)
    m = r["mesures"]["gpu_total"]
    assert m["tg_ts"] == 14.4 and m["n"] == 2
    assert [e["tg_ts"] for e in m["echantillons"]] == [14.0, 14.8]
    assert (
        m["echantillons"][0]["prompt_n"] == 4000
        and m["echantillons"][0]["predicted_n"] == 96
    )
    assert m["echantillons"][0]["mem_mb"] > 0
    # Dispersion = étendue relative à la moyenne : (14,8 - 14,0) / 14,4 = 5,6 %.
    assert m["tg_disp_pct"] == 5.6
    assert r["mesures"]["experts_cpu"]["tg_disp_pct"] == 0.0


def test_l_ordre_des_candidats_alterne_entre_les_repetitions():
    make = _usine({"experts_cpu": [(12.0, 200.0)], "gpu_total": [(14.4, 260.0)]})
    probe_placement(make, [CPU, GPU], reps=2)
    assert make.journal == ["experts_cpu", "gpu_total", "experts_cpu", "gpu_total"]


def test_classement_net_pas_de_repetition_supplementaire():
    make = _usine({"experts_cpu": [(10.0, 200.0)], "gpu_total": [(14.0, 260.0)]})
    r = probe_placement(make, [CPU, GPU], reps=2)
    assert (
        make.journal.count("experts_cpu") == 2 and make.journal.count("gpu_total") == 2
    )
    assert r["placement"] is GPU


def test_classement_incertain_repete_jusqu_au_plafond():
    # Moyennes égales (10,3), dispersions 6 % et 4 % : l'écart est sous le bruit.
    make = _usine(
        {
            "experts_cpu": [(10.0, 200.0), (10.6, 200.0), (10.2, 200.0), (10.4, 200.0)],
            "gpu_total": [(10.5, 205.0), (10.1, 205.0), (10.3, 205.0), (10.3, 205.0)],
        }
    )
    r = probe_placement(make, [CPU, GPU], reps=2)
    assert make.journal.count("experts_cpu") == PLACEMENT_MAX_REPS
    assert make.journal.count("gpu_total") == PLACEMENT_MAX_REPS
    assert r["mesures"]["gpu_total"]["n"] == PLACEMENT_MAX_REPS
    # Indécis sur les DEUX axes (prefill équivalent) : la base est conservée, le
    # mécanisme le dit.
    assert r["placement"] is CPU and "indécis" in r["mecanisme"]


def test_budget_temps_epuise_pas_de_repetition_supplementaire():
    make = _usine(
        {
            "experts_cpu": [(10.0, 200.0), (10.6, 200.0)],
            "gpu_total": [(10.5, 260.0), (10.1, 260.0)],
        }
    )
    probe_placement(make, [CPU, GPU], reps=2, time_budget_s=0)
    assert (
        make.journal.count("experts_cpu") == 2 and make.journal.count("gpu_total") == 2
    )


def test_seul_le_tandem_incertain_est_remesure():
    # Trois candidats : le 3e est loin derrière, seuls les deux premiers sont affinés.
    p10 = Placement("experts_partiel", 999, n_cpu_moe=10, estime=True)
    make = _usine(
        {
            "experts_cpu": [(10.0, 200.0), (10.6, 200.0), (10.2, 200.0), (10.4, 200.0)],
            "gpu_total": [(10.5, 205.0), (10.1, 205.0), (10.3, 205.0), (10.3, 205.0)],
            "experts_partiel_n10": [(6.0, 100.0)],
        }
    )
    probe_placement(make, [CPU, GPU, p10], reps=2)
    assert make.journal.count("experts_partiel_n10") == 2
    assert make.journal.count("gpu_total") == PLACEMENT_MAX_REPS


# ── décision : gain contre dispersion ────────────────────────────────────────────


def _m(tg, pp, disp):
    return {"tg_ts": tg, "pp_ts": pp, "tg_disp_pct": disp}


def test_generation_equivalente_le_prefill_departage_meme_contre_la_base():
    """Ornith, 3e /rebench (2026-10-10) : gpu_total@ub2048 (base) 11,2 t/s / 124 t/s de
    prefill ; gpu_total@ub512 11,3 / 212. Génération équivalente, prefill +71 % : garder
    l'actuel vaut pour l'INCERTITUDE, pas quand un axe est net sans perte sur l'autre."""
    from loom.setup.placement import PLACEMENT_PP_TIEBREAK_PCT

    base = Placement("gpu_total", 999, ubatch=2048, batch=4096, actuel=True)
    alt = Placement("gpu_total", 999, ubatch=512, batch=2048)
    best, mecanisme = pick_placement(
        {
            "gpu_total@ub2048@b4096": _m(11.2, 124.0, 1.8),
            "gpu_total@ub512@b2048": _m(11.3, 212.1, 0.0),
        },
        [base, alt],
    )
    assert best is alt
    assert "équivalente" in mecanisme and "prefill" in mecanisme and "+71" in mecanisme
    assert PLACEMENT_PP_TIEBREAK_PCT == 10.0


def test_les_quatre_mesures_reelles_d_ornith_elisent_gpu_total_ub512():
    """Revue P1 (2026-10-10) : rejouées telles quelles depuis la session 4f2047f991f0,
    la règle de 2ff84b9 élisait experts_cpu@ub2048 (-6,7 % de génération, dispersion
    6,7 %, prefill 217 > 212). La dispersion n'élargit pas la perte acceptable : seule
    la marge explicite borne l'équivalence."""
    base = Placement("gpu_total", 999, ubatch=2048, batch=4096, actuel=True)
    cands = [
        base,
        Placement("experts_cpu", 999, cpu_moe=True, ubatch=2048, batch=4096),
        Placement("gpu_total", 999, ubatch=512, batch=2048),
        Placement("experts_cpu", 999, cpu_moe=True, ubatch=512, batch=2048),
    ]
    reel = {
        "gpu_total@ub2048@b4096": _m(11.2, 124.0, 1.8),
        "experts_cpu@ub2048@b4096": _m(10.45, 217.2, 6.7),
        "gpu_total@ub512@b2048": _m(11.3, 212.1, 0.0),
        "experts_cpu@ub512@b2048": _m(10.65, 180.55, 0.9),
    }
    best, mecanisme = pick_placement(reel, cands)
    assert best.key == "gpu_total@ub512@b2048"
    assert "équivalente" in mecanisme and "+71" in mecanisme


def test_une_forte_dispersion_n_elargit_pas_la_perte_de_generation_acceptable():
    base = Placement("gpu_total", 999, ubatch=2048, batch=4096, actuel=True)
    alt = Placement("experts_cpu", 999, cpu_moe=True, ubatch=2048, batch=4096)
    best, mecanisme = pick_placement(
        {
            "gpu_total@ub2048@b4096": _m(11.2, 124.0, 1.8),
            "experts_cpu@ub2048@b4096": _m(10.45, 217.2, 6.7),
        },
        [base, alt],
    )
    assert best is base and "conservé" in mecanisme


def test_generation_equivalente_prefill_equivalent_garde_la_base():
    base = Placement("gpu_total", 999, ubatch=2048, batch=4096, actuel=True)
    alt = Placement("gpu_total", 999, ubatch=512, batch=2048)
    best, mecanisme = pick_placement(
        {
            "gpu_total@ub2048@b4096": _m(11.2, 124.0, 1.8),
            "gpu_total@ub512@b2048": _m(11.3, 130.0, 0.0),
        },
        [base, alt],
    )
    assert best is base and "conservé" in mecanisme


def test_generation_equivalente_prefill_pire_garde_la_base():
    base = Placement("gpu_total", 999, ubatch=2048, batch=4096, actuel=True)
    alt = Placement("gpu_total", 999, ubatch=512, batch=2048)
    best, _ = pick_placement(
        {
            "gpu_total@ub2048@b4096": _m(11.2, 124.0, 1.8),
            "gpu_total@ub512@b2048": _m(11.3, 90.0, 0.0),
        },
        [base, alt],
    )
    assert best is base


def test_gain_sous_la_dispersion_est_indecis_base_conservee():
    # Prefill équivalent (205 contre 200) : seule la génération compte, et elle est indécise.
    best, mecanisme = pick_placement(
        {"experts_cpu": _m(12.0, 200.0, 12.0), "gpu_total": _m(13.0, 205.0, 10.0)},
        [CPU, GPU],
    )
    # +8 % dépasse la marge de 5 %, mais pas la dispersion mesurée (12 %).
    assert best is CPU
    assert (
        "indécis" in mecanisme
        and "dispersion" in mecanisme
        and "gpu_total" in mecanisme
    )


def test_gain_au_dela_de_la_dispersion_est_adopte():
    best, mecanisme = pick_placement(
        {"experts_cpu": _m(12.0, 200.0, 2.0), "gpu_total": _m(14.4, 260.0, 3.0)},
        [CPU, GPU],
    )
    assert best is GPU and "adopté" in mecanisme


def test_sans_dispersion_connue_la_marge_seule_decide():
    best, _ = pick_placement(
        {
            "experts_cpu": {"tg_ts": 12.0, "pp_ts": 200.0},
            "gpu_total": {"tg_ts": 13.0, "pp_ts": 260.0},
        },
        [CPU, GPU],
    )
    assert best is GPU


# ── la sonde remonte les tokens réellement traités ───────────────────────────────


def test_probe_result_porte_les_tokens_traites(monkeypatch):
    from loom.runtime.hardware import HardwareProfile
    from loom.setup import topology as topo
    from loom.setup.topology import TOPO_MOE_HYBRIDE, ServerProbe

    monkeypatch.setattr(topo.time, "sleep", lambda s: None)
    probe = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=HardwareProfile(True, "GPU", 1000, 16, vram_total_mb=48_000),
        kill=lambda p: None,
    )

    class _Proc:
        pid = 1

    def fake_completion(prompt, n_predict, cache_prompt=False):
        return {
            "timings": {
                "predicted_per_second": 14.4,
                "prompt_per_second": 262.0,
                "prompt_n": 4011,
                "predicted_n": n_predict,
            }
        }

    monkeypatch.setattr(probe, "_start", lambda ctx: _Proc())
    monkeypatch.setattr(probe, "_measure_mem", lambda proc: 100)
    monkeypatch.setattr(probe, "_completion", fake_completion)
    monkeypatch.setattr(probe, "_tokens_of", lambda text: 12)
    r = probe.run(8192, 4096)
    assert r.prompt_n == 4011 and r.predicted_n == 96
