# tests/test_placement_batchs.py
"""Couples placement × batchs sur les FINALISTES (revue du 2026-10-10, chantier 3).

Vu sur Ornith : la comparaison des placements tournait avec le ubatch machine (2048),
puis la sonde ubatch, séparée, trouvait 512 bien meilleur pour l'élu — à un autre
contexte, une autre profondeur, et sans jamais mesurer experts-CPU en 512. Ici les
finalistes sont comparés × deux couples (ubatch, batch) : quatre configurations
complètes au même contexte, à la même profondeur, avec les mêmes slots, tours alternés,
échantillons conservés, décision à la génération. La configuration ACTUELLE exacte
(placement + batchs de l'exécutant) est la base ; en cas d'indécision elle reste.
"""

from __future__ import annotations

from loom.setup.placement import (
    PLACEMENT_PROBE_CTX,
    PLACEMENT_PROBE_PROMPT,
    Placement,
    batch_couples,
    final_depth,
    probe_placement,
)
from loom.setup.topology import ProbeResult

# ── identité d'une configuration complète ───────────────────────────────────────


def test_la_cle_porte_le_couple_de_batchs():
    assert Placement("gpu_total", 999, ubatch=512, batch=2048).key == "gpu_total@ub512"
    assert (
        Placement("experts_partiel", 999, n_cpu_moe=20, ubatch=2048, batch=4096).key
        == "experts_partiel_n20@ub2048"
    )
    assert Placement("gpu_total", 999).key == "gpu_total"  # sans couple : inchangé
    assert (
        "ub 512/b 2048"
        in Placement("gpu_total", 999, ubatch=512, batch=2048).describe()
    )


def test_couples_de_batchs_actuel_puis_alternative():
    # L'exécutant tourne en 2048/4096 ([server] machine) : actuel d'abord, puis l'autre.
    assert batch_couples((2048, 4096)) == [(2048, 4096), (512, 2048)]
    # Rien d'explicite : les défauts llama-server (512/2048) sont l'actuel.
    assert batch_couples(None) == [(512, 2048), (2048, 4096)]
    assert batch_couples((None, None)) == [(512, 2048), (2048, 4096)]
    # Un couple exotique reste la base ; deux couples au plus.
    assert batch_couples((1024, 2048)) == [(1024, 2048), (512, 2048)]


# ── sonde : finalistes × couples ────────────────────────────────────────────────


class _Sonde:
    """Rejoue (tg, pp) par (clé complète, ctx). La clé complète lit `self.ubatch`, posé
    par la sonde de placement sur une COPIE de l'objet (comme sur ServerProbe)."""

    def __init__(self, placement, table, journal, ubatch=None, batch=None):
        self.placement, self.table, self.journal = placement, table, journal
        self.ubatch, self.batch = ubatch, batch

    def _key(self):
        base = self.placement.key.split("@")[0]
        return f"{base}@ub{self.ubatch}" if self.ubatch else base

    def run(self, ctx, depth):
        k = self._key()
        self.journal.append((k, ctx, depth))
        val = (
            self.table.get((k, ctx)) or self.table.get(k) or self.table[k.split("@")[0]]
        )
        if val[0] == "boom":
            raise RuntimeError("ErrorOutOfDeviceMemory")
        return ProbeResult(ctx=ctx, mem_mb=1234, tg_ts=val[0], pp_ts=val[1])


def _usine(table, ubatch=2048, batch=4096):
    journal = []

    def make(placement):
        # La fabrique rend une sonde aux batchs ACTUELS de l'exécutant.
        return _Sonde(placement, table, journal, ubatch=ubatch, batch=batch)

    make.journal = journal
    return make


ACTUEL = Placement("gpu_total", 999, actuel=True)
CPU = Placement("experts_cpu", 999, cpu_moe=True)
COUPLES = [(2048, 4096), (512, 2048)]


def test_finalistes_compares_x_couples_au_meme_contexte_et_slots():
    table = {
        # Présélection (batchs actuels 2048) à 8 192.
        ("gpu_total@ub2048", PLACEMENT_PROBE_CTX): (12.8, 138.0),
        ("experts_cpu@ub2048", PLACEMENT_PROBE_CTX): (10.0, 292.0),
        # Quatre configurations complètes à 32 768 / 16 384.
        ("gpu_total@ub2048", 32_768): (11.9, 119.0),
        ("gpu_total@ub512", 32_768): (11.8, 205.0),
        ("experts_cpu@ub2048", 32_768): (10.55, 219.0),
        ("experts_cpu@ub512", 32_768): (10.6, 250.0),
    }
    make = _usine(table)
    r = probe_placement(
        make, [ACTUEL, CPU], useful_ctx=32_768, reps=1, batch_couples=COUPLES
    )
    assert r["couples"] == COUPLES
    # La base est la configuration actuelle EXACTE : placement + batchs de l'exécutant.
    assert r["baseline"] == "gpu_total@ub2048"
    profond = [j for j in make.journal if j[1] == 32_768]
    # Quatre configurations, même contexte, même profondeur, ordre alterné.
    assert [j[0] for j in profond] == [
        "gpu_total@ub2048",
        "experts_cpu@ub2048",
        "gpu_total@ub512",
        "experts_cpu@ub512",
    ]
    assert all(j[2] == final_depth(32_768) for j in profond)
    assert set(r["mesures"]) == {j[0] for j in profond}
    # Génération équivalente entre gpu_total@2048 et @512 (-0,8 %) : la base (actuelle)
    # reste ; le prefill 205 contre 119 est une information, pas la décision.
    assert r["placement"].key == "gpu_total@ub2048"
    assert (r["placement"].ubatch, r["placement"].batch) == (2048, 4096)
    assert "indécis" in r["mecanisme"] or "conservé" in r["mecanisme"]


def test_un_autre_couple_peut_gagner_et_porte_ses_batchs():
    table = {
        "gpu_total@ub2048": (10.0, 119.0),
        "gpu_total@ub512": (11.5, 205.0),  # +15 % de génération : adopté
        "experts_cpu@ub2048": (9.0, 219.0),
        "experts_cpu@ub512": (9.1, 250.0),
    }
    r = probe_placement(
        _usine(table), [ACTUEL, CPU], useful_ctx=32_768, reps=1, batch_couples=COUPLES
    )
    assert r["placement"].key == "gpu_total@ub512"
    assert (r["placement"].ubatch, r["placement"].batch) == (512, 2048)
    assert r["placement"].label == "gpu_total" and r["placement"].ngl == 999
    assert r["gain_pct"] == 15.0


def test_contexte_utile_court_les_couples_sont_quand_meme_compares():
    table = {
        "gpu_total@ub2048": (12.0, 138.0),
        "gpu_total@ub512": (12.1, 300.0),
        "experts_cpu@ub2048": (10.0, 292.0),
        "experts_cpu@ub512": (10.1, 310.0),
    }
    make = _usine(table)
    r = probe_placement(
        make,
        [ACTUEL, CPU],
        useful_ctx=PLACEMENT_PROBE_CTX,
        reps=1,
        batch_couples=COUPLES,
    )
    assert len(r["mesures"]) == 4 and r["ctx_final"] == PLACEMENT_PROBE_CTX
    assert all(j[2] == PLACEMENT_PROBE_PROMPT for j in make.journal)


def test_sans_couples_le_comportement_est_inchange():
    make = _usine(
        {"gpu_total": (12.8, 138.0), "experts_cpu": (10.0, 292.0)}, ubatch=None
    )
    r = probe_placement(make, [ACTUEL, CPU], reps=1)
    assert set(r["mesures"]) == {"gpu_total", "experts_cpu"} and r.get("couples") == []
    assert r["placement"].ubatch is None


def test_un_seul_candidat_est_valide_avec_les_deux_couples():
    table = {"gpu_total@ub2048": (8.0, 60.0), "gpu_total@ub512": (8.1, 90.0)}
    make = _usine(table)
    r = probe_placement(
        make, [ACTUEL], useful_ctx=32_768, reps=1, batch_couples=COUPLES
    )
    # Rien à comparer entre placements, mais le couple de batchs, lui, se mesure.
    assert set(r["mesures"]) == {"gpu_total@ub2048", "gpu_total@ub512"}
    assert r["placement"].label == "gpu_total" and r["compare"] is True
