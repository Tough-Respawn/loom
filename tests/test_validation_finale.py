# tests/test_validation_finale.py
"""Validation du RÉGLAGE FINAL complet (revue du 2026-10-10, recommandation 1).

Le verdict assemblait des résultats obtenus avec des paramètres différents : placement
comparé à 32 768 / 1 slot / ub 2048, contexte calibré jusqu'à 65 536 / 2 slots, ubatch
choisi sur un prompt de 4 096 à ctx 8 192, cache vérifié à 4 096. La combinaison
proposée n'avait jamais tourné telle quelle. Ici : la configuration finale (placement
élu, slots décidés, batchs mesurés) est mesurée au contexte CALIBRÉ, à la profondeur de
la comparaison, deux fois ; sa génération est confrontée à celle qui a fait élire le
placement ; le cache est vérifié avec ce contexte alloué.
"""

from __future__ import annotations

from dataclasses import dataclass

from loom.setup.placement import Placement, validate_final
from loom.setup.topology import ProbeResult


def test_placement_depuis_les_flags_d_une_sonde():
    assert Placement.from_flags(999, False, None, 41).key == "gpu_total"
    # -ngl 41 sur 41 couches laisse la sortie sur CPU : réglage exact conservé, pas 999.
    assert Placement.from_flags(41, False, None, 41).key == "gpu_partiel_ngl41"
    assert Placement.from_flags(999, True, None, 41).key == "experts_cpu"
    assert Placement.from_flags(999, False, 20, 41).key == "experts_partiel_n20"
    assert Placement.from_flags(8, False, None, 41).key == "gpu_partiel_ngl8"
    assert Placement.from_flags(0, False, None, 41).key == "cpu"


@dataclass
class _SondeFinale:
    ngl: int = 999
    cpu_moe: bool = False
    n_cpu_moe: object = None
    n_parallel: int = 2
    ubatch: object = 512
    batch: object = 2048
    tg: float = 11.7
    pp: float = 280.0
    boom: bool = False

    def __post_init__(self):
        self.journal: list[tuple[int, int]] = []

    def run(self, ctx, depth):
        self.journal.append((ctx, depth))
        if self.boom:
            raise RuntimeError("ErrorOutOfDeviceMemory")
        return ProbeResult(
            ctx=ctx,
            mem_mb=40_000,
            tg_ts=self.tg,
            pp_ts=self.pp,
            prompt_n=depth,
            predicted_n=96,
        )


def test_validate_final_mesure_la_configuration_complete_au_contexte_calibre():
    sonde = _SondeFinale()
    f = validate_final(sonde, ctx=65_536, depth=16_384, n_layers=41, reference_tg=11.9)
    assert sonde.journal == [(65_536, 16_384), (65_536, 16_384)]
    assert f["tg_ts"] == 11.7 and f["pp_ts"] == 280.0 and f["n"] == 2
    assert f["ctx"] == 65_536 and f["depth"] == 16_384
    # La configuration mesurée est nommée : placement, slots, batchs.
    assert f["placement"] == "gpu_total" and f["slots"] == 2
    assert (f["ubatch"], f["batch"]) == (512, 2048)
    # Confrontée à la génération qui a fait élire le placement : -1,7 %, cohérent.
    assert f["reference_tg"] == 11.9 and f["ecart_pct"] == -1.7
    assert f["coherent"] is True
    assert len(f["echantillons"]) == 2 and f["echantillons"][0]["prompt_n"] == 16_384


def test_validate_final_incoherent_quand_la_generation_chute():
    f = validate_final(
        _SondeFinale(tg=9.0), ctx=65_536, depth=16_384, n_layers=41, reference_tg=11.9
    )
    assert f["coherent"] is False and f["ecart_pct"] < -20


def test_validate_final_sans_reference_ne_juge_pas_la_coherence():
    f = validate_final(_SondeFinale(), ctx=32_768, depth=16_384, n_layers=41)
    assert f["coherent"] is None and "ecart_pct" not in f


def test_validate_final_echec_nomme_jamais_fatal():
    f = validate_final(_SondeFinale(boom=True), ctx=65_536, depth=16_384, n_layers=41)
    assert f["echec"].startswith("RuntimeError") and f["ctx"] == 65_536
