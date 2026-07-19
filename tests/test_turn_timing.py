# Décomposition CHIFFRÉE du tour (turn.timing) : prefill/génération mesurés par
# llama-server (extension `timings`), chargement déduit de l'attente avant 1er octet.
# Retour user 2026-07-19 : « plus jamais un 1min52 opaque ».
from loom.agent.client import _turn_timing_fields


def test_decomposition_complete():
    tim = {
        "cache_n": 6200,
        "prompt_n": 9775,
        "prompt_ms": 27300.0,
        "predicted_n": 443,
        "predicted_ms": 41000.0,
    }
    out = _turn_timing_fields(tim, first_byte_ms=114886.0)
    assert out["cache_tok"] == 6200
    assert out["prefill_s"] == 27.3 and out["prefill_tok"] == 9775
    assert out["prefill_tps"] == 358.1
    assert out["generation_s"] == 41.0 and out["generation_tok"] == 443
    assert out["generation_tps"] == 10.8
    # 114,9 s d'attente - 27,3 s de prefill = ~87,6 s de chargement/queue devant
    assert out["chargement_s"] == 87.6
    assert out["total_s"] == 68.3  # prefill + génération (le chargement s'ajoute)


def test_sans_first_byte_ni_division_par_zero():
    out = _turn_timing_fields({"prompt_n": 0, "prompt_ms": 0}, first_byte_ms=None)
    assert out["prefill_tps"] == 0.0 and out["generation_tps"] == 0.0
    assert "chargement_s" not in out
