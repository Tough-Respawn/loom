"""llama-swap tue un serveur qui n'a pas répondu à /health en 120 s (défaut) puis le
relance. Vécu 2026-10-02 sur la 2060 : ornith 1.5 Q8 (36 Go lus en RAM avec 37 Go libres,
PC chargé) a dépassé 2 min au premier chargement, a été tué, relancé (86 s) : 138 s
perdues et une requête en 500. Loom pose un délai explicite et large."""

from __future__ import annotations

from loom.runtime import swap


def test_delai_de_sante_explicite_et_large(monkeypatch):
    cfg = swap.build_swap_config(
        [], profile=None, llama_bin="x", models_dir="d", context=1
    )
    assert cfg["healthCheckTimeout"] == swap.SWAP_HEALTH_TIMEOUT_S >= 600
    assert (
        list(cfg)[0] == "healthCheckTimeout"
    )  # clé globale, avant la liste des modèles
    assert "healthCheckTimeout:" in swap.dump_yaml(cfg)
