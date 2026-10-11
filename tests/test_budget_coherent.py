# tests/test_budget_coherent.py
"""Budget mémoire et contexte VISÉ cohérents (lot 2 du bench de placement).

Le filtre de faisabilité estimait le KV en f16 pour 65 536 tokens pendant que la
comparaison tournait à 8 192 : il pouvait éliminer un placement avant sa mesure. En
mémoire unifiée, le budget comptait la VRAM Vulkan (48 Go) sans la borner par la RAM.
Et la calibration renvoyait 4 096 sans dire que rien n'avait été VALIDÉ en vitesse.
"""

from __future__ import annotations

from loom.runtime.model_profile import ModelProfile
from loom.setup.placement import kv_estimate_mb, useful_context
from loom.setup.topology import TOPO_MOE_HYBRIDE, TOPO_RAM, calibrate, memory_budget_mb
from tests.test_topology import GOLDEN_BUDGET, GOLDEN_META, FakeProbe


def test_budget_uma_compte_la_ram_une_fois():
    # Radeon 860M : Vulkan annonce 48 789 Mo, la machine a 64 Go -> la VRAM borne.
    assert memory_budget_mb(TOPO_MOE_HYBRIDE, 48_789, 64_000, 640, uma=True) == 48_149
    # Même iGPU sur 32 Go : la RAM moins la marge OS borne, pas l'annonce Vulkan.
    assert memory_budget_mb(TOPO_MOE_HYBRIDE, 48_789, 32_000, 640, uma=True) == (
        32_000 - 3_072 - 640
    )
    # GPU discret : inchangé ; CPU seul : inchangé.
    assert memory_budget_mb(TOPO_MOE_HYBRIDE, 6_144, 64_000, 640) == 5_504
    assert memory_budget_mb(TOPO_RAM, 0, 64_000, 640, uma=True) == 64_000 - 3_072


def test_contexte_utile_modele_puis_machine_puis_plancher():
    assert useful_context(32_768, 8_192, 262_144) == 32_768
    assert useful_context(None, 16_384, 262_144) == 16_384
    assert useful_context(None, None, 262_144) == 8_192
    # Borné par la limite du modèle, jamais sous 4 096.
    assert useful_context(65_536, 8_192, 16_384) == 16_384
    assert useful_context(1_024, None, None) == 4_096


def test_kv_estime_au_contexte_utile_avec_le_type_de_cache_de_l_executant():
    meta = {
        "architecture": "qwen35moe",
        "n_layers": 40,
        "head_count_kv": 2,
        "key_length": 256,
        "value_length": 256,
        "full_attention_interval": 4,
        "recurrent": True,
    }
    prof = ModelProfile.from_meta(meta)
    # 10 couches d'attention x 32 768 tokens x 2 têtes x 512 x 1,0625 o = 340 Mio.
    assert kv_estimate_mb(prof, 32_768, gpu_tuning=True, slots=1) == 340
    # Sans profil GPU l'exécutant garde le KV f16 (64 Mio/couche) ; 2 slots doublent.
    assert kv_estimate_mb(prof, 32_768, gpu_tuning=False, slots=2) == 1_280


def test_calibrate_distingue_contexte_valide_et_repli():
    ok = calibrate(
        FakeProbe(), GOLDEN_META, topology=TOPO_MOE_HYBRIDE, budget_mb=GOLDEN_BUDGET
    )
    assert ok["valide"] is True
    repli = calibrate(
        FakeProbe(),
        GOLDEN_META,
        topology=TOPO_MOE_HYBRIDE,
        budget_mb=GOLDEN_BUDGET,
        time_budget_s=0,
    )
    assert repli["context"] == 4096 and repli["valide"] is False
    assert "NON validé" in repli["mecanisme"]
