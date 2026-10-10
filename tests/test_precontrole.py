# tests/test_precontrole.py
"""Précontrôle mémoire AVANT tout chargement (revue n°16, 2026-10-10).

Le relecteur : « Avant tout chargement, estimer la faisabilité des configurations
utilisées pour démarrer les sondes : placement, petit contexte, slots et mémoire
récurrente. Cela doit couvrir aussi llama-bench. […] Des métadonnées incomplètes
doivent rester signalées comme incertaines. Le test décisif serait : impossibilité
établie → aucun lancement de processus modèle, verdict explicite et archive. »

Ce fichier couvre les fonctions PURES (aucun processus) : complétude, bornes, verdicts.
"""

from __future__ import annotations

from loom.runtime.hardware import HardwareProfile
from loom.runtime.model_profile import ModelProfile
from loom.setup.placement import (
    AucunPlacementFaisable,
    DemarrageImpossible,
    demarrage_isolation,
    filtre_llama_bench,
    inconnues_decisives,
    plan_placements,
    precontrole,
    precontrole_texte,
)

MIB = 1024 * 1024


def _catalogue(
    n, *, attention, recurrentes=(), nextn=(), par=100 * MIB, sortie=50 * MIB
):
    return {
        "total": n * par + 2 * sortie,
        "familles": {"embeddings": sortie, "output": sortie},
        "par_couche": [par] * n,
        "experts_par_couche": [0] * n,
        "couches_attention": list(attention),
        "couches_recurrentes": list(recurrentes),
        "couches_nextn": list(nextn),
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }


def _meta_hybride(**over):
    """Forme RÉELLE d'un en-tête qwen35 (Bonsai 2) : sliding_window et expert_count
    ABSENTS (None), dimensions ssm présentes, catalogue complet."""
    meta = {
        "architecture": "qwen35",
        "n_layers": 4,
        "context_length": 262144,
        "expert_count": None,
        "expert_used_count": None,
        "head_count": 16,
        "head_count_kv": 4,
        "embedding_length": 2048,
        "key_length": 256,
        "value_length": 256,
        "sliding_window": None,
        "sliding_window_pattern": None,
        "full_attention_interval": 4,
        "recurrent": True,
        "ssm_conv_kernel": 4,
        "ssm_inner_size": 4096,
        "ssm_state_size": 128,
        "ssm_group_count": 16,
        "split_count": None,
        "key_length_mla": None,
        "kv_lora_rank": None,
        "shared_kv_layers": None,
        "key_length_swa": None,
        "value_length_swa": None,
        "head_count_kv_array": False,
        "arrays": {},
        "weights": _catalogue(4, attention=[3], recurrentes=[0, 1, 2]),
    }
    meta.update(over)
    return meta


def _inconnues(meta):
    return inconnues_decisives(ModelProfile.from_meta(meta), meta)


def test_forme_reelle_d_un_hybride_valide_est_complete():
    """Une clé optionnelle ABSENTE (sliding_window, expert_count) n'est pas une donnée
    manquante : Ornith et Bonsai ne doivent pas rester « incertains » à chaque bench."""
    assert _inconnues(_meta_hybride()) == []
    dense = _meta_hybride(
        architecture="llama",
        recurrent=False,
        full_attention_interval=None,
        ssm_conv_kernel=None,
        ssm_inner_size=None,
        ssm_state_size=None,
        ssm_group_count=None,
        weights=_catalogue(4, attention=[0, 1, 2, 3]),
    )
    assert _inconnues(dense) == []


def test_metadonnees_absentes_toutes_nommees():
    out = " ; ".join(_inconnues({}))
    assert "catalogue" in out and "KV" in out


def test_chaque_cas_incertain_est_nomme():
    cas = {
        "parties": _meta_hybride(split_count=2),
        "catalogue incomplet": _meta_hybride(
            weights=_catalogue(3, attention=[2], recurrentes=[0, 1])
        ),
        "MLA": _meta_hybride(key_length_mla=192),
        "partagé": _meta_hybride(shared_kv_layers=2),
        "SWA": _meta_hybride(key_length_swa=128),
        "tableau": _meta_hybride(head_count_kv_array=True),
        "fenêtre": _meta_hybride(sliding_window=4096),
        "architecture": _meta_hybride(architecture="mamba2"),
        "état récurrent": _meta_hybride(ssm_inner_size=None, ssm_state_size=None),
    }
    for attendu, meta in cas.items():
        out = " ; ".join(_inconnues(meta))
        assert attendu in out, (attendu, out)


# ── verdicts du précontrôle (lot L3) ─────────────────────────────────────────────


def _dense(n=40, par_mb=300, sortie_mb=300, emb_mb=300):
    """Dense de n couches, catalogue complet (forme réelle : clés optionnelles None)."""
    w = {
        "total": (n * par_mb + sortie_mb + emb_mb) * MIB,
        "familles": {"embeddings": emb_mb * MIB, "output": sortie_mb * MIB},
        "par_couche": [par_mb * MIB] * n,
        "experts_par_couche": [0] * n,
        "couches_attention": list(range(n)),
        "couches_recurrentes": [],
        "couches_nextn": [],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    return _meta_hybride(
        architecture="llama",
        n_layers=n,
        head_count_kv=8,
        key_length=128,
        value_length=128,
        recurrent=False,
        full_attention_interval=None,
        ssm_conv_kernel=None,
        ssm_inner_size=None,
        ssm_state_size=None,
        ssm_group_count=None,
        weights=w,
    )


def _moe(n=8, dense_mb=900, experts_mb=4000, sortie_mb=300, emb_mb=300):
    w = {
        "total": (n * (dense_mb + experts_mb) + sortie_mb + emb_mb) * MIB,
        "familles": {"embeddings": emb_mb * MIB, "output": sortie_mb * MIB},
        "par_couche": [(dense_mb + experts_mb) * MIB] * n,
        "experts_par_couche": [experts_mb * MIB] * n,
        "couches_attention": list(range(n)),
        "couches_recurrentes": [],
        "couches_nextn": [],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }
    return dict(_dense(n=n), expert_count=64, weights=w)


def _taille(meta):
    return int(meta["weights"]["total"] // MIB)


#: GPU discret NVIDIA (CUDA) : Loom impose --load-mode none, poids CPU résidents.
NVIDIA_6G = HardwareProfile(
    True,
    "RTX",
    6000,
    16,
    vram_total_mb=6144,
    backend="CUDA",
    vram_is_discrete=True,
    gpu_count=1,
)
NVIDIA_8G = HardwareProfile(
    True,
    "RTX",
    8000,
    16,
    vram_total_mb=8192,
    backend="CUDA",
    vram_is_discrete=True,
    gpu_count=1,
)
NVIDIA_24G = HardwareProfile(
    True,
    "RTX",
    24000,
    16,
    vram_total_mb=24576,
    backend="CUDA",
    vram_is_discrete=True,
    gpu_count=1,
)
#: Vulkan non NVIDIA : mémoire unifiée seulement PRÉSUMÉE par Loom.
VULKAN = HardwareProfile(
    True, "Radeon", 46350, 16, vram_total_mb=48789, backend="Vulkan", gpu_count=1
)
CPU_SEUL = HardwareProfile(False, None, 0, 16)


def _pc(meta, hw, *, ram, base_slots=1, ctx_checkpoints=None, vram=None, mmproj_mb=0):
    prof = ModelProfile.from_meta(
        meta, model_size_mb=_taille(meta) if meta.get("weights") else 0
    )
    return precontrole(
        prof,
        meta,
        model_size_mb=_taille(meta) if meta.get("weights") else 12_600,
        hw=hw,
        gpu_backend=hw.has_gpu,
        vram_total_mb=hw.vram_total_mb if vram is None else vram,
        ram_total_mb=ram,
        uma=hw.has_gpu and not hw.vram_is_discrete,
        headroom_mb=640,
        base_slots=base_slots,
        ctx_checkpoints=ctx_checkpoints,
        current=None,
        mmproj_mb=mmproj_mb,
    )


def test_impossibilite_etablie_par_la_somme_des_allocations_certaines():
    """12 600 Mo de poids résidents (--load-mode none) ne tiennent dans aucune
    répartition de 6 144 Mo de VRAM + 4 000 Mo de RAM : établi, pour tout placement."""
    res = _pc(_dense(), NVIDIA_6G, ram=4000)
    assert res["verdict"] == "impossible" and res["etabli"] is True
    assert res["complet"] is True and res["residence"]["certaine"] is True
    assert res["borne"]["total_mb"] == 12_600
    assert "mémoire physique" in res["raison"]
    assert "12600" in res["raison"] and "10144" in res["raison"]
    assert "tout placement" in res["raison"]


def test_metadonnees_incompletes_jamais_impossibles():
    meta = dict(_dense(), weights=None)
    res = _pc(meta, NVIDIA_6G, ram=4000)
    assert res["verdict"] == "incertain" and res["etabli"] is False
    assert any("catalogue" in i for i in res["inconnues"])


def test_memoire_unifiee_presumee_ne_rend_jamais_l_impossibilite_etablie():
    """Loom classe « unifiée » toute carte Vulkan sans CUDA : sur une carte discrète, le
    mmap reste actif et les poids CPU sont des pages de fichier. Pas d'« établi » ; le
    plan au plancher, vide, donne le verdict budgétaire."""
    res = _pc(_dense(), VULKAN, ram=4000)
    assert res["etabli"] is False and res["residence"]["certaine"] is False
    assert res["verdict"] == "hors_budget"
    assert "contexte plancher" in res["raison"] and "4096" in res["raison"]


def test_capacite_inconnue_ou_non_modelisee_rend_incertain():
    vram_inconnue = HardwareProfile(
        True, "RTX", 6000, 16, vram_is_discrete=True, gpu_count=0
    )
    res = _pc(_dense(), vram_inconnue, ram=4000)
    assert res["verdict"] == "incertain" and any("VRAM" in i for i in res["inconnues"])
    deux_gpu = HardwareProfile(
        True,
        "RTX",
        6000,
        16,
        vram_total_mb=6144,
        backend="CUDA",
        vram_is_discrete=True,
        gpu_count=2,
    )
    res = _pc(_dense(), deux_gpu, ram=4000)
    assert res["verdict"] == "incertain"
    assert any("plusieurs GPU" in i for i in res["inconnues"])


def test_hors_budget_au_plancher_sans_impossibilite_physique():
    """9 300 Mo tiennent physiquement dans 6 144 + 4 000 Mo, mais aucun placement ne tient
    dans les budgets (marges comprises) même à 4096 × 1 : l'étape 2 refuserait de toute
    façon (monotonie) — sortie avant tout chargement, sans prétendre à l'« établi »."""
    res = _pc(_dense(n=30, par_mb=290), NVIDIA_6G, ram=4000)
    assert res["verdict"] == "hors_budget" and res["etabli"] is False
    assert res["plan_plancher"]["candidats"] == []
    assert "contexte plancher" in res["raison"] and "1 slot" in res["raison"]
    assert "postes" in res["raison"]


def test_faisable_sur_une_machine_qui_porte_le_modele():
    res = _pc(_dense(), NVIDIA_24G, ram=64_000)
    assert res["verdict"] == "faisable" and res["etabli"] is False
    assert "gpu_total" in res["plan_plancher"]["candidats"]


def test_plancher_vide_implique_etape_2_vide():
    """Monotonie : contexte utile >= 4096 et slots >= base, mêmes postes — un plan vide au
    plancher l'est aussi à l'étape 2. C'est ce qui autorise la sortie « hors budget »."""
    from loom.setup.placement import memory_estimate_mb

    meta = _dense(n=30, par_mb=290)
    prof = ModelProfile.from_meta(meta, model_size_mb=_taille(meta))
    vus_vides = 0
    for ram in (4000, 8000, 12000, 16000):
        for hw in (NVIDIA_6G, NVIDIA_8G):
            res = _pc(meta, hw, ram=ram)
            if res["plan_plancher"]["candidats"]:
                continue
            vus_vides += 1
            est = memory_estimate_mb(prof, 32768, gpu_tuning=True, slots=2)
            plan2 = plan_placements(
                moe=False,
                n_layers=30,
                model_size_mb=_taille(meta),
                kv_mb=est["device_mb"],
                host_extra_mb=est["host_mb"],
                gpu_backend=True,
                vram_total_mb=hw.vram_total_mb,
                ram_total_mb=ram,
                uma=False,
                headroom_mb=640,
                profile=prof,
            )
            assert plan2.candidates == [], (ram, hw.vram_total_mb)
    assert vus_vides >= 2  # le cas est vraiment exercé


def test_texte_du_precontrole_par_verdict():
    assert "impossible" in precontrole_texte(_pc(_dense(), NVIDIA_6G, ram=4000))
    t = precontrole_texte(_pc(dict(_dense(), weights=None), NVIDIA_6G, ram=4000))
    assert "incertain" in t and "flux inchangé" in t
    assert "faisable" in precontrole_texte(_pc(_dense(), NVIDIA_24G, ram=64_000))


def test_exception_de_demarrage_impossible_porte_son_verdict():
    exc = DemarrageImpossible(
        "raison", etabli=False, details={"verdict": "hors_budget"}
    )
    assert str(exc) == "raison" and exc.etabli is False
    assert exc.details["verdict"] == "hors_budget"
    # Pas une sous-classe : la branche AucunPlacementFaisable du worker forcerait sinon
    # l'étape « placement » et un libellé qui ne lui correspond pas.
    assert not isinstance(exc, AucunPlacementFaisable)


def test_texte_de_l_etape_2_distingue_contexte_demande_et_demarrage():
    """Revue n°16 : « le contexte demandé ne tient pas » n'est pas « impossible de
    démarrer ». Le message d'étape 2 nomme le contexte, les slots, les postes et
    rappelle que le démarrage au plancher tenait (précontrôle)."""
    from loom.setup.placement import (
        PlacementPlan,
        memory_estimate_mb,
        texte_etape2,
    )

    meta = _dense(n=30, par_mb=290)
    prof = ModelProfile.from_meta(meta, model_size_mb=_taille(meta))
    est = memory_estimate_mb(prof, 32768, gpu_tuning=True, slots=2)
    plan = PlacementPlan(
        [], [{"key": "cpu", "raison": "non exploré : ne tient pas — x"}]
    )
    t = texte_etape2(plan, ctx=32768, slots=2, estimation=est, precontrole="faisable")
    assert t.startswith(
        "le contexte utile 32768 (2 slots) ne tient avec aucun placement"
    )
    assert "aucun placement faisable" in t  # raison du plan conservée
    assert f"KV {est['kv_mb']} Mo" in t and "postes à ce contexte" in t
    assert "démarrage au plancher" in t and "faisable" in t
    sans = texte_etape2(plan, ctx=8192, slots=1, estimation=est, precontrole=None)
    assert "(1 slot)" in sans and "démarrage au plancher" not in sans


# ── démarrage de la sonde d'isolation (lot L3) ───────────────────────────────────


def _iso(meta, hw, flags, *, ram, complet=True):
    prof = ModelProfile.from_meta(meta, model_size_mb=_taille(meta))
    return demarrage_isolation(
        prof,
        meta,
        flags=flags,
        complet=complet,
        model_size_mb=_taille(meta),
        gpu_backend=hw.has_gpu,
        vram_total_mb=hw.vram_total_mb,
        ram_total_mb=ram,
        uma=hw.has_gpu and not hw.vram_is_discrete,
        headroom_mb=640,
        gpu_tuning=hw.has_gpu,
    )


PREVU_GPU = {"ngl": 999, "cpu_moe": False, "n_cpu_moe": None}


def test_isolation_toujours_a_un_slot_sur_le_demarrage_prevu_s_il_tient():
    d = _iso(_dense(), NVIDIA_24G, PREVU_GPU, ram=64_000)
    assert d["lancer"] is True and d["modeste"] is False
    assert d["slots"] == 1 and d["ctx"] == 4096 and d["flags"] == PREVU_GPU


def test_isolation_sur_un_demarrage_plus_modeste_si_le_prevu_ne_tient_pas():
    d = _iso(_dense(), NVIDIA_8G, PREVU_GPU, ram=32_000)
    assert d["lancer"] is True and d["modeste"] is True
    assert 0 < d["flags"]["ngl"] < 40  # un offload partiel qui tient
    assert "ne tient pas" in d["raison"]


def test_isolation_non_lancee_pour_une_memoire_recurrente_si_le_prevu_ne_tient_pas():
    meta = dict(_dense(), recurrent=True)
    d = _iso(meta, NVIDIA_8G, PREVU_GPU, ram=32_000)
    assert d["lancer"] is False and "imposé" in d["raison"]


def test_isolation_metadonnees_incompletes_demarrage_prevu_inchange():
    d = _iso(_dense(), NVIDIA_8G, PREVU_GPU, ram=32_000, complet=False)
    assert d["lancer"] is True and d["modeste"] is False and d["flags"] == PREVU_GPU


def test_flags_bruts_ngl_0_cpu_moe_rien_sur_le_device():
    """Critique de conception : Placement.from_flags fait passer --cpu-moe avant -ngl 0
    (experts_cpu en ngl 999). Les flags bruts « -ngl 0 --cpu-moe » ne mettent RIEN sur
    le device : sur une machine sans GPU, le démarrage tient en RAM."""
    flags = {"ngl": 0, "cpu_moe": True, "n_cpu_moe": None}
    d = _iso(_moe(), CPU_SEUL, flags, ram=64_000)
    assert d["lancer"] is True and d["modeste"] is False and d["flags"] == flags


# ── filtre des -ngl de llama-bench (lot L3) ──────────────────────────────────────


def _bench(meta, hw, ngl, ncmoe=0, *, complet=True, meme_binaire=True):
    prof = ModelProfile.from_meta(meta, model_size_mb=_taille(meta))
    return filtre_llama_bench(
        prof, ngl=ngl, ncmoe=ncmoe, complet=complet, hw=hw, meme_binaire=meme_binaire
    )


def test_llama_bench_retire_les_ngl_impossibles_sur_le_device():
    f = _bench(_dense(), NVIDIA_8G, [0, 22, 99])
    assert f["ngl"] == [0, 22]
    assert [r["ngl"] for r in f["retires"]] == [99]
    assert "VRAM" in f["retires"][0]["raison"]


def test_llama_bench_inchange_si_incomplet_ou_autre_binaire():
    assert _bench(_dense(), NVIDIA_8G, [0, 22, 99], complet=False)["ngl"] == [0, 22, 99]
    assert _bench(_dense(), NVIDIA_8G, [0, 22, 99], meme_binaire=False)["ngl"] == [
        0,
        22,
        99,
    ]


def test_llama_bench_repli_sur_ngl_0_si_tout_est_impossible():
    # MoE : les denses seuls (8 x 900 Mo + sortie) dépassent 6 144 Mo de VRAM.
    f = _bench(_moe(), NVIDIA_6G, [999], ncmoe=8)
    assert f["ngl"] == [0] and f["ncmoe"] == 0 and "repli" in f["note"]
