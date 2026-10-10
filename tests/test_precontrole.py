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
    Placement,
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
    # Revue adverse : le fichier d'échange (Windows) et le repli de la mémoire épinglée
    # CUDA pourraient charger au-delà de la RAM — l'« établi » vaut SANS pagination.
    assert "sans pagination" in res["raison"]


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


def test_capacite_inconnue_empeche_seulement_l_etabli():
    """Revue adverse : une capacité inconnue (VRAM lue par nvidia-smi seulement, plusieurs
    GPU listés) ne rend pas les MÉTADONNÉES incomplètes. Elle interdit l'« établi » ; le
    plan au plancher, sur la même VRAM que l'étape 2 (repli compris), décide du reste —
    sinon le démarrage modeste et le filtre llama-bench sautaient à tort."""
    vram_inconnue = HardwareProfile(
        True, "RTX", 6000, 16, vram_is_discrete=True, gpu_count=0
    )
    res = _pc(_dense(), vram_inconnue, ram=4000, vram=6144)
    assert res["complet"] is True and res["capacite_connue"] is False
    assert res["etabli"] is False and res["verdict"] == "hors_budget"
    assert any("VRAM" in i for i in res["inconnues"])
    assert "capacité" in res["raison"]
    assert _pc(_dense(), vram_inconnue, ram=64_000, vram=24_576)["verdict"] == (
        "faisable"
    )
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
    assert res["etabli"] is False and res["verdict"] == "hors_budget"
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
    # La raison du « non établi » est la VRAIE : la somme certaine tient physiquement
    # (pas « GPU discret : poids résidents », qui rendrait l'établi possible).
    assert "allocations certaines 9300 Mo ≤ mémoire physique 10144 Mo" in res["raison"]


def test_slots_de_base_decisifs_au_plancher():
    """Revue adverse (mutation) : forcer base_slots à 1 laissait la suite verte. Dense de
    7 800 Mo, 6 Go + 6 000 Mo : 1 slot tient au plancher, 2 slots ([server] n_parallel)
    non."""
    meta = _dense(n=48, par_mb=150)
    un = _pc(meta, NVIDIA_6G, ram=6000, base_slots=1)
    assert un["verdict"] == "faisable" and un["plan_plancher"]["candidats"] == [
        "gpu_partiel_ngl33"
    ]
    deux = _pc(meta, NVIDIA_6G, ram=6000, base_slots=2)
    assert deux["verdict"] == "hors_budget" and "2 slots" in deux["raison"]


def _hybride(n=64, par_mb=100):
    """qwen35 : 1 couche d'attention sur 4, état récurrent de Bonsai 2 (149,6 Mio)."""
    att = [i for i in range(n) if (i + 1) % 4 == 0]
    rec = [i for i in range(n) if (i + 1) % 4 != 0]
    w = {
        "total": (n * par_mb + 600) * MIB,
        "familles": {"embeddings": 300 * MIB, "output": 300 * MIB},
        "par_couche": [par_mb * MIB] * n,
        "experts_par_couche": [0] * n,
        "couches_attention": att,
        "couches_recurrentes": rec,
        "couches_nextn": [],
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
        "sortie_liee": False,
    }
    return _meta_hybride(
        n_layers=n,
        ssm_conv_kernel=4,
        ssm_inner_size=6144,
        ssm_state_size=128,
        ssm_group_count=16,
        weights=w,
    )


def test_etat_recurrent_et_checkpoints_decisifs_au_plancher():
    """Mutations sans effet avant : état récurrent forcé à 0, checkpoints forcés à 0.
    Hybride, 8 Go + 8 000 Mo : 32 checkpoints par slot (4 788 Mo) ne tiennent pas, 6 si —
    et le conseil nomme le poste qui domine."""
    meta = _hybride()
    defaut = _pc(meta, NVIDIA_8G, ram=8000)
    assert defaut["borne"]["etat_mb"] == 149
    assert defaut["verdict"] == "hors_budget"
    assert defaut["plan_plancher"]["postes"]["checkpoints_mb"] == 4788
    assert "les checkpoints dominent" in defaut["raison"]
    six = _pc(meta, NVIDIA_8G, ram=8000, ctx_checkpoints=6)
    assert six["verdict"] == "faisable" and six["plan_plancher"]["candidats"] == [
        "gpu_total"
    ]


def test_mmproj_decisif_pour_l_etabli():
    """Mutation sans effet avant : mmproj forcé à 0. Dense de 10 200 Mo sur 6 Go +
    4 200 Mo : sans mmproj, hors budget ; avec 900 Mo de mmproj (allocation hôte
    certaine), au-delà de la mémoire physique."""
    meta = _dense(n=40, par_mb=240)
    assert _pc(meta, NVIDIA_6G, ram=4200)["verdict"] == "hors_budget"
    res = _pc(meta, NVIDIA_6G, ram=4200, mmproj_mb=900)
    assert res["verdict"] == "impossible" and res["borne"]["mmproj_mb"] == 900
    assert res["borne"]["total_mb"] == 11_100


def test_mmproj_absent_ou_rejete_bloquant_type_inconnu_incertain(tmp_path):
    """Le mmproj que la sonde passera (--mmproj) : absent ou en-tête rejeté, llama-server
    échoue au chargement (load_model renvoie false quand mtmd_init_from_file échoue) —
    bloquant ; type de valeur inconnu du lecteur : taille inconnue, pas un échec."""
    from loom.setup.placement import lire_mmproj
    from tests.test_gguf_profile import _gguf

    absent = lire_mmproj(tmp_path / "mmproj.gguf")
    assert absent["bloquant"] and "absent" in absent["bloquant"]
    (tmp_path / "html.gguf").write_bytes(b"<html>404</html>")
    rejete = lire_mmproj(tmp_path / "html.gguf")
    assert rejete["bloquant"] and "pas un fichier GGUF" in rejete["bloquant"]
    bon = _gguf(
        tmp_path / "bon.gguf",
        {"general.architecture": "clip"},
        [("v.blk.0.attn_k.weight", 2 * MIB)],
    )
    assert lire_mmproj(bon) == {"mb": 2, "bloquant": None}
    import struct

    inconnu = tmp_path / "inconnu.gguf"
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1)
    blob += struct.pack("<Q", 3) + b"cle" + struct.pack("<I", 99)  # type 99 inconnu
    inconnu.write_bytes(blob)
    assert lire_mmproj(inconnu) == {"mb": None, "bloquant": None}


def test_mmproj_de_taille_inconnue_rend_incertain():
    res = _pc(_dense(), NVIDIA_24G, ram=64_000, mmproj_mb=None)
    assert res["verdict"] == "incertain" and any(
        "mmproj" in i for i in res["inconnues"]
    )


def test_faisable_sur_une_machine_qui_porte_le_modele():
    res = _pc(_dense(), NVIDIA_24G, ram=64_000)
    assert res["verdict"] == "faisable" and res["etabli"] is False
    assert "gpu_total" in res["plan_plancher"]["candidats"]


def test_plancher_vide_implique_etape_2_vide():
    """Monotonie : contexte utile >= 4096 et slots >= base, mêmes postes — un plan vide au
    plancher l'est aussi à l'étape 2. C'est ce qui autorise la sortie « hors budget »."""
    from loom.setup.placement import memory_estimate_mb

    vus_vides = 0
    for meta in (_dense(n=30, par_mb=290), _hybride()):
        prof = ModelProfile.from_meta(meta, model_size_mb=_taille(meta))
        n = meta["n_layers"]
        for ram in (4000, 8000, 12000, 16000):
            for hw in (NVIDIA_6G, NVIDIA_8G):
                res = _pc(meta, hw, ram=ram)
                if res["plan_plancher"]["candidats"]:
                    continue
                vus_vides += 1
                est = memory_estimate_mb(prof, 32768, gpu_tuning=True, slots=2)
                plan2 = plan_placements(
                    moe=False,
                    n_layers=n,
                    model_size_mb=_taille(meta),
                    kv_mb=est["device_mb"],
                    host_extra_mb=est["host_mb"],
                    gpu_backend=True,
                    vram_total_mb=hw.vram_total_mb,
                    ram_total_mb=ram,
                    uma=False,
                    headroom_mb=640,
                    current=Placement("gpu_total", 999, actuel=True),
                    profile=prof,
                )
                assert plan2.candidates == [], (n, ram, hw.vram_total_mb)
    assert vus_vides >= 4  # le cas est vraiment exercé, dense ET hybride


def test_texte_du_precontrole_par_verdict():
    assert "impossible" in precontrole_texte(_pc(_dense(), NVIDIA_6G, ram=4000))
    t = precontrole_texte(_pc(dict(_dense(), weights=None), NVIDIA_6G, ram=4000))
    assert t.startswith("précontrôle incertain : ") and "flux inchangé" in t
    assert "précontrôle : précontrôle" not in t  # pas de double préfixe
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
    faisable_1 = {"verdict": "faisable", "plan_plancher": {"slots": 1}}
    t = texte_etape2(plan, ctx=32768, slots=2, estimation=est, precontrole=faisable_1)
    assert t.startswith(
        "le contexte utile 32768 (2 slots) ne tient avec aucun placement"
    )
    assert "aucun placement faisable" in t  # raison du plan conservée
    assert f"KV {est['kv_mb']} Mo" in t and "postes à ce contexte" in t
    assert "le démarrage au plancher (4096 par slot, 1 slot) passait" in t
    # Leviers RÉELS : contexte au-dessus du plancher, slots retenus > slots du plancher.
    assert "un context plus bas" in t and "moins de slots" in t
    sans = texte_etape2(plan, ctx=8192, slots=1, estimation=est, precontrole=None)
    assert "(1 slot)" in sans and "démarrage au plancher" not in sans
    # Revue adverse : « incertain » n'établit RIEN au plancher — jamais « passait ».
    inc = texte_etape2(
        plan,
        ctx=8192,
        slots=1,
        estimation=est,
        precontrole={"verdict": "incertain", "plan_plancher": {"slots": 1}},
    )
    assert "passait" not in inc and "non établie" in inc
    # Au plancher déjà (4096), baisser le context ne peut rien.
    plancher = texte_etape2(
        plan, ctx=4096, slots=2, estimation=est, precontrole=faisable_1
    )
    assert "un context plus bas" not in plancher and "moins de slots" in plancher


# ── démarrage de la sonde d'isolation (lot L3) ───────────────────────────────────


def _iso(meta, hw, flags, *, ram, complet=True, ctx_checkpoints=None):
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
        ctx_checkpoints=ctx_checkpoints,
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
    assert d["prevu_tient"] is None  # inconnu : rien n'est conclu


def test_isolation_dit_si_le_demarrage_prevu_tient():
    """`prevu_tient` : False quand le démarrage prévu est refusé par l'estimation (données
    complètes) — la suite ne doit jamais y revenir en repli (revue adverse)."""
    assert _iso(_dense(), NVIDIA_24G, PREVU_GPU, ram=64_000)["prevu_tient"] is True
    assert _iso(_dense(), NVIDIA_8G, PREVU_GPU, ram=32_000)["prevu_tient"] is False


def test_isolation_non_lancee_si_rien_ne_tient_a_4096():
    d = _iso(_dense(), NVIDIA_6G, PREVU_GPU, ram=4000)
    assert d["lancer"] is False and d["prevu_tient"] is False
    assert "aucun démarrage plus modeste" in d["raison"]


def test_isolation_checkpoints_bornes_par_le_ctx_checkpoints_du_modele():
    """La sonde crée au plus 6 checkpoints, jamais plus que le `ctx_checkpoints` passé au
    serveur (il évince au-delà). Hybride sur 8 Go + 4 000 Mo : 6 x 150 Mo de checkpoints
    ne tiennent pas côté hôte (928 Mo), 0 si — compter 6 refusait à tort un démarrage
    qui tient, et `prevu_tient` décide désormais d'une sortie."""
    meta = _hybride()
    assert _iso(meta, NVIDIA_8G, PREVU_GPU, ram=4000)["prevu_tient"] is False
    d = _iso(meta, NVIDIA_8G, PREVU_GPU, ram=4000, ctx_checkpoints=0)
    assert d["prevu_tient"] is True and d["lancer"] is True


def _repli(meta, hw, flags, *, ram, slots, complet=True, ctx_checkpoints=None):
    from loom.setup.placement import repli_calibration

    prof = ModelProfile.from_meta(meta, model_size_mb=_taille(meta))
    return repli_calibration(
        prof,
        meta,
        flags=flags,
        complet=complet,
        slots=slots,
        ctx=8192,
        ctx_checkpoints=ctx_checkpoints,
        model_size_mb=_taille(meta),
        gpu_backend=hw.has_gpu,
        vram_total_mb=hw.vram_total_mb,
        ram_total_mb=ram,
        uma=hw.has_gpu and not hw.vram_is_discrete,
        headroom_mb=640,
        gpu_tuning=hw.has_gpu,
    )


def test_repli_de_la_calibration_juge_a_son_premier_chargement():
    """Revue adverse : la garde du repli regardait 4096 x 1 (la sonde d'isolation),
    alors que la calibration charge d'abord à 8192 x slots retenus. 40 x 168 Mo + 280 +
    280 sur 8 Go : tout GPU tient à 4096 (7 340 Mo) mais pas à 8192 (7 680 pour 7 552)
    — repli condamné. 40 x 130 Mo tient à 8192 (6 160) : même refusé par l'étape 2 à un
    contexte utile plus grand, la calibration trouvera un contexte qui tient."""
    lourd = _dense(n=40, par_mb=168, sortie_mb=280, emb_mb=280)
    r = _repli(lourd, NVIDIA_8G, PREVU_GPU, ram=32_000, slots=1)
    assert r["tient"] is False and r["ctx"] == 8192 and r["slots"] == 1
    assert "7680 Mo device" in r["raison"]
    assert _iso(lourd, NVIDIA_8G, PREVU_GPU, ram=32_000)["prevu_tient"] is True
    leger = _dense(n=40, par_mb=130, sortie_mb=280, emb_mb=280)
    assert _repli(leger, NVIDIA_8G, PREVU_GPU, ram=32_000, slots=1)["tient"] is True
    inconnu = _repli(lourd, NVIDIA_8G, PREVU_GPU, ram=32_000, slots=1, complet=False)
    assert inconnu["tient"] is None


def test_flags_bruts_ngl_0_cpu_moe_rien_sur_le_device():
    """Critique de conception : Placement.from_flags fait passer --cpu-moe avant -ngl 0
    (experts_cpu en ngl 999). Les flags bruts « -ngl 0 --cpu-moe » ne mettent RIEN sur
    le device : sur une machine sans GPU, le démarrage tient en RAM."""
    flags = {"ngl": 0, "cpu_moe": True, "n_cpu_moe": None}
    d = _iso(_moe(), CPU_SEUL, flags, ram=64_000)
    assert d["lancer"] is True and d["modeste"] is False and d["flags"] == flags


# ── filtre des -ngl de llama-bench (lot L3) ──────────────────────────────────────


def _bench(meta, hw, ngl, ncmoe=0, *, complet=True, meme_binaire=True, vram=None):
    prof = ModelProfile.from_meta(meta, model_size_mb=_taille(meta))
    return filtre_llama_bench(
        prof,
        ngl=ngl,
        ncmoe=ncmoe,
        complet=complet,
        hw=hw,
        vram_total_mb=hw.vram_total_mb if vram is None else vram,
        ram_total_mb=64_000,
        uma=hw.has_gpu and not hw.vram_is_discrete,
        headroom_mb=640,
        meme_binaire=meme_binaire,
    )


def test_llama_bench_retire_les_ngl_impossibles_sur_le_device():
    f = _bench(_dense(), NVIDIA_8G, [0, 22, 99])
    assert f["ngl"] == [0, 22]
    assert [r["ngl"] for r in f["retires"]] == [99]
    assert "VRAM" in f["retires"][0]["raison"]
    assert "budget device" in f["retires"][0]["raison"]


def test_llama_bench_juge_contre_le_meme_budget_que_le_precontrole():
    """Revue adverse : juger contre la VRAM BRUTE gardait un -ngl 999 -ncmoe 48 de
    7 788 Mo sur 8 192 Mo (budget 7 552 : 640 Mo de marge pour calcul et pilote), le
    placement même que le précontrôle refusait — et un seul échec perdait tout le bench."""
    moe = _moe(n=48, dense_mb=155, experts_mb=400)
    f = _bench(moe, NVIDIA_8G, [999], ncmoe=48)
    assert f["ngl"] == [0] and f["ncmoe"] == 0 and "repli" in f["note"]


def test_llama_bench_filtre_aussi_sur_la_vram_de_repli():
    """Revue adverse de L7 : VRAM lue par nvidia-smi seulement (profil de repli, total
    0) — le filtre sautait encore, et llama-bench chargeait -ngl 999 -ncmoe 48 (7 788 Mo
    pour un budget de 7 552), le placement que le précontrôle et la sonde, sur la MÊME
    VRAM, refusaient. Sans aucune VRAM connue : liste inchangée, et dit."""
    repli = HardwareProfile(True, "RTX 8G", 7900, 16, vram_is_discrete=True)
    moe = _moe(n=48, dense_mb=155, experts_mb=400)
    f = _bench(moe, repli, [999], ncmoe=48, vram=8192)
    assert f["ngl"] == [0] and f["ncmoe"] == 0 and "repli" in f["note"]
    f = _bench(moe, repli, [999], ncmoe=48, vram=0)
    assert f["ngl"] == [999] and "inconnue" in f["note"]


def test_texte_faisable_dit_une_capacite_non_etablie():
    """Revue adverse de L7 : depuis que la capacité inconnue ne rend plus « incertain »,
    la console affichait « [ok] faisable » sans dire que le plan repose sur une
    capacité que Loom ne connaît pas."""
    vram_inconnue = HardwareProfile(
        True, "RTX", 6000, 16, vram_is_discrete=True, gpu_count=0
    )
    t = precontrole_texte(_pc(_dense(), vram_inconnue, ram=64_000, vram=24_576))
    assert t.startswith("précontrôle : faisable")
    assert "capacité physique non établie" in t and "VRAM" in t
    assert "capacité" not in precontrole_texte(_pc(_dense(), NVIDIA_24G, ram=64_000))


def test_llama_bench_moe_n_cpu_moe_pris_en_compte():
    """Mutation sans effet avant (gpu_bytes sans n_cpu_moe). MoE de 8 couches sur 8 Go :
    tous les experts en RAM (-ncmoe 8) tiennent ; deux couches d'experts sur le device
    (-ncmoe 6), non."""
    f = _bench(_moe(), NVIDIA_8G, [999], ncmoe=8)
    assert f["ngl"] == [999] and f["ncmoe"] == 8 and f["retires"] == []
    f = _bench(_moe(), NVIDIA_8G, [999], ncmoe=6)
    assert f["ngl"] == [0] and [r["ngl"] for r in f["retires"]] == [999]


def test_llama_bench_note_quand_seul_ngl_0_reste():
    """Revue adverse : « [attention] llama-bench : . » (note vide) quand 0 restait seul."""
    petit = HardwareProfile(
        True,
        "RTX 2G",
        2000,
        16,
        vram_total_mb=2048,
        backend="CUDA",
        vram_is_discrete=True,
        gpu_count=1,
    )
    f = _bench(_dense(), petit, [0, 22, 99])
    assert f["ngl"] == [0] and f["note"]


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
