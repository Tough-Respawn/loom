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

from loom.runtime.model_profile import ModelProfile
from loom.setup.placement import inconnues_decisives

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
