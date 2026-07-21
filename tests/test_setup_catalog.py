# Installeur loom-setup : estimation de taille depuis le nom de repo et filtrage
# des résultats de recherche par le budget machine — SANS réseau.
import pytest

from loom.runtime.hf_catalog import HfCatalogError
from loom.setup.catalog import (
    budget_mb,
    estimate_min_mb,
    filter_by_budget,
    fitting_entries,
    resolve_entry,
)


def test_estimate_min_mb_noms_reels():
    # Cas réels observés dans une recherche « qwen q4 » (450 Mo/milliard).
    assert estimate_min_mb("meshllm/Qwen3.5-397B-A17B-UD-Q4_K_XL-layers") == 178_650
    assert estimate_min_mb("Joshua65535/qwen2.5-1.5b-instruct-q4_k_m.gguf") == 675
    assert estimate_min_mb("agentscope-ai/QwenPaw-Flash-9B-Q4_K_M") == 4_050
    # MoE : l'empreinte suit les paramètres TOTAUX (80), pas les actifs (3)
    assert estimate_min_mb("meshllm/Qwen3-Next-80B-A3B-Thinking") == 36_000
    # « Q4 »/« Q4_K_M » ne sont PAS des tailles ; sans taille -> None (pas masqué)
    assert estimate_min_mb("org/mon-modele-GGUF") is None


def test_filter_by_budget():
    hits = [
        {"repo_id": "big/Qwen3.5-397B-A17B", "downloads": 1},
        {"repo_id": "ok/qwen2.5-1.5b-instruct", "downloads": 2},
        {"repo_id": "inconnu/mystere-GGUF", "downloads": 3},
    ]
    kept, hidden = filter_by_budget(hits, budget=1_600)
    assert hidden == 1
    assert [h["repo_id"] for h in kept] == [
        "ok/qwen2.5-1.5b-instruct",
        "inconnu/mystere-GGUF",  # taille inconnue -> gardé, dans le doute
    ]
    assert kept[0]["est_mb"] == 675 and kept[1]["est_mb"] is None
    # l'entrée d'origine n'est pas mutée
    assert "est_mb" not in hits[1]


def test_resolve_entry_repo_epingle_gagne():
    entry = {"repo": "org/fixe-GGUF", "query": "jamais utilisé"}

    def boom(q):
        raise AssertionError("repo épinglé -> pas de recherche")

    assert resolve_entry(entry, boom, budget=1) == "org/fixe-GGUF"


def test_resolve_entry_query_resolue_et_filtree():
    entry = {"query": "qwen3 4b instruct gguf"}
    hits = [
        {"repo_id": "big/Qwen3-400B", "downloads": 99},  # trop gros -> filtré
        {"repo_id": "ok/Qwen3-4B-Instruct-GGUF", "downloads": 5},
    ]
    assert resolve_entry(entry, lambda q: hits, budget=3_000) == (
        "ok/Qwen3-4B-Instruct-GGUF"
    )
    # rien de jouable -> None
    assert resolve_entry(entry, lambda q: hits[:1], budget=3_000) is None

    # erreur réseau/HF -> PROPAGÉE (le CLI montre le message, diagnostic proxy
    # inclus) au lieu d'un None trompeur « famille disparue ? »
    def offline(q):
        raise HfCatalogError("hors-ligne")

    with pytest.raises(HfCatalogError):
        resolve_entry(entry, offline, budget=3_000)


def test_budget_et_catalogue():
    assert budget_mb(6000, 24000) == 25_904
    assert budget_mb(0, 2000) == 0  # jamais négatif
    # plus gourmand d'abord ; rien ne tient -> liste vide
    cat = [{"min_budget_mb": 4000}, {"min_budget_mb": 18000}]
    assert fitting_entries(25_904, cat) == [
        {"min_budget_mb": 18000},
        {"min_budget_mb": 4000},
    ]
    assert fitting_entries(1_600, cat) == []


def test_parse_hf_repo():
    from loom.setup.catalog import parse_hf_repo

    # URL complète, avec ou sans chemin/slash final
    assert (
        parse_hf_repo("https://huggingface.co/llmfan46/Ornith-1.0-35B-GGUF")
        == "llmfan46/Ornith-1.0-35B-GGUF"
    )
    assert (
        parse_hf_repo("https://huggingface.co/org/repo-GGUF/tree/main")
        == "org/repo-GGUF"
    )
    assert parse_hf_repo("hf.co/org/repo/") == "org/repo"
    # id collé tel quel
    assert (
        parse_hf_repo("deepreinforce-ai/Ornith-1.0-35B")
        == "deepreinforce-ai/Ornith-1.0-35B"
    )
    # recherche libre : pas un repo
    assert parse_hf_repo("qwen 4b instruct") is None
    assert parse_hf_repo("ornith") is None
    assert parse_hf_repo("") is None
