# loom/setup/catalog.py
"""Recommandation d'un premier modèle qui FIT la machine.

Petit catalogue curaté (modèles du parc éprouvés sur l'offload MoE) filtré par
le budget mémoire (VRAM libre + RAM − marge, même heuristique que
recommend_quant). Le quant EXACT est ensuite choisi sur les tailles réelles du
repo (list_gguf_files + recommend_quant). Un repo injoignable est simplement
sauté : le catalogue dégrade proprement, offline compris."""

from __future__ import annotations

import re

from loom.runtime.hf_catalog import list_gguf_files

# Shortlist par PALIER de budget, pensée pour qui ne sait pas quoi chercher :
# des libellés lisibles, du plus gourmand au plus léger — chaque machine a
# toujours au moins une proposition. `query` (pas de repo figé) : le repo réel
# est résolu EN LIVE sur HF au moment du choix (top téléchargements qui fit),
# donc la liste ne périme pas quand un repo est renommé/retiré. `min_budget_mb`
# ≈ plus petit quant utilisable + marge : en dessous, l'entrée n'apparaît pas.
CATALOG: list[dict] = [
    {
        "query": "qwen3.6 35b a3b instruct gguf",
        "label": "Qwen3.6 35B A3B (MoE + vision) — machines larges, le parc",
        "min_budget_mb": 18_000,
    },
    {
        "query": "gemma4 26b a4b gguf",
        "label": "Gemma4 26B A4B (MoE) — polyvalent",
        "min_budget_mb": 14_000,
    },
    {
        "query": "qwen3 8b instruct gguf",
        "label": "~8B instruct — bon équilibre qualité/mémoire",
        "min_budget_mb": 5_500,
    },
    {
        "query": "qwen3 4b instruct gguf",
        "label": "~4B instruct — machines modestes, tool-use correct",
        "min_budget_mb": 2_600,
    },
    {
        "query": "qwen2.5 1.5b instruct gguf",
        "label": "~1.5B instruct — très léger (dépannage/découverte)",
        "min_budget_mb": 900,
    },
]


def budget_mb(vram_free_mb: int, ram_mb: int, margin_mb: int = 4096) -> int:
    """Budget mémoire pour un modèle : VRAM libre + RAM − marge système (les
    MoE tournent experts en RAM, cf. recommend_quant)."""
    return max(0, vram_free_mb + ram_mb - margin_mb)


def fitting_entries(budget: int, catalog: list[dict] | None = None) -> list[dict]:
    """Entrées du catalogue qui tiennent dans `budget`, plus gourmandes d'abord
    (la 1re = la recommandation)."""
    cat = CATALOG if catalog is None else catalog
    fit = [e for e in cat if e["min_budget_mb"] <= budget]
    return sorted(fit, key=lambda e: e["min_budget_mb"], reverse=True)


def probe_repo(repo: str, lister=list_gguf_files) -> list[dict] | None:
    """Fichiers GGUF réels du repo (quants + mmproj éventuel), ou None si le repo
    n'expose rien. Une erreur réseau/HF PROPAGE HfCatalogError (message montrable,
    diagnostic proxy inclus) — un None trompeur ferait conclure « repo disparu »
    quand c'est la connexion qui casse."""
    return lister(repo) or None


# Taille en milliards dans un nom de repo : « 7B », « 1.5b », « 397B-A17B ».
# \b évite les faux amis type « Q4 » ; on prend le MAX des occurrences : sur un
# MoE « 80B-A3B », l'empreinte mémoire suit les paramètres TOTAUX (80), pas les
# actifs (3) — tous les experts doivent tenir en RAM.
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
# Mo par milliard de paramètres en quant AGRESSIVE (~Q3/Q4 bas) : volontairement
# optimiste — ce filtre ÉLIMINE l'impossible, la taille réelle des quants
# (list_gguf_files + recommend_quant) tranche ensuite.
_MB_PER_B = 450


def estimate_min_mb(repo_id: str) -> int | None:
    """Taille minimale ESTIMÉE (Mo) d'un modèle d'après son nom de repo, ou None
    si le nom ne porte aucune taille (« mon-modele-GGUF ») — dans le doute, on
    ne masque pas."""
    hits = [float(m) for m in _PARAMS_RE.findall(repo_id)]
    if not hits:
        return None
    return int(max(hits) * _MB_PER_B)


def filter_by_budget(hits: list[dict], budget: int) -> tuple[list[dict], int]:
    """Sépare les résultats de recherche HF : (jouables annotés `est_mb`, nombre
    de masqués). Masqué = taille estimée du nom > budget. Taille inconnue =
    gardé (est_mb None)."""
    kept: list[dict] = []
    hidden = 0
    for h in hits:
        est = estimate_min_mb(h["repo_id"])
        if est is not None and est > budget:
            hidden += 1
            continue
        kept.append(dict(h, est_mb=est))
    return kept, hidden


def resolve_entry(entry: dict, searcher, budget: int) -> str | None:
    """Repo HF réel d'une entrée du catalogue : repo épinglé si présent, sinon
    résolution live (recherche de `query`, filtrée par le budget, top
    téléchargements). None si rien de jouable (famille disparue) ; une erreur
    réseau/HF PROPAGE HfCatalogError — même raison que probe_repo."""
    if entry.get("repo"):
        return entry["repo"]
    hits = searcher(entry["query"])
    kept, _ = filter_by_budget(hits, budget)
    return kept[0]["repo_id"] if kept else None


def pick_mmproj(files: list[dict]) -> str | None:
    """Nom du plus petit mmproj du repo (projecteur vision), s'il y en a un."""
    mm = [f for f in files if f["is_mmproj"]]
    return min(mm, key=lambda f: f["size_mb"])["filename"] if mm else None
