# loom/setup/catalog.py
"""Discovery seeds, filtered by memory before checking exact repository artifacts.
Catalogue order is not a measured quality ranking; local task evaluations decide
when comparable evidence is available."""

from __future__ import annotations

import re

from loom.runtime.hf_catalog import list_gguf_files

# Search seeds, not a quality ranking. Exact artifacts are checked independently.
CATALOG: list[dict] = [
    {"query": "qwen3.6 35b a3b instruct gguf", "label": "Qwen3.6 35B A3B (MoE + vision), à évaluer", "min_budget_mb": 18000},
    {"query": "gemma4 26b a4b gguf", "label": "Gemma4 26B A4B (MoE), à évaluer", "min_budget_mb": 14000},
    {"query": "Qwen2.5-1.5B-Instruct-GGUF", "label": "Qwen2.5 1.5B — petit budget, à évaluer", "min_budget_mb": 1000},
    {"query": "Ternary-Bonsai-2-4B", "label": "Bonsai 2 4B — dense compact, à évaluer", "min_budget_mb": 1600},
    {"query": "Qwen3.5-4B-GGUF", "label": "Qwen3.5 4B — dense, à évaluer", "min_budget_mb": 2600},
    {"query": "Qwen3.5-9B-GGUF", "label": "Qwen3.5 9B — dense, à évaluer", "min_budget_mb": 5500},
    {"query": "gemma-4-E4B", "label": "Gemma 4 E4B — à évaluer", "min_budget_mb": 5500},
]


def budget_mb(vram_free_mb: int, ram_mb: int, margin_mb: int = 4096) -> int:
    """Budget mémoire pour un modèle : VRAM libre + RAM − marge système (les
    MoE tournent experts en RAM, cf. recommend_quant)."""
    return max(0, vram_free_mb + ram_mb - margin_mb)


def fitting_entries(budget: int, catalog: list[dict] | None = None) -> list[dict]:
    """Filtrer puis proposer les paliers de capacité décroissants, sans score de qualité."""
    cat = CATALOG if catalog is None else catalog
    fit = [e for e in cat if e["min_budget_mb"] <= budget]
    return sorted(fit, key=lambda entry: entry["min_budget_mb"], reverse=True)


# URL HF (huggingface.co / hf.co, http(s) optionnel) -> les 2 premiers segments
# de chemin = org/repo ; le reste (/tree/main, ?query…) est ignoré.
_HF_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/([^/\s]+/[^/\s?#]+)"
)
_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def parse_hf_repo(text: str) -> str | None:
    """`org/repo` depuis une URL Hugging Face ou un id collé tel quel — None si
    le texte ressemble à une recherche libre (le champ de recherche du setup
    accepte les deux)."""
    t = text.strip().rstrip("/")
    m = _HF_URL_RE.match(t)
    if m:
        return m.group(1)
    if _REPO_ID_RE.match(t):
        return t
    return None


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
