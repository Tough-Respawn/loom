# loom/runtime/hf_catalog.py
"""Catalogue Hugging Face pour /add-model : shortlist de repos GGUF + inventaire
des quants d'un repo. API publique (pas de clé pour les repos publics). Toute
erreur réseau/HF est ramenée à HfCatalogError au message MONTRABLE dans le chat
(même contrat que ModelUnavailable dans models_fetch.py) — jamais de stacktrace."""

from __future__ import annotations

import re

# Multi-parties llama.cpp : "...-00001-of-00003.gguf". On regroupe sous la 1re partie
# (celle que charge llama-server) avec la taille CUMULÉE.
_PART_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


class HfCatalogError(Exception):
    """Recherche/inventaire HF impossible — porte un message prêt à montrer."""


def _api(api):
    if api is not None:
        return api
    from huggingface_hub import HfApi

    return HfApi()


def _err(what: str, exc: Exception) -> HfCatalogError:
    head = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return HfCatalogError(
        f"{what} impossible ({type(exc).__name__}: {head}) — vérifie la connexion, "
        "ou réessaie."
    )


def search_models(query: str, limit: int = 8, api=None) -> list[dict]:
    """Top repos GGUF pour `query`, triés par téléchargements décroissants."""
    try:
        hits = _api(api).list_models(
            search=query, filter="gguf", sort="downloads", direction=-1, limit=limit
        )
        return [
            {
                "repo_id": m.id,
                "downloads": int(m.downloads or 0),
                "likes": int(m.likes or 0),
            }
            for m in hits
        ]
    except Exception as exc:  # noqa: BLE001 - tout devient un message actionnable
        raise _err("recherche Hugging Face", exc) from exc


def list_gguf_files(repo_id: str, api=None) -> list[dict]:
    """Fichiers .gguf du repo (tailles réelles), multi-parties REGROUPÉES."""
    try:
        info = _api(api).model_info(repo_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        raise _err(f"inventaire du repo « {repo_id} »", exc) from exc
    groups: dict[str, dict] = {}
    for s in info.siblings or []:
        fn = s.rfilename
        if not fn.lower().endswith(".gguf"):
            continue
        m = _PART_RE.search(fn)
        base = fn[: m.start()] if m else fn
        g = groups.setdefault(base, {"part_files": [], "size": 0})
        g["part_files"].append(fn)
        g["size"] += int(s.size or 0)
    out = []
    for g in groups.values():
        parts = sorted(g["part_files"])
        out.append(
            {
                "filename": parts[0],
                "part_files": parts,
                "size_mb": max(1, g["size"] // (1024 * 1024)),
                "is_mmproj": "mmproj" in parts[0].lower(),
            }
        )
    return sorted(out, key=lambda f: f["size_mb"])
