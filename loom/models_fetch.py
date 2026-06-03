# loom/models_fetch.py
"""Garantit la présence locale d'un fichier GGUF (download si absent)."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


def ensure_model(repo: str, filename: str, models_dir: str | Path) -> Path:
    """Renvoie le chemin local du GGUF, en le téléchargeant depuis HF si absent."""
    models_dir = Path(models_dir)
    target = models_dir / filename
    if target.exists():
        return target
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(models_dir),
    )
    return target
