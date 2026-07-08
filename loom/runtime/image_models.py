# loom/runtime/image_models.py
"""Découverte des modèles IMAGE : loom/models/_IMAGE/<id>/ (model.toml + workflow.json).

Troisième type de modèle (après local llama-swap et distant API) : un dossier par
modèle, patron des LLM. Le préfixe '_' du parent exclut ces dossiers de la découverte
llama-swap (convention _TEMPLATE/_REMOTE). Le workflow.json est un graphe ComfyUI au
FORMAT API avec les placeholders {PROMPT} et {SEED} (remplacés à la soumission)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class ImageModel:
    id: str
    label: str
    dir: str
    width: int
    height: int
    comfy_dir: str  # racine de l'install ComfyUI (pour la démarrer nous-mêmes)
    comfy_port: int
    workflow_path: str
    # id d'un modèle Loom (petit local décensuré de préférence) qui RÉÉCRIT la demande
    # utilisateur (toute langue) en prompt de diffusion anglais avant la génération.
    # Vide = pas d'affinage, le prompt part brut. Séquence VRAM sûre : le refiner est
    # servi par llama-swap PUIS déchargé avant la diffusion (jamais deux résidents).
    refiner: str = ""


def discover_image_models(models_dir: Path | None = None) -> list[ImageModel]:
    """Scanne loom/models/_IMAGE/*/ ; dossier sans model.toml OU sans workflow.json
    -> ignoré (message console, pas d'exception : un dossier cassé ne bloque pas l'app)."""
    base = (models_dir or MODELS_DIR) / "_IMAGE"
    out: list[ImageModel] = []
    if not base.is_dir():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        toml_p, wf_p = d / "model.toml", d / "workflow.json"
        if not (toml_p.is_file() and wf_p.is_file()):
            print(f"[loom] modèle image ignoré (fichier manquant) : {d.name}")
            continue
        try:
            data = tomllib.loads(toml_p.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"[loom] modèle image illisible ({d.name}) : {exc}")
            continue
        out.append(
            ImageModel(
                id=d.name,
                label=str(data.get("label") or d.name),
                dir=str(d),
                width=int(data.get("width") or 1024),
                height=int(data.get("height") or 1024),
                comfy_dir=str(data.get("comfy_dir") or "C:/tools/ComfyUI"),
                comfy_port=int(data.get("comfy_port") or 8188),
                workflow_path=str(wf_p),
                refiner=str(data.get("refiner") or ""),
            )
        )
    return out
