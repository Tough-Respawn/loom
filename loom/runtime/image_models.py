# loom/runtime/image_models.py
"""Découverte des modèles IMAGE/VIDÉO : <models_root>/local/{image,video}/<id>/
(model.toml + workflow.json).

Types de modèles ComfyUI (après local llama-swap et distant API) : un dossier par
modèle, patron des LLM. Le workflow.json est un graphe ComfyUI au FORMAT API avec
les placeholders {PROMPT}, {SEED} et, pour l'édition/i2v, {IMAGE} (remplacés à la
soumission)."""

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
    # Budget d'attente de la génération (s). 600 suffit à une image ; un modèle VIDÉO
    # (Wan) sur petit GPU se compte en dizaines de minutes -> à monter dans model.toml.
    timeout: int = 600
    # "image" ou "video" : dérivé du dossier parent (local/image vs local/video).
    # Sert au préfixe du sélecteur UI ; le comportement runtime est identique.
    kind: str = "image"
    # RÔLE en une ligne (model.toml `description`) : infobulle du sélecteur UI.
    description: str = ""


def discover_image_models(models_root: Path | None = None) -> list[ImageModel]:
    """Scanne les modèles ComfyUI de la racine : local/image/*/ et local/video/*/
    (même format — la sortie vidéo est portée par le workflow). Dossier sans model.toml
    OU sans workflow.json -> ignoré (message console, pas d'exception : un dossier
    cassé ne bloque pas l'app)."""
    root = Path(models_root or MODELS_DIR)
    bases = [root / "local" / "image", root / "local" / "video"]
    out: list[ImageModel] = []
    dirs = [p for base in bases if base.is_dir() for p in base.iterdir() if p.is_dir()]
    for d in sorted(dirs, key=lambda p: p.name):
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
                timeout=int(data.get("timeout") or 600),
                kind=d.parent.name if d.parent.name in ("image", "video") else "image",
                description=str(data.get("description") or ""),
            )
        )
    return out
