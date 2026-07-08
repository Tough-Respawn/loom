# Modèles Loom

La racine des modèles est **configurable** : `[storage] models_root` dans
`config/local.toml` (ex. `E:/loom-models`). À défaut, c'est ce dossier du package
(`loom/models/`). L'arborescence ci-dessous est LA convention, où que soit la racine.
Le repo ne suit que `_TEMPLATE/` et ce README : les modèles sont personnels, jamais
poussés.

## Arborescence

```
<models_root>/
  local/
    text/    <id>/model.toml (+ .gguf, profile.md)   -> LLM servis par llama.cpp
    image/   <id>/model.toml + workflow.json         -> ComfyUI, un message = une image
    video/   <id>/model.toml + workflow.json         -> ComfyUI, un message = un clip
  remote/    <id>/profile.md                         -> profils des modèles API
  _TEMPLATE/ gabarit d'un modèle texte
```

## Modèles TEXTE (`local/text/<id>/`)
- `model.toml` — métadonnées (source GGUF, offload, contexte, vision). Voir `_TEMPLATE/`.
- `profile.md` *(optionnel)* — consignes/quirks du modèle, injectés dans son contexte.
- le `.gguf` (et un `*.mmproj-*.gguf` pour la vision) — téléchargé par Loom si absent.

`<id>` (le nom du dossier) apparaît dans le sélecteur UI et sert à `default_model`.

## Modèles IMAGE / VIDÉO (`local/image/`, `local/video/`)
Même format dans les deux : `model.toml` (label, taille, racine/port ComfyUI, `refiner`,
`timeout`) + `workflow.json` (graphe ComfyUI **format API** avec `{PROMPT}`, `{SEED}`
et, pour l'édition/i2v, `{IMAGE}`). Le moteur est ComfyUI (install séparée, venv privé) ;
Loom le démarre et lui parle en HTTP. Les POIDS ComfyUI (unet/encodeurs/vae) vivent dans
leur propre dossier (ex. `E:/comfyui-models`, référencé par `extra_model_paths.yaml`
dans l'install ComfyUI).

`refiner = "<id d'un modèle Loom>"` *(optionnel)* : avant la diffusion, ce modèle réécrit
ta demande — quelle que soit la langue — en UN prompt anglais propre (description, ou
instruction d'édition si une photo est jointe), affiché sous le résultat. Recommandé :
un petit local décensuré (ex. `gemma4-e4b-heretic`) — servi PUIS déchargé avant la
diffusion, jamais deux résidents. `{IMAGE}` : mets le CHEMIN de ta photo dans le message
(guillemets si espaces). `timeout` : budget d'attente (s), à monter pour la vidéo.

## Ajouter un modèle
- Texte : copie `_TEMPLATE/` vers `local/text/<ton-id>/`, édite `model.toml`.
- Image/vidéo : copie un dossier existant de `local/image/`, édite les deux fichiers.
