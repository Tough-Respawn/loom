# Modèles Loom

La ou LES racines des modèles sont **configurables** : `[storage] models_root` dans
`config/local.toml` — chaîne ou liste. À défaut, c'est ce dossier du package
(`loom/models/`). Avec plusieurs disques, **ordonne du plus rapide au plus lent**
(ex. `["C:/loom-models", "E:/loom-models"]` : NVMe puis T7 USB) :

- la **première racine gagne** quand un même id existe sur plusieurs disques ;
- les **nouveaux modèles s'installent sur la première** (`/add-model` propose le
  choix du disque quand il y a plusieurs racines ; les distants — quelques lignes
  de config — vont toujours sur la première, sans question).

L'arborescence ci-dessous est LA convention, identique sous chaque racine. Le repo
ne suit que `_TEMPLATE/` et ce README : les modèles sont personnels, jamais poussés.

## Arborescence

```
<models_root>/            (répétée sous chaque racine)
  local/
    text/    <id>/model.toml (+ .gguf, profile.md)   -> LLM servis par llama.cpp
    image/   <id>/model.toml + workflow.json         -> ComfyUI, un message = une image
    video/   <id>/model.toml + workflow.json         -> ComfyUI, un message = un clip
  remote/    <id>/model.toml (+ profile.md)          -> API distantes OpenAI-compatibles
  _TEMPLATE/ gabarits : model.toml (texte) et remote-model.toml (distant)
```

Partout, **l'`id` d'un modèle = le nom de son dossier** : il apparaît dans le
sélecteur UI et sert à `default_model`. Un modèle = un dossier autoportant ; le
`model.toml` du dossier **surcharge la config globale** (`config/defaults.toml`
puis `config/local.toml`).

## Modèles TEXTE (`local/text/<id>/`)
- `model.toml` — métadonnées (source GGUF, offload, contexte, vision). Voir `_TEMPLATE/model.toml`.
- `profile.md` *(optionnel)* — consignes/quirks du modèle, injectés dans son contexte.
- le `.gguf` (et un `*.mmproj-*.gguf` pour la vision) — téléchargé par Loom si absent.

## Modèles DISTANTS (`remote/<id>/`)
- `model.toml` — `base_url`, `model`, clé (`api_key` en clair ou `api_key_env`),
  `context`, prix, `vision`, `strong`… Voir `_TEMPLATE/remote-model.toml`. Les
  réglages standard communs à tous les distants vivent dans la table `[remote]`
  de `config/defaults.toml` (surchargeable par `config/local.toml`) ; ce fichier
  surcharge tout.
- `profile.md` *(optionnel)* — mêmes fixes déterministes que pour un local.

Géré à chaud par l'UI (panneau engrenage : ajout/édition/test/suppression) et par
`/add-model distant`. Supprimer le modèle = supprimer son dossier.

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
- **La voie royale : tape `/add-model` dans le chat** — le wizard fait tout (local :
  recherche Hugging Face, quant recommandé selon ta machine, choix du disque si
  plusieurs racines, téléchargement, model.toml généré ; distant : URL + clé,
  dossier `remote/<id>/` créé et monté à chaud).
- À la main (toujours possible) :
  - Texte : copie `_TEMPLATE/` vers `local/text/<ton-id>/`, édite `model.toml`.
  - Distant : crée `remote/<ton-id>/` et copie `_TEMPLATE/remote-model.toml`
    dedans sous le nom `model.toml`, puis édite.
  - Image/vidéo : copie un dossier existant de `local/image/`, édite les deux fichiers.
