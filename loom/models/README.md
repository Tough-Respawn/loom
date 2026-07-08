# Modèles Loom

Un modèle = un **dossier** `loom/models/<id>/` contenant :
- `model.toml` — métadonnées (source GGUF, offload, contexte, vision). Voir `_TEMPLATE/model.toml`.
- `profile.md` *(optionnel)* — consignes/quirks propres au modèle, injectées dans son contexte.
- le `<...>.gguf` — le poids (et un `*.mmproj-*.gguf` pour la vision). **Gitignoré** : jamais
  distribué, chacun fournit le sien.

`<id>` (le nom du dossier) apparaît dans le sélecteur UI et sert à `default_model`
(`[chat]` de `loom.config.toml`). La découverte scanne ce dossier au démarrage.

## Local vs distant, d'un coup d'œil
La racine de `models/` ne contient que du **LOCAL** (un dossier = un GGUF servi par
llama.cpp). Les modèles **DISTANTS** (API OpenAI-compatible, définis via le panneau
engrenage / `config/local.toml`) n'ont ici que leur éventuel `profile.md`, groupé sous
**`_REMOTE/<id>/`** — le préfixe `_` les exclut de la découverte locale, comme `_TEMPLATE`.

## Modèles IMAGE (`_IMAGE/<id>/`)
Un modèle image = un dossier sous `_IMAGE/` : `model.toml` (label, taille par défaut,
racine/port ComfyUI) + `workflow.json` (graphe ComfyUI **format API** avec `{PROMPT}`
et `{SEED}`). Sélectionnable dans l'UI comme un LLM : un message = une image. Le moteur
est ComfyUI (install séparée, venv privé) ; Loom le démarre et lui parle en HTTP.
Ajouter un modèle image = copier un dossier, éditer deux fichiers — comme les GGUF.

`refiner = "<id d'un modèle Loom>"` *(optionnel)* : avant la diffusion, ce modèle réécrit
ta demande — quelle que soit la langue — en UN prompt de diffusion anglais propre (sujet,
cadrage, lumière, style), affiché sous l'image. Recommandé : un petit modèle local
décensuré (ex. `gemma4-e4b-heretic`) — il est servi PUIS déchargé avant la diffusion,
jamais deux modèles résidents. Absent ou indisponible -> le prompt part brut.

## Ajouter ton modèle
1. `cp -r loom/models/_TEMPLATE loom/models/<ton-id>` (ou copie le dossier à la main).
2. Édite `loom/models/<ton-id>/model.toml` (`repo`/`filename`, `n_layers`, `size_mb`, et
   `cpu_moe`/`n_cpu_moe`/`context` selon ta VRAM).
3. Dépose le `.gguf` dans le dossier (ou laisse Loom le télécharger depuis `repo`/`filename`).
4. (Optionnel) pointe `default_model = "<ton-id>"` dans `loom.config.toml`.

Les dossiers de modèles personnels sont **gitignorés** (seul `_TEMPLATE/` est suivi) : le repo
reste agnostique, tu branches les tiens sans les pousser.
