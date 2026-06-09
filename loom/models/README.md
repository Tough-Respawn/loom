# Modèles Loom

Un modèle = un **dossier** `loom/models/<id>/` contenant :
- `model.toml` — métadonnées (source GGUF, offload, contexte, vision). Voir `_TEMPLATE/model.toml`.
- `profile.md` *(optionnel)* — consignes/quirks propres au modèle, injectées dans son contexte.
- le `<...>.gguf` — le poids (et un `*.mmproj-*.gguf` pour la vision). **Gitignoré** : jamais
  distribué, chacun fournit le sien.

`<id>` (le nom du dossier) apparaît dans le sélecteur UI et sert à `default_model`
(`[chat]` de `loom.config.toml`). La découverte scanne ce dossier au démarrage.

## Ajouter ton modèle
1. `cp -r loom/models/_TEMPLATE loom/models/<ton-id>` (ou copie le dossier à la main).
2. Édite `loom/models/<ton-id>/model.toml` (`repo`/`filename`, `n_layers`, `size_mb`, et
   `cpu_moe`/`n_cpu_moe`/`context` selon ta VRAM).
3. Dépose le `.gguf` dans le dossier (ou laisse Loom le télécharger depuis `repo`/`filename`).
4. (Optionnel) pointe `default_model = "<ton-id>"` dans `loom.config.toml`.

Les dossiers de modèles personnels sont **gitignorés** (seul `_TEMPLATE/` est suivi) : le repo
reste agnostique, tu branches les tiens sans les pousser.
