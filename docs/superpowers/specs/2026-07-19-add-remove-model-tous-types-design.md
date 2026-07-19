# /add-model et /remove-model : tous les types de modèles

Date : 2026-07-19 · Statut : validé (design approuvé dans le chat)

## Problème

Le wizard `/add-model` ne sait ajouter que deux types : local texte (GGUF Hugging
Face) et distant API. `/remove-model` ne liste que les locaux texte et les
distants ajoutés via l'UI. Or le sélecteur affiche quatre types (home, remote,
image, video) et les distants déclarés dans `config/local.toml` (glm-*) ne sont
supprimables nulle part en slash command. Objectif : **tout modèle listé dans le
sélecteur doit pouvoir être ajouté et supprimé par slash command**, avec un
filtre par type à l'entrée du wizard.

## Décisions actées

- Un seul point d'entrée `/add-model` avec étape « type » (pas de commandes
  dédiées par type).
- Recette ComfyUI (`workflow.json`) : « les deux » — l'utilisateur colle un
  chemin vers son export ComfyUI (format API), OU répond « plus tard » et le
  wizard scaffolde le dossier à compléter.
- Les poids ComfyUI (`E:/comfyui-models`) ne sont JAMAIS téléchargés ni
  supprimés par le wizard : seul le dossier Loom (model.toml + workflow.json)
  est géré.
- Les distants de `config/local.toml` deviennent supprimables via tomlkit
  (commentaires préservés), comme ils sont déjà éditables (`upsert_remote_in_toml`).

## 1. /add-model : filtre par type

`start()` (wizard.py) évolue :

- `/add-model` sans argument → étape `kind` à 4 choix :
  1. local (GGUF Hugging Face), 2. distant (API), 3. image (ComfyUI),
  4. video (ComfyUI). Réponse : numéro ou mot-clé.
- Raccourcis : `/add-model image`, `/add-model video` sautent l'étape kind
  (comme `distant` aujourd'hui). `/add-model local <recherche>` accepté.
- Back-compat inchangée : `/add-model <texte libre>` = recherche HF locale
  directe ; URL brute = flux distant.

## 2. Nouveau flux image/vidéo (steps `i_*`)

Étapes (kind déjà connu : image ou video) :

1. `i_id` — id kebab (validé `_valid_id`, unicité `deps.existing_ids`).
   Détection reprise : si le dossier `<base>/<id>/` existe déjà avec un
   `workflow.json` valide (cas « plus tard » complété), proposer de le monter
   directement (action `mount_image`).
2. `i_dims` — dimensions `LxH` (défaut proposé : 1024x1024 image, 832x480 video ;
   « ok » = défaut).
3. `i_desc` — description une ligne (infobulle sélecteur ; « non » = vide).
4. `i_workflow` — « colle le chemin de ton export ComfyUI (format API), ou
   “plus tard” ».

Fin de flux → action `install_image` :

- Dossier créé : `<racine>/local/{image|video}/<id>/` où `<racine>` = la racine
  de `models_root` qui héberge déjà des modèles de ce type, sinon la première.
- `model.toml` généré : `label`, `width`, `height`, `description`, et
  `comfy_dir`/`comfy_port`/`refiner`/`timeout` recopiés d'un modèle image
  existant (même install ComfyUI) ; à défaut, valeurs du dataclass `ImageModel`.
- Chemin fourni → copie en `workflow.json` + validation légère : JSON parsable
  et placeholder `{PROMPT}` présent (sinon warning dans la réponse, copie quand
  même) ; pour `video` i2v le placeholder `{IMAGE}` est signalé s'il manque
  (warning, pas blocage). Montage à chaud : ajout aux registres image
  (S.image_models/_image_by_id, image/video_model_ids, S.models,
  model_descriptions — même périmètre que ce que lit `_models_payload`)
  + SSE `models`.
- « plus tard » → scaffold sans workflow.json (PAS de fichier vide : la
  découverte au boot ignore les dossiers incomplets avec un message console,
  comportement conservé). La réponse donne le chemin exact où déposer l'export
  et rappelle : « relance /add-model image et redonne le même id pour le monter,
  sinon il sera découvert au prochain démarrage ».

Le montage à chaud est factorisé dans routes.py (`_mount_image`, symétrique de
`_mount_local`/`_mount_remote`), avec re-découverte du dossier via
`discover_image_models` restreint au dossier créé (réutilise le parseur
existant, pas de duplication).

## 3. /remove-model : liste complète

`_removable_models` (routes.py) retourne quatre familles, étiquetées :

| kind | label | effet à la confirmation |
|---|---|---|
| `local` | `<id> — local, X Go sur disque` | inchangé : rmtree dossier (GGUF compris), regen llama-swap |
| `remote` | `<id> — distant (<model>)` | inchangé : store JSON + clé + démontage |
| `remote_config` | `<id> — distant (<model>, config/local.toml)` | NOUVEAU : `delete_remote_in_toml` (tomlkit, commentaires préservés) + `_forget_remote` |
| `image` / `video` | `<id> — image|video (ComfyUI), définition seule` | NOUVEAU : rmtree du dossier Loom (~Ko) + démontage des registres image |

Confirmations (étape `d_pick`) adaptées :

- `remote_config` : « il sera retiré de config/local.toml (sa clé avec) et
  démonté » + si `default_model` de local.toml = cet id, avertir : « c'est le
  modèle par défaut : au prochain boot, repli sur le premier modèle installé ».
- `image`/`video` : « sa définition Loom (model.toml + workflow.json) sera
  supprimée ; les poids ComfyUI partagés (comfyui-models) ne sont PAS touchés ».

`delete_remote_in_toml(local_path, model_id)` : nouvelle fonction model_store,
symétrique d'`upsert_remote_in_toml` (retire l'entrée de l'AoT
`[[remote_models]]` par id, préserve le reste du document).

Démontage image à chaud (`_forget_image`) : retrait de S.image_models,
_image_by_id, image/video_model_ids, S.models, model_descriptions ; SSE
`models` ensuite (mécanique existante `models_changed`).

## 4. Corrections embarquées

- **Persistance du résultat** : le suffixe `✅/❌` (`extra_reply`) est aujourd'hui
  émis en SSE mais jamais persisté → au rechargement le fil s'arrête sur
  « Suppression de … » (et un ❌ d'erreur serait perdu). Correction : persister
  l'échange final APRÈS calcul d'`extra_reply` (journal + conversation), une
  seule écriture.
- **Lisibilité de la liste 1/2** : une ligne de rappel sous la liste : « (les
  distants déclarés dans config/local.toml sont marqués ; images/vidéos :
  définition seule, poids ComfyUI non touchés) » — formulation courte.

## Hors périmètre

- Téléchargement/suppression des poids ComfyUI (unet/vae/encodeurs).
- Édition (modifier un modèle image existant) : passe par les fichiers.
- Ajout image/vidéo via le panneau engrenage (slash command seulement ici).

## Tests

- **Unit wizard** (pattern `test_wizard_add_model.py`, deps stubbées) : étape
  kind à 4 choix + raccourcis ; flux i_* complet (chemin fourni, « plus tard »,
  reprise sur dossier complet, id invalide/dupliqué, workflow sans {PROMPT}) ;
  d_pick/d_confirm pour les 4 kinds (messages adaptés, avertissement
  default_model).
- **Unit model_store** : `delete_remote_in_toml` (suppression par id, document
  et commentaires préservés, id absent = no-op).
- **Routes** (pattern `test_add_model_routes.py`) : effets `install_image`,
  `mount_image`, `remove` × {remote_config, image} ; persistance d'`extra_reply`.
- **E2E réel** (Playwright sur loom.web qui tourne) : ajout d'un modèle image
  factice (workflow.json minimal), visible dans le sélecteur sans reload ;
  suppression du même modèle ; suppression d'un distant config sur un
  local.toml de TEST (jamais le vrai) ; vérif du fil après reload (✅ persisté).
