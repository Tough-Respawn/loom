# Modèles Loom

Loom découvre les modèles dans des dossiers configurables. Aucun poids ni catalogue
de modèles préinstallés n'est livré : le dépôt contient ce guide et les gabarits de
[`_TEMPLATE/`](_TEMPLATE/). Les dossiers de modèles sous `loom/models/` sont ignorés
par Git, à l'exception de ces gabarits.

## Emplacement des modèles

Dans `config/local.toml`, `[storage] models_root` accepte un chemin ou une liste :

```toml
[storage]
models_root = "/chemin/vers/les/modeles"
# Plusieurs emplacements, dans l'ordre de priorité :
# models_root = ["/chemin/stockage-rapide/modeles", "/chemin/stockage-secondaire/modeles"]
```

Remplacer ces chemins par ceux de l'installation ; les chemins TOML peuvent utiliser
`/`, y compris sous Windows. Sans ce réglage, Loom utilise
`loom/models/`. Les chemins relatifs sont résolus depuis le répertoire de lancement.

La première racine est prioritaire lorsqu'un même identifiant existe dans plusieurs
racines. Elle reçoit aussi les nouveaux modèles par défaut ; `/add-model` permet de
choisir une autre racine pour un modèle texte local. Les définitions distantes sont
créées dans la première racine.

Chaque racine utilise cette arborescence :

```text
<models_root>/
  local/
    text/<id>/model.toml     (+ poids GGUF et profile.md optionnel)
    image/<id>/model.toml    (+ workflow.json obligatoire)
    video/<id>/model.toml    (+ workflow.json obligatoire)
  remote/<id>/model.toml     (+ profile.md optionnel)
```

Utiliser un identifiant unique par modèle et le nommer par son dossier, sans champ
`id` dans le TOML. Cet identifiant apparaît dans le sélecteur et peut être choisi
avec `[chat] default_model` dans `config/local.toml`.

La configuration commune est lue depuis `config/defaults.toml`, puis surchargée par
`config/local.toml`. Les réglages propres au modèle prennent ensuite priorité pour
les champs pris en charge ; le `model.toml` n'est pas une copie de la config globale.

## Modèles texte locaux

Le dossier `local/text/<id>/` contient :

- `model.toml` : source des poids et réglages du serveur llama.cpp. Voir le
  [gabarit texte](_TEMPLATE/model.toml).
- Le GGUF indiqué par `filename` et, pour la vision, le projecteur indiqué par
  `mmproj_filename`. Loom télécharge les fichiers absents depuis `repo`.
- `profile.md`, facultatif : son frontmatter active des correctifs déjà implémentés
  dans [`models_profile.py`](../runtime/models_profile.py), tels que la normalisation
  des guillemets. Son texte n'est pas injecté comme consigne dans le contexte.

`repo`, `filename`, `n_layers` et `size_mb` doivent être présents dans une définition
manuelle. Les marqueurs du gabarit doivent être remplacés par les données du GGUF ;
les zéros de `n_layers` et `size_mb` indiquent des valeurs à renseigner.
Un fichier incomplet ou illisible est ignoré avec un message dans la console.

Les réglages de performance dépendent du modèle **et de la machine** : contexte,
couches GPU, placement des experts MoE, threads, batchs et isolation du cache.
`loom-setup` et `/rebench` permettent de mesurer ces réglages ; ne pas reprendre les
valeurs d'une autre installation comme référence. L'ajout d'un modèle extrait des
métadonnées du GGUF, mais ne remplace pas cette calibration.

## Modèles distants

Le dossier `remote/<id>/` contient un `model.toml` avec `base_url` et `model`, ainsi
que l'authentification et les options nécessaires au fournisseur. Voir le
[gabarit distant](_TEMPLATE/remote-model.toml). `profile.md` est également facultatif.

Les options communes peuvent être définies dans `[remote]` de la configuration
globale ; le fichier du modèle les surcharge champ par champ. Les modèles distants
sont gérés à chaud par le panneau des modèles de l'interface et par
`/add-model distant`.

## Modèles image et vidéo

Les dossiers `local/image/<id>/` et `local/video/<id>/` contiennent un `model.toml`
et un `workflow.json` exporté depuis ComfyUI au **format API**. ComfyUI doit être
installé séparément avec les poids et les nœuds nécessaires au workflow ; Loom
démarre le service et communique avec lui en HTTP.

Le `model.toml` accepte notamment `label`, `description`, `width`, `height`,
`comfy_dir`, `comfy_port`, `timeout`, `refiner` et `refine_hints`. `comfy_dir` désigne
l'installation ComfyUI ; les poids restent dans les emplacements configurés par
ComfyUI, éventuellement via son `extra_model_paths.yaml`.

Dans le JSON du workflow, utiliser les chaînes `"{PROMPT}"`, `"{SEED}"` et, pour
l'édition ou la vidéo à partir d'une image, `"{IMAGE}"` aux entrées correspondantes.
Pour ce dernier cas, indiquer le chemin de l'image dans le message, entre guillemets
s'il contient des espaces.

`refiner` peut désigner un modèle texte disponible dans Loom pour reformuler la
demande avant la génération. `refine_hints` apporte des indications propres au
workflow. Les modèles texte locaux sont déchargés avant la diffusion. `timeout`
fixe le délai d'attente en secondes, à adapter à la durée de génération.

## Ajouter un modèle

- **Première installation** : lancer `uv run loom-setup` depuis le dépôt pour être
  guidé dans l'installation et la calibration d'un modèle texte local.
- **Depuis le chat** : `/add-model` propose les modèles texte locaux, distants,
  image et vidéo. Pour le texte local, l'assistant recherche les GGUF sur Hugging
  Face, propose une quantification et télécharge les fichiers. Pour l'image et la
  vidéo, il demande un export ComfyUI au format API ; les poids restent à installer
  dans ComfyUI.
  La recommandation lit à distance l'en-tête GGUF d'un quant du dépôt (dense ou MoE,
  part des experts, cache KV par token, fenêtre glissante) et la liste des formats du
  llama.cpp installé (`llama-quantize`). La VRAM passe d'abord : le plus gros quant d'au
  moins 4 bits par poids qui y tient entier avec 16k de contexte, pleine précision
  comprise. Sinon, pour un modèle dense, le plus gros qui y tient entier (un dense qui
  déborde de la VRAM n'est jamais recommandé) ; pour un MoE, experts en RAM, le plus
  gros dont la génération estimée atteint 12 t/s, à défaut 8 t/s.
  Jamais un format inconnu de ce llama.cpp (quant d'un fork) ni la classe 1 bit. Chaque
  quant affiche « tout en VRAM » ou sa part lue en RAM et sa vitesse ESTIMÉE (débits
  fixes calés sur une RTX 2060 et de la DDR4, `/rebench` mesure ensuite la vraie).
  Quand rien ne convient, que le recommandé reste sous 12 t/s ou sous 4 bits (hors
  format ternaire natif), l'assistant cherche dans la même famille (nom du dépôt, puis
  modèle de base) des dépôts qui tournent ici et les propose par lettre.
- **Manuellement** : créer le dossier correspondant au type de modèle. Pour le
  texte local, copier uniquement `_TEMPLATE/model.toml` ; pour un distant, copier
  `_TEMPLATE/remote-model.toml` sous le nom `model.toml`. Pour l'image ou la vidéo,
  fournir les deux fichiers décrits ci-dessus. Relancer Loom après une modification
  manuelle pour recharger la configuration.

## Vérifier le travail d'une session

Le menu de l'onglet propose **Télécharger le journal des outils (.md)**. L'export ZIP contient
également `journal.md`, issu de `timeline.jsonl`, avec les entrées et sorties
persistées des outils. Un appel réussi ne prouve pas que la tâche a été accomplie.
En console :

```bash
uv run python -m loom.agent.transcript var/sessions/<id>
```
