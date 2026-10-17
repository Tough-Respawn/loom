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
- **Depuis l’interface** : configuration → **Gérer les modèles** → **Ajouter un modèle**.
  `/add-model` ouvre ce même panneau, y compris pendant une génération. Les étapes
  d’ajout et `/cancel` restent hors du fil du modèle. Le panneau propose les modèles
  texte locaux, distants, image et vidéo. Pour le texte local, l'assistant recherche les GGUF sur Hugging
  Face, propose une quantification et télécharge les fichiers. Pour l'image et la
  vidéo, il demande un export ComfyUI au format API ; les poids restent à installer
  dans ComfyUI.
  La recommandation vérifie les tailles des fichiers, les formats annoncés par le
  moteur installé et les métadonnées GGUF. Le budget GPU suit le lancement de
  llama.cpp : poids placés sur le GPU, cache `q8_0` au contexte configuré (au moins
  16k), slots et état récurrent. Les embeddings d'entrée séparés et le projecteur
  vision restent en RAM ; les embeddings partagés avec la sortie comptent en VRAM.
  Un dense dont les couches débordent en RAM n'est pas proposé comme candidat GPU ;
  sans métadonnées, le calcul reste conservateur.
  Si aucun quant ne convient, les alternatives sont cherchées dans la même famille
  et, si elle est déclarée, celle du modèle de base. Sans évaluation comparable,
  les candidats sont ordonnés par nombre de paramètres décroissant : c'est une
  heuristique de capacité, pas une preuve de qualité. Une alternative publiée par un
  tiers qui a modifié le modèle (affinage, distillation, abliteration, ternarisation) est
  marquée **dérivé tiers de** son modèle officiel : filiation `base_model` déclarée sur
  Hugging Face, ou nom de dépôt différent de celui du modèle qu'il dit quantifier. Une
  simple quantification (unsloth, bartowski…) n'est pas marquée.
  L'interface distingue **candidat estimé** et **évalué dans Loom**. Les évaluations
  locales comparables classent d'abord la réussite des tâches, puis leur durée.
  Sans mesure, la qualité reste explicitement inconnue. La vitesse d'offload est
  une estimation à débits fixes ; `/rebench` mesure les performances réelles.
- **Manuellement** : créer le dossier correspondant au type de modèle. Pour le
  texte local, copier uniquement `_TEMPLATE/model.toml` ; pour un distant, copier
  `_TEMPLATE/remote-model.toml` sous le nom `model.toml`. Pour l'image ou la vidéo,
  fournir les deux fichiers décrits ci-dessus. Relancer Loom après une modification
  manuelle pour recharger la configuration.

## Retirer un modèle

Dans la configuration, ouvrir **Gérer les modèles** et confirmer la suppression.
`/remove-model` ouvre ce même panneau : aucun message n'entre dans le fil du modèle,
et la génération continue. Un modèle en cours d'utilisation ne peut pas être supprimé.
Les modèles texte locaux sont effacés du disque ; pour les modèles image et vidéo,
seule leur définition Loom est retirée, les poids ComfyUI partagés sont conservés.

## Vérifier le travail d'une session

Le menu de l'onglet propose **Télécharger le journal des outils (.md)**. L'export ZIP contient
également `journal.md`, issu de `timeline.jsonl`, avec les entrées et sorties
persistées des outils. Un appel réussi ne prouve pas que la tâche a été accomplie.
En console :

```bash
uv run python -m loom.agent.transcript var/sessions/<id>
```

Pour mesurer la qualité agentique et alimenter la recommandation, voir le
[harnais d'évaluation](../../evals/README.md).
