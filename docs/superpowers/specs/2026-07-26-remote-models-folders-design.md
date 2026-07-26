# Distants en dossiers par modèle — `<racine>/remote/<id>/model.toml`

**Date** : 2026-07-26 · **Statut** : validé (conversation add-model/kimi)

## Problème

Les modèles locaux (texte, image, vidéo) sont découverts par dossier
(`<racine>/local/…/<id>/model.toml`) : un modèle = un dossier autoportant, et son
`model.toml` surcharge la config globale. Les distants, eux, vivent en
`[[remote_models]]` dans `config/local.toml` (depuis l'unification du 21/07 qui a
replié l'ancien store `var/remote_models.json`). Résultat : deux logiques, et des
modèles mélangés aux réglages machine (VRAM, chemins) — une virgule ratée dans
`local.toml` casse tout le boot. L'arbo prévoyait déjà `remote/<id>/` (profile.md
seulement) : on la termine.

## Décisions

1. **Un distant = un dossier** `<racine>/remote/<id>/model.toml` (+ `profile.md`
   optionnel, déjà géré). Découverte multi-racines identique aux locaux :
   scan de `remote/` sous chaque `models_root`, première racine gagnante par id.
   L'`id` est le nom du dossier (comme en local) ; un `id` dans le toml est ignoré.
2. **Couches** : `defaults.toml` peut porter une table `[remote]` (réglages
   standard appliqués à tous les distants, ex. `max_tokens`) ; `local.toml` peut
   la surcharger (deep-merge existant) ; le `model.toml` du dossier surcharge tout.
   Champs par modèle : `base_url`, `model`, `api_key` OU `api_key_env`, `context`,
   `max_tokens`, `vision`, `strong`, `enable_thinking_param`, `price_in/out/cached`,
   `description` (mêmes champs que `RemoteModelConfig`).
3. **Écritures** : le wizard `/add-model` distant et le panneau engrenage écrivent
   `<racine[0]>/remote/<id>/model.toml` (tomlkit, commentaires préservés à
   l'édition). Suppression = suppression du dossier (rmtree). L'UI ne change pas.
4. **Migration au boot** (dans `load_config`, comme le 21/07) :
   `[[remote_models]]` de `local.toml` → dossiers sur la racine prioritaire, puis
   retrait de `local.toml` ; un `var/remote_models.json` résiduel → dossiers
   directement. Idempotent ; un dossier existant du même id n'est pas écrasé.
   Coupe franche : après migration, `[[remote_models]]` n'est plus une source —
   mais les entrées encore présentes (échec d'écriture, FS en lecture seule)
   restent fusionnées en mémoire pour CE boot (dégradation douce, pas de perte).
5. **Choix du disque** (`/add-model` local) : s'il y a PLUSIEURS racines, une
   étape propose où installer — défaut = racine[0] (la plus rapide, C: ici),
   espace libre affiché. Une seule racine = pas de question. Les distants vont
   toujours sur racine[0] sans question (quelques lignes, aucun enjeu perf).
6. **Gabarits + doc** : `_TEMPLATE/remote/model.toml` commenté champ par champ ;
   `loom/models/README.md` réécrit (multi-racines : liste ordonnée, racine[0] =
   prioritaire = cible des installs ; section `remote/` complète) ;
   `config/local.example.toml` purgé des `[[remote_models]]`.

## Hors périmètre

Pas de changement d'UI (mêmes formulaires/boutons), pas de changement des
locaux (déjà en dossiers), pas de gestion de quota/clé par provider.

## Validation

Pytest (migrations, découverte, écritures) + E2E runtime réel sur config
sandbox : boot qui migre des `[[remote_models]]` en dossiers, ajout/édition/
suppression par les routes HTTP et le wizard, `local.toml` intact hors modèles.
