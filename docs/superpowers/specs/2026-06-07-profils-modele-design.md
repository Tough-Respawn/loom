# Spec — Profils par modèle (`loom/models/<id>/` + `profile.md` + fixes déterministes)

Date : 2026-06-07
Statut : design validé, prêt pour plan
Périmètre : structure des modèles + couche de correctifs déterministes par modèle

## 1. Problème

Chaque modèle local a des **travers de sortie déterministes propres** que le harness doit
corriger — quel que soit le prompt. Exemple vérifié en live : **Qwen3.5-4B émet des
guillemets typographiques** `’ ‘ ” “` (U+2019/2018/201D/201C) au lieu d'apostrophes/
guillemets ASCII → `SyntaxError` en Python, et il les **ré-émet** même quand on le lui
interdit (un prompt ne suffit pas). Gemma E4B, lui, n'a pas ce travers (« très bon sans
custom »). Aujourd'hui : aucun endroit pour porter un correctif spécifique à un modèle, et
les modèles sont éclatés dans `loom.config.toml` (`[[models]]`).

## 2. Objectif

1. **Un dossier par modèle** `loom/models/<id>/` regroupant TOUT son par-modèle : sa def
   (`model.toml`), son profil (`profile.md`), et son GGUF téléchargé.
2. **Correctifs déterministes par modèle** : le `profile.md` active des fixes **déjà codés**
   dans le harness ; le harness les applique au contenu produit par le modèle. 1er fix :
   `normalize_quotes` (débloque Qwen).
3. Gemma : `profile.md` sans fix → comportement inchangé.

## 3. Non-objectifs

- Mini-langage de transformation dans le `.md` : NON. Le `.md` **active** des fixes
  curatés codés en dur ; il ne contient pas de logique.
- Injection de prose dans le system prompt : NON (on a vu que le prompt n'arrête pas le
  travers de Qwen ; le fix déterministe suffit).
- Toucher la boucle/les autres outils : NON. Seul le contenu d'écriture est concerné.

## 4. Structure cible

```
loom/models/<id>/
  model.toml      # def du modèle (ex- [[models]] de loom.config.toml)
  profile.md      # frontmatter de flags + prose (peut être absent = aucun fix)
  <filename>.gguf # GGUF téléchargé ici (+ mmproj éventuel)
```

- `model.toml` : mêmes champs qu'une table `[[models]]` actuelle (`repo`, `filename`,
  `n_layers`, `size_mb`, `mmproj_filename`, `id` = nom du dossier, `n_gpu_layers`).
- **Découverte** : `config.py` scanne `loom/models/*/model.toml` pour construire
  `cfg.models` (remplace le bloc `[[models]]` de `loom.config.toml`). `default_model`
  reste dans `[chat]` de `loom.config.toml`.
- **Téléchargement** : `models_fetch.ensure_model` / `serve.py` téléchargent le GGUF dans
  `loom/models/<id>/` (au lieu du dossier plat actuel).
- **Migration** : créer `loom/models/gemma-uncensored/` et
  `loom/models/qwen3.5-4b-abliterated/` (model.toml issus des `[[models]]` actuels) ;
  déplacer/retélécharger les GGUF dans leur dossier ; retirer `[[models]]` de
  `loom.config.toml`.

## 5. `profile.md`

Frontmatter YAML (flags des fixes built-in) + corps prose (le « pourquoi »).

Exemple `loom/models/qwen3.5-4b-abliterated/profile.md` :

```markdown
---
fixes:
  normalize_quotes: true
---
Qwen3.5-4B émet des guillemets typographiques (’ ‘ ” “) au lieu d'ASCII, ce qui casse la
syntaxe Python. On normalise le contenu écrit dans les fichiers de code.
```

Gemma : pas de `profile.md` (ou frontmatter `fixes: {}`) → aucun fix.

## 6. Fixes déterministes built-in (curatés)

Un registre codé en dur de fixes nommés. Au lancement, le profil du modèle actif liste
ceux à activer.

- **`normalize_quotes`** : remplace `’ ‘` → `'`, `” “` → `"` dans le `content` écrit, pour
  les **fichiers de code uniquement** (exclut `.md/.markdown/.txt/.rst` où ces caractères
  peuvent être voulus).

La liste est **extensible** : on ajoute un fix nommé quand on identifie un nouveau travers.

## 7. Application

- Un module `loom/models_profile.py` : `load_profile(model_id) -> Profile` (lit
  `loom/models/<id>/profile.md`, parse le frontmatter, résout les fixes actifs) et
  `apply_fixes(profile, tool_name, args, suffix) -> args` (applique les fixes au `content`/
  `old_string`/`new_string` des outils d'écriture selon l'extension).
- `build_registry` reçoit l'**id du modèle actif** (`conv.model`, déjà dispo dans
  `make_registry`), charge le profil, et le passe à `ToolRegistry`.
- `ToolRegistry.run` : après `validate_and_coerce`, applique `apply_fixes` avant
  `spec.run(args)` — choke point unique, comme la coercition centrale.
- Aucun profil / aucun fix → passe-plat (Gemma inchangé).

## 8. Vérification (smokes — pas de suite de tests)

1. `load_profile('qwen3.5-4b-abliterated')` → fix `normalize_quotes` actif ;
   `load_profile('gemma-uncensored')` → aucun fix.
2. `apply_fixes` sur un `write_file` `.py` avec `game_state[’revealed’]` →
   `game_state['revealed']` ; sur un `.md` → inchangé.
3. End-to-end : registre construit avec `active_model='qwen3.5-4b-abliterated'`,
   `replace_lines` d'un bloc Python à guillemets typo → le fichier écrit compile.
4. `config.py` découvre les 2 dossiers modèles ; `serve.py` génère le swap yaml comme avant.

## 9. Fichiers touchés

- Créer : `loom/models_profile.py` ; `loom/models/<id>/model.toml` + `profile.md` (×2).
- Modifier : `loom/config.py` (découverte des dossiers au lieu de `[[models]]`),
  `loom/tools/base.py` (`ToolRegistry` porte le profil + applique dans `run`),
  `loom/tools/__init__.py` (`build_registry` reçoit `active_model`, charge le profil),
  `loom/web/__main__.py` (passe `conv.model` à `make_registry`/`build_registry`),
  `loom/models_fetch.py`/`loom/serve.py` (téléchargement dans `loom/models/<id>/`),
  `loom.config.toml` (retire `[[models]]`).

## 10. Phasage (pour le plan)

- **Phase A (valeur)** : `models_profile.py` + fixes + application au registre + les 2
  `profile.md`, en lisant `loom/models/<id>/profile.md` **sans** encore déplacer la def.
  Débloque Qwen immédiatement.
- **Phase B (refonte structurelle)** : migrer les `[[models]]` en `model.toml` + découverte
  `config.py` + GGUF dans le dossier.

Les deux phases servent la même cible §4 ; la Phase A livre le déblocage sans risque sur le
chemin de lancement.
