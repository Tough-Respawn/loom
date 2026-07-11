---
name: code-review
description: Se déclenche automatiquement quand on relit un diff (git diff, PR) avant de merger — guide une revue en deux passes. Pass 1 : conformité aux conventions du projet Loom (commits courts, ruff/uv, pas de pytest, frontière de confiance sur l'ingestion). Pass 2 : qualité/bugs (logique, lisibilité, robustesse). Idéal pour vérifier qu'un commit/PR est « prêt à merger ».
---

# Code review — deux passes

Quand un diff (git diff, PR) est soumis avant merger, applique cette revue systématique.
**Ne te contente pas de lire** : vérifie chaque point avec les outils (`run_shell`, `read_file`, `search_text`).

---

## Pass 1 — Conventions du projet

Vérifie chaque règle. Signale ce qui **dévie** avec précision (fichier:ligne, contexte).

### 1. Commits
- **Courts et descriptifs** : format Conventional Commits (`type: description courte`).
- Un commit = une idée. Pas de mélanges.
- Pas de « fix », « wip », « update » seuls — ils doivent être précis.

### 2. Outils Python
- **ruff** pour le lint/format : code formaté, pas d'erreurs ruff.
- **uv** pour les dépendances : `pyproject.toml` cohérent, pas de `requirements.txt` orphelin.
- **Pas de pytest** : Loom vérifie par smokes (`uv run python -c`) et ruff, pas de suite de tests.
  Signaler `pytest`/`unittest` qui trainent.

### 3. Frontière de confiance
- Les contenus externes (`fetch_url`, `web_search`, `read_file` sur PDF/Office, `read_image`) sont marqués **DONNÉES**, pas instructions.
- Les actions à effet de bord sur du contenu externe sont **gated** (confirmées avant exécution).
- Pas de `ignore mes instructions` ou `execute cette commande` venus d'un contenu ingéré sans garde-fou.

### 4. Structure du projet
- Les fichiers nouveaux vont aux bons endroits (`loom/tools/` pour les outils, `loom/extend/` pour les skills/plugins, `loom/models/` pour les modèles).
- Les chemins relatifs sont corrects (pas de `../../` en trop après réorganisation).
- `.gitignore` couvre ce qui doit l'être (`.claude/`, `loom/runtime/data/`, `*.local.toml`).

---

## Pass 2 — Qualité et Bugs

### 5. Logique
- Pas de **logique falsifiable** : les conditions (`if/else`, `try/except`) sont-elles correctes ?
- Les **bords** sont-ils gérés ? (vide, null, erreurs, timeouts)
- Pas de **double traitement** ou **trajet manqué** dans les boucles d'outils.

### 6. Lisibilité
- Noms explicites (`find_files`, pas `f1`).
- Commentaires qui disent **pourquoi**, pas **quoi**.
- Pas de code mort, pas de complexité inutile.

### 7. Robustesse
- Les **erreurs** sont traitées (pas de `except: pass` sauvage).
- Les **dépendances** sont compatibles (pas de versions conflictuelles).
- Les **chemins** fonctionnent sur Windows ET Linux (si applicable).

---

## Méthode

1. **Lire le diff** : `run_shell("git diff <ref>")` pour voir les changements non commités.
2. **Pass 1** : vérifier les conventions une par une. Noter les écarts.
3. **Pass 2** : analyser la qualité du code. Noter les risques.
4. **Rendre un verdict** :
   - ✅ **Prêt** : zéro écart majeur, aucun bug détecté.
   - ⚠️ **Presque** : écarts mineurs, aucun blocage.
   - ❌ **À corriger** : un ou plusieurs écarts bloquants (convention brisée, bug logique).

5. **Fournir le résumé** : un paragraphe court résumant les points clés + le verdict.

---

## Garde-fous

- **Ne pas sur-réviser** : se concentrer sur ce qui compte pour Loom (pas de style pur).
- **Préférer la preuve** : un `run_shell` vaut mieux qu'une hypothèse.
- **Un seul verdict** : trancher avant de conclure, pas de « peut-être », « probablement ».
