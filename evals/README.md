# Éval des prompts de Loom

Harnais d'évaluation à la Anthropic : jeu de cas figé + grader, pour comparer deux
variantes du *system prompt* sur le comportement réel du modèle local.

## Principe

- **Eval set figé** (`cases.py`) : 6 cas, chacun ciblant un travers documenté du petit
  modèle (échecs `edit_file`, confabulation d'exécution, unix-ismes, recherche
  inutile, sur-outillage, preuve HTML).
- **A/B** : variante `old` = `git HEAD` des prompts, variante `new` = version sur disque.
- **N runs par cas** (le modèle est stochastique : un run ne prouve rien).
- **Double grader** :
  - *code* (déterministe) : marqueurs objectifs depuis la trajectoire d'outils — c'est le
    poids principal ;
  - *juge LLM* (le modèle lui-même) : note « la tâche est-elle accomplie » contre la
    rubrique du cas. Supplémentaire, faillible (petit modèle juge).

## Lancer

Le serveur modèle doit tourner (port de `loom/loom.config.toml`). Lance la stack toi-même,
puis :

```bash
# hors-ligne : valide la mécanique des graders, sans modèle
uv run python -m evals.run_eval --self-test

# éval réelle, ancien vs nouveau prompt, 3 runs/cas
uv run python -m evals.run_eval --runs 3

# variantes / options
uv run python -m evals.run_eval --runs 5 --variant new
uv run python -m evals.run_eval --runs 3 --no-judge        # graders code seuls
uv run python -m evals.run_eval --cases edit_block,html_counter
uv run python -m evals.run_eval --model qwen3.6-35b-a3b-abliterated
```

## Sortie

Tableau comparatif (runs réussis / total par cas, ancien vs nouveau) + juge moyen + tours
moyen. Détail dans `evals/out/report.json` et un transcript Markdown par run sous
`evals/out/<variante>/`.

## Lire les résultats

- Le grader **code** est l'étalon objectif. Un cas « réussi » = tous ses checks critiques
  (nom sans `_`) verts.
- Le **taux** (ex. `4/5`) compte, pas un run isolé. Forte variance = prompt sous-spécifié.
- La **preuve E2E** finale (la page tourne, le test passe) reste à constater toi-même sur
  les artefacts : le grader vérifie déjà `add(2,3)==5` (cas edit_block), mais relis les
  transcripts des cas à artefact.

## Ajouter un cas

Dans `cases.py`, ajoute un `EvalCase(id, prompt, rubric, setup, check)`. `setup(ws)` sème
le workspace temporaire ; `check(traj, ws)` renvoie `{nom: bool}` (préfixe `_` = check
informatif non bloquant). Garde les cas **bénins** (pas de shell destructeur).
