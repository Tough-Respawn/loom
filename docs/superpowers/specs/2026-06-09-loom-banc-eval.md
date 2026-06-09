# Loom — Banc d'éval (design léger, MVP différé)

**Date :** 2026-06-09
**Statut :** design léger validé en principe — **construction différée** (pas la priorité ;
le skill de debug passe avant). Ce doc fige les décisions pour ne pas reperdre la réflexion.

## Objectif

Remplacer l'évaluation **n=1 au feeling** (« gemma direct → vite », « qwen → jamais un
démineur ») par un **instrument répétable** : rejouer un petit jeu de tâches sur plusieurs
**variantes** (modèle, prompt) et produire des **chiffres comparables**, pour trancher sur de
la donnée. Le modèle tourne **librement** (boucle tool-use pure, zéro bridage) ; on ne juge
qu'**après coup**.

## Décisions actées

- **Notation = juge LLM, PAS de déterministe.** Un grader déterministe rigide serait le même
  piège que le rail qu'on a retiré. Un modèle capable (le plus costaud dispo en local) lit
  l'intention de la tâche + l'artefact produit (+ une capture via `check_page` pour le web) et
  rend un score (ex. 0-3 ou pass/fail) avec justification. Variance maîtrisée en répétant.
- **Signaux objectifs gratuits** à côté du jugement (mesure, pas bridage) : a-t-il terminé,
  nombre de tours, tokens, secondes.
- **Tâches : petit jeu divers** (3-5) couvrant le généraliste : 1 artefact web (type
  démineur), 1 script/CLI, 1 fonction python, 1 tâche généraliste (Q&A/refacto). On agrandit
  ensuite.
- **Variantes : simples.** Une variante = `{nom, modèle, prompt_système optionnel}`. Un run =
  tâches × variantes × N répétitions. (Axes de compa avancés — régression par commit, configs
  arbitraires — **déférés**.)
- **Pilotage headless** via `client.stream_chat_tools` (le vrai chemin tool-use), pas la web.

## Forme pressentie (non figée — à détailler quand on construira)

- `bench/tasks/<id>/task.toml` : `prompt`, `type` (web|cli|python|qa), `intention` (critère en
  langage naturel pour le juge), `seed/` optionnel (fichiers de départ du workspace).
- Runner : pour chaque (tâche × variante × N) → workspace temp, lance la boucle, capture
  artefacts + métriques.
- Juge : modèle fort local → score + justification.
- Rapport : tableau tâche × variante → score médian + métriques.
- CLI : `python -m loom.bench run --variants … --repeat N`.

## Hors scope (différé)

Catalogue de tâches étoffé · axes de comparaison avancés · dashboard · juge multi-modèles.

## Lien

Le banc consommera idéalement le **skill de debug** (item suivant) comme l'une des tâches, et
mesurera l'effet de tout changement de harnais/prompt. Voir le spec du skill de debug une fois
écrit. [[loom-banc-eval]]
