# Archive — specs/plans de fonctionnalités RETIRÉES

Ces documents décrivent du code qui **n'existe plus** dans Loom. Ils sont conservés comme
trace des décisions, pas comme doc active. Ne pas s'y référer pour l'état courant.

- **`2026-06-03-loom-moteur-unique*`** — moteur de *build* déterministe (plan→code→review,
  pipeline multi-agent, vérificateur, fan-out). Supprimé le 2026-06-04 : trop étroit (web
  uniquement), bridait le modèle. Loom = boucle tool-use pure.
- **`2026-06-07-harnais-reflexion*`** et **`2026-06-07-complete-program*`** — rail de
  réflexion (décompose→exécute→vérifie→intègre, `submit_spec`, gates, preuve déterministe).
  Supprimé le 2026-06-09 : la donnée a montré qu'il desservait le modèle (×12 le coût, 0
  résultat). Voir le commit « refactor: retire le rail de réflexion ».

État courant : voir les specs/plans **non archivés** sous `docs/superpowers/` + `ETAT_PROJET.md`.
