# ADR 0004 — Anti-sur-vérification par signal souple, pas par garde bloquant

- Statut : Accepté
- Date : 2026-07-08

## Contexte
Observé en éval instrumentée (baseline `1f513e7`) : sur `html_counter`, un run à
**20 tours / 230k tokens / arrêt max_iters** dont 14 `check_interactive` verts d'affilée
sur une page qui marchait dès le premier check. La sur-vérification compulsive était le
premier poste de coût anormal chiffrable.

Deux remèdes possibles :
1. compter les checks dans le détecteur de non-progrès (couper au N-ième identique) ;
2. un signal informatif dans le résultat d'outil, le modèle restant décisionnaire.

## Décision
**Signal souple dans le résultat d'outil** (option 2). Un compteur de checks navigateur
(`check_page`, `check_interactive`, `serve_and_check`) VERTS consécutifs ; remis à zéro
par tout outil qui change l'état (write/edit/append/shell/format) ou par un check raté
(échec = information nouvelle) ; les lectures n'y touchent pas. À partir du 3e check
vert d'affilée, le résultat porte : « la preuve est faite — conclus, ne re-vérifie que
si tu modifies quelque chose ». Local seulement (`strong` coupé), testé par injection
dans le self-test des évals.

## Pourquoi PAS le garde anti-non-progrès (option 1)
Les outils de vérification sont **volontairement exclus** du détecteur de non-progrès
(`_VERIFY_TOOLS`) : relancer la même preuve est légitime après un changement d'état ou
un échec (« relance jusqu'à 3 runs verts » ; coupure à tort vécue sur le test LRU).
Réintroduire les checks dans une coupe mécanique rouvrirait cette décision et
transformerait la vérification — le comportement qu'on veut ENCOURAGER — en risque
d'interruption. Cohérent avec la leçon fondatrice : fiabiliser via prompt/erreurs/outils,
jamais via un orchestrateur qui décide à la place du modèle.

## Conséquences
- Validé en éval (baseline `5606795`) : `html_counter` 8/8 runs sans déraillement,
  maximum 3 checks consécutifs, ~7 tours / 54k tokens de moyenne (vs 20/230k au pire).
- 8 runs n'excluent pas une queue rare : le mécanisme est déterministe (la note part au
  3e check), son EFFET sur le modèle reste probabiliste — surveillé par les baselines.
- Si un modèle ignorait systématiquement le signal, l'escalade serait un nudge de tour
  (patron ACT_NUDGE), toujours pas une coupe.
