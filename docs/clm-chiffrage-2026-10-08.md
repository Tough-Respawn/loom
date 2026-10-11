# Étude CLM (Context Language Models) : chiffrage du coût d'une édition de contexte

Date : 2026-10-08. Papier : « Context Language Models », Meta Superintelligence Labs, UW, MIT, Trillium Labs,
arXiv 2609.37725 (29/09/2026). Principe : le contexte vivant du modèle est un fichier que le modèle édite
librement ; chaque édition est synchronisée dans son contexte au tour suivant. Côté serveur, les auteurs ont
ajouté à SGLang un « Suffix Cache Reuse » (réutilisation du KV des tokens survivants après une édition).

Question posée ici : que coûte une édition de contexte avec llama-server tel qu'il est, sur les modèles de
Loom, avant de bâtir un mode CLM dans Loom ?

## Protocole

- Binaire : `C:/tools/llama-loom-b11507/llama-server.exe` (master c811cb8f0, PR #26004 fusionnée).
- Modèle : ornith-1.5-35b-a3b Q8_0 (hybride récurrent, `--cpu-moe`, RTX 2060, flags de Loom via llama-swap).
- Payload : une VRAIE requête Loom capturée (`LOOM_DUMP_REQUESTS`, hook ajouté dans `loom/agent/calltrace.py`)
  au cours d'une session à outils : 43 messages, 19 résultats d'outils, 27 schémas d'outils, system prompt de
  17,8k caractères, 22 311 tokens au total. Préfixe stable (system + schémas) : 10 374 tokens.
- Rejeu direct sur le slot 1 (`id_slot = 1`, `max_tokens = 1`), chaque variante précédée d'un renvoi du
  payload de base pour repartir d'un cache plein. Mesure : `timings.prompt_n` (tokens re-préfillés) et
  `prompt_ms` renvoyés par llama-server. Script : scratchpad `replay_edit_cost.py`.

## Résultats (ornith, contexte de 22 311 tokens)

| Variante | Position approx. de l'édition | Tokens re-préfillés | Temps |
|---|---|---|---|
| Ajout sans édition (tour suivant) | fin | 4 | 0,4 s |
| Un mot ajouté au dernier message utilisateur | ~19,3k (3,1k tokens le suivent) | 3 097 | 13,5 à 24 s |
| Un résultat d'outil du milieu remplacé par une note | ~15,6k | 11 931 | 56 à 77 s |
| Le plus ancien résultat d'outil remplacé | ~10,9k | 11 935 | 52 à 72 s |
| Tous les résultats d'outils vidés sauf les 2 derniers (style microcompact) | ~10,9k | 8 422 | 36 à 57 s |
| Un mot changé dans le system prompt | 0 | 22 312 | 97 s |

Débit de prefill observé entre 130 et 230 tokens/s selon la longueur et la passe (prompt court mesuré à
258 t/s le même soir). Deux séries complètes (`replay_ornith.jsonl`, `replay_ornith_cms2048.jsonl`), mêmes
tokens re-préfillés à 50 tokens près.

## Pourquoi l'édition du milieu coûte autant que la plus ancienne

Sur un modèle hybride, le serveur ne peut pas rembobiner l'état récurrent : il recule jusqu'au dernier
checkpoint situé AVANT le point d'édition et re-préfille tout ce qui suit. Or la politique actuelle de
llama-server (lue dans `server-context.cpp`, PR #24176 de juin 2026 et #20288 de mars 2026) ne crée des
checkpoints qu'à deux endroits :

1. au début de chaque message utilisateur (espacés d'au moins `--checkpoint-min-step`) ;
2. près de la fin du prompt (à `4 + n_ubatch` et à `4` tokens de la fin).

Les checkpoints « au milieu » d'une boucle d'outils sont sautés volontairement. Dans le rejeu, les
checkpoints observés étaient à 10 374 (fin de l'amorçage), 19 218 (début du dernier message utilisateur),
20 259 et 22 307 (fin du prompt). Toute édition entre 10 374 et 19 218 recule donc à 10 374, soit 11,9k
tokens à refaire. Passer `checkpoint_min_step` de 8 192 à 2 048 n'a rien changé (vérifié, réglage retiré).

`--cache-reuse` (réutilisation par décalage de KV après une divergence) n'aide pas non plus : il est désactivé
par le serveur quand un mmproj est chargé (ornith, qwen) et sur les contextes qui ne savent pas décaler
(SWA et hybrides : gemma4, ornith, qwen). Sur le parc de Loom, une édition au milieu = re-prefill de tout ce
qui suit le checkpoint précédent, environ une minute par édition à 22k de contexte.

## Conséquences pour un mode CLM dans Loom

- Les éditions proposées par le papier (vider de vieux résultats, réécrire un tableau de bord en place)
  tombent presque toutes « au milieu » d'une boucle d'outils. Chez nous, chacune coûte ~1 min sur ornith.
  Le papier compte 1,4 à 2,6 éditions par tâche : 1 à 3 minutes par tâche longue.
- Deux éditions restent bon marché : à la fin du contexte (après le dernier résultat d'outil) et au début
  d'un tour utilisateur (un checkpoint y existe toujours). Une stratégie « compacter au début du tour,
  avant le nouveau message utilisateur » recule au checkpoint de ce tour, pas plus loin.
- Le préfixe stable (system + schémas, 10,4k tokens ici) ne doit jamais être touché : 97 s.

## Ce que llama.cpp pourrait offrir (côté serveur)

1. **Checkpoints à d'autres frontières** : option pour en créer aussi au début de chaque résultat d'outil
   (ou tous les N tokens), bornée par `--ctx-checkpoints`. Coût : 62,8 Mio de RAM et ~25 ms par checkpoint
   sur ornith. Changement local à la boucle de prompt de `server-context.cpp`. C'est le levier le moins cher
   et il suffit à ramener le coût d'une édition au milieu de « tout depuis le tour précédent » à « depuis le
   résultat d'outil précédent ».
2. **Réutilisation du suffixe** (l'équivalent du Suffix Cache Reuse) : décalage du KV des couches
   d'attention pour les tokens survivants + reprise de l'état récurrent depuis le checkpoint précédent
   (approximation acceptée par le papier pour les couches linéaires). Gros chantier, à proposer en
   discussion upstream avec ces chiffres plutôt qu'en PR directe.

## Ce que coûte la compaction actuelle, en vraies sessions (mesuré le 2026-10-09)

Recoupement sur les 19 sessions conservées dans `var/sessions` (debug.log, `prefixe=diverge` dans
`call.request` = cache KV perdu à partir d'un point, `prefill_tok` de l'appel = tokens refaits) :

| Mesure | Valeur |
|---|---|
| Appels modèle (tours) | 70 |
| Divergences de préfixe | 3 |
| La plus chère (compaction au début de la conversation, ornith) | 8 118 tokens re-préfillés, 49,7 s |
| Événements de microcompaction dans les debug.log | 0 |

La compaction de Loom est une édition « au milieu » au sens CLM et paie le même prix (49,7 s observés),
mais elle ne se déclenche presque jamais sur ces sessions.

## Verdict

Pas de mode CLM dans Loom pour l'instant. Avec llama-server tel qu'il est, le papier généralise
exactement l'opération la plus chère sur ce matériel (1 à 3 minutes par tâche longue sur ornith), sans
gain mesuré en face. La part du concept qui reste bon marché, éditer à la frontière d'un tour utilisateur,
est déjà couverte par les notes (`write_note`) et les todos ré-émis à chaque tour.

Condition pour rouvrir : le levier 1 ci-dessus mergé upstream (checkpoint au début de chaque résultat
d'outil). Les chiffres de ce doc sont la matière pour le proposer en discussion.

## Note annexe : le checkpoint en fin de dernier message utilisateur est déjà sur master

Couvert par #24176 (checkpoint au début de chaque message utilisateur) et #20288 (deux checkpoints en
fin de prompt). Observé dans Loom : après une reprise à chaud, la requête suivante repart de 10 398 pour
un contexte de 10 448, soit 40 tokens re-préfillés. Pas de PR de suivi à ouvrir sur ce point.

## Fichiers

- Hook de capture : `loom/agent/calltrace.py`, `LOOM_DUMP_REQUESTS=<dossier>` (une copie JSON de chaque
  requête envoyée au modèle, tout `kwargs` sauf `stream`). Le dump contient le prompt complet, mémoire
  SOUL/USER et fichiers lus compris : données personnelles, jamais dans le dépôt.
- Script de rejeu : `evals/clm/replay_edit_cost.py` (`uv run python evals/clm/replay_edit_cost.py
  <dump.json> [port] [slot]`).
- Mesures archivées : `evals/clm/replay_ornith.jsonl` et `evals/clm/replay_ornith_cms2048.jsonl`
  (deux séries complètes, la seconde avec `checkpoint_min_step` 2048, réglage retiré depuis).
