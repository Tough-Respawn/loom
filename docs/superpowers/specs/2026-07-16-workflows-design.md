# Workflows dynamiques dans Loom

Date : 2026-07-16
Statut : validé (Python + exec, dégradation local/distant)

## Le problème

`dispatch_agent` sait déjà fan-out : le modèle émet N appels dans un tour, et
`_run_tools_parallel` les exécute en threads concurrents côté distant. Ce qu'il ne
sait pas faire, structurellement :

1. **Sortir les résultats du contexte.** Chaque synthèse de sous-agent atterrit dans
   la conversation du parent. Le contexte local est 24576 tokens (borné par 6 Go de
   VRAM) : un audit de 100 fichiers est physiquement impossible aujourd'hui. C'est le
   manque décisif, et il concerne le LOCAL autant que le distant.
2. **Boucler jusqu'à convergence.** « Continue jusqu'à ce que deux tours d'affilée ne
   trouvent rien de neuf » demande au modèle de tenir un compteur à travers des tours
   qui, chacun, mangent son contexte.
3. **Reprendre.** Un run interrompu repart de zéro.

Un workflow déplace le plan dans du code : la boucle, le branchement et les résultats
intermédiaires vivent dans des variables du script ; seul le verdict final remonte.

## Ce que ce n'est pas

Ce n'est PAS le retour de l'orchestrateur déterministe supprimé le 2026-06-04
(cf. `loom-lecon-harness-deterministe`). Ce qui bridait le modèle, c'était un
orchestrateur **fixe, écrit à l'avance**, qui imposait sa forme à toute tâche. Ici le
script est écrit **par le modèle, pour la tâche du moment**. Le modèle garde la
décision ; il l'exprime en code plutôt que tour par tour. Le tool-use pur reste le
chemin par défaut : `run_workflow` est un outil de plus, appelé quand le modèle juge
que la tâche le mérite.

`dispatch_agent` n'est pas remplacé — il est la brique que le workflow orchestre.
`agent()` dans un script EST un sous-agent.

## Architecture

### 1. Sortie structurée : l'outil de sortie

Loom n'a aucun `response_format` et n'en aura pas : un modèle local via llama.cpp ne
le supporte pas uniformément, et ça dupliquerait une validation qui existe déjà.

À la place, quand `agent(task, schema=...)` reçoit un schéma, on injecte dans le
registre du sous-agent un outil `submit_result` dont les `parameters` SONT le schéma
demandé. Le sous-agent le remplit, `validate_and_coerce` le valide et le coerce
gratuitement, la valeur est capturée et `agent()` renvoie un dict au lieu du texte.

Le schéma d'un outil est le seul mécanisme de structure que Loom possède déjà, et il
est mieux respecté qu'une consigne en prose (cf. `loom-consolidation-contexte-2026-07`
— la mécanique va dans les schémas, pas dans le prompt).

### 2. `loom/tools/agent.py` : extraire `run_subagent`

`make_dispatch_agent` contient aujourd'hui toute la machinerie (tiers, KV slot,
relève api_error) dans une closure. On l'extrait en une fonction `run_subagent(...)`
réutilisable, et `make_dispatch_agent` devient un `ToolSpec` mince par-dessus.
Comportement inchangé pour `dispatch_agent` — c'est un refactor par extraction.

### 3. `loom/workflow/runtime.py` : le moteur

`run_workflow(source, *, agent_fn, args=None, is_remote=False)` :

- **Parsing.** `ast.parse` du script. `meta` est extrait statiquement par
  `ast.literal_eval` (littéral pur exigé, comme Claude Code). Puis le corps entier est
  enveloppé dans un `FunctionDef` **au niveau AST** — pas par ré-indentation de texte,
  qui corromprait les chaînes multi-lignes. Le script peut donc faire `return`.
- **Globals injectés** : `agent`, `parallel`, `pipeline`, `phase`, `log`, `args`.
- **Pas d'async.** `agent()` bloque. C'est le vrai gain du choix Python sur JS : le
  script est du Python synchrone ordinaire, sans `await`, donc bien plus facile à
  écrire juste pour un modèle moyen.
- **Concurrence** : `parallel(thunks)` via `ThreadPoolExecutor`. `max_workers` =
  `min(16, cpu-2)` si le modèle est distant, **1 si local** — un slot llama-swap,
  règle cardinale `loom-distant-pas-limites-locales`. Même script, même sémantique,
  perf différente. C'est la condition `is_remote` qu'applique déjà
  `_run_tools_parallel`.
- **Tolérance aux pannes** : un thunk qui lève donne `None` dans la liste de résultats,
  jamais une exception qui tue le run (contrat identique à Claude Code).
- **Caps** : 16 concurrents, 1000 agents par run (backstop anti-runaway).

### 4. `loom/tools/workflow.py` : l'outil `run_workflow(path, args=None)`

Le modèle écrit le script avec `write_file`, puis appelle `run_workflow(path=...)`.

Passer le script en argument JSON obligerait un petit modèle à échapper 50 lignes de
Python dans une string — le piège d'échappement le ferait tomber. Un chemin de fichier
est trivial à émettre, réutilise `write_file`, et donne gratuitement la propriété que
la doc Claude Code met en avant : le script est un fichier qu'on peut lire, diffter et
relancer. Un workflow sauvegardé est simplement un fichier gardé.

Outil **streamant** (`run_stream`) : les events des sous-agents remontent en direct.

### 5. UI : aucun changement

Un outil streamant voit ses events passer par `_stream_tool_events` (`client.py:252`),
qui les aplatit en lignes d'activité via `_sub_activity_line` et **accumule les
`content` pour en faire le résultat**. Deux conséquences qui dictent le design :

- l'arène côté à côté n'est PAS accessible depuis un outil (elle vient d'un event
  `parallel` émis par la boucle elle-même) — la progression s'affiche donc dans le flux
  live de la pastille `run_workflow`, comme pour `dispatch_agent` ;
- les `content` des sous-agents doivent être **filtrés**, sinon leurs synthèses
  deviendraient le résultat du workflow — ce qui annulerait exactement la propriété
  recherchée. `phase()`/`log()` passent donc par `tool_call`/`tool_result`, et le seul
  `content` émis est le retour final du script.

### 6. `submit_result` doit être autorisé d'office (trouvé en E2E)

Incident observé deux fois le 2026-07-16 : `agent(schema=…)` rendait `None` pour **tous**
les agents, et le modèle en concluait « le schéma bloque » puis abandonnait la sortie
structurée. Cause racine : `submit_result` n'appartenait à aucune catégorie de
`loom/permissions.py`, donc il tombait dans la branche « outil inconnu -> ask » — **même
en mode `allow`**. Or un sous-agent tourne sans UI, et tout `ask` sans `confirm` y est
refusé par défaut (`client.py:1454`). Le résultat n'était donc jamais enregistré.

Il rejoint `READ_TOOLS` : il ne fait que déposer une valeur en mémoire pour son appelant
— ni disque, ni process, ni réseau. Rendre son résultat n'est pas une action à autoriser.

Leçon transférable : un outil INTERNE injecté dans un registre doit être classé dans la
politique de permission, sinon le défaut prudent (`ask`) le transforme en échec
silencieux sur le chemin sans UI. Deux tests de régression verrouillent ça.

### 7. Validation du schéma (durcissement séparé)

`agent(schema=…)` vérifie la forme du schéma avant l'appel API — racine `type: object`,
`properties` non vide, `required` cohérent. Un schéma malformé partirait sinon tel quel
comme `parameters` de `submit_result`, l'API rejetterait l'appel, et les agents
mourraient tous en `api_error`. Même classe de panne que ci-dessus (globale et muette),
autre voie ; ce n'est pas le correctif de l'incident.

## Hors périmètre (v1)

- **Reprise par `runId`** : demande un journal persistant des résultats d'agents.
  Réel, mais séparable — la valeur principale (résultats hors contexte, boucles) ne
  l'exige pas.
- **Isolation worktree** : Loom n'a pas de sandbox par décision (`loom-no-sandbox`).
- **`budget`** : Loom n'a pas de directive de budget par tour.
- **Workflows sauvegardés comme commandes `/nom`** : les skills couvrent déjà ce rôle.

## Sécurité

`exec()` de code écrit par le modèle n'ouvre aucune porte que `run_shell("python -c
…")` n'ouvre déjà. Le bac à sable a été supprimé délibérément (`loom-no-sandbox`) ; la
deny-list de `loom/permissions.py` reste le garde-fou assumé, et les sous-agents
héritent de la même politique de permission que le fil principal.

À dire franchement plutôt que de se le cacher : Claude Code, lui, **interdit** fs et
shell au script (« No direct filesystem or shell access from the workflow itself »).
Ici le script a les builtins entiers, donc un `open()` ou un `subprocess` écrit
directement dedans ne repasse par AUCUNE deny-list — seules les actions des ouvriers
sont contrôlées. Approuver `run_workflow` revient donc à approuver du code arbitraire,
au même titre que `run_shell`. C'est cohérent avec la posture du projet, mais ce n'est
pas la garantie qu'offre Claude Code, et l'outil est marqué `danger: True` pour ça.

## Vérification

Tests unitaires sur le runtime avec un `agent_fn` bouchonné : extraction de `meta`,
`return` au niveau du script, `parallel` concurrent vs sérialisé, `pipeline` sans
barrière, thunk qui lève → `None`, cap d'agents, script syntaxiquement faux → erreur
lisible.

Puis **test E2E réel** : un vrai modèle, un vrai script, un vrai résultat observé.
Rien n'est déclaré fonctionnel avant ça.
