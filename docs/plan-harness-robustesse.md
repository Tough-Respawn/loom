# Plan de robustesse du harness (post mortem Snake)

Snake = banc d'essai. Le morpion (3 fichiers, simple) passe ; Snake (5 fichiers, ES
modules, boucle temporelle, clavier) a révélé les vraies fragilités. Plan posé après
recul, pas de patch au fil de l'eau.

## 1. Synthèse des modes d'échec observés (runs Snake)

| # | Symptôme | Cause racine | Statut |
|---|---|---|---|
| F1 | tool-call write_file géant tronqué → 500 ×7 | tool-loop séquentiel + ctx 8192 | réglé (fan-out) |
| F2 | `erreur de planification: Expecting ',' delimiter` | snippets de code dans un design JSON | réglé (format délimité + parseur 3 niveaux + temp 0.2) |
| F3 | `game.js` tronqué (`Unexpected end of input`) | `max_tokens=2048` < taille fichier | réglé (gen DÉRIVÉ via compute_budget, P1) |
| F4 | fix → `Context size has been exceeded` | N requêtes concurrentes × gros prompt > pool `-c` partagé (kv_unified) | réglé (budget + workers DÉRIVÉS + prompt fix borné, P1) |
| F5 | fix → `Connection error` (tous les fichiers) | 4 requêtes × ~7K tok > 24576 → serveur lâche | réglé (workers dérivés + retry transitoire + last-good, P1+P3) |
| F6 | `#board` vide + `Unexpected token export` | le modèle impose des **ES modules** ; jsdom n'exécute PAS `type=module` | réglé (rejet déterministe `_check_no_es_modules`, P2) |
| F7 | run coupé en pleine révision | timeout fixe (curl) + rounds | cosmétique |

**Deux patterns de fond, pas 7 bugs indépendants :**
- **P-BUDGET** (F3,F4,F5) : aucune gestion PRINCIPÉE du budget contexte. On hand-tune
  `-c`, `max_tokens`, `max_workers` à chaque run. Tant que ce n'est pas DÉRIVÉ d'une
  formule, un fichier un peu plus gros ou un fichier de plus refait tout sauter.
- **P-VERIF** (F6) : le verifier jsdom ne reflète pas un vrai navigateur (pas d'ES
  modules, pas de canvas, exécution partielle). Et on se bat contre le modèle pour
  l'empêcher d'utiliser des modules — bataille qu'on perd. Or **dans un vrai navigateur
  (http), les modules MARCHENT**. On combat un faux problème.

## 2. Le plan (corrige les PATTERNS, pas les symptômes)

### P1 — Budget de contexte principé (tue F3/F4/F5 d'un coup) — ✓ FAIT
`compute_budget(context, n_parallel, n_files)` dérive `(max_workers, gen_max_tokens,
file_char_cap)` ; `[server] n_parallel` épinglé via `--parallel` est la source de vérité
partagée serveur↔harness ; `run_build` consomme ces valeurs et borne le prompt de fix.

Rendre les tailles DÉRIVÉES, plus jamais hand-tunées. Le serveur expose `-c` (24576) et
`n_parallel` (4) → pool partagé. Règle :
```
slot_budget   = context // n_parallel           # ex 24576//4 = 6144
gen_max_tokens = slot_budget - reserve_prompt    # ex 6144 - 2048 = 4096
max_workers    = max(1, context // (prompt_cap + gen_max_tokens))  # borne la concurrence
```
- Caper le prompt de génération/fix à `reserve_prompt` (ex 2048 tok) — déjà fait pour le
  fix, à généraliser + mesurer en tokens (approx chars/4), pas en chars.
- Exposer `context`/`n_parallel` du serveur au harness (via config) pour calculer, au lieu
  de constantes magiques dans `run_build`.
- Résultat : on ne peut PLUS déborder, quelle que soit la taille des fichiers / leur nombre.

### P2 — Vérification dans un VRAI navigateur (keystone : tue F6 ET la bataille des modules)
Remplacer/augmenter jsdom par **Chromium headless (Playwright)** :
- servir le workspace sur un http local éphémère, charger `index.html` dans un vrai
  navigateur, y rejouer les mêmes tests (rendu + clic + clavier + temps).
- Avantages : **ES modules marchent**, canvas marche, timers/clavier réels → la preuve
  « jouable » devient FIDÈLE. Et surtout : **on arrête de forcer le modèle à éviter les
  modules** (on retire la règle no-modules du prompt → moins de contraintes ignorées).
- Garder jsdom en **fallback offline** (si pas de Chromium). Le verifier choisit : browser
  si dispo, sinon jsdom (dégradé, sans le check module).
- Playwright est déjà une dépendance du projet (utilisé pour les démos).

### P3 — Boucle de fix robuste (tue les `Connection error` résiduels) — ✓ FAIT
- **Retry borné** par fichier sur erreur transitoire (connection/timeout) : 2 essais
  (`_gen_with_retry`, `transient = (APIConnectionError, APITimeoutError)`).
- **Ne jamais écraser** un fichier par une erreur : état persistant `path -> {content,
  abspath}`, un gen/fix raté garde la version précédente (last-good).
- Verify sur l'**UNION** des fichiers connus (pas seulement le dernier round) ; un fichier
  planifié manquant relance une correction (borné par `max_rounds`).

### P4 — Temps/rounds dérivés
- `max_rounds` et l'attente dépendent du nombre de fichiers (pas de constante).
- (Pas de curl --max-time arbitraire ; c'est l'UI/SSE qui pilote.)

### P5 (plus tard) — Sessions / reprise
- Persister un run (plan + fichiers + défauts) pour reprendre après coupure. Le fan-out
  étant sans état par fichier, c'est moins critique, mais utile pour les gros builds.

## 3. Séquencement recommandé
1. **P1 (budget)** + **P3 (robustesse)** : tuent toute la flakiness infra. Petit, sûr.
2. **P2 (vrai navigateur)** : le gros gain de correction — verification fidèle + fin de la
   bataille des modules. Plus gros, mais c'est LE keystone pour les apps complexes.
3. Re-tester Snake : doit converger jouable (modules OK dans un vrai navigateur).
4. P4/P5 ensuite.

## 4. Principe directeur
Arrêter de contraindre le modèle dans une boîte (no-modules, petits fichiers) pour
satisfaire un verifier faible. **Renforcer le verifier (vrai navigateur) + borner les
budgets par calcul** → le harness encaisse ce que le modèle produit, au lieu de lutter
contre lui.
