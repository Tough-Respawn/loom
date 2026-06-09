# Spec — Moteur unique Loom (fan-out durci)

> Issu d'un brainstorming + relecture adversariale multi-agent (4 lentilles : robustesse modèle faible,
> architecture, saturation contexte, migration) ancrée dans le code réel. 32 constats, 7 blockers traités.
> Voir aussi `docs/harness-strategy.md` et la memory `harness-design-anthropic`.

## 0. Décisions finales (arbitrées avec l'utilisateur)

| Sujet | Décision |
|---|---|
| Fusion des deux moteurs | **Oui**, sur la base fan-out (`run_build`). `run_pipeline` reste deprecated derrière `mode=='pipeline'`, retrait en PR finale isolée. |
| Déclencheur `rewrite` | **Objectif** : rewrite seulement si le fichier existant **échoue déjà le verify**. Le 4B ne juge jamais « dégueulasse » seul (la review l'a écarté). |
| Périmètre | **Web + Python** (tout ce que `verify.py` sait checker déterministiquement). `review_semantic` = gate **soft** par-dessus. |
| best-of-N | **Réparation + fichiers d'ancrage en passe 1** (N>1 sur ce dont d'autres dépendent, N=1 sur feuilles). |
| Garde-fou `rewrite` | `> N lignes`, **N = 200** par défaut. |
| Stop anti-divergence | comparaison sur l'**ensemble des `location`** des défauts (pas `len`). |
| `build_step_messages` | conservé **uniquement si** le handoff EXPLORE→PLAN le réutilise, sinon supprimé avec ses tests. |

## 1. Objectif & contexte

Loom est un assistant IA **local, offline**, piloté par un modèle **faible** (Gemma 4B, ctx **8192**). Principe directeur (`docs/harness-strategy.md`) : le scaffolding est **proportionnel à la faiblesse du modèle** — on tient le modèle en laisse courte (contraintes machine, pas prompt) et on lui rend des erreurs **exploitables**.

But de ce spec : **fusionner** les deux moteurs existants en **un seul**, bâti sur la base fan-out de `run_build` (`loom/orchestrator.py:172`) :
- `run_pipeline` (séquentiel plan→code→review, tool-use, reviewer LLM, gate texte `is_blocking`) — `orchestrator.py:27` ;
- `run_build` (fan-out par fichier, gate 100% déterministe) — `orchestrator.py:172`.

Le moteur fusionné ajoute : EXPLORE (ground truth), mode par fichier (create/patch/rewrite), edit ciblé, best-of-N en réparation, reviewer sémantique non bloquant. **La review adversariale a montré que plusieurs de ces ajouts, mal bornés, ré-introduisent exactement les trous de contexte/robustesse que le fan-out avait fermés.** Ce spec les borne chiffré.

## 2. Décisions verrouillées (les 5)

1. **Un seul moteur**, bâti sur le fan-out. `run_pipeline` n'est **pas supprimé immédiatement** : il survit derrière le flag `mode=='pipeline'` (déjà existant, `app.py:288/316`) marqué *deprecated*, et sa machinerie tool-use (`stream_chat_tools` + `permission`/`confirm`) est **réemployée par EXPLORE**. Retrait final en PR séparée une fois la parité prouvée (cf. §9).
2. **Mode par fichier** `create | patch | rewrite`, **dérivé déterministiquement par le harness** (pas porté par le texte du plan).
3. **Lecture auto OU outillée** selon la tâche, **routage déterministe** (pas un méta-jugement du 4B).
4. **best-of-N en réparation**, **plus extension aux fichiers d'ancrage en passe 1** (cf. §8). N=1 ailleurs.
5. **Reviewer sémantique réformé non bloquant**, **gate = verify déterministe seul**, plus de `VERDICT: OK` textuel.

## 3. Architecture du pipeline

```
TASK
 │
 ├─[EXPLORE] (optionnel, routage DÉTERMINISTE)
 │     précise (paths explicites sur disque)  → lecture AUTO bornée de ces paths
 │     vague  (workspace non vide, sinon skip) → boucle outillée stream_chat_tools BORNÉE
 │     SORTIE = résumé ground-truth (paths + signatures + flag verify-fail/par fichier)
 │              JAMAIS le contenu brut réinjecté
 │
 ├─[PLAN] 1 appel → contrat design (greenfield) OU design + résumé EXPLORE (brownfield, borné)
 │     → list[FileSpec(path, role)]   (mode N'est PAS parsé ici)
 │
 ├─[DÉRIVATION MODE]  (harness, déterministe, post-plan, SANS LLM)
 │     existe pas sur disque         → create
 │     existe + verify échoue déjà   → rewrite (déclencheur OBJECTIF)
 │     existe + verify OK            → patch   (défaut le moins destructeur)
 │
 ├─[GÉNÉRATION] fan-out parallèle, 1 appel isolé/fichier (budget mesuré, §6) :
 │     create  → generate_one          (fichier entier)        [best-of-N si ANCRAGE]
 │     patch   → edit_one NOUVEAU      (read→diff→apply→fallback rewrite)
 │     rewrite → generate_one borné    (garde-fou > 200 lignes)
 │
 ├─[VÉRIFIE] verify_files déterministe = LE GATE (seul à valider/bloquer)
 │
 ├─[REVIEW SÉM.] reviewer LLM OPTIONNEL, SEULEMENT si verify déjà vert
 │     → Defect(kind='semantic') dans une LISTE SÉPARÉE. Ne bloque/débloque JAMAIS.
 │
 ├─ défauts_det = verify ; défauts_sem = review (séparés)
 │     si défauts_det vide ET incomplete=False → DONE
 │
 └─[FIX] fan-out sur les SEULS fichiers en défaut + rapport agrégé
       best-of-N candidats/fichier, verify_syntax_file garde le valide
       │
       └─ retour [VÉRIFIE]
          STOP si: round >= max_rounds  OU  ensemble des location ne DÉCROÎT PAS (anti-divergence)
          défauts_sem : AU PLUS 1 passe de fix, ne maintiennent JAMAIS la boucle ouverte
```

## 4. Unités & frontières

| Nom | Rôle | Signature | Dépend de | Statut |
|---|---|---|---|---|
| `explore` | Ground truth bornée (auto ou outillée) | `explore(client, task, workspace, *, model, budget) -> ExploreResult` (résumé + `{path: hint}`) | `stream_chat_tools`, `read_file` (max_bytes EXPLORE bas), budget pur sur `list[dict]` | **nouveau** |
| `plan_files` | Contrat design + liste fichiers | `plan_files(client, task, *, model, max_tokens, explore_summary='') -> (design, list[FileSpec])` | `_parse_plan` (inchangé) | **étendu** (param `explore_summary`, prompt brownfield séparé) |
| `derive_modes` | Calcule mode/fichier (déterministe) | `derive_modes(specs, workspace, verifier) -> list[PlannedFile]` | `os.path.exists`, `verifier` | **nouveau** |
| `_complete_file` | Squelette partagé appel-modèle + `extract_code` | `_complete_file(client, prompt, *, model, max_tokens) -> str` | `client.complete`, `extract_code` | **nouveau (factorisation)** |
| `generate_one` | Fichier entier (create/rewrite) | `generate_one(client, design, spec, all_paths, *, model, max_tokens) -> (path, content)` | `_complete_file`, `_file_prompt` (+ clip, §6) | **étendu** (clip prompt) |
| `edit_one` | Patch ciblé read→diff→apply→fallback | `edit_one(client, design, spec, workspace, *, model, max_tokens, file_char_cap) -> (path, content)` | read disque, logique `edit_file`, fallback `generate_one` | **nouveau** |
| `fix_one` | Régénère fichier corrigé | `fix_one(client, design, spec, current, defects, *, model, max_tokens, file_char_cap) -> (path, content)` | `_complete_file`, `_fix_prompt` | **inchangé** (passe par `_complete_file`) |
| `review_semantic` | Défauts sémantiques (optionnel) | `review_semantic(client, design, current_files, *, model) -> list[Defect]` (pur, sans I/O, `kind='semantic'`) | `client.complete` | **nouveau** |
| `verify_files` | Gate déterministe | inchangé (`verifier(abs_paths) -> VerifyReport`) | `verify.py` | **inchangé** |
| `compute_budget` | Budget dérivé | `reserve_prompt_tokens` devient **mesuré** (§6) | `estimate_tokens` | **étendu** |

**Non touché** : `verify.py` (gate), UI Preact (events SSE, voir §9 matrice), `_parse_plan` (3 niveaux + tests verts).

**FileSpec** reste `(path, role)`. Le mode est porté par un type distinct `PlannedFile(spec, mode)` produit par `derive_modes` **après** `plan_files` — n'altère ni le dataclass ni les 3 chemins de `_parse_plan` (`parallel.py:88-121`), ni `test_parse_plan_*`.

## 5. Stratégie par fichier create/patch/rewrite

Règle de dérivation (déterministe, `derive_modes`, **sans LLM**) :

| Signal disque | Signal verify | Mode | Justification |
|---|---|---|---|
| `Path(workspace/path).exists()` == False | — | **create** | 100% fiable (`os.path.exists`) |
| exists + verify FAIL | syntaxe/runtime cassée | **rewrite** | déclencheur **objectif**, pas esthétique |
| exists + verify OK | — | **patch** | défaut le **moins destructeur** |

**Écarté (validé par l'utilisateur) : « rewrite = jugement EXPLORE dégueulasse ».** Confier un jugement esthétique subjectif au maillon le plus faible (4B) sur un fichier non trivial → rewrite qui casse les contrats inter-fichiers que patch préservait. Le hint EXPLORE peut **suggérer** rewrite, mais ne le déclenche que si verify échoue déjà. En cas de doute → **patch**.

Garde-fous :
- **edit_one (patch)** — unité à 2 temps **dans le harness** :
  1. **read déterministe** du fichier cible → contenu exact passé au modèle (borné `<= file_char_cap/2`, §6) ;
  2. modèle renvoie `old_string`/`new_string` ;
  3. application via la logique `edit_file` (`fs.py:117-137`) avec ses erreurs exploitables (n° de ligne sur ambiguïté, hint CRLF) ;
  4. **fallback** : si `old_string` introuvable/ambigu → **bascule automatique en `generate_one` borné** (dégénère vers rewrite). Jamais de fichier non patché silencieux.
  - **Contrat d'état** : `edit_one` **relit le fichier après edit** et renvoie `(path, new_full_content)` pour peupler `state[path] = {content, abspath}` **comme generate_one**. Sans ça, `_verify_phase` (union), `_incomplete()` et le last-good ne capturent pas les patches → boucle infinie.
- **rewrite** — `generate_one` avec **garde-fou `> 200 lignes`** (refuse/dégrade la réécriture intégrale d'un gros fichier) + passage par `verify_syntax_file` post-write (déjà branché dans `make_write_file`, `fs.py:44`).

## 6. Gestion du contexte & bornes (chiffrées)

Cible : `prompt + génération <= 0.9 · 8192` **par requête**, multi-slot KV unifié partagé.

**6.1 `compute_budget` — coût prompt mesuré (corrige reserve fixe 2048).**
- `reserve_prompt_tokens` ne doit **plus être une constante**. En brownfield, calculer `reserve = max(estimate_tokens(design), estimate_tokens(plus_gros_fichier_injecté))` (`context.estimate_tokens`, `context.py:13`) **après** connaissance des tailles, et recalculer `gen_max_tokens` ensuite.
- Conséquence (risque R1) : brownfield force souvent `n_parallel` effectif = 1, `gen_max_tokens` plafonné bas.

**6.2 Clip prompt côté GÉNÉRATION (corrige `generate_one` sans clip).**
- `_file_prompt` (`parallel.py:189`) doit **clipper le contenu injecté à `file_char_cap`** comme `_fix_prompt` le fait déjà (`parallel.py:251-285`). Aujourd'hui seul `fix_one` clippe → brownfield déborde silencieusement.

**6.3 edit_one — fenêtre, pas fichier entier.**
- Contenu injecté **`<= file_char_cap/2`** (borne dure). Si le fichier dépasse, extraire une **fenêtre** (offset/limit autour de la zone) ou un patch ciblé sur symbole. **Jamais** les 200 KB de `read_file` (`read.py:33`, `loom.config.toml:64`) = ~50k tokens = 6× le contexte.
- **CRLF-exactness** : le contenu passé au modèle doit être byte-exact (pas de normalisation), sinon tout `old_string` devient introuvable (`fs.py:119`).

**6.4 EXPLORE outillée — borne dure (corrige `stream_chat_tools` sans budget).**
- `max_iters` EXPLORE **≤ 3** (pas 8, `client.py:187/202`).
- `read_file` EXPLORE : `max_bytes` dédié **8–16 KB** (param distinct du 200 KB global).
- Avant chaque tour : budget pur sur le `convo` local. `summarize` (`context.py:65`) **n'est PAS réutilisable** tel quel (opère sur `Conversation`). → **extraire** la logique de budget en fonction pure sur `list[dict]` réutilisant `estimate_tokens`/`conversation_tokens` (déjà génériques, `context.py:42-50`).
- Ne réinjecter que des **previews** de `read_file`, pas le fichier entier (en préservant l'appariement `tool_call_id ↔ tool`).
- Garde-fou dur : `conversation_tokens > 0.6 · context` → **stop EXPLORE, passe au PLAN avec ce qu'on a**.
- **Défaut = EXPLORE OFF** (lecture auto). La boucle outillée ne s'active que sur routage déterministe (§6.5).

**6.5 Routage EXPLORE (déterministe, corrige le méta-jugement).**
- *précise* = la requête mentionne des **paths explicites présents sur disque** → lecture auto de ces paths.
- *vague* = sinon, **et seulement si** le workspace contient déjà des fichiers pertinents → boucle outillée bornée.
- **On ne demande jamais au modèle « cette tâche est-elle vague ? »**

**6.6 PLAN brownfield.**
- Séparer le prompt PLAN brownfield du gabarit greenfield (`parallel.py:133-168` est 100% greenfield, max_tokens=2048 déjà consommé par les consignes).
- PLAN brownfield reçoit le **résumé EXPLORE borné `<= 1500 tok`** (paths + signatures, pas les fichiers entiers).

**6.7 best-of-N — concurrence.**
- Les N candidats d'un fichier sont joués **séquentiellement DANS un worker**, pas ajoutés à la concurrence. Invariant : **nb requêtes simultanées (workers) <= fit** de `compute_budget` ; N n'entre pas dans la concurrence.

## 7. Reviewer sémantique réformé

**Rôle** : répondre « ça fait VRAIMENT le bon truc ? » → défauts **sémantiques** que le verify déterministe (syntaxe/ESM/runtime jsdom, `verify.py:252-273`) est aveugle à détecter.

**Signature pure, testable isolément** : `review_semantic(client, design, current_files, *, model) -> list[Defect]`, sans I/O ni état.

**Ce qu'il NE fait PAS** :
- Il **ne débloque jamais** (pas un gate) et **ne bloque jamais** non plus.
- Il **ne s'active QUE si verify déterministe est déjà vert** (sinon le bruit sémantique noie les vrais défauts).
- Ses Defect portent `kind='semantic'` et vont dans une **liste séparée `semantic_defects[]`**, jamais mélangés à `verify_defects[]`.
- Les défauts sémantiques déclenchent **AU PLUS UNE** passe de fix et **ne maintiennent jamais la boucle ouverte**. La condition de continuation est pilotée par les **défauts déterministes uniquement** — sinon un 4B reviewer (faux positifs garantis) régénère indéfiniment des fichiers valides.

**Périmètre Web + Python (décidé)** : le moteur couvre tout ce que `verify.py` checke déterministiquement (web jsdom/DOM **et** Python `py_compile`). Pour une tâche **sans artefact exécutable** (rédactionnel pur, `verify` renvoie `None`), `review_semantic` sert de **gate soft documenté** (signale, ne bloque pas) — seul filet restant après suppression du `is_blocking` textuel.

## 8. best-of-N en réparation + ancrage

| Aspect | Règle |
|---|---|
| **Déclencheur principal** | FIX (réparation) |
| **Extension (décidée)** | **Aussi** sur le(s) fichier(s) d'**ancrage en passe 1** (typiquement `index.html` / le fichier définissant ids/sélecteurs DOM, référencé par d'autres dans `all_paths`). Critère : **N>1 sur les fichiers dont d'autres dépendent**, N=1 sur les **feuilles**. Motif : un ancrage cassé en passe 1 contamine tous les fichiers parallèles qui ont déjà pris sa référence ; `_check_no_es_modules` existe précisément parce que la passe 1 produit régulièrement de l'ESM interdit. |
| **N** | petit (**2**), joué **séquentiellement** dans le worker (§6.7) |
| **Sélection** | `verify_syntax_file` (`fs.py:44`) garde le **premier candidat valide** ; si aucun valide, garde le last-good |

## 9. Plan de migration incrémental & impact tests

**Base = `run_build`. `run_pipeline` reste branché sous `mode=='pipeline'` (deprecated) jusqu'à l'étape finale.** Chaque PR verte avant la suivante (214 tests existants à ne pas casser d'un bloc).

| PR | Contenu | Tests |
|---|---|---|
| 1 | `PlannedFile(spec, mode)` + `derive_modes` (exists/verify-fail) | nouveaux tests dérivation ; `_parse_plan`/`test_parse_plan_*` **inchangés** |
| 2 | `edit_one` + contrat `state[path]` + fallback rewrite | test edit OK / fallback / état peuplé |
| 3 | `rewrite` borné + garde-fou 200 lignes | test garde-fou |
| 4 | `compute_budget` mesuré + clip `generate_one` (§6.1-6.2) | test budget brownfield |
| 5 | `explore()` borné (cap 8192, §6.4-6.6) + extraction budget pur `list[dict]` | test bornes EXPLORE |
| 6 | `review_semantic` optionnel non-bloquant + **handler app.js** + liste séparée | test review pur + test UI event |
| 7 | best-of-N réparation + ancrage (§8) | test sélection candidat valide |
| 8 | stop anticipé non-décroissance (ci-dessous) | 2 tests (stagne→stop ; décroît→max_rounds) |
| **9 (finale, PR isolée)** | retrait `run_pipeline` + `is_blocking`/`is_reviewer`/`build_step_messages` + `[[agents]]` orphelins + `mode=='pipeline'` | mapping test par test ci-dessous |

**Stop anticipé non-décroissance (NOUVEAU, absent du code actuel).** `run_build` boucle sur `while rounds < max_rounds and (not report.ok or _incomplete())` (`orchestrator.py:344-346`), **sans comparer les défauts**. À ajouter : conserver l'**ensemble des `location`** du round précédent ; si `set(location) actuel` n'est **pas un sous-ensemble strict** du précédent (un fix peut résoudre A et créer B à `len` égal) → **arrêter**. `max_rounds` bas (**2-3**).

**Mapping des 11 tests `test_run_pipeline_*` (test_orchestrator.py:52-344,551)** — à statuer en PR 9 :
- ordre d'agents (l.63), max_tokens override (l.108), thinking par agent (l.171/190), routage tools vs plain (l.127), propagation content-pas-reasoning (l.70) → **supprimés** (spécifiques au pipeline séquentiel) ;
- gate verify > texte reviewer (l.228), boucle review bornée (l.302/551) → **réécrits pour le fan-out** (gate verify-only + stop non-décroissance les remplacent) ;
- `is_blocking`/`is_reviewer` (test_agents.py:16-45) + `build_step_messages` (l.75-95) → **supprimés avec leurs tests** (sauf si EXPLORE/handoff réutilise `build_step_messages`). Suppressions **attendues, pas des régressions**.

**Orphelins à traiter en PR 9** : route `/run mode=pipeline`, config `[[agents]]` + `default_pipeline` + `max_revisions`, `resolve_agents`/`compose_agent_system_prompt`/`AgentRun.steps`/`RunStep.written`, et la **seule voie** branchant `stream_chat_tools`+`permission`+`confirm`. **EXPLORE doit explicitement réutiliser `stream_chat_tools`+`permission`+`confirm`** (décision 1) — sinon code mort. Conserver une source `build_model` **indépendante de `[[agents]]`** (ex. `models[0]`), car `selected[0].model` (`app.py:289`) en dépend encore.

**Matrice events SSE × phase (l'affirmation « mêmes events SSE » du design était partiellement FAUSSE).** `run_pipeline` émet `reasoning`/`usage`/`tool_call`/`tool_request` que `run_build` n'émet pas ; `app.js:447-509` n'a **pas de branche `default`** → tout event inconnu est **silencieusement ignoré**. Règle : **réutiliser les events existants, ne pas en créer de non-rendus.**

| Phase | Events émis (existants, déjà rendus app.js) |
|---|---|
| EXPLORE | `agent_start`(role='Exploration')/`agent_done` + `tool_begin`/`tool_request`/`tool_result`/`reasoning` |
| PLAN | `agent_start`/`content`/`agent_done` |
| GÉN/FIX | `agent_start`/`tool_begin`/`tool_result`(name='write_file'\|'edit_file')/`agent_done`/`revision` |
| VERIFY | `verify_start`/`verify` `{ok, defects:[{location,kind,evidence}]}` |
| REVIEW SÉM. | `agent_start`(role='Relecture')/`content` + défauts via event `verify`-like réutilisé **OU** nouvel event **avec handler app.js ajouté en PR 6** |

**Test route /run manquant** (`test_web.py` ne couvre ni `run_build` ni `run_pipeline`). Avant la fusion : ajouter un test d'intégration `/run` mode build (fakes client/write/verifier) figeant le contrat SSE + le câblage `workspace=` (`app.py:294-311`).

## 10. Risques résiduels

- **R1 — Brownfield = n_parallel effectif 1.** Mesurer le prompt réel (§6.1) force souvent la sérialisation. Acceptable, à documenter (perf brownfield < greenfield).
- **R2 — edit_one fallback rewrite peut casser des contrats.** Un patch qui dégénère en rewrite reprend le risque destructeur de §5 ; le garde-fou 200 lignes le borne sans l'éliminer.
- **R3 — review_semantic faux positifs.** Bornés à 1 passe + hors condition de boucle, mais un faux défaut peut provoquer 1 régénération inutile. Acceptable, pas nul.

## 11. Critères d'acceptation

1. `derive_modes` ne consulte **jamais** le LLM ; rewrite **uniquement** sur verify-fail. *(test)*
2. `edit_one` peuple `state[path]={content, abspath}` identiquement à `generate_one`. *(test)*
3. `edit_one` avec `old_string` introuvable/ambigu → bascule rewrite, **aucun fichier non patché silencieux**. *(test)*
4. Aucune requête de génération/fix n'injecte > `file_char_cap` chars ; `generate_one` clippe comme `fix_one`. *(test)*
5. EXPLORE outillée : `max_iters <= 3`, `read_file` EXPLORE `<= 16 KB`, stop dur à `0.6·context`, previews seulement. Aucune saturation 8192 sur 3 fichiers moyens. *(test)*
6. Routage EXPLORE 100% déterministe, aucun appel « tâche vague ? » au modèle. *(test)*
7. `review_semantic` : signature pure, `kind='semantic'`, liste séparée, actif **seulement** si verify vert, hors condition de boucle. *(test)*
8. Boucle FIX : stop si l'ensemble des `location` ne décroît pas ; `max_rounds <= 3` ; défauts sémantiques ≤ 1 passe. *(2 tests)*
9. best-of-N : N candidats **séquentiels** dans le worker (workers <= fit) ; ancrage N>1, feuilles N=1 ; `verify_syntax_file` sélectionne le valide. *(test)*
10. `run_pipeline` reste accessible via `mode=='pipeline'` jusqu'à la PR finale ; chaque PR 1-8 verte avant la suivante. *(CI)*
11. UI inchangée : tout event émis a un handler dans `app.js`. *(test UI)*
12. Test d'intégration `/run` mode build présent et vert avant la fusion. *(test)*
