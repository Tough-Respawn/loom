# CHANGELOG — Loom

<!-- RÔLE : historique des versions (commits, fonctionnalités, corrections, refactoring). Pitch public : README.md. Suivi interne : ETAT_PROJET.md. Carte technique : loom.md. -->

> Commits du 2026-04 à 2026-06 — `feat/loom-memory-foundation` (HEAD), `master` à `be69fb6`.

---

## 2026-07-31 — éclatement des god-files

### Refactoring
- **Trois god-files éclatés par domaine, à comportement constant** (branche `refactor/god-files`, mergée en fast-forward) :
  - `loom/agent/client.py` **3014 → 1353 L** : fonctions libres sorties en modules cohésifs (`debuglog`, `errors`, `toolsets`, `compaction`, `guards`, `streaming`, `toolrun`), layering acyclique. `LoomClient` + réexports gardés en façade dans `client.py`.
  - `loom/web/routes.py` **4003 L → package `loom/web/routes/`** : `helpers` (cœur partagé) + `config`/`skills`/`chat`/`models`/`misc`/`sessions`/`soul`. Cycle `models↔chat` rompu par import tardif dans `_register_model_routes` ; `_index_context` basculé en `skills`.
  - `loom/web/static/app.js` **4144 → 1313 L** : ES modules `render`/`state`/`components`/`sse`/`chat`/`tabs`/`panes`/`panels` + `shared.js` (5 variables mutables partagées via setters, seule façon de réassigner un binding importé en ESM).
- **Méthode** : découpage mécanique piloté par AST (Python) / acorn (JS), imports croisés calculés, cycles détectés avant écriture. Filet : 501 tests pytest verts après chaque split ; app.js vérifié en navigateur réel (chat, palette, drawer, compteurs, 0 erreur console).
- **Hygiène** : `audit-dette.md` déplacé dans `docs/`, 2 lambdas assignées converties en `def` (ruff `E731`), ruff `loom` clean.

---

## 2026-07-18 — boot remote-only

### Améliorations
- **`loom.web` boote sans modèle local** quand un modèle DISTANT existe (`[[remote_models]]` de `config/local.toml` OU store UI `var/remote_models.json`) : `load_config` tolère `models=[]` dans ce cas (le serveur llama.cpp local ne sert que les locaux et démarre à la demande), `default_model` retombe sur le premier distant si le défaut configuré pointe un local disparu. Machine vraiment vierge (ni local ni distant) : comportement inchangé — installeur guidé (`maybe_bootstrap`) ou `ValueError` claire.
- La fusion du store UI (`var/remote_models.json`) dans `cfg.remote_models` vit désormais dans `load_config` (plus dans `build_app`) : une seule source pour tous les consommateurs, y compris le reload à chaud ; dérivation du chemin centralisée (`config.remote_store_path`).
- `serve.py` (moteur llama.cpp) continue d'exiger un modèle local : garde explicite « aucun modèle LOCAL à servir » (code 1) au lieu d'un crash, `cfg.model`/`model_by_id` lèvent une erreur claire sur parc local vide.

---

## 2026-07-18 — loom-setup

### Nouvelles fonctionnalités
- **`loom-setup`** : installeur console guidé (`uv run loom-setup`) — détecte OS/GPU/VRAM/RAM, télécharge la dernière release stable de llama.cpp (build CUDA/Vulkan/CPU selon la machine, DLL cudart incluses) dans `var/runtime/llama/<tag>/`, écrit `[server] bin` dans `config/local.toml`, puis propose un premier modèle qui fit le budget mémoire. Chaque action lourde est expliquée et confirmée (`--yes` pour tout accepter) ; bilan ✅/⏭️/❌/🔧 + journal `var/logs/setup.log`. Idempotent : relançable, ne refait que ce qui manque.
- **Shortlist de modèles par palier de budget** : le catalogue porte des FAMILLES éprouvées en tool-use (du 35B MoE au ~1.5B), pas des repos figés — le repo réel est résolu en live sur HF au moment du choix (top téléchargements qui fit), donc la liste ne périme pas. Chaque machine a toujours au moins une proposition ; recherche libre et `/add-model` restent pour les initiés.
- **Recherche filtrée par la machine** : les résultats HF impossibles (taille estimée depuis le nom, « 397B » sur 1,6 Go de budget…) sont masqués avec un compteur, les jouables annotés `~X Mo mini` ; la taille réelle des quants (`recommend_quant`) tranche après le choix.
- **« Libère ta RAM » avant le choix du modèle** : le setup liste les plus gros consommateurs (groupés par nom, via psutil), laisse l'utilisateur fermer LUI-MÊME ce qu'il n'utilise pas (rien n'est jamais tué automatiquement), re-mesure sur Entrée et recalcule le budget — la shortlist en profite immédiatement.

### Améliorations
- **Bootstrap auto au premier run** : `serve.py` détecte la machine vierge (binaire ou modèle manquant) et enchaîne tout seul sur l'installeur guidé dans le terminal ; hors terminal interactif (service), il guide vers `uv run loom-setup` et sort proprement.
- Hint de `serve.py` quand le binaire manque : pointe vers `loom-setup`.
- **Couleurs terminal** (`loom/runtime/term.py`) : bannières/étapes en gras, ✅ vert, ❌/ERREUR rouge, ⏭️ estompé, 🔧 jaune — uniquement sur un vrai TTY (jamais de codes ANSI dans les fichiers de log ni les pipes), `NO_COLOR` respecté, mode VT activé sur les vieilles consoles Windows. Branché sur loom-setup et serve.
- Le premier modèle installé devient `[chat] default_model` dans `config/local.toml` (sinon le défaut de defaults.toml peut pointer un modèle d'une autre machine).
- Boucle « libère ta RAM » avant le choix du modèle : liste les plus gros consommateurs (groupés par nom), l'utilisateur ferme lui-même, re-mesure — le budget et la shortlist en profitent.
- **Étape 4 : bench du matériel** (`loom/setup/bench.py`) — après l'install, `llama-bench` (livré avec la release) MESURE la vitesse réelle sur le modèle installé : combinaisons de threads (physiques/logiques/moitié — machines hybrides P+E) et d'offload GPU (`-ngl` si backend Vulkan/CUDA présent). La génération (tg) tranche, le prefill départage. Le contexte est CALCULÉ (KV/token lu du header GGUF → le plus grand cache qui tient en RAM sans swap ; vécu : 24576 par défaut sur 16 Go → swap → 0,8 t/s). Résultats écrits dans `config/local.toml` (`[override] threads`/`n_gpu_layers`, `[server] context`, table `[bench]` pour l'idempotence).
- **Debug actif par défaut** : le trace des échanges modèle (stderr + `sessions/<id>/debug.log`) n'exige plus `LOOM_DEBUG=1` — c'est le premier outil de diagnostic, coût négligeable. `LOOM_DEBUG=0` pour couper.
- **Terminal tenable en debug** : les blocs volumineux (dump complet de la requête avec system prompt, save/restore de slot KV) ne vont plus QUE dans `debug.log` — le terminal garde les lignes compactes (`turn.request`, `usage`, premier byte) et la RÉPONSE du modèle. Les polls `GET /machine_state` rejoignent `/sysmon` dans le filtre werkzeug.
- Docs d'install (Windows/Linux) : voie rapide `loom-setup` + correction du chemin de config obsolète (`loom.config.personnel.toml` → `config/local.toml`).

---

## 2026-07-17 — /add-model

### Nouvelles fonctionnalités
- **`/add-model`** : ajout d'un modèle en une commande (wizard chat déterministe, zéro LLM) — local : recherche Hugging Face, choix du quant recommandé selon VRAM/RAM, téléchargement avec progression, `model.toml` généré (n_layers/MoE lus du GGUF), montage à chaud dans le sélecteur ; distant : URL + clé, persisté via le store du panneau engrenage.

---

## 2026-06-09 — Mémoire & Reflection

### Nouvelles fonctionnalités
- **Mémoire persistante** : provider local SQLite FTS5 (`remember`/`recall`), interface `MemoryProvider` (local v1), bloc identité SOUL/USER/MEMORY toujours injecté au prompt, cache mtime des fichiers identité.
- **Étape `reflect`** : capitalisation post-tour (validation pure + écritures), cles config dédiées (`reflect`, `skills_learned`, `summarization_recall`), trace visible (ran/retenu/erreur).
- **Catalogue skills appris** : namespace `learned:`, frontmatter étendu, câblage au catalogue web.
- **Outils mémoire** : `recall`, `remember`, `write_note`, `read_note` (permission `evaluate()` autorisée d'office).
- **Pastille outil IN/OUT** dans l'UI (nom du tool, commande + sortie dépliables bloc par bloc).

### Améliorations
- Injecte le dossier de travail courant au prompt (anti-tatonnement git/chemins).
- Adopte un repo cité par son nom (sous-dossier du workspace racine).
- Rendu LaTeX offline (MathJax) + compteur live tool-calls.
- Défaut = dernier modèle + robustesse vision.
- Compteur live tokens envoyés/reçus + débit.
- Badge d'alerte quand le mode permission est `allow`.
- Keep-warm du modèle pour éviter le cold start.

### Corrections
- Valide le nom des skills appris (anti traversée de chemin dans `reflect`).
- Autorise d'office `recall`/`remember`/`write_note`/`read_note`.
- Rattrapage des tool-calls textuels + règle 0 assouplie.
- Termine l'arbre de process sur Ctrl+C.
- Durcit les outils web + frontière de confiance native.

### Refactoring
- Range le cœur dans `loom/agent/` et les extensions dans `loom/extend/`.
- Regroupe l'infra de service dans `loom/runtime/`.
- Renomme `loom.config.local.toml` en `.personnel.toml` (suivi par git).

### Évaluation
- Harnais d'eval des prompts (eval set + double grader, A/B git).
- Harnais d'eval du skill code-review (rappel/FP/verdict).

### Modèles
- Ne garder que les MoE 24B+ (retire Gemma E4B + Qwen 4B).
- Template `_TEMPLATE/model.toml` + README.

### Documentation
- Présentation de Loom générée par le modèle local (Qwen3.6).
- Actualise README + ETAT_PROJET (tool-use pur, plugins, skills déclenchés, MoE 24B+).
- Archive les specs/plans de code retirés.
- Design closed-learning-loop + plans mémoire/reflect.
- Spec skill de debug généraliste + `check_page` localisant.

### Debug
- `debug` : imprime le système prompt en entier (LOOM_DEBUG), messages bornes.
- Skill de debug généraliste + `check_page` localisant.

---

## Commits détaillés

| Date | Commit | Catégorie | Description |
|------|--------|-----------|-------------|
| 06-09 | 6ac1207 | debug | imprime le system prompt en entier (LOOM_DEBUG) |
| 06-09 | 89e4c1f | feat(ui) | pastille outil en IN/OUT (nom du tool, commande + sortie dépliables) |
| 06-09 | 7db1867 | feat(web) | injecte le dossier de travail courant au prompt |
| 06-09 | 1027862 | fix(permissions) | autorise d'office recall/remember/write_note/read_note |
| 06-09 | 4840ca9 | feat(web) | trace visible de reflect au lieu du except: pass muet |
| 06-09 | cbc7b88 | perf(memory) | cache mtime des fichiers identite |
| 06-09 | d602a83 | feat(web) | adopte un repo cite par son nom |
| 06-09 | d0ec7a1 | docs(prompt) | presente les skills appris (learned:) au modele |
| 06-09 | 365b281 | feat(web) | cable catalogue skills appris + resumeur recall |
| 06-09 | 7e6bbaf | fix(security) | valide le nom des skills appris (anti traversée) |
| 06-09 | a211918 | feat(agent) | etape reflect (validation pure + ecritures) |
| 06-09 | 1d66ab2 | feat(prompts) | prompt de reflexion (capitalisation post-tour) |
| 06-09 | da8d291 | feat(config) | cles reflect + skills appris + summarization recall |
| 06-09 | 40a527d | feat(skills) | agrege skills_learned (namespace learned) |
| 06-09 | 25d3dd5 | feat(evals) | harnais d'eval du skill code-review |
| 06-09 | 7d62a43 | docs(plans) | design closed-learning-loop + plans memoire |
| 06-09 | bdc844e | docs(prompt) | presente remember/recall + memoire persistante |
| 06-09 | 561d10f | feat(web) | cable la memoire + injecte le bloc identite |
| 06-09 | a909032 | feat(tools) | outils recall/remember (memoire persistante) |
| 06-09 | 0add967 | feat(config) | bloc [memory] + identity_max_tokens |
| 06-09 | 08240a7 | feat(memory) | identite markdown SOUL/USER/MEMORY |
| 06-09 | 6f72662 | feat(memory) | provider local SQLite FTS5 |
| 06-09 | a8f9e6e | feat(memory) | interface MemoryProvider + selection |
| — | be69fb6 | feat(evals) | harnais d'eval des prompts (master) |
| — | d133b46 | feat(prompts) | durcit chat/subagent + corrige la liste d'outils |
| — | a9151bb | refactor(config) | renomme .local.toml en .personnel.toml |
| — | 6447316 | feat(web) | rendu LaTeX offline (MathJax) + compteur live |
| — | 0712b47 | feat(web) | défaut = dernier modèle + robustesse vision |
| — | 0b408fa | feat(agent) | rattrapage des tool-calls textuels |
| — | 4832d8b | docs | présentation de Loom générée par Qwen3.6 |
| — | 88a04f1 | feat(web) | compteur live tokens envoyés/reçus |
| — | 05fdbbe | docs(bench) | abandonne le drafter MTP, garde le sweep MoE |
| — | 74a4cb2 | feat(web) | badge d'alerte mode permission |
| — | 88f86b0 | feat(skills) | ajoute le skill code-review |
| — | 3c1a706 | fix(securite) | durcit les outils web |
| — | 3116b42 | feat(tools) | write_note/read_note + fix du thrash |
| — | 5263016 | feat(web) | keep-warm du modèle |
| — | a85b479 | fix(runtime) | termine l'arbre de process sur Ctrl+C |
| — | 36b6fae | loom | correction des chemins relatifs |
| — | 92ba919 | docs(loom) | carte du package dans \_\_init\_\_ |
| — | 38d68dd | refactor | range le coeur dans loom/agent/ |
| — | 72f63e0 | refactor | regroupe l'infra de service dans loom/runtime/ |
| — | 165766d | docs | actualise README + ETAT_PROJET |
| — | febc20a | chore(models) | template \_TEMPLATE/model.toml |
| — | 4021291 | docs | archive les specs/plans de code retirés |
| — | 67c87e1 | chore(models) | ne garder que les MoE 24B+ |
| — | 773e8fd | feat(debug) | skill de debug + check_page |
| — | 3b70c8a | docs(debug) | spec skill de debug |
| — | 8584d0c | docs(eval) | design du banc d'eval |
| — | fd2ea61 | chore(skills) | retire le skill exemple |
