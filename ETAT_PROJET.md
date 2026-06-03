# État du projet — Loom

> Dernière mise à jour : 2026-06-03

Assistant IA **local, multimodal et offline** : rendre un petit modèle open-source réellement utile
par de la plomberie (runtime, mémoire, skills, vision, contexte). Voir [README.md](README.md) pour le
pitch et le démarrage.

## Livré

### A — Fondation runtime
- Package **Loom** (`loom/`, installable hatchling). Runtime **llama.cpp** (`llama-server`), API
  OpenAI-compatible sur `:8080`. Lanceur auto-adaptatif `uv run loom/serve.py` (GPU sinon CPU,
  offload réglé selon la VRAM libre via `nvidia-smi`).
- **1 modèle → `llama-server` direct** ; **2+ → `llama-swap`** (généré depuis le registre `[[models]]`).
- Décision : [docs/adr/0001-llamacpp-vs-ollama.md](docs/adr/0001-llamacpp-vs-ollama.md).
  Perfs : ~18-21 tok/s GPU (RTX 2060).

### Modèle
- **Gemma 4 E4B *ablitéré* (non censuré)** — `mradermacher/gemma-4-E4B-it-uncensored-GGUF`, Q4_K_M
  ~5 Go. **Vision** via `mmproj-F16.gguf`.

### Loom Chat (v1 → v4 + polish)
- **Chat web** Flask + HTMX + SDK `openai`, **streaming SSE**, markdown rendu, mémoire JSON persistée.
- **Vision** (coller un screenshot → extraction de données).
- **Skills** injectables (`loom/skills/<nom>/SKILL.md`, format Claude Code, activation par conversation).
- **Hardening** : verrou anti-concurrence (429), retries/timeout SDK, save atomique, **résumé auto du
  contexte** (`context.py`).
- **Multi-modèles** : registre `[[models]]` + **sélecteur UI** + `swap.py` (un modèle par requête).
- **Interruption** : soumettre un nouveau message stoppe net la génération en cours (flag d'annulation
  serveur + `AbortController`), la réponse partielle est conservée.
- **Thinking** : bloc de réflexion **animé** + **toggle 🧠** (désactive via `enable_thinking=false`
  → réponse directe). **Bouton 📋 copier** sur chaque réponse + par bloc de code.
- Lancement : `uv run python -m loom.web` (:8000). **86 tests verts**, ruff clean.

### Boucle tool-use (livré, désactivée par défaut)
- `loom/tools/` (package) : `base` (`ToolSpec`/`ToolRegistry`/`_resolve_in_root`), `read`
  (`read_file`/`verify`), `fs` (write/edit), `shell`, `web` ; `__init__` ré-exporte l'API publique
  + `build_registry`. `client.stream_chat_tools()` :
  boucle agentique (reconstruction des `tool_calls` streamés, exécution, réinjection `role:tool`,
  relance, garde `max_iters`). `serve.py` inclut `--jinja`.
- Outils : `read_file`, `write_file`/`edit_file` (atomiques, exact-unique, workspace-bound),
  `run_shell` (double barrière deny-list **avant** subprocess, OS detect), `web_search`
  (SearXNG/Tavily/ddgs + trafilatura, dégradé hors-ligne).
- **Mode permission** (`loom/permissions.py`) : `evaluate()` pur + `DEFAULT_DENY` (regex
  incontournable : `rm -rf`, `Remove-Item -Recurse -Force`, `format`, `dd if=`…) +
  **confirmation interactive** (bulle Autoriser/Refuser, pause SSE + `/tool_decision`).
  Périmètre = racine des vrais projets (`workspace_dir`).
- **Activables depuis l'UI** (panneau 🛠️ Outils, par conversation). Validé en live :
  read_file/web_search/run_shell fonctionnent (`--jinja` actif).
- **Multi-agent** (`loom/agents.py` + `loom/orchestrator.py`) : pipeline plan→code→review,
  route `/run` (streaming live). Les agents ont leurs **propres outils** (le développeur écrit
  via `stream_chat_tools` + confirmation ; le relecteur lit/teste) et une **boucle review→fix**
  bornée (verdict `BLOQUANT`/`OK`). `[[agents]]` définis dans le config.

### Moteur unique fan-out (branche `feat/moteur-unique`, à valider sur le vrai modèle)
> Fusion de `run_pipeline` (séquentiel) et `run_build` (fan-out) en **un seul moteur bâti sur le
> fan-out**. Spec : `docs/superpowers/specs/2026-06-03-loom-moteur-unique-design.md`. Plan :
> `docs/superpowers/plans/2026-06-03-loom-moteur-unique.md`. **346 tests verts**, ruff clean.

- **Mode par fichier déterministe** (`derive_modes`/`cap_rewrites`, `loom/parallel.py`) : fichier
  absent → `create` (génération complète) ; existant + verify KO → `rewrite` (borné > 200 lignes →
  dégradé en `patch`) ; existant + verify OK → `patch`. Aucun jugement confié au modèle.
- **`edit_one`** : patch ciblé en 2 temps (read déterministe → `{old_string,new_string}` → application
  via `make_edit_file` → **fallback** réécriture complète si introuvable/ambigu/JSON invalide).
- **`explore()`** (`loom/explore.py`) : ground-truth brownfield déterministe (lecture bornée des
  fichiers cités), injectée dans le PLAN via `plan_files(explore_summary=…)`.
- **best-of-N en réparation** (`best_of`, N=2) : garde le 1er candidat qui passe `verify_syntax_file`.
- **Stop anti-divergence** : la boucle FIX s'arrête si l'ensemble des `location` de défauts ne
  décroît pas (un 4B oscille sinon). `max_rounds` borné.
- **`review_semantic`** (advisory, non bloquant, **off par défaut**) : signale les défauts
  comportementaux que le verify déterministe ne voit pas. Activable via le flag form `semantic`.
- **`run_pipeline` conservé deprecated** sous `mode=='pipeline'` ; son retrait (PR9) attend la preuve
  de parité du moteur fusionné sur 3 tâches réelles avec le vrai Gemma 4B.
- **214 → 346 tests verts**, ruff clean.

## Reste
1. **Valider le moteur fusionné sur le vrai Gemma 4B** (greenfield Démineur + brownfield/patch sur
   projet existant). Si parité OK → PR9 : retrait de `run_pipeline` + mapping des 11 tests
   `test_run_pipeline_*`. Brancher `semantic_review` dans l'UI si souhaité. Boucle outillée
   d'EXPLORE (différée).
2. `llama-swap` installé + 2ᵉ modèle (agents du pipeline sur des modèles distincts).
3. **SearXNG** self-host pour un `web_search` fiable (`ddgs` rate-limite — `fetch_pages=false`
   en attendant).
4. Persister/rejouer les runs multi-agent (aujourd'hui éphémères). RAG (skills volumineux), audio.

## Conventions
- Toolchain : **`uv`** (`uv run` / `uvx`) + **`ruff`** (hook PostToolUse lint+format PEP8).
- Garde-fous `.claude/settings.json` : deny `rm -rf *` / `Remove-Item -Recurse -Force *`, lecture des
  secrets (`.env`, `*.pem`, `*.key`). Dépôt pas (encore) sous git.
