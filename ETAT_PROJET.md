# État du projet — Loom

> Dernière mise à jour : 2026-06-01

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
- **214 tests verts**, ruff clean.

## Reste
1. `llama-swap` installé + 2ᵉ modèle (agents du pipeline sur des modèles distincts).
2. **SearXNG** self-host pour un `web_search` fiable (`ddgs` rate-limite — `fetch_pages=false`
   en attendant).
3. Persister/rejouer les runs multi-agent (aujourd'hui éphémères). RAG (skills volumineux), audio.

## Conventions
- Toolchain : **`uv`** (`uv run` / `uvx`) + **`ruff`** (hook PostToolUse lint+format PEP8).
- Garde-fous `.claude/settings.json` : deny `rm -rf *` / `Remove-Item -Recurse -Force *`, lecture des
  secrets (`.env`, `*.pem`, `*.key`). Dépôt pas (encore) sous git.
