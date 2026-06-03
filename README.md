# 🧵 Loom — un assistant IA local, multimodal et offline

> Tisser plusieurs fils (modèle, mémoire, skills, vision) en une intelligence cohérente —
> **100 % sur ta machine, sans internet.**

Loom rend un **petit modèle open-source local** réellement utile en comblant l'écart avec les gros
modèles cloud par de la **plomberie** : runtime auto-adaptatif, mémoire persistée, injection de
connaissance (skills), vision, et gestion de contexte. Le pari : *rester productif même internet
coupé*, et démontrer le savoir-faire d'« internaliser le harness » sur des modèles autres que Claude.

## Ce que ça fait

- 💬 **Chat web** local (Flask + HTMX) avec **streaming** token-par-token et **markdown** rendu.
- 🧠 **Mémoire** de conversation persistée (JSON), avec **résumé automatique** quand le contexte
  devient long.
- 🛠️ **Outils (agent)** : `read_file`, `write_file`/`edit_file`, `run_shell`, `web_search`,
  **activables par conversation** depuis l'UI. Le modèle **lit, écrit et exécute** sur tes vrais
  projets — pas juste du chat.
- 🔒 **Mode permission** : deny-list dure incontournable (`rm -rf`, `format`…) + **confirmation
  interactive** (bulle Autoriser/Refuser avant chaque action sensible, comme Claude Code).
- 🤖 **Multi-agent** : pipeline **plan→code→review** où chaque agent a ses propres outils (le
  développeur écrit les fichiers, le relecteur lance les tests) + **boucle review→fix** bornée.
- 🖼️ **Vision** : colle un screenshot → le modèle lit et extrait les données (Gemma 4 multimodal).
- 🧩 **Skills** (format `SKILL.md` façon Claude Code) : injecte ta connaissance (ton archi, tes
  conventions) dans le contexte, activable à la volée.
- 💭 **Raisonnement** : le « thinking » du modèle s'affiche dans un bloc animé, et se **désactive**
  d'un clic (toggle 🧠) pour des réponses directes et instantanées.
- ⏹️ **Interruption** : soumets un nouveau message pendant que le modèle déroule → la génération en
  cours s'arrête net (la réponse partielle est conservée).
- 🔀 **Multi-modèles** : registre `[[models]]` + **sélecteur dans l'UI** ; un modèle par requête
  (base pour le multi-agent), hot-swap via llama-swap quand plusieurs modèles sont installés.
- 🔓 **Modèle non censuré** (Gemma 4 E4B *ablitéré*) par défaut.
- 🛡️ **Robuste** : verrou anti-concurrence, retries, save atomique, gestion des réponses vides.

## Architecture

```
Navigateur ──HTTP──► Loom Chat (Flask :8000) ──OpenAI API──► llama-server (:8080) ──► GGUF (GPU+CPU)
                         │
                         ├─ client.py        (SDK openai + boucle tool-use stream_chat_tools)
                         ├─ conversation.py  (mémoire + modèle + thinking + outils, persisté JSON)
                         ├─ context.py       (budget tokens + résumé auto)
                         ├─ skills.py        (connaissance injectable)
                         ├─ tools*.py        (read_file, write/edit, run_shell, web_search)
                         ├─ permissions.py   (deny-list dure + allow/ask/deny)
                         ├─ orchestrator.py  (pipeline multi-agent plan→code→review)
                         └─ swap.py          (registre [[models]] → llama-swap.yaml)
```

- **Runtime** : [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`), derrière une
  **API OpenAI-compatible** (couche au-dessus 100 % agnostique). Choix documenté dans
  [docs/adr/0001-llamacpp-vs-ollama.md](docs/adr/0001-llamacpp-vs-ollama.md).
- **Lanceur auto-adaptatif** : `loom/serve.py` détecte le GPU (sinon CPU) et règle l'offload.
- **Modèle** : Gemma 4 E4B (Q4_K_M, ~5 Go) + projecteur vision `mmproj`.

## Démarrer

Prérequis : [`uv`](https://docs.astral.sh/uv/), et le binaire `llama-server`
(voir [docs/install-windows.md](docs/install-windows.md) / [docs/install-linux.md](docs/install-linux.md)).

```bash
uv sync                      # dépendances + installe le package loom
uv run loom/serve.py         # télécharge le modèle au 1er run + sert sur :8080
uv run python -m loom.web    # interface chat sur :8000
```
Puis ouvre **http://127.0.0.1:8000**.

Mesurer le débit : `uv run loom/benchmark.py`.

## Ajouter un skill

Crée `loom/skills/<nom>/SKILL.md` :
```markdown
---
name: dagster
description: Mon archi Dagster
---
<ta connaissance ici>
```
Coche-le dans le panneau **🧩 Skills** de l'interface.

## Outils & multi-agent

- **Active les outils** dans le panneau **🛠️ Outils** (par conversation). Le modèle peut alors
  lire/écrire/exécuter — chaque action sensible te demande **Autoriser / Refuser**.
- **Périmètre** = `workspace_dir` (config). La lecture peut être large (Loom est offline → aucune
  exfiltration possible) ; l'écriture et le shell sont gardés par le **mode permission**.
- **Pipeline multi-agent** : panneau **🤖 Multi-agent** → décris une tâche ; le *planner* planifie,
  le *coder* écrit les fichiers (avec confirmation), le *reviewer* relit et lance les tests, avec
  une boucle de correction. Les rôles/outils sont dans `[[agents]]` du config.
- **Prérequis** : `llama-server` doit tourner avec `--jinja` (déjà dans `serve.py`) pour que le
  modèle émette des appels d'outils structurés.

## Configuration

Tout est dans [loom/loom.config.toml](loom/loom.config.toml) (modèle, contexte, port, skills,
robustesse). Les réglages spécifiques à une machine (chemin du binaire, override GPU) vont dans
`loom/loom.config.local.toml` (gitignoré).

## Statut & roadmap

État détaillé : [ETAT_PROJET.md](ETAT_PROJET.md). Specs & plans : [docs/superpowers/](docs/superpowers/).

- ✅ Runtime, chat, vision, skills, hardening, **interruption**, **toggle thinking**,
  **multi-modèles** (registre + sélecteur UI + llama-swap).
- ✅ **Boucle tool-use** : `read_file`, `write_file`/`edit_file`, `run_shell`, `web_search`,
  activables par conversation depuis l'UI (panneau 🛠️ Outils).
- ✅ **Mode permission** : deny-list dure incontournable + **confirmation interactive**
  (bulle Autoriser/Refuser avant chaque action sensible, comme Claude Code).
- ✅ **Multi-agent** (`/run`) : pipeline **plan→code→review** où les agents ont leurs propres
  **outils** (le développeur écrit les fichiers, le relecteur lance les tests, avec confirmation)
  et une **boucle review→fix** bornée.
- 🔜 SearXNG (web_search fiable), llama-swap + 2ᵉ modèle (agents sur modèles distincts), RAG.

## Stack

Python 3.12+ · `uv` · `ruff` · Flask · HTMX · SDK `openai` (Stainless) · llama.cpp · Gemma 4.
