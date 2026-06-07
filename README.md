# 🧵 Loom — un agent IA local, multimodal et offline

> Un petit modèle open-source qui **agit** sur ta machine avec des outils —
> **100 % en local, sans internet.**

Loom est un **agent tool-use** local : une boucle qui donne à un petit modèle (Gemma 4B)
les bons outils et la logique de les enchaîner, pour qu'il **localise, lise, modifie et
exécute** sur tes vrais projets au lieu de seulement discuter. Le pari : rester productif
même internet coupé, et démontrer le savoir-faire d'« internaliser le harness » sur un
modèle autre que Claude.

Le harness, c'est ça et rien d'autre : **les outils, et l'intelligence d'appeler le bon
au bon moment.** Pas de pipeline déterministe, pas de mode « build » séparé — un seul
chemin, l'agent qui agit.

## Ce que ça fait

- 💬 **Chat web** local (Flask + Preact/htm, zéro build) avec **streaming SSE** et markdown.
- 🧰 **18 outils** exposés au modèle, regroupés par usage :
  - **Localiser** : `find_files` (glob), `search_text` (grep), `list_dir`.
  - **Lire** : `read_file` (texte), `read_document` (PDF / Excel / Word → texte),
    `read_image` (voir une image du disque : capture, schéma).
  - **Planifier / déléguer** : `manage_todos` (bloc-notes de plan, mémoire de travail),
    `dispatch_agent` (sous-agent à contexte isolé qui fait un gros chantier et ne renvoie
    qu'une synthèse — son activité est visible en direct).
  - **Modifier / créer** : `write_file`, `append_file` (ajout en fin), `edit_file`
    (remplacement exact-unique), `replace_lines` / `insert_lines` (édition par numéro de
    ligne, indentation préservée), `format_code` (ruff Python / prettier web).
  - **Exécuter** : `run_shell` (PowerShell/bash, deny-list dure, tue l'arbre au timeout).
  - **Web** : `web_search`, `fetch_url` (dégradés proprement hors-ligne).
  - **Vérifier le rendu** : `check_page` (charge une page HTML en navigateur headless,
    exécute le JS, renvoie erreurs console + compte d'éléments).
- 🧭 **Boucle agentic, pas déterministe** : l'arrêt suit le **stop naturel** du modèle
  (il répond sans appel d'outil → fini). Par-dessus, des **garde-fous** non-bloquants :
  plafond de tours, mur de temps, détecteur de non-progrès (anti-boucle).
- 🔒 **Mode permission** : deny-list dure incontournable (`rm -rf`, `format`, `dd if=`…)
  + confirmation interactive (Autoriser/Refuser) en mode `ask`, ou autonomie en `allow`.
- 🛡️ **Sécurité de l'ingestion** (active même hors-ligne) : garde **anti-SSRF** (refus des
  adresses internes), et **frontière de confiance** — tout contenu externe (URL, PDF,
  image) est marqué comme DONNÉE à analyser, jamais comme instructions à exécuter.
- 🖼️ **Vision** : colle un screenshot dans le chat, *ou* laisse l'agent lire une image du
  workspace lui-même via `read_image` (Gemma 4 multimodal, projecteur `mmproj`).
- 🗂️ **Sessions** : un fil persistant par projet (historique + outils actifs), bascule/
  création/suppression depuis l'UI.
- 🧩 **Skills** (format `SKILL.md` façon Claude Code) : injecte ta connaissance (archi,
  conventions) dans le contexte, activable à la volée.
- 💭 **Raisonnement** : le « thinking » s'affiche dans un bloc animé, désactivable d'un clic.
- ⏹️ **Interruption** : soumettre un nouveau message stoppe net la génération en cours
  (la réponse partielle est conservée).
- 🔀 **Multi-modèles** : registre `[[models]]` + sélecteur UI ; hot-swap via llama-swap.
- 🔓 **Modèle non censuré** (Gemma 4 E4B *ablitéré*) par défaut.

## Architecture

```
Navigateur ──HTTP──► Loom (Flask :8000) ──OpenAI API──► llama-server (:8080) ──► GGUF (GPU+CPU)
                        │
                        ├─ client.py        (SDK openai + boucle tool-use stream_chat_tools + garde-fous)
                        ├─ conversation.py  (mémoire + modèle + thinking + outils, persisté JSON)
                        ├─ session.py       (un fil persistant par projet)
                        ├─ context.py       (budget tokens + résumé auto)
                        ├─ skills.py        (connaissance injectable)
                        ├─ tools/           (localiser, lire, modifier, exécuter, web, todos, sous-agent)
                        ├─ permissions.py   (deny-list dure + allow/ask/deny)
                        └─ swap.py          (registre [[models]] → llama-swap.yaml)
```

- **Runtime** : [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`), derrière une
  **API OpenAI-compatible** (couche au-dessus 100 % agnostique). Choix documenté dans
  [docs/adr/0001-llamacpp-vs-ollama.md](docs/adr/0001-llamacpp-vs-ollama.md).
- **Lanceur auto-adaptatif** : `loom/serve.py` détecte le GPU (sinon CPU) et règle l'offload
  selon la VRAM libre (`nvidia-smi`). Inclut `--jinja` (requis pour les appels d'outils) et
  `--mmproj` (vision).
- **Modèle** : Gemma 4 E4B (Q4_K_M, ~5 Go) + projecteur vision `mmproj-F16.gguf`.

## Démarrer

Prérequis : [`uv`](https://docs.astral.sh/uv/), et le binaire `llama-server`
(voir [docs/install-windows.md](docs/install-windows.md) / [docs/install-linux.md](docs/install-linux.md)).

```bash
uv sync                      # dépendances + installe le package loom
uv run loom/serve.py         # télécharge le modèle au 1er run + sert sur :8080
uv run python -m loom.web    # interface chat sur :8000
```
Puis ouvre **http://127.0.0.1:8000**.

## Utiliser l'agent

- Les outils sont **armés par défaut** : décris une tâche (« résume cette facture »,
  « où est défini X et corrige-le », « regarde ce screenshot »), l'agent enchaîne les
  outils tout seul.
- **Périmètre** = `workspace_dir` (config). La lecture peut être large ; l'écriture et le
  shell sont gardés par le **mode permission** (`allow` = autonome, `ask` = confirmation).
- **Déléguer** : pour un gros chantier (analyser/modifier beaucoup), l'agent peut lancer
  `dispatch_agent` — un sous-agent à contexte isolé qui travaille puis renvoie une synthèse.

## Ajouter un skill

Crée `loom/skills/<nom>/SKILL.md` :
```markdown
---
name: dagster
description: Mon archi Dagster
---
<ta connaissance ici>
```
Coche-le dans le panneau **Skills** de l'interface.

## Configuration

Tout est dans [loom/loom.config.toml](loom/loom.config.toml) (modèle, contexte, port, outils
armés, permissions). Les réglages spécifiques à une machine (chemin du binaire, override GPU)
vont dans `loom/loom.config.local.toml` (gitignoré).

## Statut

État détaillé : [ETAT_PROJET.md](ETAT_PROJET.md).

- ✅ Runtime auto-adaptatif, sessions, vision, skills, thinking, interruption, multi-modèles.
- ✅ Agent tool-use : 18 outils, politique de décision + séquencement dans le prompt système.
- ✅ Garde-fous de boucle (stop naturel + plafonds + mur de temps + anti-répétition).
- ✅ Sécurité d'ingestion : anti-SSRF + frontière de confiance (active même hors-ligne).
- 🔜 SearXNG (web_search fiable), 2ᵉ modèle plus costaud, RAG (skills volumineux), audio.

## Stack

Python 3.12+ · `uv` · `ruff` · Flask · Preact/htm · SDK `openai` · llama.cpp · Gemma 4.
