# Loom — agent IA local, multimodal et offline

<!-- RÔLE : carte technique (arborescence, stack, conventions, points d'attention). Pitch public : README.md. Suivi interne : ETAT_PROJET.md. Historique versions : CHANGELOG.md. -->

## But

Loom est un **agent tool-use** local : une boucle qui donne à un modèle local des outils et la logique de les enchaîner, pour qu'il localise, lise, modifie et exécute sur de vrais projets au lieu de seulement discuter. Objectif affiché : rester productif internet coupé, en « internalisant le harness » (outils + intelligence d'appel) autour d'un modèle open-source autre que Claude. 100 % offline par défaut.

Le pari architectural : pas d'orchestrateur déterministe, pas de rail de réflexion, pas de mode build séparé — un seul chemin, la boucle `stream_chat_tools` où le modèle décide. Deux orchestrateurs rigides ont été essayés puis supprimés (juin 2026) car ils bridaient le modèle.

## Stack

- **Python 3.12+**, gestionnaire `uv`, build `hatchling`.
- **Runtime LLM** : `llama.cpp` (`llama-server`), API OpenAI-compatible sur `:8080` ; `llama-swap` pour le hot-swap multi-modèles. MoE 24B+ rendus jouables même sur petite VRAM par offload des experts en RAM (`--cpu-moe`).
- **Web/UI** : Flask + Preact/htm (zéro build, libs vendorées en `loom/web/static/`), streaming SSE, markdown, MathJax (LaTeX offline). Multi-onglets (une session par onglet, génération concurrente pour les distants), console de configuration (édition des TOML à chaud), gestionnaire de modèles (locaux + distants, ajout/édition sans redémarrer), compteur tokens/cache + jauge de remplissage du contexte, moniteur système (CPU/RAM/GPU) pour les modèles locaux.
- **SDK** : `openai` (client API), `httpx`, `huggingface-hub` (téléchargement des GGUF).
- **Ingestion docs** : `pypdf`, `openpyxl`, `python-docx`, `trafilatura`.
- **Vérification rendu** : `playwright` (headless, `check_page` / `check_interactive`).
- **Lint/format** : `ruff` (présent au runtime pour l'outil `format_code`).
- **Mémoire** : SQLite FTS5 (provider local).
- **Recherche web** (optionnel, dégradé offline) : `ddgs`.
- **Modèles** : locaux — `gemma4-26b-a4b-uncensored`, `gemma4-26b-a4b-qat` (vision), `qwen3.6-35b-a3b-abliterated` (vision) ; distants (API OpenAI-compatible) — `glm-zai` (glm-5.2, **défaut** actuel via `local.toml`), `glm-5v` (vision). Les distants se déclarent en config (`[[remote_models]]`) ou via l'UI (store `var/remote_models.json`), pas par dossier ; un distant ne subit **pas** les limites locales (VRAM) — il prend la limite de sortie du provider.

## Arborescence

```
from-claude-to-local-haranessed-llm/
├── loom/                      # package Python (hatchling)
│   ├── __init__.py            # carte du package
│   ├── config.py              # schéma + chargement config (defaults.toml + local.toml)
│   ├── permissions.py         # politique allow/ask/deny + deny-list dure
│   ├── agent/                 # cœur : boucle tool-use
│   │   ├── client.py          #   stream_chat_tools (reconstruction tool_calls, exécution, réinjection)
│   │   ├── conversation.py    #   historique
│   │   ├── context.py         #   fenêtre + résumé auto
│   │   ├── session.py         #   sessions persistantes (titre inféré)
│   │   ├── inline_image.py    #   vision (screenshot collé)
│   │   └── reflect.py         #   capitalisation post-tour (mémoire)
│   ├── tools/                 # ~23 outils appelés par le modèle
│   │   ├── search.py fs.py read.py      # localiser/lire (find_files, search_text, list_dir, read_file [texte+PDF/xlsx/docx], read_image)
│   │   ├── shell.py format.py           # exécuter (run_shell), formater (ruff/prettier)
│   │   ├── web.py                       # web_search, fetch_url (anti-SSRF)
│   │   ├── browser.py                   # check_page, check_interactive (Playwright)
│   │   ├── todo.py agent.py note.py memory.py  # manage_todos, dispatch_agent, write/read_note, recall/remember
│   │   ├── skills.py plugins.py         # use_skill, list/add/install plugins
│   │   └── trust.py                     # frontière de confiance (ingestion)
│   ├── extend/                # skills (catalogue + use_skill) + store plugins (compatible Claude Code)
│   ├── prompts/               # prompts système .md EN ANGLAIS (chat.system, subagent, reflect) — instructions EN, réponse FR imposée ; POLITIQUE seule (la mécanique des outils vit dans les schémas) + identity/ (SOUL/USER/MEMORY)
│   ├── runtime/               # serve.py (lanceur llama.cpp auto-adaptatif + regen swap yaml), swap.py, hardware.py, server_args.py, models_fetch.py, models_profile.py, platform_info.py (détection OS), sysmon.py (moniteur CPU/RAM/GPU), config_schema.py (introspection config pour la console), model_store.py (store des modèles distants gérés par l'UI)
│   ├── memory/                # provider SQLite FTS5 (local.py) + identity.py
│   ├── models/                # découvertes par dossier <id>/ (model.toml + profile.md + GGUF gitignoré) ; _TEMPLATE/ suivi
│   ├── skills/                # skills livrés : code-review/, debugging/, trust-boundary/
│   ├── plugins/               # store de plugins installés (gitignoré) + marketplaces/
│   └── web/                   # Flask + SSE (app.py, __main__.py) + templates/ + static/ (Preact, htm, marked, MathJax, DOMPurify — vendorés)
├── config/                    # defaults.toml (versionné) + local.toml (surcharge machine, gitignored) + local.example.toml
├── docs/                      # adr/ (llamacpp-vs-ollama), superpowers/ (specs/plans), install-windows.md, install-linux.md, glossaire.md
├── evals/                     # harnais d'éval des prompts (cases.py, run_eval.py) + éval du skill code-review (review_cases.py, run_review_eval.py)
├── var/                       # état machine (gitignored) : identity/, memory/, sessions/, skills_learned/, logs/, cache/
├── pyproject.toml             # dépendances + build hatchling
├── uv.lock
├── README.md  ETAT_PROJET.md  CHANGELOG.md  .gitignore
```

## Lancer / Tester

Prérequis : `uv`. `loom-setup` installe le binaire llama.cpp adapté et le premier modèle.

```powershell
uv sync                       # dépendances + installe le package loom
uv run loom-setup             # binaire + premier modèle + calibration machine
uv run python -m loom.web     # UI chat sur :8000
```

`loom.web` lance le moteur llama.cpp à la demande et l'arrête avec l'interface.
`uv run loom/runtime/serve.py` reste disponible pour servir le moteur seul. Ouvrir
http://127.0.0.1:8000. La plupart des réglages (permissions, budgets, keep-warm…)
s'appliquent sans redémarrer loom.web.

**Tests** : suite pytest, analyse Ruff, self-test d'éval hors ligne et banc E2E Playwright.

```powershell
uv run pytest tests/ -q
uv run ruff check loom tests scripts evals
uv run python -m evals.run_eval --self-test
```

Le scénario navigateur isolé est décrit dans `tests/e2e/README.md`.

**Évals** (serveur modèle doit tourner) :

```powershell
uv run python -m evals.run_eval --self-test      # hors-ligne : valide les graders sans modèle
uv run python -m evals.run_eval --runs 3 --model qwen3.6-35b-a3b-abliterated   # A/B git HEAD vs disque (le default_model est un distant → forcer le local)
uv run python -m evals.run_review_eval           # éval du skill code-review
```

## Conventions

- **Gestionnaire de paquets** : `uv` (`uv run` / `uv sync`). Pas de requirements.txt, lock dans `uv.lock`.
- **Lint/format** : `ruff` (PEP8, indentation). Formatage web via prettier (outil `format_code`).
- **Commits** : Conventional Commits courts, branche de référence `master`.
- **Config** : `config/defaults.toml` (versionné) + `config/local.toml` (surcharge machine, gitignored). Overrides GPU et chemins binaires dans `local.toml`.
- **Modèles** : découverts par dossier `loom/models/<id>/`, GGUF jamais versionné. `config/defaults.toml` → `[chat] default_model` choisit celui chargé au démarrage.
- **Front** : zéro build, libs JS vendorées dans `loom/web/static/` (Preact, htm, marked, MathJax, DOMPurify, htmx).
- **Prompts en ANGLAIS** (`chat.system.md`, `subagent.system.md`) : instructions EN (denses), **réponse dans la langue de l'utilisateur — FR par défaut, imposée dans le prompt**. Bascule validée par A/B (`evals/`).
- **Un gotcha d'outil se met dans le SCHÉMA de l'outil** (`ToolSpec(description=…)`), pas dans le prompt : le prompt ne porte que la politique. Les schémas (aussi en anglais) sont la source unique de la mécanique.
- **CI/CD** : aucune CI propre au projet détectée (pas de `.github/` à la racine du repo).

## Points d'attention

- **Mode permission livré à `allow`** : l'agent écrit et exécute du shell sans confirmation. La doc recommande de passer à `ask` (`config/defaults.toml` → `[permissions] mode`) hors environnement isolé.
- **La deny-list n'est pas une frontière de sécurité** : elle bloque les formes évidentes (`rm -rf`, `format`, `dd if=`) mais un interpréteur (`python -c`) la contourne. Ne pas s'y fier contre un modèle hostile.
- **Anti-SSRF + frontière de confiance** : `fetch_url`/`web_search` épingle l'IP et refuse les hôtes internes ; tout contenu externe est marqué DONNÉE (pas instruction). Défense en profondeur, pas garantie.
- **Validation multi-niveaux** : pytest couvre le runtime et les routes ; Ruff couvre le
  statique ; le self-test valide les graders sans modèle ; Playwright reste requis pour
  les interactions et le rendu navigateur.
- **Banc d'éval complet** : livré (graders code, juge LLM, historiques par commit).
  Le `--self-test` est hors ligne ; les comparaisons de prompts exigent un serveur modèle.
- **Éval** : `default_model` de config est un distant → forcer `--model qwen…` pour évaluer le local (seul à révéler une régression du prompt). Le jeu de cas **n'exerce pas `dispatch_agent`** → le prompt sous-agent n'est pas encore testé par l'A/B.
- **Agnostique machine** : la taille des modèles chargeables et le contexte local sont bornés par le matériel hôte (VRAM/RAM), mesurés à la calibration — aucune spec de machine n'est une constante du projet. Les analyses liées à une machine (bancs, perf, sondes) restent en local (`docs/local/`, `var/`, `config/local.toml` — gitignorés) ; le versionné ne porte que mécaniques et principes.
- **GPU multi-backends** : après installation du binaire, `llama-server --list-devices`
  est la source de vérité pour CUDA, Vulkan (AMD/Intel/NVIDIA) et Metal ; `nvidia-smi`
  n'est plus qu'un repli avant installation et pour la télémétrie NVIDIA riche.
- **Tranches plugins à venir** : hooks (PostToolUse, exécute du code tiers → porte de confiance) et agents (personas dispatchables).
- **Avant de proposer une piste** (orchestrateur, gating d'outils, édition par ligne, speculative decoding, sweeps batch…) : lire « Déjà essayé, rejeté » dans `ETAT_PROJET.md`. Plusieurs bonnes idées générales ont déjà été testées et falsifiées ici.
