# 🧵 Loom — un agent IA local, multimodal et offline

<!-- RÔLE : pitch public (présentation, démarrage rapide). Suivi interne : ETAT_PROJET.md. Carte technique : loom.md. Historique versions : CHANGELOG.md. -->

> Un modèle open-source qui **agit** sur ta machine avec des outils —
> **100 % en local, sans internet.**

Loom est un **agent tool-use** local : une boucle qui donne à un modèle local les bons outils
et la logique de les enchaîner, pour qu'il **localise, lise, modifie et exécute** sur tes vrais
projets au lieu de seulement discuter. Le pari : rester productif même internet coupé, et
démontrer le savoir-faire d'« internaliser le harness » sur un modèle autre que Claude.

Le harness, c'est ça et rien d'autre : **les outils, et l'intelligence d'appeler le bon au bon
moment.** Pas de pipeline déterministe, pas de rail de réflexion, pas de mode « build » séparé
— un seul chemin, l'agent qui agit (`stream_chat_tools`).

## Ce que ça fait

- 💬 **Chat web** local (Flask + Preact/htm, zéro build) avec **streaming SSE** et markdown.
- 🧰 **~23 outils** exposés au modèle, regroupés par usage :
  - **Localiser** : `find_files` (glob), `search_text` (grep), `list_dir`.
  - **Lire** : `read_file` (texte, et PDF / Excel / Word → texte extrait automatiquement),
    `read_image` (voir une image du disque : capture, schéma).
  - **Planifier / déléguer** : `manage_todos` (bloc-notes de plan), `dispatch_agent`
    (sous-agent à contexte isolé qui fait un gros chantier et ne renvoie qu'une synthèse).
  - **Modifier / créer** : `write_file`, `append_file`, `edit_file`
    (remplacement exact-unique), `format_code` (ruff Python / prettier web).
  - **Exécuter** : `run_shell` (PowerShell/bash, garde-fou deny-list, tue l'arbre au timeout).
  - **Web** : `web_search`, `fetch_url` (dégradés proprement hors-ligne).
  - **Vérifier le rendu** : `check_page` (charge une page HTML headless, exécute le JS,
    renvoie erreurs console + un **diagnostic de localisation** si la page hang),
    `check_interactive` (joue des clics/saisies réels + post-conditions DOM pour PROUVER
    qu'une page est jouable).
  - **Skills / plugins** : `use_skill` (charge un skill du catalogue), `list_plugins`,
    `add_marketplace`, `install_plugin`.
- 🧭 **Boucle agentic, pas déterministe** : l'arrêt suit le **stop naturel** du modèle (il
  répond sans appel d'outil → fini). Par-dessus, des **garde-fous** non-bloquants : plafond de
  tours et détecteur de non-progrès (anti-boucle). *Pas de mur de temps* (retiré : il
  décapitait le raisonnement).
- 🔒 **Mode permission** : en mode `ask`, confirmation interactive (Autoriser/Refuser) avant
  chaque écriture / commande shell ; en mode `allow`, l'agent agit sans demander. Par-dessus,
  une **deny-list** filtre quelques commandes destructrices évidentes (`rm -rf`, `format`,
  `dd if=`…). L'installation de plugins (code tiers) est gardée `ask`. Détails et limites :
  voir [Sécurité](#sécurité--périmètre-de-confiance).
- 🛡️ **Sécurité de l'ingestion** (active même hors-ligne) : garde **anti-SSRF** + **frontière
  de confiance** — tout contenu externe (URL, PDF, image, sortie d'outil) est marqué comme
  DONNÉE à analyser, jamais comme instructions à exécuter.
- 🖼️ **Vision** : colle un screenshot dans le chat, *ou* laisse l'agent lire une image du
  workspace via `read_image` (modèle multimodal + projecteur `mmproj`).
- 🗂️ **Sessions** : un fil persistant par projet (historique + outils actifs), bascule/
  création/suppression depuis l'UI ; **titre inféré** automatiquement par le modèle.
- 🧩 **Skills déclenchés par le modèle** (format `SKILL.md` façon Claude Code) : le prompt
  système annonce un **catalogue** `nom : description` ; quand un skill est pertinent, le
  modèle appelle `use_skill(name)` pour charger ses instructions. Plus de case à cocher.
- 🔌 **Plugins compatibles Claude Code** : Loom héberge son propre store (marketplaces +
  install + cache, même format que CC) ; installe **n'importe quel** plugin CC et ses **skills**
  deviennent disponibles au modèle local. (Hooks/agents : tranches suivantes.)
- 🐛 **Skill de debug** intégré : méthode reproduire → localiser → cause racine → fix minimal
  → preuve forte → réécrire-si-pourri, déclenchée quand un bug apparaît.
- 💭 **Raisonnement** : le « thinking » s'affiche dans un bloc animé, désactivable d'un clic.
- ⏹️ **Interruption** : soumettre un nouveau message stoppe net la génération en cours.
- 🔀 **Multi-modèles** : modèles découverts par dossier `loom/models/<id>/` + sélecteur UI ;
  hot-swap via llama-swap.

## Architecture

```
Navigateur ──HTTP──► Loom (Flask :8000) ──OpenAI API──► llama-server (:8080) ──► GGUF (GPU+CPU)
                        │
                        ├─ agent/           (boucle tool-use : client, conversation, context, session, inline_image)
                        ├─ tools/           (capacités appelées par le modèle : localiser, lire, modifier, exécuter, web, todos, sous-agent…)
                        ├─ extend/          (skills = catalogue + use_skill ; plugins = store compatible Claude Code)
                        ├─ permissions.py   (deny-list dure + allow/ask/deny — filtre les appels d'outils)
                        ├─ config.py        (config transverse depuis config/defaults.toml + local.toml)
                        ├─ prompts/         (prompts système)
                        ├─ runtime/         (llama.cpp : serve, swap, hardware, server_args, fetch, profils)
                        └─ web/             (serveur Flask + SSE + templates/static — la couche UI)
```

- **Runtime** : [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`), derrière une
  **API OpenAI-compatible**. Choix documenté dans
  [docs/adr/0001-llamacpp-vs-ollama.md](docs/adr/0001-llamacpp-vs-ollama.md).
- **Lanceur auto-adaptatif** : `loom/runtime/serve.py` détecte le GPU (sinon CPU) et règle l'offload
  selon la VRAM libre (`nvidia-smi`). Inclut `--jinja` (appels d'outils) et `--mmproj` (vision).
- **Modèles** : des **MoE 24B+** rendus jouables sur 6 Go de VRAM par **offload des experts en
  RAM** (`--cpu-moe` / `--n-cpu-moe` ; attention/dense sur GPU, experts routés en RAM). Par
  défaut `gemma4-26b-a4b-uncensored` ; `qwen3.6-35b-a3b-abliterated` dispo (avec vision).
  Chacun branche les siens — voir [loom/models/README.md](loom/models/README.md) et
  `loom/models/_TEMPLATE/`.

## Démarrer

Prérequis : [`uv`](https://docs.astral.sh/uv/). Le binaire `llama-server` et le
premier modèle s'installent tout seuls via `loom-setup` (ou à la main :
[docs/install-windows.md](docs/install-windows.md) / [docs/install-linux.md](docs/install-linux.md)).

```bash
uv sync                      # dépendances + installe le package loom
uv run loom-setup            # installeur guidé : OS/GPU/RAM détectés, llama.cpp
                             # + un modèle qui fit — chaque action confirmée
uv run python -m loom.web    # interface chat sur :8000
```
Puis ouvre **http://127.0.0.1:8000**. `loom.web` démarre le serveur modèle toute
seule (enfant, à la demande) et l'éteint avec elle. (`uv run loom/runtime/serve.py`
reste dispo pour servir le moteur seul — et déclenche aussi l'installeur au premier
run sur machine vierge.)

> ⚠️ Avant de t'en servir, lis [Sécurité](#sécurité--périmètre-de-confiance) : Loom donne un
> shell et l'écriture disque au modèle. En dehors d'un environnement isolé, mets
> `[permissions] mode = "ask"`.

## Utiliser l'agent

- Les outils sont **armés par défaut** : décris une tâche (« résume cette facture », « où est
  défini X et corrige-le », « regarde ce screenshot »), l'agent enchaîne les outils tout seul.
- **Périmètre** = `workspace_dir` (ou le dossier détecté dans ton message). Lecture large ;
  écriture et shell gardés par le **mode permission** (`allow` = autonome, `ask` = confirmation).
- **Déléguer** : pour un gros chantier, l'agent peut lancer `dispatch_agent` — un sous-agent à
  contexte isolé qui travaille puis renvoie une synthèse.

## Sécurité & périmètre de confiance

Loom donne à un **modèle local** un shell et l'écriture sur ton disque. À traiter comme tel : le
harness n'est **pas un bac à sable**.

- **Le mode permission est ton vrai garde-fou.** En `ask` (recommandé), tu confirmes chaque
  écriture et chaque commande shell. En `allow`, l'agent agit sans rien demander : à réserver à
  un environnement que tu acceptes de voir modifié sans confirmation (VM, conteneur, compte
  dédié). Réglage dans [config/defaults.toml](config/defaults.toml) → `[permissions] mode`.
  Le défaut livré est `allow` ; **passe-le à `ask`** si tu n'es pas en environnement isolé.
- **La deny-list n'est pas une frontière de sécurité.** Elle bloque quelques formes évidentes de
  commandes destructrices (`rm -rf`, `format`, `dd if=`) pour éviter l'**accident**. Elle ne
  résiste **pas** à un modèle hostile ou manipulé : un interpréteur (`python -c …`), un alias ou
  un autre ordre d'arguments la contournent. Si tu n'as pas confiance dans le modèle ou dans ce
  qu'il lit, isole l'exécution.
- **Anti-SSRF (solide)** : `fetch_url` / `web_search` résolvent et **épinglent** l'IP validée,
  refusent les hôtes internes (loopback, lien-local, métadonnées cloud) et ne suivent pas les
  redirections.
- **Frontière de confiance (défense en profondeur)** : tout contenu externe (URL, PDF, image,
  sortie d'outil) est marqué comme **DONNÉE à analyser, jamais instruction à exécuter**. C'est un
  rappel injecté dans le contexte, pas une contrainte dure : il réduit le risque d'injection, il
  ne l'élimine pas.
- **Plugins = code + instructions tiers.** Installer un plugin clone un repo et ses skills
  deviennent appelables par le modèle. L'install est gardée `ask` ; n'ajoute que des marketplaces
  de confiance.

En clair : `ask` par défaut, et si tu veux l'autonomie `allow`, fais-le dans un environnement isolé.

## Skills & plugins

- **Skill local** : crée `loom/skills/<nom>/SKILL.md` (frontmatter `name`/`description` + corps).
  Il rejoint le catalogue ; le modèle l'appelle via `use_skill` quand c'est pertinent.
- **Plugin Claude Code** : ajoute une marketplace puis installe un plugin —
  `python -m loom.extend.plugins marketplace add <git-url>` puis `install <plugin>` (ou via les outils
  `add_marketplace` / `install_plugin`). Ses skills apparaissent au catalogue.

## Configuration

Tout est dans [config/defaults.toml](config/defaults.toml) (`default_model`, contexte, ports,
outils armés, permissions). Les réglages spécifiques à une machine (chemin du binaire, override
GPU) vont dans [config/local.toml](config/local.toml) — gitignoré, à copier depuis
`config/local.example.toml` et adapter à ta machine (les valeurs livrées sont celles d'une RTX 2060).

### Calibration machine : mesurée, jamais supposée

`defaults.toml` ne contient **aucune valeur liée à une machine** — les vrais réglages sortent du
bench de `loom-setup` (étape 4), qui **mesure au lieu de calculer** :

- **Topologie découverte** depuis le matériel et le modèle : `moe_hybride` (MoE + GPU : experts
  en RAM, attention + cache KV en VRAM), `gpu_dense` (tout en VRAM) ou `ram` (pas de GPU).
- **Contexte par pente mesurée** : deux chargements réels de `llama-server` (avec les flags
  exacts de l'exécutant — `server_args.py`) donnent le coût mémoire *réel* par token, puis une
  échelle de **vitesse en profondeur** valide barreau par barreau. On n'écrit que du vérifié :
  aucune formule de header GGUF ne survit aux architectures modernes (attention à fenêtre
  glissante : facteur ×5 constaté entre la formule et la mesure — [le rapport](docs/bench-contexte-2026-07-18.md)).
- **Décision tracée** : `local.toml [bench]` garde le *mécanisme* (`context_mecanisme`,
  `context_pente_kb_tok`, `context_valide_jusqua`) — on sait toujours d'où sort un chiffre.
  Le contexte calibré s'écrit aussi dans le `model.toml` du modèle benché : la vérité est
  **par modèle** (la pente KV dépend de l'architecture).
- **Fail-loud** : si le contexte tourne sur le repli neutre de `defaults.toml` (machine jamais
  calibrée), `loom.web` l'annonce au boot au lieu de brider en silence.

Re-calibrer (nouveau matériel, nouveau modèle par défaut) : supprimer la table `[bench]` de
`config/local.toml` et relancer `uv run loom-setup`. Compter 5-20 min (chargements réels).

## Statut

État détaillé : [ETAT_PROJET.md](ETAT_PROJET.md).

- ✅ Runtime auto-adaptatif (offload MoE), sessions (titre inféré), vision, thinking, interruption, multi-modèles.
- ✅ Agent tool-use : ~23 outils, politique de décision + séquencement dans le prompt système.
- ✅ Garde-fous de boucle (stop naturel + plafond de tours + anti-répétition).
- ✅ Sécurité d'ingestion : anti-SSRF + frontière de confiance (active même hors-ligne).
- ✅ Skills déclenchés par le modèle (catalogue + `use_skill`) ; store de plugins compatible CC.
- ✅ Workflows : `run_workflow` — un script Python (écrit par le modèle ou canonique d'un skill)
  orchestre N sous-agents (`agent`/`parallel`/`pipeline`, sortie structurée, rôles cheap/strong) ;
  les synthèses restent hors du contexte du fil principal.
- ✅ Calibration machine agnostique (`loom-setup`) : topologie découverte + contexte par pente
  mesurée et vitesse validée, décision tracée, fail-loud sur repli.
- 🔜 Banc d'éval (juge LLM), tranches plugins (hooks, agents), SearXNG, RAG (skills volumineux).

## Stack

Python 3.12+ · `uv` · `ruff` · Flask · Preact/htm · SDK `openai` · llama.cpp · MoE 24B+ (offload experts).
