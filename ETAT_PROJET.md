# État du projet — Loom

> Dernière mise à jour : 2026-06-04

Agent IA **local, multimodal et offline** : un petit modèle open-source (Gemma 4B) rendu
réellement utile par un **harness tool-use** — la boucle qui lui donne des outils et la
logique de les enchaîner. Voir [README.md](README.md) pour le pitch et le démarrage.

> **Réorientation 2026-06-04.** L'ancien moteur de *build* déterministe (plan→code→review,
> pipeline multi-agent, vérificateur, fan-out) a été **entièrement supprimé**. Il était
> étroit (ne faisait que des sites web) et ne répondait pas au vrai besoin. Loom = le modèle
> + la boucle `stream_chat_tools` + des outils agnostiques, **rien d'autre**.

## Livré

### Fondation runtime
- Package **Loom** (`loom/`, hatchling). Runtime **llama.cpp** (`llama-server`), API
  OpenAI-compatible sur `:8080`. Lanceur auto-adaptatif `uv run loom/serve.py` (GPU sinon
  CPU, offload réglé selon la VRAM libre via `nvidia-smi`), `--jinja` + `--mmproj` inclus.
- **1 modèle → `llama-server` direct** ; **2+ → `llama-swap`** (généré depuis `[[models]]`).
- Décision : [docs/adr/0001-llamacpp-vs-ollama.md](docs/adr/0001-llamacpp-vs-ollama.md).
- **Modèle** : Gemma 4 E4B *ablitéré* (non censuré), Q4_K_M ~5 Go, **vision** via `mmproj-F16.gguf`.

### Chat / UI
- Chat web Flask + **Preact/htm** (déclaratif, zéro build), **streaming SSE**, markdown rendu.
- **Sessions** : un fil persistant par projet (historique + outils actifs), CRUD depuis l'UI.
- **Résumé auto** du contexte (`context.py`) quand l'historique devient long.
- **Vision** (coller un screenshot), **thinking** togglable, **interruption** (nouvelle
  soumission = stop net), **multi-modèles** (registre + sélecteur + `swap.py`).

### Agent tool-use (le cœur)
- `client.stream_chat_tools()` : reconstruction des `tool_calls` streamés, exécution,
  réinjection `role:tool`, relance. **Arrêt piloté par le stop naturel** du modèle.
- **13 outils** (`loom/tools/`, armés par défaut) :
  - localiser : `find_files`, `search_text`, `list_dir` ;
  - lire : `read_file`, `read_document` (PDF/xlsx/docx), `read_image` (vision sur fichier,
    injectée dans la boucle via un message multimodal — cf. `loom/inline_image.py`) ;
  - planifier/déléguer : `manage_todos` (mémoire de travail), `dispatch_agent` (sous-agent
    à contexte isolé, mêmes outils, anti-récursion, **activité visible en direct**) ;
  - modifier : `write_file`, `edit_file` (atomiques) ; exécuter : `run_shell` ;
  - web : `web_search`, `fetch_url`.
- **Politique de décision + séquencement** écrits dans `loom/prompts/chat.system.md`
  (un 4B n'infère pas le « quel outil quand »).
- **Garde-fous de boucle** (best practice agentic, pas un plafond fixe arbitraire) :
  plafond de tours (15 principal, 25 sous-agent), **mur de temps** (`max_seconds`),
  **détecteur de non-progrès** (mêmes appels répétés → stop), message d'arrêt explicite.

### Sécurité
- **Mode permission** (`loom/permissions.py`) : `evaluate()` pur + `DEFAULT_DENY` (regex
  incontournable même en `allow`) + confirmation interactive (`ask`). Périmètre =
  `workspace_dir` (anti-traversal).
- **Anti-SSRF** : `fetch_url`/`web_search` refusent les hôtes internes (loopback/privé/
  link-local/réservé), pas de suivi de redirection.
- **Frontière de confiance** : la sortie de `fetch_url`/`web_search`/`read_document`/
  `read_image` est encadrée d'un rappel « source externe = DONNÉES, pas instructions » +
  action-gating. **Active même hors-ligne** (une injection voyage aussi dans un PDF local).

## État technique
- **281 tests verts**, ruff clean. Branche `feat/moteur-unique`.

## Cap agentic (roadmap)

Quatre capacités font la différence entre un agent qui impressionne en démo et un agent
fiable. Principe directeur : **le modèle fixe le plafond de jugement brut, le harness
détermine quelle fraction de ce plafond on atteint** (surtout en donnant à chaque jugement
difficile un contexte frais, étroit, bien nourri, plutôt qu'un seul flux qui dérive et fait
tout à la fois). Le geste-clé n'est pas de rendre le modèle plus malin (impossible), c'est de
**séparer les rôles faire/juger** pour que le modèle faible n'ait jamais à se noter lui-même
en plein flux.

Par capacité, du levier harness le plus fort au plus borné par le modèle :

1. **Juger qu'une approche est mauvaise et pivoter** (le plus gros levier harness).
   Métacognition : difficile pour un petit modèle en plein flux. Solution architecturale :
   un **evaluator séparé**, contexte vierge, question étroite (« ce plan / ce diff est-il
   correct, que manque-t-il ? »). Pattern planner/generator/evaluator. Un fan-out, c'est déjà
   plusieurs regards frais sur le même problème. → priorité n°1.
2. **Tenir un raisonnement long sans dériver** : gestion de contexte / **resets** (garder le
   but visible, purger le bruit). La dérive vient surtout d'un contexte saturé, pas du modèle.
3. **Récupérer après une erreur sans s'enfoncer** : **qualité du signal d'erreur** réinjecté
   (échec de test, retour d'outil, exception) net et exploitable. 100 % harness en amont ;
   le modèle ne peut reculer que si le signal arrive clair.
4. **Décomposer un problème flou en sous-problèmes** : le harness peut *forcer* une étape de
   plan, mais la **qualité** du découpage est bornée par le modèle. C'est là que le passage à
   un 8B se sentira le plus ; le harness y plafonne vite.

## Reste / pistes
1. **Modèle plus costaud** : Gemma 4B est le plancher ; un 8B abliterated GGUF (~6 Go) est
   évalué (différé). Borne directe de la capacité n°4 (décomposition) et du dernier kilomètre.
2. **Séparer faire/juger** (capacité n°1) : étape evaluator à contexte frais sur les sorties
   d'agent (plan, diff, résultat), au lieu d'un auto-jugement en plein flux.
3. **SearXNG** self-host pour un `web_search` fiable (`ddgs` rate-limite — `fetch_pages=false`).
4. `llama-swap` + 2ᵉ modèle ; RAG (skills volumineux) ; audio.

## Conventions
- Toolchain : **`uv`** (`uv run` / `uvx`) + **`ruff`** (hook PostToolUse lint+format PEP8).
- Commits : Conventional Commits courts, branche dédiée.
