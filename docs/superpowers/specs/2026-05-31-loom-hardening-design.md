# Loom v3.3 — Hardening (robustesse + UI + doc) — Design

> Spec de design — 2026-05-31
> Statut : **approuvé verbalement (Tout + résumé auto), spec à relire**

## 0. Contexte

Loom chat (v1/v2) + skills (v3.2) fonctionnent. v3.3 **durcit le harness** (priorité), nettoie
l'**UI**, et complète la **doc**. Le contexte long est géré par **résumé automatique** (choix
utilisateur).

## 1. Pilier 1 — Harness ultra robuste

### 1.1 `loom/client.py` — résilience
- **Retry 503 / connexion** : si le serveur renvoie 503 (« Loading model ») ou est injoignable,
  réessayer avec backoff jusqu'à `request_timeout` (le modèle peut être en chargement).
- **`max_tokens`** : `stream_chat(messages, system_prompt, max_tokens)` — passé à l'API (fin des
  réponses tronquées).
- **timeout / max_retries** : client `OpenAI(..., timeout=..., max_retries=...)` depuis la config.

### 1.2 `loom/conversation.py` — save atomique
- `save` écrit un fichier `.tmp` puis `os.replace` (atomique) → pas de JSON corrompu si crash.

### 1.3 `loom/context.py` 🆕 — fenêtre de contexte par résumé auto
- `estimate_tokens(text) -> int` (heuristique ~ `len/4`).
- `conversation_tokens(system_prompt, messages) -> int`.
- `needs_summary(system_prompt, messages, budget) -> bool` (pur, testable).
- `summarize(conversation, client, budget, keep_recent)` : si au-dessus du budget, prend les vieux
  messages (tous sauf les `keep_recent` derniers), demande au modèle un **résumé concis**, et les
  **remplace par un seul message** `{"role":"user","content":"[Résumé de la conversation: …]"}`.
  Les `keep_recent` récents restent intacts. Mute la conversation + sauvegarde.
- Config `[chat]` : `context_token_budget` (déf. 3000), `keep_recent_messages` (déf. 6).

### 1.4 `loom/web/app.py` — concurrence + réponses vides
- **Verrou** `threading.Lock` : `/chat` acquiert le verrou (sinon **429 « occupé »**), le relâche à
  la fin du stream (`finally`) → un échange à la fois, pas de corruption de la conversation partagée.
- **Intégration contexte** : avant de streamer, appeler `context.summarize(...)` si nécessaire.
- **Réponse vide** (thinking-only) : si le contenu final est vide, persister/afficher un placeholder
  clair (« (le modèle a seulement réfléchi — augmente max_tokens) »).
- `max_tokens` passé à `client.stream_chat`.

## 2. Pilier 2 — UI clean

### `loom/web/templates/index.html` + statiques
- **Markdown rendu** : `marked.min.js` + `purify.min.js` (vendored offline). Pendant le stream, le
  texte s'accumule en brut ; au `done`, la bulle est re-rendue via
  `DOMPurify.sanitize(marked.parse(text))`. **Bouton « copier »** sur les blocs de code.
- **Historique de saisie** : ↑/↓ rappellent les messages envoyés (tableau JS).
- **Détails** : **Entrée** = envoyer, **Shift+Entrée** = nouvelle ligne ; indicateur « répond… ».

## 3. Pilier 3 — Doc

- **`README.md`** 🆕 : pitch (productivité offline via plomberie), archi (runtime llama.cpp +
  chat Flask + skills), comment lancer (`uv run loom/serve.py` + `uv run python -m loom.web`),
  modèle non censuré + vision, lien vers specs/ADR.
- **`ETAT_PROJET.md`** : section à jour (runtime A + chat v1/v2 + skills v3.2 + hardening v3.3 +
  roadmap llama-swap/picker).
- **`loom/benchmark.py`** 🆕 : comble le fichier référencé mais jamais créé (débit + validité JSON,
  comme prévu au T9 du runtime).

## 4. Découpage en 2 workflows
- **Workflow A — Robustesse + contexte** : §1.1–1.4 (client, conversation, context.py, web) + tests.
- **Workflow B — UI + doc** : §2 + §3 (template/statiques, README, ETAT, benchmark.py).

## 5. Tests
- `client` : retry sur 503 mocké, `max_tokens` transmis.
- `conversation` : save atomique (le fichier final est complet ; pas de `.tmp` résiduel).
- `context` : `estimate_tokens`, `needs_summary` (seuils), `summarize` (client mocké → vieux
  messages remplacés par 1 résumé, récents intacts).
- `web` : 429 si verrou pris ; réponse vide → placeholder ; résumé déclenché quand au-dessus budget
  (context mocké).
- UI : non testée unitairement (template) ; vérifiée au lancement réel.

## 6. Config `[chat]` enrichie
`max_tokens`, `request_timeout`, `max_retries`, `context_token_budget`, `keep_recent_messages`.

## 7. Dépendances
Aucune nouvelle côté Python (threading/os = stdlib). UI : `marked` + `dompurify` vendored (JS).

## 8. Hors-scope (roadmap)
llama-swap / picker multi-modèles, RAG, audio, agents/outils.
