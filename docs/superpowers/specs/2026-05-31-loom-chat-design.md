# Loom Chat (v1) — Design

> Spec de design — 2026-05-31
> Statut : **approuvé verbalement (Flask + SDK OpenAI), spec à relire**

## 0. Contexte

Première brique d'interaction avec Loom (sous-projet A = runtime livré). Objectif assumé de
l'utilisateur : **« quelque chose qui répond, basta »** — une interface simple, qu'on enrichira
plus tard. On ne touche presque pas au harness ici ; on pose juste la **mémoire** + le **client
modèle** + une **UI web minimale**. Inspiré des bons patterns du chatbot WassaSim (historique
`{role, content}`, streaming SSE, system prompt, gestion d'erreur), **allégé** (pas de DB,
rate-limit, idempotency, escalation, dashboard).

## 1. Objectif

Une interface web locale pour **parler à Loom**, avec :
- **mémoire de conversation persistée sur disque (JSON)** — survit au redémarrage,
- **vrai streaming token-par-token**,
- bouton **reset**,
- **system prompt configurable**.

## 2. Décisions techniques

- **Client modèle** : **SDK officiel `openai`** (généré par Stainless), pointé sur l'endpoint
  OpenAI-compatible de Loom (`http://127.0.0.1:8080/v1`, clé bidon). Pas de hand-rolling HTTP.
- **Framework web** : **Flask** (simple, sync, streaming via générateur).
- **Front** : HTMX pour la structure + un petit JS pour consommer le flux SSE et afficher les
  tokens en direct.
- Ports : Flask sur **8000**, llama-server sur **8080** (pas de collision).

## 3. Composants (3 unités à responsabilité unique)

### 3.1 `loom/client.py` — `LoomClient` (I/O modèle)
- Construit un `openai.OpenAI(base_url, api_key="loom-local")` depuis la config (port).
- `stream_chat(messages, system_prompt) -> Iterator[str]` : injecte le system prompt en tête,
  appelle `chat.completions.create(stream=True)`, yield `chunk.choices[0].delta.content` non vide.
- Lève une erreur claire si l'endpoint est injoignable (Loom non lancé).

### 3.2 `loom/conversation.py` — `Conversation` (la MÉMOIRE)
- État : `system_prompt: str`, `messages: list[dict{role, content}]`.
- `add(role, content)`, `reset()` (vide les messages, garde le system prompt),
  `to_messages() -> list[dict]`.
- `save(path)` / `load(path)` : sérialise/charge `{system_prompt, messages}` en JSON.
- Robuste au fichier absent/corrompu (repart d'une conversation vide).

### 3.3 `loom/web/app.py` — Flask (thin) + `loom/web/templates/index.html` (HTMX)
- `GET /` : rend la page avec l'historique courant.
- `POST /chat` : valide le message (non vide, ≤ 5000 car.), `conversation.add("user", msg)` +
  `save`, puis renvoie une **réponse SSE** (`text/event-stream`) via un générateur qui :
  - streame les tokens (`data: {"type":"text","text": "..."}`),
  - à la fin, `conversation.add("assistant", full)` + `save`, puis `data: {"type":"done"}`,
  - en cas d'erreur (endpoint down…), `data: {"type":"error","message": "..."}`.
- `POST /reset` : `conversation.reset()` + `save`, renvoie la page fraîche.

### Arborescence
```
loom/
  client.py
  conversation.py
  web/
    app.py
    templates/index.html
  data/                      # gitignoré
    conversation.json
```

## 4. Configuration

Nouvelle section dans `loom.config.toml` :
```toml
[chat]
system_prompt = "Tu es un assistant utile, concis et factuel. Réponds en français."
history_path = "loom/data/conversation.json"
web_port = 8000
```
(`conversation.py`/`web` lisent ces valeurs via `loom.config`.)

## 5. Flux de données
```
Navigateur ──GET /──► page (historique rendu)
Navigateur ──POST /chat (message)──► add(user)+save ──► LoomClient.stream_chat
                                                              │ (SDK openai, stream=True)
                                              ◄── tokens SSE ─┘
            ◄── affichage live ── done ──► add(assistant)+save
Navigateur ──POST /reset──► reset()+save ──► page fraîche
```

## 6. Robustesse (repris de WassaSim, allégé)
- Validation taille/vacuité du message.
- Générateur SSE tolérant : toute exception → event `error` ; l'UI affiche
  *« Loom n'est pas démarré ? Lance `uv run loom/serve.py` »*.
- Sauvegarde après chaque tour (user ET assistant) → rien n'est perdu en cas de crash.

## 7. Tests
- `conversation.py` : `add`, `reset`, round-trip `save`/`load`, fichier absent → conversation vide.
- `client.py` : extraction du delta depuis un chunk mocké (le SDK `openai` est mocké) ;
  filtrage des deltas vides.
- `web/app.py` : Flask `test_client` — `GET /` rend la page ; `POST /reset` vide l'historique ;
  `POST /chat` (client mocké) renvoie un flux SSE contenant `text` puis `done`.

## 8. Lancement
`uv run flask --app loom.web.app run --port 8000` (ou un petit `loom/web/app.py` exécutable) →
ouvrir `http://127.0.0.1:8000`. Prérequis : Loom (llama-server) lancé sur 8080.

## 9. Hors-scope v1 (→ sous-projet C / harness)
- Outils / boucle agentique (tool-use) — la structure `messages` est prête à l'accueillir.
- RAG, multi-conversations, auth, rate-limit.
- **Gestion de la fenêtre de contexte** : on envoie tout l'historique. ⚠️ Au-delà de ~4096 tokens
  (contexte Gemma), il faudra *trimmer*/résumer — prévu en sous-projet C. Limitation assumée en v1.

## 10. Dépendances ajoutées
`openai`, `flask` (+ `pytest` déjà présent pour les tests).
