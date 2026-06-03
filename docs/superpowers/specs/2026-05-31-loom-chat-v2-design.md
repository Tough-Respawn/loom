# Loom Chat v2 — Vision (screenshots) + Thinking — Design

> Spec de design — 2026-05-31
> Statut : **approuvé verbalement, spec à relire**

## 0. Contexte

Loom Chat v1 (Flask + HTMX + SDK openai, mémoire JSON, streaming) est livré et fonctionne.
v2 ajoute deux capacités, **scope focalisé** (pas d'agents/rôles — reportés) :
1. **Vision** : lire un screenshot → Gemma 4 (multimodal natif) en extrait les données ; l'utilisateur
   itère ensuite dessus via la mémoire existante.
2. **Thinking** : Gemma 4 émet un raisonnement (`reasoning_content`) avant la réponse ; on l'affiche
   dans une zone grisée repliable au lieu de laisser l'UI muette.

Faisabilité vérifiée : `unsloth/gemma-4-E4B-it-GGUF` fournit `mmproj-F16.gguf` (~990 Mo) ;
llama-server fait la vision via `-m model.gguf --mmproj mmproj.gguf` + l'API OpenAI (image_url base64).

## 1. Décisions

- **Vision = native Gemma 4 via mmproj** (pas d'OCR séparé). Téléchargement du mmproj + flag
  `--mmproj` sur llama-server.
- **VRAM 6 Go serrée** : par défaut, l'encodeur vision reste sur **CPU** (`--no-mmproj-offload`)
  pour garder le LLM sur GPU ; le traitement image (une fois par message) y est un peu plus lent,
  acceptable. Réglable.
- **Thinking = affiché** (grisé/repliable) + réponse finale.

## 2. Changements par unité (extension de l'existant)

### 2.1 `loom/config.py` + `loom/loom.config.toml`
- `[model]` : ajout `mmproj_filename = "mmproj-F16.gguf"` (chaîne, peut être `""` pour désactiver).
- `ModelConfig` gagne `mmproj_filename: str`.

### 2.2 `loom/serve.py`
- Si `mmproj_filename` non vide : `ensure_model` le télécharge aussi, et `build_server_args` ajoute
  `--mmproj <path>` + `--no-mmproj-offload`.
- `build_server_args` gagne un paramètre optionnel `mmproj_path: str | None = None`.

### 2.3 `loom/conversation.py` — messages multimodaux
- `add(role, content)` accepte `content: str | list` (texte simple OU liste de parts OpenAI).
- `save`/`load` : le JSON gère str ET list sans changement de logique (validé par tests).

### 2.4 `loom/client.py` — streamer reasoning + content
- `_iter_events(stream)` (remplace l'usage de `_iter_deltas`) yield des tuples taggés :
  `("reasoning", txt)` si `delta.reasoning_content`, `("content", txt)` si `delta.content`
  (ignore les vides). Lecture via `getattr(delta, "reasoning_content", None)` (robuste au SDK).
- `stream_chat(messages, system_prompt) -> Iterator[tuple[str, str]]` : passe les messages tels
  quels (multimodaux compris) et yield les events.

### 2.5 `loom/web/app.py` — `/chat` v2
- Accepte `message` (texte) + **image optionnelle** (`request.files["image"]`, encodée base64 →
  data URI). Construit le `content` user : `str` si pas d'image, sinon
  `[{type:"text",text:msg}, {type:"image_url",image_url:{url:"data:<mime>;base64,<...>"}}]`.
- SSE émet `{type:"reasoning",text}`, `{type:"text",text}`, puis `{type:"done"}` /
  `{type:"error",message}`. L'assistant persisté = le `content` (texte final) uniquement.

### 2.6 `loom/web/templates/index.html` — UI v2
- **Coller un screenshot** (handler `paste`) OU bouton fichier → aperçu miniature → envoyé en
  `multipart/form-data` avec le message.
- Rendu : bloc **« 🧠 réflexion »** grisé repliable (rempli par les events `reasoning`) + la réponse
  (events `text`). Le rendu de l'historique gère un `content` liste (texte + vignette image).

## 3. Flux (avec image)
```
Coller screenshot ──POST /chat (message + image)──► add(user, [text,image]) + save
   ──► LoomClient.stream_chat ──► llama-server (mmproj) 
        ◄─ SSE: reasoning… puis text… ─┘
   ──► UI: réflexion grisée + réponse ──► add(assistant, texte) + save
```

## 4. Tests
- `conversation` : `add`/`save`/`load` round-trip avec `content` **liste multimodale** (en plus du str).
- `client` : `_iter_events` sépare reasoning et content depuis des chunks mockés
  (SimpleNamespace avec `reasoning_content` et `content`) ; ignore les vides.
- `web` : `/chat` avec client mocké émettant reasoning+text → le flux SSE contient `reasoning`
  PUIS `text` PUIS `done` ; `/chat` avec une image (fichier mocké) construit un message `content`
  liste ; l'historique avec image se rend sans crash.

## 5. Robustesse
- Image trop grosse (> ~10 Mo) → rejet 400 clair.
- Type non-image → rejet.
- mmproj absent/désactivé → vision indisponible mais le chat texte marche (dégradation propre).
- Erreurs de stream → event `error` (comme v1).

## 6. VRAM / lancement
- 1er run v2 : `ensure_model` télécharge le mmproj (~990 Mo) en plus.
- `serve.py` ajoute `--mmproj <path> --no-mmproj-offload`. On valide à l'exécution que ça tient
  (LLM GPU + vision CPU). Si trop serré, baisser `n_gpu_layers` via l'override.

## 7. Hors-scope v2
Agents/rôles/outils, RAG, **audio** (Gemma 4 le gère mais on s'en tient à l'image), gestion de la
fenêtre de contexte (un screenshot consomme beaucoup de tokens visuels — limitation assumée).

## 8. Dépendances
Aucune nouvelle (base64 = stdlib ; openai/flask déjà là).
