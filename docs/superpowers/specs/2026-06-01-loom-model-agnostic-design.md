# Loom v4 — Model-agnostic (llama-swap + registre + picker) — Design

> Spec de design — 2026-06-01
> Statut : **approuvé verbalement, spec à relire**

## 0. Contexte

L'utilisateur veut que **le modèle soit un paramètre de premier ordre** : pluggable, changeable
selon le besoin/contexte, et ciblable par du code (fondation du **multi-agent** — différents
modèles par agent, lot suivant). Décisions actées :
- **Mécanisme = `llama-swap`** (proxy OpenAI Go qui charge/décharge à la demande selon `model`).
- **Multi-agent visé = différents modèles par agent** → on dimensionne pour le multi-modèle.
- Sur 6 Go VRAM : **un seul modèle chargé à la fois** (swap manuel/à la demande) ; sur meilleure
  infra le swap devient transparent. Limite assumée.

## 1. Objectif v4

- Un **registre de modèles** dans la config.
- **llama-swap** sert l'API OpenAI sur `:8080` et swappe le bon modèle selon le champ `model`.
- Le modèle est **sélectionnable dans l'UI** (par conversation) et **paramétrable par appel** (pour
  les futurs agents).

## 2. Architecture
```
chat → llama-swap (:8080, /v1/* + /v1/models) → swap auto → llama-server[id]
```
`serve.py` ne lance plus llama-server directement : il **génère le `llama-swap.yaml`** depuis le
registre et **lance llama-swap**.

## 3. Composants

### 3.1 `loom/config.py` + `loom/loom.config.toml` — registre
- `[model]` (unique) → **`[[models]]`** (liste). Chaque entrée :
  `id`, `repo`, `filename`, `mmproj_filename` (optionnel), `n_layers`, `size_mb`,
  `n_gpu_layers` (override optionnel par modèle).
- `default_model = "<id>"` (dans `[chat]`).
- `ModelConfig` gagne `id`. `RuntimeConfig` : `models: list[ModelConfig]` + `default_model: str`,
  + helper `model_by_id(id) -> ModelConfig`.
- On démarre avec **1 entrée** (le Gemma non censuré actuel). Ajouter un modèle = ajouter une
  entrée (il se télécharge au prochain `serve`) → **pas de gros download forcé maintenant**.

### 3.2 `loom/swap.py` 🆕 — génération du yaml
- `build_swap_config(models, profile, llama_bin, models_dir) -> dict` : pour chaque modèle, calcule
  `n_gpu_layers` (auto via `resolve_n_gpu_layers`, sauf override), construit la commande
  `llama-server` (réutilise `build_server_args`, port = macro `${PORT}` de llama-swap, `--mmproj`
  si présent), et renvoie le dict `{models: {id: {cmd: "..."}}}`.
- `write_swap_yaml(config: dict, path)` : sérialise en YAML (mini-sérialiseur maison ou `pyyaml` ;
  on tranche au plan — préférence : pas de nouvelle dép si simple).

### 3.3 `loom/serve.py`
- Pour chaque modèle du registre : `ensure_model` (download GGUF + mmproj si besoin).
- Génère le `llama-swap.yaml` via `swap.py`.
- **Lance llama-swap** (binaire configurable `swap_bin`, comme `server.bin`).

### 3.4 `loom/client.py`
- `build_create_kwargs(..., model)` et `stream_chat(messages, system_prompt, max_tokens, model)` :
  **le modèle est un paramètre de l'appel** (plus de `"local"` figé).

### 3.5 `loom/conversation.py`
- `Conversation` gagne `model: str` (modèle de la conversation), persisté ; défaut = `default_model`.
- `set_model(id)`.

### 3.6 `loom/web/app.py` + UI
- `GET /` passe la liste des modèles + le modèle actif.
- `POST /model` : `conversation.set_model(id)` + save.
- `/chat` : passe `conversation.model` à `client.stream_chat`.
- UI : **menu déroulant de modèles** (peuplé depuis la config / `/v1/models`). Changer → `POST /model`.
  L'UI affiche « ⏳ chargement du modèle… » possible au 1er usage (llama-swap recharge).

## 4. Flux
```
Sélection modèle (dropdown) → POST /model → conversation.model = id + save
→ /chat → client.stream_chat(..., model=id) → llama-swap charge/route le modèle id → réponse
```

## 5. Étapes machine (hors workflow)
- Télécharger le binaire **llama-swap** (release Windows), le référencer dans `loom.config.local.toml`
  (`[server] swap_bin = "..."`).
- `uv run loom/serve.py` : génère le yaml + lance llama-swap.

## 6. Tests
- `config` : parse `[[models]]` + `default_model` + `model_by_id`.
- `swap` : `build_swap_config` génère la bonne structure (cmd avec `${PORT}`, `--mmproj` sur les
  modèles vision, `-ngl` par modèle).
- `client` : `model` transmis dans les kwargs.
- `conversation` : `model` persisté (round-trip + défaut + vieux JSON sans le champ).
- `web` : `POST /model` met à jour la conv ; `/chat` envoie le bon `model` au client (mock).

## 7. Robustesse
- Modèle inconnu demandé → 400 / fallback sur `default_model`.
- GGUF manquant pour un modèle du registre → llama-swap échoue à le charger ; message clair.
- Swap = latence (reload) : l'UI indique le chargement.

## 8. Hors-scope (roadmap, ordonné)
1. **Couche outils / tool-use** 🔝 — un **tool `read_file`** (+ boucle tool-use façon Claude Code /
   WassaSim) pour que Loom **lise réellement les fichiers** (aujourd'hui un chemin n'est que du
   texte). Le plus gros déblocage pour bosser sur de vrais projets. **Prochain lot après v4.**
2. **Multi-agent** (rôles planner/coder/reviewer, chacun ciblant un modèle du registre).
3. RAG (gros skills), audio.

## 9. Dépendances
Binaire **llama-swap** (externe, installé). Python : éventuellement `pyyaml` (à trancher au plan ;
sinon mini-sérialiseur maison).
