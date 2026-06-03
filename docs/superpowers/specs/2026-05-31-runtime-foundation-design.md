# Sous-projet A — Fondation runtime (local LLM)

> Spec de design — 2026-05-31
> Statut : **approuvé en design, en attente de relecture du spec écrit**

## 0. Contexte & vision globale

Objectif global du projet `from-claude-to-local-haranessed-llm` : **rester productif sur tous mes
projets même sans internet**, en reproduisant mon usage agentique de Claude Code avec des
**modèles open-source locaux** (Mistral, Qwen, DeepSeek, Gemma…). Enjeu secondaire mais réel :
**démontrer le savoir-faire d'« internaliser le harness »** sur des modèles autres que Claude.

Le pari assumé : un petit modèle local (7B–14B) n'a **pas** la qualité brute d'Opus. L'écart se
comble par la **plomberie** : sortie contrainte, orchestration déterministe, décomposition en
rôles, RAG, mémoire. C'est ça la valeur du projet.

### Découpage en 3 sous-projets

| # | Sous-projet | Livrable | Statut |
|---|---|---|---|
| **A** | **Fondation runtime** | Un modèle local benchmarké, servi via API OpenAI-compatible, reproductible multi-machines | **CE SPEC** |
| B | Boucle agentique | Réutiliser un harness (Aider/Cline/OpenHands) branché sur le runtime local pour éditer un vrai projet, offline | À venir |
| C | Couche d'orchestration « maison » | Multi-agents à rôles + workflows + RAG + **mémoire/contexte** = le harness par-dessus | À venir |

Chaque sous-projet a son propre cycle spec → plan → implémentation. **Ce document ne couvre que A.**

---

## 1. Objectif du sous-projet A

Avoir **un modèle open-source qui tourne sur la machine, servi derrière une API
OpenAI-compatible, benchmarké, reproductible sur une autre machine, et avec le choix de runtime
documenté** — pour que le sous-projet B puisse s'y brancher, **100 % hors-ligne**.

A est une **fondation d'infrastructure** : un squelette reproductible, pas une application.

## 2. Principe d'architecture directeur

**Frontière d'abstraction = l'API OpenAI-compatible (`/v1/chat/completions`).**

- C'est le standard de facto → on ne réinvente pas la roue.
- Tout ce qui sera construit au-dessus (harness B, agents C) ne parle **qu'à ce contrat**.
- Le runtime en dessous devient **interchangeable** (llama.cpp aujourd'hui, Ollama/vLLM demain)
  sans rien casser au-dessus. **Zéro lock-in.**

## 3. Choix de runtime : llama.cpp (`llama-server`)

Runtime de référence retenu : **llama.cpp**, via son binaire `llama-server`.
Décision à formaliser dans `docs/adr/0001-llamacpp-vs-ollama.md`. Rationale ci-dessous.

### Comparaison llama.cpp vs Ollama (cible : RTX 2060 6 Go / Windows + VPS Linux CPU)

| Critère | **llama.cpp** (`llama-server`) | Ollama |
|---|---|---|
| Install | Binaires prébuildés (CUDA/CPU) ou compilation. Friction moyenne | Installeur, daemon auto. Friction quasi nulle |
| Gestion modèles | **GGUF brut** depuis HuggingFace (fichier nu, aucun registre) | Registre + format de blobs propriétaire (`ollama pull`) |
| Agnosticisme / lock-in | ✅ Maximal — c'est *le moteur sous* Ollama/LM Studio | ⚠️ Wrapper par-dessus ; dépendance à Ollama |
| API OpenAI-compat | ✅ Native | ✅ Native |
| Sortie contrainte | ✅ **GBNF complet** (grammaires arbitraires) | ⚠️ JSON-schema seulement |
| Contrôle (offload GPU, contexte, flags) | ✅ Total (`-ngl`, `-c`, `-t`…) | ⚠️ Largement automatique, peu réglable |
| Vitesse 10 min → productif | ⚠️ Plus de setup manuel | ✅ Imbattable |

**Décision : llama.cpp**, car il satisfait les deux contraintes structurantes de l'utilisateur :
1. **Agnosticisme / pas de techno « installe un modèle »** — un GGUF est un fichier nu ; et comme
   tout passe par l'API OpenAI standard, on pourra brancher Ollama plus tard sans rien recasser.
2. **Ne pas réinventer la roue** — llama.cpp *est* la roue de référence ; Ollama en est un wrapper.

Coût accepté : ~15-20 min de setup en plus au départ. Bénéfice : contrôle total + GBNF (clé pour
fiabiliser un petit modèle) + couche supérieure 100 % agnostique.

> Note : **Kimi (K2)** envisagé initialement est un MoE ~1 000 Md de paramètres → **impossible**
> en local sur ce hardware. Écarté.

## 4. Exigence transverse : reproductibilité multi-machines

Cible double, **CPU/GPU-agnostique** :
- **Laptop** : Windows 11, RTX 2060 **6 Go VRAM**, 32 Go RAM, i7-10750H.
- **VPS** : Linux, **64 Go RAM, pas de GPU** (pour l'instant).

Le squelette ne doit **jamais supposer un GPU**. Mieux : `serve.py` **auto-détecte** le hardware
et choisit le meilleur profil (GPU si présent, sinon CPU) — donc la **même config versionnée
tourne de façon optimale sur les deux machines sans aucune édition**. Le seul delta réel est le
binaire `llama-server` (build CUDA sur le laptop vs build CPU sur le VPS), installé par machine.

## 5. Composants

1. **`llama-server`** (binaire llama.cpp) — serveur d'inférence. **Spécifique plateforme** (build
   CUDA sur le laptop, build CPU sur le VPS), installé hors-dépôt, version épinglée et documentée.
2. **`runtime/serve.py`** — lanceur **cross-platform et auto-adaptatif** (exécuté via `uv run`) :
   - **auto-détecte les ressources de la machine** et choisit le meilleur réglage *tout seul* :
     - cherche un GPU NVIDIA (présence de `nvidia-smi` + VRAM libre) ;
     - **si GPU trouvé** → estime combien de couches y tiennent (taille GGUF + cache KV vs VRAM)
       et règle `n_gpu_layers` au max raisonnable ;
     - **sinon** → bascule CPU (`n_gpu_layers = 0`) et règle `threads` selon les cœurs dispo ;
   - lit `runtime.config.toml` pour les paramètres **non liés au hardware** (modèle, contexte,
     port) ; les valeurs hardware de la config sont des **surcharges/plafonds optionnels**, pas
     des prérequis ;
   - vérifie la présence du GGUF, le **télécharge s'il manque** (via `huggingface_hub`) ;
   - démarre `llama-server` avec les flags calculés, et **logue** le profil détecté (GPU/CPU,
     n_gpu_layers, threads) pour la traçabilité.
   - Un seul point d'entrée, identique sur Windows et Linux. **Objectif : `uv run serve.py` sur
     une machine neuve démarre de façon optimale sans aucune édition.**
3. **`runtime/runtime.config.toml`** — config déclarative et versionnée :
   `model` (repo HF + fichier GGUF + quant, **épinglés**), `context`, `port`, chemin du binaire,
   et surcharges hardware **optionnelles** (`n_gpu_layers`, `threads`) si on veut forcer/plafonner
   l'auto-détection. Surcharge par machine via `runtime.config.local.toml` (gitignoré).
4. **`runtime/benchmark.py`** — mini-harness de mesure (via `uv`) : tok/s, *time-to-first-token*,
   et **smoke-test qualité** dont une requête **contrainte GBNF** vérifiant qu'on obtient du JSON
   garanti valide.
5. **Documentation** : `docs/install-windows.md`, `docs/install-linux.md`,
   `docs/adr/0001-llamacpp-vs-ollama.md`.

### Arborescence cible

```
runtime/
  models/                      # .gguf — gitignorés (trop lourds)
  serve.py                     # lanceur cross-platform (uv)
  benchmark.py                 # uv: tok/s, TTFT, validité GBNF/JSON
  runtime.config.toml          # config versionnée (laptop GPU par défaut)
  runtime.config.local.toml    # surcharge machine (gitignoré) — ex. VPS CPU
docs/
  adr/0001-llamacpp-vs-ollama.md
  install-windows.md
  install-linux.md
pyproject.toml                 # deps uv (huggingface_hub, requests/httpx…)
ETAT_PROJET.md                 # mis à jour
.gitignore                     # models/, *.local.toml, .venv, caches
```

## 6. Flux de données

```
client (curl / benchmark.py / futur harness B)
        │  HTTP localhost:<port>/v1/chat/completions
        ▼
   llama-server  ──►  GGUF chargé sur GPU (≤6 Go) + CPU/RAM
        │
        ▼  réponse (streaming ou non)
   AUCUN appel réseau sortant
```

## 7. Stratégie modèle

Modèle par défaut : **Qwen2.5-Coder-7B-Instruct** (quant `Q4_K_M`, ~4.7 Go) — meilleur codeur
open + bon tool-use dans cette taille, tient en VRAM 6 Go.

Comparants à benchmarker (mêmes critères) :
- **DeepSeek-Coder-V2-Lite** (16B MoE, ~2.4B actifs) — offload partiel, rapide malgré la taille.
- **Mistral-7B-Instruct** — baseline généraliste.
- **Gemma 2 9B** — généraliste ; serré sur 6 Go (offload partiel).

Stratégie VRAM (laptop) : 7B `Q4_K_M` → poids ~4.7 Go ; régler `n_gpu_layers` au max qui tient avec
le cache KV, reste sur CPU. Le benchmark détermine le contexte max qui tient. **Sur le VPS :
`n_gpu_layers = 0` (tout CPU), contexte plus large possible grâce aux 64 Go de RAM.**

## 8. Critères de succès (mesurables)

- [ ] `curl` vers `/v1/chat/completions` répond correctement, **wifi coupé** (laptop).
- [ ] **≥ ~15-20 tok/s** sur le 7B en GPU (assez fluide pour de l'interactif).
- [ ] Une requête **GBNF** renvoie un JSON **garanti valide** (parse OK 10/10).
- [ ] **Relance en une commande** : `uv run runtime/serve.py`.
- [ ] `serve.py` **auto-détecte** le hardware et logue le profil retenu (GPU → `n_gpu_layers`
      estimé ; CPU → `threads`) sans intervention.
- [ ] Le **même dépôt**, sans aucune édition, démarre **de façon optimale** sur une 2ᵉ machine
      (cible : VPS CPU → bascule CPU automatique).
- [ ] `benchmark.py` produit un rapport reproductible (tok/s, TTFT, validité JSON).
- [ ] **ADR + guides d'install committés.**

## 9. Tests

- `benchmark.py` sert aussi de test d'intégration : il échoue (exit ≠ 0) si l'endpoint ne répond
  pas, si tok/s < seuil configurable, ou si la sortie GBNF n'est pas un JSON valide.
- Code Python passé par le hook `ruff` (lint `--fix` + format) déjà en place.

## 10. Hors-scope (explicite)

- **Le harness agentique** (lecture/édition de fichiers, boucle outils) → sous-projet **B**.
- **Multi-agents à rôles, workflows, RAG** → sous-projet **C**.
- **Mémoire & gestion fine du contexte** → relève de **B/C** (noté ici pour ne pas l'oublier ;
  ce n'est *pas* une responsabilité du runtime A).
- **Fine-tuning / entraînement** de modèles.
- Support d'OS au-delà de Windows (laptop) et Linux (VPS).

## 11. Risques & points ouverts

- **Vitesse CPU sur VPS** : sans GPU, un 7B en CPU peut être lent (quelques tok/s). Acceptable pour
  A (objectif = squelette reproductible) ; à réévaluer pour B. Un MoE (DeepSeek-Coder-V2-Lite) ou
  un modèle plus petit pourrait être préférable côté VPS.
- **Build `llama-server`** : disponibilité d'un binaire CUDA prébuildé adapté (sinon compilation).
- **Épinglage GGUF** : enregistrer repo + nom de fichier + quant exacts (et idéalement un hash)
  pour une vraie reproductibilité.
