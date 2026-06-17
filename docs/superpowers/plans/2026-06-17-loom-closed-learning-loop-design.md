# Loom — Closed Learning Loop + Mémoire à deux étages (façon Hermes Agent)

> Spec de design. Statut : en cours de validation (itération 2, 2026-06-17).
> Prochaine étape : plan d'implémentation (skill `writing-plans`).

## 1. Intention

Faire passer Loom d'un harness **stateless dans le temps** à un harness qui
**capitalise l'expérience** et **devient propre à l'utilisateur**, en important
l'état d'esprit de **Hermes Agent** :

- L'intelligence vit dans le **harness**, pas dans les poids du modèle. Rien ne
  fine-tune ; tout s'accumule en fichiers + store mémoire.
- La boucle se **referme** : chaque tour de travail laisse une trace qui rend
  l'agent plus compétent au tour suivant.
- La **mémoire est l'identité** : l'agent a sa propre persona (`SOUL.md`), sait
  qui est l'utilisateur (`USER.md`), et se souvient de ce qu'ils ont fait.
- La valeur (skills appris + mémoire) est **portable** et indépendante du modèle
  sous la loop.

Périmètre **strictement limité** à deux chantiers (le reste de Hermes — CLI /
Telegram / Discord / Slack, multi-backends d'exécution — est hors scope) :

1. **Skills auto-créés / auto-améliorés** (closed learning loop).
2. **Mémoire persistante à deux étages** : identité interne (markdown) +
   recall pluggable (provider local par défaut, externe en option).

## 2. Contraintes (non négociables)

- **Offline — non négociable.** Loom reste **100% offline**, sans réseau et sans
  process externe. **v1 ne livre QUE le provider `local`** (SQLite `sqlite3`
  stdlib + fichiers markdown). L'interface `MemoryProvider` existe pour garder la
  porte ouverte, mais les providers externes (mem0 / supermemory / redis) sont de
  **simples points d'extension documentés, non implémentés en v1, jamais activés
  par défaut**. Aucun chemin de code par défaut ne touche le réseau.
- **Pas de rail dans la loop d'action** : `stream_chat_tools` reste pure. Aucun
  « mur de temps », rien qui décapite le raisonnement. L'apprentissage est une
  **phase séparée** exécutée une fois l'agent au repos.
- **L'identité reste locale** : `SOUL.md` / `USER.md` ne sont **jamais** délégués
  à un service externe, même quand un provider de recall externe est branché.
  L'âme ne quitte pas la machine.
- **Gouvernable** : ce que l'agent s'auto-crée est isolé, lisible et retirable
  sans toucher aux skills « officiels ».
- **Rétro-compatible** : sessions JSON et skills existants marchent sans
  migration.

## 3. Architecture d'ensemble

```
        Check → Decide → Act → Observe → Evaluate
                  ▲                          │
                  │                          ▼
          (prochain tour)            reflect (post-tour)
          enrichi par      ┌───────────┴───────────┐
       identité+skills  skills appris          mémoire
                        (dossier séparé)   ┌───────┴────────┐
                                       identité          recall
                                    SOUL.md/USER.md   MemoryProvider
                                    (interne, always-on)  (local|mem0|…)
```

Deux sous-systèmes, **un cœur partagé** : l'étape de réflexion post-tour
`reflect`, qui referme la boucle.

## 4. Sous-système A — Skills auto-créés / auto-améliorés

### 4.1 Stockage séparé et gouvernable

- Nouveau dossier `loom/skills_learned/<name>/SKILL.md` (clé config
  `learned_skills_dir`, défaut `loom/skills_learned`).
- **Jamais** mélangé avec `loom/skills/` (officiels) ni les skills de plugins.
- Curation **réactive** : dossier séparé = inspection et purge triviales. **Live
  immédiatement** au catalogue, **retirable facilement**. Pas de
  staging/quarantaine (volontairement simple).

### 4.2 Namespace et catalogue

- Réutilise `collect_skills(namespace=...)` (`loom/extend/skills.py:48-49`,
  `:67-81`). Les skills appris reçoivent le namespace `learned` → nom catalogue
  `learned:<name>`.
- `render_catalog` (`skills.py:84-99`) affiche un **marqueur visuel** (ex. suffixe
  `⟳`) pour distinguer ce que l'agent s'est forgé.

### 4.3 Frontmatter enrichi

Extension de `_parse_skill_md` (`skills.py:23`) et du dataclass `Skill`
(`skills.py:15-20`) :

```markdown
---
name: swap-debug
description: Diagnostiquer un écart de baseline sur un swap…
learned: true
created_at: 2026-06-17T10:00:00Z
updated_at: 2026-06-17T10:00:00Z
uses: 3
---
<corps>
```

Champs nouveaux **optionnels** → rétro-compat totale avec les skills existants.

### 4.4 Cycle de vie

- **Créer** : `reflect` propose un skill quand le tour a révélé une procédure
  **réutilisable et non-triviale** → écrit un `SKILL.md`.
- **Améliorer** : quand l'agent a **utilisé** un skill appris (`use_skill` sur un
  `learned:*`) et que le tour révèle un raffinement, `reflect` réécrit le corps,
  incrémente `uses`, met à jour `updated_at`. (« improves them during use ».)
- **Anti-trivial / anti-doublon** : validation avant écriture (§6.3).

## 5. Sous-système B — Mémoire à deux étages

La mémoire se décompose en **trois couches** de durée de vie croissante. Les deux
premières sont **toujours internes** ; seule la couche recall est pluggable.

### 5.1 Couche session (working / court terme) — inchangée

Les sessions JSON existantes (`loom/data/sessions/<id>/session.json`) restent la
mémoire de travail d'un fil. Aucune modification. Hors scope d'apprentissage.

### 5.2 Couche markdown always-on (interne) — identité + mémoire dense

**Trois** fichiers markdown, **toujours locaux**, **lisibles et éditables à la
main**, injectés au system prompt **à chaque tour** (bornés par
`identity_max_tokens`, défaut ~600 au total), au même point que le catalogue de
skills (`app.py:386-397`) :

| Fichier (config) | Rôle | Qui écrit |
|---|---|---|
| `SOUL.md` (`soul_path`) | **identité/persona de l'agent** : caractère, ton, façon d'interagir | `reflect` (propositions conservatrices) **+** utilisateur, à la main |
| `USER.md` (`user_path`) | **profil utilisateur** : qui tu es, préférences, façons de bosser | `reflect` (promotion de faits récurrents) **+** utilisateur, à la main |
| `MEMORY.md` (`memory_md_path`) | **mémoire arbitraire / générale durable** : faits de haute valeur qui ne sont ni agent ni user (conventions projet, détails d'environnement, consignes permanentes) | `reflect` (consolidation depuis les épisodes) **+** utilisateur, à la main |

Markdown plutôt que base : c'est petit, dense, ça doit être **vu et corrigé**
sans SQL, et c'est git-friendly. Ces trois fichiers ne partent **jamais** vers un
provider externe.

**MEMORY.md vs épisodes (pas de doublon)** : `MEMORY.md` est le **condensé dense
always-on** (« le meilleur », borné) ; le store épisodique (§5.3) est la **longue
traîne cherchable** (détail brut, pull). `reflect` **promeut** les insights
récurrents de la longue traîne **vers `MEMORY.md`** (consolidation §6.5), pendant
que le store conserve tout le détail interrogeable via `recall`.

### 5.3 Couche recall (épisodes, long terme) — `MemoryProvider` pluggable

Interface unique, plusieurs implémentations sélectionnées par config
(`memory_provider`) :

```python
class MemoryProvider(Protocol):
    def remember(self, text: str, *, kind: str = "episodic",
                 source: str = "") -> None: ...
    def recall(self, query: str, *, k: int = 5) -> list[Snippet]: ...
```

| Provider (`memory_provider`) | Stockage | Offline | Statut |
|---|---|---|---|
| `local` *(défaut)* | **SQLite FTS5** `loom/data/memory.db` | ✅ total, stdlib | recommandé, zéro dépendance |
| `mem0` | mem0 self-hosted (lib ou serveur REST) + vector store local (Qdrant/Chroma) | ✅ possible (Ollama + local) | open-source, opt-in |
| `supermemory` | binaire self-hosted, embeddings locaux | ✅ self-hosted | **closed-source, self-host sous accord enterprise** → signalé |
| `redis` | **Redis Stack + RediSearch** (FTS + vector) | ⚠️ serveur séparé, pas natif Windows (WSL/Docker/Memurai) | opt-in, pour qui fait déjà tourner Redis / veut du vector / multi-machine. **Pas un défaut** : aucun gain perceptible en mono-utilisateur (le goulot est l'inférence LLM, pas le datastore) |

- **`local` = le SEUL provider livré en v1.** `sqlite3` + module **FTS5** (présent
  dans les builds CPython Windows). Full-text natif, offline, fichier unique, pas
  d'embeddings, pas de serveur. C'est la brique par défaut **et unique** de v1.
- **`mem0` / `supermemory` / `redis` = points d'extension, PAS v1.** L'interface
  les rend branchables plus tard (pour qui veut du retrieval sémantique ou du
  multi-machine) sans toucher au reste du code. Ils restent **opt-in, jamais par
  défaut, et hors périmètre d'implémentation v1** (cf. §2 : on reste offline).
  Un provider externe ne porterait jamais que la couche **épisodes**, jamais
  l'identité.

### 5.4 Rappel (choix validé : identité always-on + recherche à la demande)

- **Always-on** : bloc `SOUL.md` + `USER.md` concaténé et borné, injecté au prompt
  à chaque tour.
- **À la demande** : outil **`recall(query)`** → `provider.recall()`, renvoie le
  top-K épisodes pertinents, **résumés** en une synthèse dense au-delà d'un seuil
  (FTS5 → résumé LLM, cf. §6.6) pour ne pas noyer le modèle local. Le modèle
  interroge quand il en a besoin (« pas de rail »).

### 5.5 Écriture

- Outil **`remember(text, kind)`** (`kind` ∈ `episodic|memory|profile|soul`) :
  - `episodic` → `provider.remember()` (couche recall, longue traîne).
  - `memory`   → append/maj `MEMORY.md` (mémoire arbitraire durable).
  - `profile`  → append/maj `USER.md`.
  - `soul`     → append/maj `SOUL.md`.
  Le modèle peut capitaliser **à tout moment** dans la loop.
- L'étape `reflect` écrit les épisodes du tour, **promeut** les faits récurrents
  vers `USER.md` / `MEMORY.md`, et propose prudemment des touches d'identité dans
  `SOUL.md`.

### 5.6 Assemblage du contexte par tour (ordre déterministe)

Le contexte envoyé au modèle à chaque tour est une **pile ordonnée et stable**.
Les couches always-on vivent dans le **system prompt** (point d'injection
existant `app.py:386-397`, là où le catalogue de skills est déjà concaténé) ;
le reste est l'historique de messages.

```
┌─ SYSTEM PROMPT ─────────────────────────────────────┐
│ 1. Politique de base            (chat.system.md)     │
│ 2. Catalogue skills    (officiels + plugins +        │
│                         learned:*  avec marqueur ⟳)  │
│ 3. # Ton moteur                 (modèle local courant)│
│ 4. # SOUL.md                    (identité de l'agent) │  ← always-on
│ 5. # USER.md                    (profil utilisateur)  │  ← always-on
│ 6. # MEMORY.md                  (mémoire durable dense)│  ← always-on
└──────────────────────────────────────────────────────┘
┌─ MESSAGES (historique de la session) ───────────────┐
│ user / assistant / tool …                            │
│ ↑ les résultats de recall() apparaissent ICI comme   │
│   messages `tool`, à la demande (pas une couche fixe)│
└──────────────────────────────────────────────────────┘
```

- **SOUL.md / USER.md / MEMORY.md** : injectés **après** le catalogue et
  l'identité moteur, **avant** l'historique. Concaténés et bornés ensemble par
  `identity_max_tokens`.
- **`recall()`** n'est **pas** une couche fixe du prompt : ses résultats entrent
  dans l'historique comme messages `tool` quand le modèle l'appelle. La couche
  recall est donc *pull*, l'identité est *push*.
- **Interaction avec la microcompaction** (`loom/agent/context.py`,
  `client._microcompact_tools`) : la summarization et la microcompaction ne
  touchent **que l'historique** (résultats d'outils anciens, vieux messages) —
  **jamais le system prompt**. Donc identité, profil et catalogue **survivent
  toujours**, par construction. Aucun risque qu'un SOUL.md soit « compacté ».
- **Budget** : catalogue + SOUL + USER comptent dans le budget de contexte ;
  d'où le bornage `identity_max_tokens` et la troncature des descriptions du
  catalogue déjà en place (`render_catalog`).

## 6. Cœur partagé — l'étape `reflect` (post-tour)

### 6.1 Emplacement et déclenchement

- Se greffe **après `_persist()`** (`loom/web/app.py` ~ligne 549), hors de
  `stream_chat_tools`.
- **Garde-fou coût** : ne s'exécute **que si le tour a fait du vrai travail** —
  au moins `reflect_min_actions` appels d'outils mutants / éditions. Un tour de
  chat trivial ne déclenche aucun appel modèle supplémentaire. (Aligné sur le
  « agent-curated memory with periodic nudges » de Hermes : pas à chaque tour.)
- Activable via `reflect_enabled`.

### 6.2 Mécanique

- **Un seul appel modèle borné**, prompt dédié (`loom/prompts/reflect.system.md`).
  Entrée : trajectoire du tour (messages + outils appelés + résultats clés).
  Sortie : un JSON strict.

```json
{
  "new_skills":      [{"name": "...", "description": "...", "body": "..."}],
  "improved_skills": [{"name": "learned:...", "body": "..."}],
  "episodes":        [{"text": "..."}],
  "memory_updates":  ["fait durable arbitraire (→ MEMORY.md)…"],
  "user_updates":    ["fait stable sur l'utilisateur (→ USER.md)…"],
  "soul_updates":    ["touche d'identité de l'agent (→ SOUL.md)…"]
}
```

### 6.3 Validation avant écriture

- **Schéma strict** : JSON non conforme rejeté en silence (pas de crash, pas
  d'écriture). Un échec de `reflect` n'affecte jamais la réponse déjà rendue.
- **Anti-trivial** : pas de skill qui réinvente un outil existant (liste de motifs
  + longueur minimale du corps).
- **Anti-doublon** : épisodes → recherche préalable via le provider ; quasi
  identique → maj plutôt que duplication. Skills → clé = `name`. `USER.md` /
  `SOUL.md` → dédup ligne par ligne, croissance bornée (`identity_max_tokens`).
- **Permissions** : `reflect` écrit dans le répertoire de données de Loom
  (`skills_learned/`, `memory.db`, `SOUL.md`, `USER.md`, `MEMORY.md`), **pas**
  dans le workspace utilisateur → écriture interne, hors mode `ask`. Tout est
  loggé et prunable.

### 6.4 Pas de rail dans la loop

`reflect` est une **phase séparée** déclenchée une fois l'agent au repos.
`stream_chat_tools` reste strictement inchangée : pas de mur de temps, rien
décapité, « stop naturel » préservé.

### 6.5 Barre de qualité — ne garder que le meilleur

Principe directeur : **une mémoire qui accumule du bruit est un fardeau, pas un
atout.** Chaque écriture (par `remember()` en cours de tour comme par `reflect`
en fin de tour) passe une barre de sélectivité. On capitalise **ce qu'il faut de
mieux**, pas la transcription.

Critères de rétention (un souvenir/skill n'est écrit que s'il les satisfait) :

- **Durable, pas éphémère** : ce qui resservira (préférence stable, cause racine,
  piège récurrent, décision structurante), pas un détail transitoire du tour.
- **Réutilisable** : un skill seulement s'il **généralise** au-delà du cas présent.
- **Haut signal** : la *leçon*, pas le *log* (« le champ `dContract__c` est en
  anglais », pas le dump SOQL complet).
- **Consolidation > accumulation** : si un souvenir proche existe, on **fusionne /
  met à jour** plutôt que d'empiler (anti-doublon §6.3). Les fichiers markdown
  (`SOUL.md` / `USER.md` / `MEMORY.md`) sont **bornés** : à saturation, `reflect`
  **resserre** (réécrit plus dense) au lieu de gonfler. La longue traîne va dans
  le store épisodique, pas dans les fichiers always-on.
- **Préférer raffiner un skill existant** plutôt qu'en créer un quasi-redondant.

Le prompt `reflect.system.md` encode ces critères, et la validation §6.3 les
applique mécaniquement (longueur minimale, anti-trivial, anti-doublon, bornage).
Effet recherché : la mémoire **gagne en densité de valeur** avec le temps, elle
ne grossit pas linéairement.

### 6.6 Compaction & consolidation (aligné sur Hermes)

Constat (repo `nousresearch/hermes-agent`) : Hermes compacte le **contexte** à la
main (`/compress`), garde le **store épisodique non-élagué** (FTS5 +
**summarization au recall**, *« without aggressive pruning »*), et **consolide
par création de skills**. On s'aligne.

**Compaction de contexte — déjà acquise, on ne touche pas.** Loom a déjà la
microcompaction (`client._microcompact_tools`) + la summarization
(`context.summarize`) **automatiques** sur l'historique (cf. §5.6). C'est plus
avancé que le `/compress` manuel de Hermes. Aucun ajout.

**Store épisodique — pas de pruning par défaut.** FTS5 encaisse l'échelle ; on
**garde l'archive cherchable** plutôt que de supprimer. On ne fait qu'une **dédup
légère** (fusion des quasi-doublons, déjà couverte par l'anti-doublon §6.3). Un
nettoyage manuel reste possible (éditer/vacuum le `.db`), hors défaut.

**`recall` summarisé — la pièce clé volée à Hermes.** Le tool `recall(query)` ne
**dumpe pas** le top-K brut (ça noie un modèle local). Il fait **FTS5 → puis
résumé LLM** des hits en une **synthèse dense** :

- mode `summarize` (défaut `true`, configurable `recall_summarize`) : au-delà de
  `recall_summarize_threshold` hits, un appel modèle borné condense les extraits
  en une réponse courte et citée ; en deçà, renvoi direct.
- économe en contexte, et fidèle à « ne garder que le meilleur » **au moment de
  la lecture** aussi, pas seulement de l'écriture.

**La consolidation, c'est la promotion — pas la suppression.** La vraie
consolidation Hermes : les **épisodes récurrents deviennent des skills appris**
(`reflect`, §4.4) et les **faits récurrents montent dans `MEMORY.md`** (§5.2). On
**densifie vers le haut** (skills + markdown always-on) au lieu d'élaguer vers le
bas. Le store épisodique reste la longue traîne brute, interrogeable.

Hors v1 (optionnel ultérieur) : un équivalent `/compress` manuel pour le store, et
des compteurs `recall_count` / `last_recalled` si un pruning par usage devient
souhaitable. Pas nécessaires tant qu'on ne pruge pas.

## 7. Surface de modification (chirurgical)

| Fichier | Modification |
|---|---|
| `loom/extend/skills.py` | frontmatter étendu ; `collect_skills` agrège `skills_learned/` (namespace `learned`) ; marqueur catalogue |
| `loom/memory/__init__.py` *(nouveau)* | interface `MemoryProvider`, `Snippet`, sélection par config |
| `loom/memory/local.py` *(nouveau)* | provider `local` : SQLite FTS5 (`remember`/`recall`, schéma + triggers) |
| ~~`loom/memory/{mem0,supermemory,redis}_provider.py`~~ | **HORS v1** — points d'extension documentés seulement, non implémentés (on reste offline §2) |
| `loom/memory/identity.py` *(nouveau)* | lecture/écriture `SOUL.md` / `USER.md`, `identity_block(max_tokens)` |
| `loom/agent/reflect.py` *(nouveau)* | étape post-tour : trajectoire, appel modèle borné, validation, écriture |
| `loom/tools/memory.py` *(nouveau)* | outils `recall`, `remember` (pattern `ToolSpec`) |
| `loom/tools/__init__.py` | enregistre `recall` / `remember` dans `build_registry` |
| `loom/tools/base.py` | `recall` / `remember` dans `AVAILABLE_TOOLS` |
| `loom/web/app.py` | injecte le bloc identité au prompt (~386) ; appelle `reflect()` post-tour (~549) |
| `loom/config.py` | clés (§9) |
| `loom/prompts/reflect.system.md` *(nouveau)* | prompt de réflexion |
| `loom/prompts/chat.system.md` | mention mémoire, identité, skills appris |

## 8. Unités et frontières (testabilité)

- **`loom/memory/local.py`** — store pur, sans Flask ni modèle. Testable sur
  SQLite `:memory:`.
- **`loom/memory/identity.py`** — IO markdown pures pour `SOUL.md` / `USER.md` /
  `MEMORY.md` (fichiers ↔ texte), append/dédup, bornage du bloc always-on.
  Testable en isolation.
- **`loom/memory/__init__.py`** — `MemoryProvider` (Protocol) ; providers externes
  en **import paresseux** (jamais requis au démarrage si non sélectionnés).
- **`loom/agent/reflect.py`** — `reflect(trajectory, client, stores)
  -> ReflectResult` ; `validate_reflect_json` **pure** (testable sans modèle) ;
  appel modèle injecté/mockable.
- **`loom/tools/memory.py`** — `ToolSpec` minces délégant à
  `loom/memory/`.

## 9. Configuration (clés nouvelles, `[chat]` / `[memory]`)

```toml
[chat]
learned_skills_dir = "loom/skills_learned"
reflect_enabled    = true
reflect_min_actions = 1
identity_max_tokens = 400

[memory]
provider       = "local"         # local | mem0 | supermemory
db_path        = "loom/data/memory.db"
soul_path      = "loom/data/SOUL.md"
user_path      = "loom/data/USER.md"
memory_md_path = "loom/data/MEMORY.md"
recall_summarize           = true   # FTS5 → résumé LLM des hits (§6.6)
recall_summarize_threshold = 5      # au-delà de N hits, on condense
# provider externe (opt-in) :
# [memory.mem0]   base_url = "...", offline = true, vector_store = "qdrant"
# [memory.supermemory]  base_url = "...", api_key = "..."
```

Ajout dans `ChatConfig` + un `MemoryConfig` dans `load_config`
(`loom/config.py:166-220`), même pattern que l'existant.

## 10. Stratégie de test

- **`memory/local.py`** : création/recherche/dédup sur SQLite `:memory:` ; synchro
  triggers FTS (insert/update/delete).
- **`memory/identity.py`** : append/dédup `SOUL.md` / `USER.md` ; bornage du bloc.
- **`memory/__init__.py`** : sélection de provider ; provider externe absent →
  fallback propre / erreur explicite, jamais de crash au boot.
- **`reflect.validate_*`** : JSON malformé rejeté ; anti-trivial ; anti-doublon ;
  amélioration vs création.
- **`skills.py`** : `skills_learned/` agrégé sous `learned:` ; frontmatter étendu ;
  rétro-compat (skill sans champs `learned`).
- **Intégration légère** : tour ≥ `reflect_min_actions` → `reflect` déclenché ;
  tour trivial → non déclenché.

## 11. Gestion des erreurs

- Toute défaillance de `reflect` (modèle, JSON, IO, provider) est **non
  bloquante** : logguée, jamais propagée à la réponse utilisateur.
- Provider externe injoignable → log + dégradation : `recall`/`remember`
  renvoient un message neutre ; on n'interrompt pas le tour. (Option de fallback
  `local` configurable ultérieurement, hors scope v1.)
- SQLite : `memory.db` créé/migré à l'init (idempotent) ; lock → log + skip
  (best-effort, comme `last_model`).
- `recall` sur base vide / sans résultat → message neutre, jamais d'erreur.

## 12. Hors scope (rappel explicite)

- Gateways CLI / Telegram / Discord / Slack.
- Multi-backends d'exécution (Docker / SSH / Modal / Daytona / Singularity).
- Staging / quarantaine des skills appris (curation réactive suffit).
- UI dédiée de gestion mémoire/skills (purge = éditer les fichiers / le `.db`).
- **Implémentation des providers externes (mem0 / supermemory / redis)** — v1
  livre **uniquement `local`**. Ils restent des points d'extension documentés. On
  reste offline (§2).
- Fallback automatique provider externe → local (v2 éventuelle).

## 13. Principes respectés (contrôle final)

- Intelligence dans le harness, pas dans les poids ✅
- Pas de rail dans la loop d'action ✅
- Offline-first par défaut ; externes opt-in et self-hostables ✅
- Identité (SOUL/USER) toujours locale, jamais exfiltrée ✅
- Gouvernable (skills appris isolés ; identité en markdown lisible ; épisodes en
  un store inspectable) ✅
- Portable / no lock-in (`.md` + provider interchangeable = la valeur) ✅
```

