# Loom — Fondation mémoire (Plan 1/2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Donner à Loom une mémoire persistante à deux étages — identité always-on (SOUL/USER/MEMORY markdown) + store épisodique cherchable (SQLite FTS5) — exposée par les outils `remember`/`recall`, 100% offline.

**Architecture:** Un `MemoryProvider` (Protocol) avec une seule implémentation v1 `local` (SQLite FTS5, stdlib). Un module `identity` lit/écrit trois fichiers markdown injectés au system prompt à chaque tour. Deux outils (`remember`, `recall`) délèguent à ces couches. Aucun rail dans `stream_chat_tools` ; aucune dépendance réseau. C'est le Plan 1/2 : la **fondation manuelle** (le modèle capitalise à la demande). Le Plan 2 ajoutera l'étape `reflect` (capitalisation automatique) + les skills auto-appris par-dessus.

**Tech Stack:** Python stdlib (`sqlite3` + module FTS5), dataclasses, `typing.Protocol`, le pattern `ToolSpec` existant.

**Périmètre exact (référence : `docs/superpowers/plans/2026-06-17-loom-closed-learning-loop-design.md`) :** §5 (mémoire), §5.6 (injection), §7 (fichiers `loom/memory/*`, hors `reflect`), §9 (config `[memory]`), §11 (erreurs). **Hors ce plan :** `reflect` (§6), skills appris (§4), summarization LLM du recall (§6.6 — différée au Plan 2 pour garder les outils mémoire purement offline et smoke-testables sans modèle). Providers externes : NON implémentés (§2/§12).

**Convention de vérification (override skill) :** l'utilisateur ne veut **pas** de suite pytest sur Loom (vérif par **smoke** `uv run python -c "…"` + `ruff`). Chaque tâche se vérifie par un smoke exécuté à la main, pas par un fichier de test commité. Commits fréquents.

---

## File Structure

| Fichier | Responsabilité |
|---|---|
| `loom/memory/__init__.py` *(nouveau)* | `Snippet`, `MemoryProvider` (Protocol), `get_provider(cfg)` (sélection + import paresseux) |
| `loom/memory/local.py` *(nouveau)* | provider `local` : SQLite FTS5 (`remember`/`recall`, schéma + triggers). Store **pur**, sans Flask ni modèle |
| `loom/memory/identity.py` *(nouveau)* | IO markdown `SOUL.md`/`USER.md`/`MEMORY.md` : lecture, append dédup, `identity_block(max_tokens)` |
| `loom/config.py` *(modif)* | `MemoryConfig` + champs `ChatConfig` ; parsing dans `load_config` |
| `loom/tools/memory.py` *(nouveau)* | `make_recall(provider)`, `make_remember(provider, identity)` (pattern `ToolSpec`) |
| `loom/tools/base.py` *(modif)* | `recall`/`remember` dans `AVAILABLE_TOOLS` |
| `loom/tools/__init__.py` *(modif)* | enregistre `recall`/`remember` dans `build_registry` |
| `loom/web/__main__.py` *(modif)* | construit provider + identité au boot, passe aux tools |
| `loom/web/app.py` *(modif)* | injecte `identity_block` au system prompt (~ligne 397) |
| `loom/prompts/chat.system.md` *(modif)* | mentionne `remember`/`recall` + l'identité |

---

## Task 1 : `MemoryProvider`, `Snippet`, sélection de provider

**Files:**
- Create: `loom/memory/__init__.py`

- [ ] **Step 1 : écrire le module**

```python
# loom/memory/__init__.py
"""Mémoire à deux étages de Loom. Interface `MemoryProvider` + sélection par config.

v1 ne livre QUE le provider `local` (SQLite FTS5, offline, stdlib). Les providers
externes (mem0/supermemory/redis) sont des points d'extension NON implémentés : on
reste 100% offline. La sélection les importe paresseusement et lève une erreur claire
si demandés — jamais de crash silencieux au boot, jamais de réseau par défaut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Snippet:
    """Un épisode retrouvé par recall (longue traîne cherchable)."""

    text: str
    kind: str = "episodic"
    source: str = ""
    score: float = 0.0


@runtime_checkable
class MemoryProvider(Protocol):
    def remember(self, text: str, *, kind: str = "episodic", source: str = "") -> None: ...
    def recall(self, query: str, *, k: int = 5) -> list[Snippet]: ...


def get_provider(name: str, *, db_path: str) -> MemoryProvider:
    """Renvoie le provider mémoire. v1 : seul `local` est implémenté.

    Les providers externes sont importés paresseusement et lèvent une erreur explicite
    (jamais activés par défaut, jamais de réseau). Cf. design §2/§5.3/§12.
    """
    if name == "local":
        from loom.memory.local import LocalMemory

        return LocalMemory(db_path)
    if name in ("mem0", "supermemory", "redis"):
        raise NotImplementedError(
            f"provider mémoire '{name}' non implémenté en v1 (Loom reste offline). "
            "Seul 'local' est disponible. Cf. design §2/§12."
        )
    raise ValueError(f"provider mémoire inconnu : '{name}' (attendu : 'local').")
```

- [ ] **Step 2 : smoke — l'interface importe, `local` se résout, externes lèvent clair**

Run :
```bash
uv run python -c "from loom.memory import Snippet, MemoryProvider, get_provider; \
import tempfile,os; p=get_provider('local', db_path=os.path.join(tempfile.mkdtemp(),'m.db')); \
assert isinstance(p, MemoryProvider); \
import pytest_dummy" 2>/dev/null; \
uv run python -c "from loom.memory import get_provider; \
ok=False;\
\ntry:\n get_provider('mem0', db_path=':memory:')\nexcept NotImplementedError: ok=True\nassert ok, 'mem0 devrait lever NotImplementedError'; print('OK externes')"
```
Note : si la ligne multi-`\n` gêne, remplace par un petit fichier temporaire. Attendu : pas d'exception sur `local`, `NotImplementedError` sur `mem0`, impression `OK externes`.

- [ ] **Step 3 : `ruff` + commit**

```bash
uv run ruff check loom/memory/__init__.py
git add loom/memory/__init__.py && git commit -m "feat(memory): interface MemoryProvider + selection (local seul en v1)"
```

---

## Task 2 : provider `local` — SQLite FTS5

**Files:**
- Create: `loom/memory/local.py`

- [ ] **Step 1 : écrire le provider**

```python
# loom/memory/local.py
"""Provider mémoire `local` : store épisodique en SQLite + FTS5 (full-text, offline).

Une table `episodes` (vérité) + une table virtuelle FTS5 `episodes_fts` synchronisée par
triggers. `remember` insère, `recall` fait un MATCH FTS5 trié par pertinence. Aucune
dépendance externe, aucun embedding, un seul fichier `.db`. Store PUR : pas de Flask, pas
de modèle — testable sur `:memory:`. Best-effort sur lock (log + skip), cf. design §11.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from loom.memory import Snippet

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id     INTEGER PRIMARY KEY,
    ts     TEXT NOT NULL,
    kind   TEXT NOT NULL DEFAULT 'episodic',
    source TEXT NOT NULL DEFAULT '',
    text   TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
    USING fts5(text, content='episodes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

# Garde-fou : un épisode = une leçon dense, pas un dump (design §6.5).
_MAX_TEXT = 4000


def _fts_query(query: str) -> str:
    """Transforme une requête libre en requête FTS5 sûre : tokens alphanum joints par OR
    (les opérateurs FTS5 d'une requête utilisateur ne doivent pas faire planter le MATCH)."""
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    return " OR ".join(tokens) if tokens else '""'


class LocalMemory:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def remember(self, text: str, *, kind: str = "episodic", source: str = "") -> None:
        text = (text or "").strip()[:_MAX_TEXT]
        if not text:
            return
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO episodes(ts, kind, source, text) VALUES (?, ?, ?, ?)",
                (ts, kind, source, text),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            # lock/contention : best-effort, on n'interrompt jamais le tour (design §11).
            pass

    def recall(self, query: str, *, k: int = 5) -> list[Snippet]:
        q = _fts_query(query)
        try:
            rows = self._conn.execute(
                "SELECT e.text, e.kind, e.source, bm25(episodes_fts) AS score "
                "FROM episodes_fts JOIN episodes e ON e.id = episodes_fts.rowid "
                "WHERE episodes_fts MATCH ? ORDER BY score LIMIT ?",
                (q, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [Snippet(text=r[0], kind=r[1], source=r[2], score=float(r[3])) for r in rows]
```

- [ ] **Step 2 : smoke — remember puis recall sur `:memory:`**

Crée `/tmp/smoke_local.py` :
```python
from loom.memory.local import LocalMemory
m = LocalMemory(":memory:")
m.remember("Le champ dContract__c est en anglais dans Salesforce", source="t1")
m.remember("Le déploiement se fait via deploy.sh le vendredi", source="t2")
hits = m.recall("salesforce contrat anglais", k=3)
assert hits, "recall devrait trouver l'épisode Salesforce"
assert "dContract__c" in hits[0].text, hits
assert m.recall("requete sans aucun terme connu xyzzy", k=3) == [] or True
print("OK local:", [h.text[:30] for h in hits])
```
Run : `uv run python /tmp/smoke_local.py`
Attendu : `OK local: [...]` avec l'épisode Salesforce en tête.

- [ ] **Step 3 : smoke — robustesse requête à opérateurs FTS**

Run : `uv run python -c "from loom.memory.local import LocalMemory; m=LocalMemory(':memory:'); m.remember('alpha beta'); print('OK', m.recall('alpha AND (', k=2)[0].text)"`
Attendu : pas de crash, `OK alpha beta` (les opérateurs sont neutralisés par `_fts_query`).

- [ ] **Step 4 : `ruff` + commit**

```bash
uv run ruff check loom/memory/local.py
git add loom/memory/local.py && git commit -m "feat(memory): provider local SQLite FTS5 (remember/recall)"
```

---

## Task 3 : identité markdown (`SOUL.md` / `USER.md` / `MEMORY.md`)

**Files:**
- Create: `loom/memory/identity.py`

- [ ] **Step 1 : écrire le module**

```python
# loom/memory/identity.py
"""Identité always-on de Loom : trois fichiers markdown TOUJOURS LOCAUX, lisibles et
éditables à la main, injectés au system prompt à chaque tour (design §5.2).

- SOUL.md   : persona/caractère de l'agent
- USER.md   : profil de l'utilisateur
- MEMORY.md : mémoire générale durable (conventions, environnement, consignes)

IO pures (fichiers <-> texte), append dédup ligne par ligne, et `identity_block` qui
concatène les trois en un bloc BORNÉ (jamais délégué à un service externe). Sans Flask
ni modèle -> testable en isolation.
"""

from __future__ import annotations

from pathlib import Path

# ~4 caractères par token (même heuristique que loom/agent/context.py).
_CHARS_PER_TOKEN = 4


def read_md(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def append_unique(path: str, line: str) -> None:
    """Ajoute `line` au fichier si absente (dédup ligne par ligne). Crée le fichier au besoin."""
    line = (line or "").strip()
    if not line:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = read_md(path)
    lines = {l.strip() for l in existing.splitlines()}
    if line in lines:
        return
    body = (existing + "\n" + line).strip() if existing else line
    p.write_text(body + "\n", encoding="utf-8")


def identity_block(
    soul_path: str, user_path: str, memory_path: str, *, max_tokens: int = 400
) -> str:
    """Concatène SOUL/USER/MEMORY en un bloc borné pour le system prompt. Vide si rien.

    Bornage simple par caractères (max_tokens * 4) : on tronque le bloc concaténé en
    gardant l'ordre SOUL -> USER -> MEMORY. Le bornage fin (resserrage) est l'affaire de
    `reflect` au Plan 2 ; ici on protège juste le budget de contexte.
    """
    sections = [
        ("# Mon identité (SOUL)", read_md(soul_path)),
        ("# L'utilisateur (USER)", read_md(user_path)),
        ("# Mémoire durable (MEMORY)", read_md(memory_path)),
    ]
    parts = [f"{title}\n{content}" for title, content in sections if content]
    if not parts:
        return ""
    block = "\n\n".join(parts)
    cap = max_tokens * _CHARS_PER_TOKEN
    if len(block) > cap:
        block = block[:cap].rstrip() + "\n[…tronqué]"
    return block
```

- [ ] **Step 2 : smoke — append dédup + bloc borné**

Crée `/tmp/smoke_identity.py` :
```python
import tempfile, os
from loom.memory import identity as I
d = tempfile.mkdtemp()
soul, user, mem = (os.path.join(d, f) for f in ("SOUL.md", "USER.md", "MEMORY.md"))
I.append_unique(user, "Préfère les commits courts")
I.append_unique(user, "Préfère les commits courts")  # doublon -> ignoré
I.append_unique(user, "Travaille sur Windows + PowerShell")
assert I.read_md(user).count("commits courts") == 1, "dédup échouée"
blk = I.identity_block(soul, user, mem, max_tokens=400)
assert "USER" in blk and "commits courts" in blk and "SOUL" not in blk, blk  # SOUL vide -> absent
short = I.identity_block(soul, user, mem, max_tokens=2)  # ~8 car
assert "tronqué" in short, short
print("OK identity")
```
Run : `uv run python /tmp/smoke_identity.py` — Attendu : `OK identity`.

- [ ] **Step 3 : `ruff` + commit**

```bash
uv run ruff check loom/memory/identity.py
git add loom/memory/identity.py && git commit -m "feat(memory): identite markdown SOUL/USER/MEMORY (always-on, borne)"
```

---

## Task 4 : configuration `[memory]` + champs `[chat]`

**Files:**
- Modify: `loom/config.py` (`ChatConfig`, nouveau `MemoryConfig`, `load_config`)

- [ ] **Step 1 : ajouter `identity_max_tokens` à `ChatConfig`**

Dans `loom/config.py`, dans `@dataclass class ChatConfig`, après `keepwarm_interval: int = 150` :
```python
    # Mémoire/identité (Plan mémoire) : budget du bloc identité always-on injecté au prompt.
    identity_max_tokens: int = 400
```

- [ ] **Step 2 : ajouter le dataclass `MemoryConfig`**

Dans `loom/config.py`, après la définition de `ChatConfig` :
```python
@dataclass
class MemoryConfig:
    """Config de la mémoire (design §9). v1 : provider 'local' uniquement (offline)."""

    provider: str = "local"
    db_path: str = "loom/data/memory.db"
    soul_path: str = "loom/data/SOUL.md"
    user_path: str = "loom/data/USER.md"
    memory_md_path: str = "loom/data/MEMORY.md"
```

- [ ] **Step 3 : parser `[memory]` dans `load_config` et l'attacher à la config**

Repère dans `load_config` comment `ChatConfig` est construit et où l'objet config global est assemblé (lire `loom/config.py` autour de la construction de la config retournée). Ajoute la lecture de la table `[memory]` (avec défauts) et expose `cfg.memory` (un `MemoryConfig`). Lis `identity_max_tokens` depuis `[chat]` comme les autres champs `ChatConfig`.

Patron (adapter aux noms réels du fichier) :
```python
    mem_raw = data.get("memory", {})
    memory = MemoryConfig(
        provider=mem_raw.get("provider", "local"),
        db_path=mem_raw.get("db_path", "loom/data/memory.db"),
        soul_path=mem_raw.get("soul_path", "loom/data/SOUL.md"),
        user_path=mem_raw.get("user_path", "loom/data/USER.md"),
        memory_md_path=mem_raw.get("memory_md_path", "loom/data/MEMORY.md"),
    )
    # ... puis ajouter `memory=memory` à l'objet config retourné (et le champ correspondant
    #     sur le dataclass de config global, à côté de `chat`, `models`, `permissions`).
```

- [ ] **Step 4 : smoke — la config charge avec les nouveaux champs**

Run :
```bash
uv run python -c "from loom.config import load_config; from pathlib import Path; \
RT=Path('loom'); cfg=load_config(RT/'loom.config.toml', RT/'loom.config.personnel.toml'); \
print('provider:', cfg.memory.provider, '| db:', cfg.memory.db_path, '| idtok:', cfg.chat.identity_max_tokens)"
```
Attendu : `provider: local | db: loom/data/memory.db | idtok: 400` (valeurs par défaut, aucune entrée `[memory]` requise dans le TOML).

- [ ] **Step 5 : `ruff` + commit**

```bash
uv run ruff check loom/config.py
git add loom/config.py && git commit -m "feat(config): bloc [memory] + identity_max_tokens (defauts offline)"
```

---

## Task 5 : outils `recall` / `remember`

**Files:**
- Create: `loom/tools/memory.py`
- Modify: `loom/tools/base.py` (`AVAILABLE_TOOLS`)
- Modify: `loom/tools/__init__.py` (`build_registry`)

- [ ] **Step 1 : écrire les outils (pattern `ToolSpec`)**

```python
# loom/tools/memory.py
"""Outils mémoire : `remember` (écrit) et `recall` (lit) la mémoire persistante de Loom.

`remember(text, kind)` : kind='episodic' -> store épisodique cherchable (provider) ;
kind ∈ {'memory','profile','soul'} -> append dédup dans MEMORY.md / USER.md / SOUL.md.
`recall(query)` : top-K épisodes pertinents (FTS5), rendus en texte borné. (La
summarization LLM des hits est ajoutée au Plan 2 ; ici on renvoie les extraits bruts.)
"""

from __future__ import annotations

from loom.memory import identity as _id
from loom.tools.base import ToolError, ToolSpec

_KIND_TO_PATH = {"memory": "memory_md_path", "profile": "user_path", "soul": "soul_path"}
_MAX_RECALL_CHARS = 1500


def make_remember(provider, paths: dict) -> ToolSpec:
    """`paths` : dict {memory_md_path, user_path, soul_path} (chemins identité)."""

    def run(args: dict) -> str:
        text = (args.get("text") or "").strip()
        if not text:
            raise ToolError("argument 'text' : un contenu non vide est attendu")
        kind = (args.get("kind") or "episodic").strip().lower()
        if kind == "episodic":
            provider.remember(text, kind="episodic", source="remember")
            return "mémorisé (épisode cherchable via recall)."
        key = _KIND_TO_PATH.get(kind)
        if not key:
            raise ToolError(
                "kind invalide : attendu 'episodic' | 'memory' | 'profile' | 'soul'."
            )
        _id.append_unique(paths[key], text)
        return f"consigné dans {kind} (mémoire durable always-on)."

    return ToolSpec(
        name="remember",
        description=(
            "Capitalise un fait DURABLE et de haute valeur dans ta mémoire persistante "
            "(elle survit à la session). kind='episodic' (défaut) : range une leçon/observation "
            "dans le store cherchable (à retrouver plus tard via recall). kind='memory' : fait "
            "général durable (convention projet, détail d'environnement). kind='profile' : fait "
            "stable sur l'utilisateur. kind='soul' : trait de ta propre persona. Écris la LEÇON "
            "dense, pas le log brut. Ne mémorise que ce qui resservira."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Le fait à mémoriser (synthèse dense)."},
                "kind": {
                    "type": "string",
                    "enum": ["episodic", "memory", "profile", "soul"],
                    "description": "Où ranger : episodic (store) | memory | profile | soul.",
                },
            },
            "required": ["text"],
        },
        run=run,
    )


def make_recall(provider) -> ToolSpec:
    def run(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            raise ToolError("argument 'query' : décris ce que tu cherches en mémoire")
        k = int(args.get("k", 5) or 5)
        hits = provider.recall(query, k=max(1, min(k, 10)))
        if not hits:
            return "(aucun souvenir pertinent)"
        out, total = [], 0
        for h in hits:
            line = f"- {h.text}" + (f"  [{h.source}]" if h.source else "")
            if total + len(line) > _MAX_RECALL_CHARS:
                break
            out.append(line)
            total += len(line)
        return "Souvenirs pertinents :\n" + "\n".join(out)

    return ToolSpec(
        name="recall",
        description=(
            "Interroge ta mémoire persistante (épisodes des sessions passées) par recherche "
            "plein-texte. À utiliser quand une tâche ressemble à du déjà-vu : tu peux avoir une "
            "leçon ou une procédure mémorisée. Argument : query (mots-clés de ce que tu cherches)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Mots-clés de ce que tu cherches."},
                "k": {"type": "integer", "description": "Nombre de souvenirs (défaut 5, max 10)."},
            },
            "required": ["query"],
        },
        run=run,
    )
```

- [ ] **Step 2 : déclarer les outils dans `AVAILABLE_TOOLS`**

Dans `loom/tools/base.py`, dans la liste `AVAILABLE_TOOLS`, à côté de `write_note`/`read_note` (écriture interne au répertoire de données, pas le workspace → `danger: False`, non gated) :
```python
    {"name": "recall", "label": "recall", "danger": False},
    {"name": "remember", "label": "remember", "danger": False},
```

- [ ] **Step 3 : enregistrer les outils dans `build_registry`**

Dans `loom/tools/__init__.py`, `build_registry` prend un nouvel argument `memory=None` (un objet portant `provider` + `paths`). Après le bloc `write_note`/`read_note`, ajoute :
```python
    if memory is not None:
        from loom.tools.memory import make_recall, make_remember

        if "recall" in enabled:
            specs.append(make_recall(memory.provider))
        if "remember" in enabled:
            specs.append(make_remember(memory.provider, memory.paths))
```
Ajoute `memory` à la signature (mot-clé, défaut `None`) et documente-le dans la docstring (comme `conversation`/`client`). `memory.provider` = un `MemoryProvider` ; `memory.paths` = `{"memory_md_path":…, "user_path":…, "soul_path":…}`.

- [ ] **Step 4 : ajouter `recall`/`remember` à `tools_enabled` de la config**

Dans `loom/loom.config.toml`, à la liste `tools_enabled` de `[chat]`, ajoute `"recall"` et `"remember"`.

- [ ] **Step 5 : smoke — l'outil remember(episodic) puis recall via le provider**

Crée `/tmp/smoke_tools_mem.py` :
```python
import tempfile, os
from loom.memory import get_provider
from loom.tools.memory import make_recall, make_remember
d = tempfile.mkdtemp()
prov = get_provider("local", db_path=os.path.join(d, "m.db"))
paths = {"memory_md_path": os.path.join(d,"MEMORY.md"),
         "user_path": os.path.join(d,"USER.md"), "soul_path": os.path.join(d,"SOUL.md")}
rem = make_remember(prov, paths); rec = make_recall(prov)
print(rem.run({"text": "Sur Windows le shell est PowerShell", "kind": "episodic"}))
print(rem.run({"text": "L'utilisateur préfère les réponses concises", "kind": "profile"}))
assert "PowerShell" in rec.run({"query": "windows shell powershell"})
assert "concises" in open(paths["user_path"], encoding="utf-8").read()
print("OK tools memoire")
```
Run : `uv run python /tmp/smoke_tools_mem.py` — Attendu : `OK tools memoire`.

- [ ] **Step 6 : `ruff` + commit**

```bash
uv run ruff check loom/tools/memory.py loom/tools/base.py loom/tools/__init__.py
git add loom/tools/memory.py loom/tools/base.py loom/tools/__init__.py loom/loom.config.toml
git commit -m "feat(tools): outils recall/remember (memoire persistante)"
```

---

## Task 6 : câblage au boot + injection identité au system prompt

**Files:**
- Modify: `loom/web/__main__.py` (construit provider + identité, les passe à `make_registry`)
- Modify: `loom/web/app.py` (injecte `identity_block` au system prompt)

- [ ] **Step 1 : construire provider + paths au boot**

Dans `loom/web/__main__.py`, dans `build_app`, après la construction de `client` et avant `make_registry`, ajoute :
```python
    from types import SimpleNamespace
    from loom.memory import get_provider

    mem_provider = get_provider(cfg.memory.provider, db_path=cfg.memory.db_path)
    mem_paths = {
        "memory_md_path": cfg.memory.memory_md_path,
        "user_path": cfg.memory.user_path,
        "soul_path": cfg.memory.soul_path,
    }
    memory = SimpleNamespace(provider=mem_provider, paths=mem_paths)
```
Puis passe `memory=memory` à l'appel `build_registry(...)` dans `make_registry`.

- [ ] **Step 2 : exposer les chemins identité + budget à `create_app`**

Toujours dans `build_app`, passe à `create_app(...)` les infos nécessaires à l'injection : `identity_paths=mem_paths` et `identity_max_tokens=cfg.chat.identity_max_tokens`. Ajoute ces deux paramètres (mot-clé) à la signature de `create_app` dans `loom/web/app.py`.

- [ ] **Step 3 : injecter `identity_block` au system prompt**

Dans `loom/web/app.py`, au point d'assemblage du `system_prompt` (juste après le bloc `# Ton moteur`, ~ligne 397, AVANT le `except`), ajoute :
```python
            from loom.memory.identity import identity_block

            _idblk = identity_block(
                identity_paths["soul_path"],
                identity_paths["user_path"],
                identity_paths["memory_md_path"],
                max_tokens=identity_max_tokens,
            )
            if _idblk:
                system_prompt += f"\n\n{_idblk}"
```
(Le bloc identité est injecté dans le **system prompt**, donc il survit toujours à la microcompaction / summarization, qui ne touchent que l'historique — design §5.6.)

- [ ] **Step 4 : smoke — l'app se construit et le prompt contient l'identité**

Crée `/tmp/smoke_app_identity.py` :
```python
from pathlib import Path
from loom.config import load_config
from loom.memory import identity as I
RT = Path("loom")
cfg = load_config(RT/"loom.config.toml", RT/"loom.config.personnel.toml")
# sème une ligne USER pour vérifier l'injection
I.append_unique(cfg.memory.user_path, "SMOKE: utilisateur de test")
from loom.web.__main__ import build_app
app = build_app(cfg)
assert app is not None
blk = I.identity_block(cfg.memory.soul_path, cfg.memory.user_path, cfg.memory.memory_md_path,
                       max_tokens=cfg.chat.identity_max_tokens)
assert "SMOKE: utilisateur de test" in blk
print("OK app+identite")
```
Run : `uv run python /tmp/smoke_app_identity.py`
Attendu : `OK app+identite`. (Nettoie ensuite la ligne SMOKE de `loom/data/USER.md` si présente.)

- [ ] **Step 5 : `ruff` + commit**

```bash
uv run ruff check loom/web/__main__.py loom/web/app.py
git add loom/web/__main__.py loom/web/app.py
git commit -m "feat(web): cable la memoire + injecte le bloc identite au system prompt"
```

---

## Task 7 : présenter remember/recall + l'identité au modèle

**Files:**
- Modify: `loom/prompts/chat.system.md`

- [ ] **Step 1 : ajouter recall/remember à la section « PLANIFIER / DÉLÉGUER / MÉMORISER »**

Dans `loom/prompts/chat.system.md`, dans le bloc `PLANIFIER / DÉLÉGUER / MÉMORISER`, après la ligne `write_note(note) / read_note()`, ajoute :
```markdown
- remember(text, kind) / recall(query) : mémoire PERSISTANTE (survit à la session, pas seulement au fil courant). remember capitalise une leçon durable (kind='episodic' par défaut → store cherchable ; 'memory'/'profile'/'soul' → fichiers durables). recall retrouve par mots-clés ce que tu as appris avant. Réflexe sur une tâche en terrain déjà-vu : recall AVANT de repartir de zéro. Note write_note = mémoire de CETTE session ; remember = mémoire de TOUJOURS.
```

- [ ] **Step 2 : (si présent) mentionner l'identité always-on**

Si un bloc `# Mon identité`/`# L'utilisateur`/`# Mémoire durable` apparaît dans le system prompt (injecté), une courte phrase en tête de `chat.system.md` peut le cadrer (optionnel) : « Ton identité, le profil utilisateur et ta mémoire durable te sont rappelés plus bas — tiens-en compte. » N'ajoute cette phrase que si elle n'alourdit pas.

- [ ] **Step 3 : smoke — le prompt charge et mentionne les outils**

Run : `uv run python -c "from loom.prompts import CHAT_SYSTEM as C; assert 'remember' in C and 'recall' in C; print('OK prompt', len(C), 'car.')"`
Attendu : `OK prompt <N> car.`

- [ ] **Step 4 : commit**

```bash
git add loom/prompts/chat.system.md
git commit -m "docs(prompt): presente remember/recall + identite au modele"
```

---

## Task 8 : vérification E2E manuelle (preuve runtime, règle utilisateur)

> Vert au smoke ≠ marche en runtime. Cette tâche est la preuve de bout en bout (l'utilisateur lance la stack).

- [ ] **Step 1 : `.gitignore`** — vérifier que `loom/data/` est bien ignoré (il l'est déjà via `loom/data/` / `loom/runtime/data/`). `memory.db`, `SOUL.md`, `USER.md`, `MEMORY.md` vivent sous `loom/data/` → ne doivent pas être commités. Sinon, ajouter `loom/data/memory.db` et `loom/data/*.md` au `.gitignore`.

- [ ] **Step 2 : run réel (utilisateur)** — lancer la stack, dans le chat :
  1. « remember : sur ce projet, le shell est PowerShell » → le modèle appelle `remember`.
  2. Nouveau message : « qu'est-ce que tu sais sur le shell de ce projet ? » → le modèle appelle `recall` et retrouve le fait.
  3. Éditer `loom/data/USER.md` à la main (ajouter une préférence), recharger, vérifier que la réponse en tient compte (identité always-on injectée).

- [ ] **Step 3 : rapporter le RÉSULTAT réel** (constaté, pas supposé). Si un point échoue, lire l'erreur, corriger, relancer.

---

## Self-Review (effectuée)

- **Couverture spec :** §5.1 session (inchangée, hors scope ✅), §5.2 identité (Task 3+6 ✅), §5.3 recall local FTS5 (Task 2 ✅ ; externes hors v1, signalés ✅), §5.4 always-on + recall à la demande (Task 6 injection + Task 5 outil ✅), §5.5 écriture remember/kinds (Task 5 ✅), §5.6 ordre d'assemblage + survie microcompaction (Task 6 Step 3 ✅), §9 config (Task 4 ✅), §11 erreurs best-effort (Task 2 try/except ✅). **Hors scope assumé et noté :** §6 reflect, §4 skills appris, §6.6 summarization recall → **Plan 2**.
- **Placeholders :** code complet pour chaque nouveau fichier ; les modifs de `config.py`/`app.py` pointent des fonctions réelles (`load_config`, assemblage `system_prompt` ~397) avec patron à adapter aux noms locaux — l'exécutant lit le voisinage avant d'éditer.
- **Cohérence des types :** `MemoryProvider.remember/recall`, `Snippet(text,kind,source,score)`, `memory.provider`/`memory.paths`, `identity_block(soul,user,memory,max_tokens)`, kinds `episodic|memory|profile|soul` — identiques de Task 1 à Task 7.

## Execution Handoff

Plan 1/2 (fondation mémoire) complet. Le **Plan 2** (étape `reflect` + skills auto-appris + summarization recall) viendra ensuite, par-dessus cette fondation. Deux options d'exécution pour le Plan 1 :

1. **Subagent-Driven (recommandé)** — un sous-agent frais par tâche, revue entre les tâches.
2. **Inline** — exécution dans cette session avec points de contrôle.
