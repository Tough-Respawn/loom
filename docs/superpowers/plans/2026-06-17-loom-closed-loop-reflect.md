# Loom — Boucle d'apprentissage : reflect + skills auto-appris (Plan 2/2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refermer la boucle : après un tour qui a fait du vrai travail, une étape `reflect` (hors de la loop d'action) capitalise automatiquement — épisodes mémorisés, faits promus vers SOUL/USER/MEMORY, et skills réutilisables auto-créés/améliorés dans un dossier isolé.

**Architecture:** Une fonction `reflect()` greffée **après** `_persist()` dans `app.py`, hors de `stream_chat_tools` (aucun rail dans la loop). Un seul appel modèle borné (`reflect.system.md`) renvoie un JSON strict ; `validate_reflect_json` (pure) le filtre (schéma, anti-trivial, anti-doublon) ; les écritures vont dans le répertoire de données interne. Les skills appris vivent dans `loom/skills_learned/` (namespace `learned:`), affichés au catalogue avec un marqueur. Toute défaillance de `reflect` est non bloquante. **Prérequis : Plan 1 (fondation mémoire) livré** — provider, identité, outils `remember`/`recall` existent.

**Tech Stack:** Python stdlib, le pattern `ToolSpec`/prompts `.md`, `client.stream_chat` (appel modèle borné), `loom/memory/*` (Plan 1).

**Périmètre exact (réf. `docs/superpowers/plans/2026-06-17-loom-closed-learning-loop-design.md`) :** §4 (skills appris), §6 (reflect : déclenchement, mécanique, validation, barre de qualité), §6.6 (summarization du recall — différée du Plan 1, livrée ici). **Hors scope :** providers externes, gateways, staging/quarantaine, pruning auto (§12).

**Convention de vérification (override skill) :** smoke `uv run python -c "…"` + `ruff`, pas de suite pytest. La logique pure (`validate_reflect_json`) se smoke sans modèle ; le tour réel `reflect` se prouve en E2E (Task 7, stack lancée par l'utilisateur). Commits fréquents.

---

## File Structure

| Fichier | Responsabilité |
|---|---|
| `loom/extend/skills.py` *(modif)* | frontmatter étendu (`learned`/`uses`/`created_at`/`updated_at`) ; `collect_skills` agrège `learned_skills_dir` (namespace `learned`) ; marqueur `⟳` dans `render_catalog` |
| `loom/config.py` *(modif)* | `ChatConfig` : `learned_skills_dir`, `reflect_enabled`, `reflect_min_actions` ; `MemoryConfig` : `recall_summarize`, `recall_summarize_threshold` |
| `loom/prompts/reflect.system.md` *(nouveau)* | prompt de l'étape de réflexion (critères de rétention + schéma JSON) |
| `loom/agent/reflect.py` *(nouveau)* | `validate_reflect_json` (pure) + `reflect(...)` (appel modèle, validation, écritures) |
| `loom/tools/memory.py` *(modif)* | `make_recall` : summarization LLM des hits au-delà d'un seuil (§6.6) |
| `loom/web/__main__.py` *(modif)* | passe `learned_skills_dir` à `collect_skills` (catalogue) et à `build_registry` ; assemble `stores` pour `reflect` |
| `loom/web/app.py` *(modif)* | appelle `reflect(...)` après `_persist()` (~539), non bloquant |
| `loom/prompts/chat.system.md` *(modif)* | mentionne les skills appris (`learned:*`) + que l'agent capitalise |

---

## Task 1 : skills appris — frontmatter, agrégation, marqueur catalogue

**Files:**
- Modify: `loom/extend/skills.py` (`Skill`, `_parse_skill_md`, `collect_skills`, `render_catalog`)

- [ ] **Step 1 : étendre le dataclass `Skill` (champs optionnels, rétro-compat)**

Dans `loom/extend/skills.py`, le `@dataclass class Skill` :
```python
@dataclass
class Skill:
    name: str
    description: str
    body: str
    base_dir: str = ""
    learned: bool = False
    uses: int = 0
    created_at: str = ""
    updated_at: str = ""
```

- [ ] **Step 2 : parser les nouveaux champs dans `_parse_skill_md`**

Dans `_parse_skill_md`, étendre la boucle de frontmatter pour capter aussi `learned`/`uses`/`created_at`/`updated_at`, et renvoyer un dict de métadonnées en plus. Changer la signature pour renvoyer `(name, description, body, meta)` où `meta = {"learned": bool, "uses": int, "created_at": str, "updated_at": str}` (valeurs par défaut si absentes → rétro-compat). Mettre à jour `_load_skill_file` pour passer `meta` au constructeur `Skill`.

```python
def _parse_skill_md(text: str, fallback_name: str):
    name, description, body = fallback_name, "", text
    meta = {"learned": False, "uses": 0, "created_at": "", "updated_at": ""}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            body = text[end + 4 :].lstrip("\r\n")
            for line in front.splitlines():
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key, val = key.strip().lower(), val.strip()
                if key == "name" and val:
                    name = val
                elif key == "description":
                    description = val
                elif key == "learned":
                    meta["learned"] = val.lower() in ("true", "1", "yes", "oui")
                elif key == "uses":
                    meta["uses"] = int(val) if val.isdigit() else 0
                elif key in ("created_at", "updated_at"):
                    meta[key] = val
    return name, description, body, meta
```
Et `_load_skill_file` :
```python
def _load_skill_file(md, namespace):
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return None
    name, desc, body, meta = _parse_skill_md(text, md.parent.name)
    if namespace:
        name = f"{namespace}:{name}"
    return Skill(name=name, description=desc, body=body, base_dir=str(md.parent), **meta)
```
**Attention rétro-compat :** d'autres appelants de `_parse_skill_md` existent (Plan 1 ne le touche pas, mais le grep `_parse_skill_md` doit confirmer que `run_review_eval.py` / autres déballent bien 4 valeurs). Adapter tout appelant qui faisait `name, desc, body = _parse_skill_md(...)` en `name, desc, body, _ = ...`.

- [ ] **Step 3 : agréger `learned_skills_dir` dans `collect_skills`**

```python
def collect_skills(local_dir, plugins_root_path=None, learned_dir=None):
    skills = _scan_dir(local_dir, namespace=None)
    if learned_dir is not None:
        skills += _scan_dir(learned_dir, namespace="learned")
    if plugins_root_path is not None:
        from loom.extend.plugins import discover_plugins
        for plugin in discover_plugins(plugins_root_path):
            for md in plugin.skills:
                sk = _load_skill_file(md, namespace=plugin.name)
                if sk:
                    skills.append(sk)
    return skills
```

- [ ] **Step 4 : marqueur catalogue pour les skills appris**

Dans `render_catalog`, la ligne par skill :
```python
    for s in skills:
        desc = " ".join(s.description.split())
        if len(desc) > 220:
            desc = desc[:217] + "…"
        mark = " ⟳" if getattr(s, "learned", False) else ""
        lines.append(f"- {s.name}{mark} : {desc}")
```

- [ ] **Step 5 : smoke — skill appris agrégé sous `learned:` avec marqueur, rétro-compat OK**

Crée `/tmp/smoke_learned.py` :
```python
import tempfile, os
from loom.extend.skills import collect_skills, render_catalog
d = tempfile.mkdtemp(); off = os.path.join(d, "off"); learned = os.path.join(d, "learned")
os.makedirs(os.path.join(off, "review")); os.makedirs(os.path.join(learned, "swap-debug"))
# skill officiel SANS champs learned (rétro-compat)
open(os.path.join(off, "review", "SKILL.md"), "w", encoding="utf-8").write(
    "---\nname: review\ndescription: relire un diff\n---\ncorps")
# skill appris AVEC frontmatter étendu
open(os.path.join(learned, "swap-debug", "SKILL.md"), "w", encoding="utf-8").write(
    "---\nname: swap-debug\ndescription: diag swap\nlearned: true\nuses: 3\n---\ncorps")
sk = collect_skills(off, None, learned_dir=learned)
names = {s.name: s for s in sk}
assert "review" in names and names["review"].learned is False
assert "learned:swap-debug" in names and names["learned:swap-debug"].uses == 3
cat = render_catalog(sk)
assert "learned:swap-debug ⟳" in cat and "review :" in cat, cat
print("OK skills appris")
```
Run : `uv run python /tmp/smoke_learned.py` — Attendu : `OK skills appris`.

- [ ] **Step 6 : `ruff` + commit**
```bash
uv run ruff check loom/extend/skills.py
git add loom/extend/skills.py && git commit -m "feat(skills): agrege skills_learned (namespace learned) + frontmatter etendu"
```

---

## Task 2 : config — clés reflect + skills appris + summarization recall

**Files:**
- Modify: `loom/config.py`

- [ ] **Step 1 : champs `ChatConfig`**

Après `identity_max_tokens` (ajouté au Plan 1) :
```python
    learned_skills_dir: str = "loom/skills_learned"
    reflect_enabled: bool = True
    reflect_min_actions: int = 1
```

- [ ] **Step 2 : champs `MemoryConfig`**

Dans `MemoryConfig` (Plan 1) :
```python
    recall_summarize: bool = True
    recall_summarize_threshold: int = 5
```

- [ ] **Step 3 : parsing**

Dans `load_config`, lire `learned_skills_dir`/`reflect_enabled`/`reflect_min_actions` depuis `[chat]` (comme les autres champs) et `recall_summarize`/`recall_summarize_threshold` depuis `[memory]` (à côté du parsing `MemoryConfig` du Plan 1).

- [ ] **Step 4 : smoke**

Run :
```bash
uv run python -c "from loom.config import load_config; from pathlib import Path; RT=Path('loom'); \
c=load_config(RT/'loom.config.toml', RT/'loom.config.personnel.toml'); \
print(c.chat.reflect_enabled, c.chat.reflect_min_actions, c.chat.learned_skills_dir, \
c.memory.recall_summarize, c.memory.recall_summarize_threshold)"
```
Attendu : `True 1 loom/skills_learned True 5`.

- [ ] **Step 5 : `ruff` + commit**
```bash
uv run ruff check loom/config.py
git add loom/config.py && git commit -m "feat(config): cles reflect + skills appris + summarization recall"
```

---

## Task 3 : prompt de réflexion `reflect.system.md`

**Files:**
- Create: `loom/prompts/reflect.system.md`

- [ ] **Step 1 : écrire le prompt** (encode les critères de rétention §6.5 + le schéma §6.2)

```markdown
Tu es la phase de RÉFLEXION de Loom, exécutée APRÈS un tour de travail, à froid. Tu n'agis pas, tu ne parles pas à l'utilisateur : tu décides ce qui mérite d'être CAPITALISÉ pour rendre l'agent plus compétent au prochain tour. Ta seule sortie est un objet JSON.

On te donne la trajectoire du tour : la demande, les outils appelés et leurs résultats clés, la réponse finale. Tu en extrais le MEILLEUR, pas la transcription.

CRITÈRES DE RÉTENTION (n'écris que ce qui les satisfait) :
- Durable, pas éphémère : une préférence stable, une cause racine, un piège récurrent, une décision structurante — pas un détail transitoire du tour.
- Réutilisable : un skill SEULEMENT s'il généralise au-delà du cas présent (une procédure que tu referais).
- Haut signal : la LEÇON, pas le log (« le champ X est en anglais », pas le dump complet).
- Consolidation > accumulation : si ça ressemble à quelque chose de déjà connu, préfère raffiner/mettre à jour plutôt qu'empiler. Ne crée pas un skill quasi-redondant : améliore l'existant.
- Sobriété : la plupart des tours ne méritent RIEN de nouveau. Un JSON avec des listes vides est une réponse normale et correcte.

Ne propose un skill que pour une PROCÉDURE réutilisable et non triviale (pas pour réinventer un outil existant comme read_file ou run_shell).

Réponds EXCLUSIVEMENT par un objet JSON sur ce schéma (toutes les clés présentes, listes éventuellement vides) :
{
  "new_skills":      [{"name": "kebab-case", "description": "…", "body": "# Titre\n…instructions…"}],
  "improved_skills": [{"name": "learned:nom-existant", "body": "# Titre\n…corps réécrit…"}],
  "episodes":        [{"text": "leçon/observation dense, autonome"}],
  "memory_updates":  ["fait durable général (convention, environnement, consigne) → MEMORY.md"],
  "user_updates":    ["fait stable sur l'utilisateur (préférence, façon de bosser) → USER.md"],
  "soul_updates":    ["touche d'identité de l'agent, prudente → SOUL.md"]
}

Pas de texte hors du JSON. Pas de commentaire. Si rien ne mérite d'être retenu, renvoie toutes les listes vides.
```

- [ ] **Step 2 : charger le prompt dans `loom/prompts/__init__.py`**

Ajouter :
```python
REFLECT_SYSTEM = _load("reflect.system.md")
```

- [ ] **Step 3 : smoke**
Run : `uv run python -c "from loom.prompts import REFLECT_SYSTEM; assert 'JSON' in REFLECT_SYSTEM and 'new_skills' in REFLECT_SYSTEM; print('OK', len(REFLECT_SYSTEM))"`
Attendu : `OK <N>`.

- [ ] **Step 4 : commit**
```bash
git add loom/prompts/reflect.system.md loom/prompts/__init__.py
git commit -m "feat(prompts): prompt de reflexion (capitalisation post-tour)"
```

---

## Task 4 : `reflect.py` — validation pure + étape de réflexion

**Files:**
- Create: `loom/agent/reflect.py`

- [ ] **Step 1 : écrire le module**

```python
# loom/agent/reflect.py
"""Étape `reflect` : capitalisation post-tour, HORS de la loop d'action (design §6).

Un seul appel modèle borné sur la trajectoire du tour -> JSON strict -> validation pure
-> écritures internes (épisodes via provider, faits vers SOUL/USER/MEMORY, skills appris).
Toute défaillance est NON bloquante : la réponse à l'utilisateur est déjà rendue.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loom.memory import identity as _id
from loom.prompts import REFLECT_SYSTEM

_KEYS = ("new_skills", "improved_skills", "episodes", "memory_updates",
         "user_updates", "soul_updates")
_MIN_SKILL_BODY = 80  # anti-trivial : un skill doit dire quelque chose
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


@dataclass
class ReflectResult:
    new_skills: list = field(default_factory=list)
    improved_skills: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    memory_updates: list = field(default_factory=list)
    user_updates: list = field(default_factory=list)
    soul_updates: list = field(default_factory=list)


def validate_reflect_json(obj) -> ReflectResult | None:
    """Filtre le JSON de reflect. Renvoie un ReflectResult propre, ou None si inexploitable.

    PURE (sans IO, sans modèle) -> testable. Rejette le hors-schéma en silence ; anti-trivial
    sur les skills (nom kebab-case, corps assez long) ; déduplication des lignes texte.
    """
    if not isinstance(obj, dict):
        return None
    res = ReflectResult()
    for s in obj.get("new_skills") or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip().lower()
        body = str(s.get("body", "")).strip()
        desc = str(s.get("description", "")).strip()
        if _NAME_RE.match(name) and len(body) >= _MIN_SKILL_BODY:
            res.new_skills.append({"name": name, "description": desc, "body": body})
    for s in obj.get("improved_skills") or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip()
        body = str(s.get("body", "")).strip()
        if name.startswith("learned:") and len(body) >= _MIN_SKILL_BODY:
            res.improved_skills.append({"name": name, "body": body})
    for e in obj.get("episodes") or []:
        text = (e.get("text", "") if isinstance(e, dict) else str(e)).strip()
        if text:
            res.episodes.append({"text": text})
    for key in ("memory_updates", "user_updates", "soul_updates"):
        seen = set()
        for line in obj.get(key) or []:
            line = str(line).strip()
            if line and line not in seen:
                seen.add(line)
                getattr(res, key).append(line)
    if not any(getattr(res, k) for k in _KEYS):
        return None
    return res


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_learned_skill(learned_dir: str, name: str, description: str, body: str,
                         *, improve: bool) -> None:
    base = name.split(":", 1)[1] if name.startswith("learned:") else name
    d = Path(learned_dir) / base
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    uses, created = 0, _now()
    if md.exists():
        from loom.extend.skills import _parse_skill_md
        _n, old_desc, _b, meta = _parse_skill_md(md.read_text("utf-8"), base)
        uses = meta.get("uses", 0) + (1 if improve else 0)
        created = meta.get("created_at") or created
        description = description or old_desc
    front = (
        f"---\nname: {base}\ndescription: {description}\nlearned: true\n"
        f"created_at: {created}\nupdated_at: {_now()}\nuses: {uses}\n---\n"
    )
    md.write_text(front + body.strip() + "\n", encoding="utf-8")


def apply_reflect(res: ReflectResult, *, provider, paths: dict, learned_dir: str) -> None:
    """Écrit le résultat validé. Best-effort par item : une écriture qui échoue n'arrête
    pas les autres (chaque write est isolé par l'appelant via try/except global)."""
    for s in res.new_skills:
        _write_learned_skill(learned_dir, s["name"], s["description"], s["body"], improve=False)
    for s in res.improved_skills:
        _write_learned_skill(learned_dir, s["name"], "", s["body"], improve=True)
    for e in res.episodes:
        provider.remember(e["text"], kind="episodic", source="reflect")
    for line in res.memory_updates:
        _id.append_unique(paths["memory_md_path"], line)
    for line in res.user_updates:
        _id.append_unique(paths["user_path"], line)
    for line in res.soul_updates:
        _id.append_unique(paths["soul_path"], line)


def _trajectory_summary(messages: list, actions: list, answer: str) -> str:
    """Condense le tour pour le prompt de reflect (borné)."""
    last_user = next((m["content"] for m in reversed(messages)
                      if m.get("role") == "user" and isinstance(m.get("content"), str)), "")
    acts = "\n".join(f"- {a}" for a in actions) or "(aucune action)"
    return (
        f"DEMANDE :\n{str(last_user)[:1000]}\n\n"
        f"ACTIONS DU TOUR :\n{acts}\n\n"
        f"RÉPONSE FINALE :\n{(answer or '')[:1500]}"
    )


def reflect(messages: list, actions: list, answer: str, *, client, model,
            provider, paths: dict, learned_dir: str) -> ReflectResult | None:
    """Un tour de réflexion complet : appel modèle borné -> validation -> écritures.

    Renvoie le ReflectResult appliqué, ou None si rien à capitaliser. NON bloquant :
    l'appelant enveloppe dans un try/except (design §11) — une erreur ici ne doit jamais
    remonter à la réponse utilisateur.
    """
    user = _trajectory_summary(messages, actions, answer)
    txt = ""
    for kind, chunk in client.stream_chat(
        [{"role": "user", "content": user}], REFLECT_SYSTEM,
        max_tokens=800, model=model, thinking=False,
    ):
        if kind == "content":
            txt += chunk
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    res = validate_reflect_json(obj)
    if res is None:
        return None
    apply_reflect(res, provider=provider, paths=paths, learned_dir=learned_dir)
    return res
```

- [ ] **Step 2 : smoke — `validate_reflect_json` (pure) accepte le bon, rejette le mauvais**

Crée `/tmp/smoke_reflect_validate.py` :
```python
from loom.agent.reflect import validate_reflect_json
# rien -> None
assert validate_reflect_json({"new_skills": [], "episodes": []}) is None
assert validate_reflect_json("pas un dict") is None
# skill trivial (corps trop court) rejeté, épisode gardé
r = validate_reflect_json({
    "new_skills": [{"name": "x", "description": "d", "body": "trop court"}],
    "episodes": [{"text": "Sur Windows le shell est PowerShell, pas bash"}],
    "user_updates": ["Préfère concis", "Préfère concis"],  # dédup
})
assert r is not None and r.new_skills == [] and len(r.episodes) == 1
assert r.user_updates == ["Préfère concis"]
# skill valide accepté
r2 = validate_reflect_json({"new_skills": [{"name": "swap-debug", "description": "diag",
    "body": "# Swap debug\n" + "étape ".ljust(90)}]})
assert r2 and r2.new_skills[0]["name"] == "swap-debug"
print("OK validate_reflect")
```
Run : `uv run python /tmp/smoke_reflect_validate.py` — Attendu : `OK validate_reflect`.

- [ ] **Step 3 : smoke — `apply_reflect` écrit skill appris + épisode + identité (sans modèle)**

Crée `/tmp/smoke_apply_reflect.py` :
```python
import tempfile, os
from loom.memory import get_provider
from loom.agent.reflect import ReflectResult, apply_reflect
d = tempfile.mkdtemp()
prov = get_provider("local", db_path=os.path.join(d, "m.db"))
paths = {"memory_md_path": os.path.join(d,"MEMORY.md"), "user_path": os.path.join(d,"USER.md"),
         "soul_path": os.path.join(d,"SOUL.md")}
learned = os.path.join(d, "skills_learned")
res = ReflectResult(
    new_skills=[{"name": "swap-debug", "description": "diag swap", "body": "# Swap\nfais X puis Y"}],
    episodes=[{"text": "deploy.sh tourne le vendredi"}],
    user_updates=["Préfère PowerShell"],
)
apply_reflect(res, provider=prov, paths=paths, learned_dir=learned)
assert os.path.exists(os.path.join(learned, "swap-debug", "SKILL.md"))
assert "learned: true" in open(os.path.join(learned,"swap-debug","SKILL.md"), encoding="utf-8").read()
assert prov.recall("deploy vendredi")[0].text.startswith("deploy.sh")
assert "PowerShell" in open(paths["user_path"], encoding="utf-8").read()
print("OK apply_reflect")
```
Run : `uv run python /tmp/smoke_apply_reflect.py` — Attendu : `OK apply_reflect`.

- [ ] **Step 4 : `ruff` + commit**
```bash
uv run ruff check loom/agent/reflect.py
git add loom/agent/reflect.py && git commit -m "feat(agent): etape reflect (validation pure + ecritures de capitalisation)"
```

---

## Task 5 : summarization du recall (§6.6)

**Files:**
- Modify: `loom/tools/memory.py` (`make_recall` prend un résumeur optionnel)

- [ ] **Step 1 : étendre `make_recall`**

Remplace la signature `make_recall(provider)` par `make_recall(provider, *, summarize=None, threshold=5)` où `summarize` est un callable `(query, hits) -> str` (ou `None` = pas de résumé). Au-delà de `threshold` hits, déléguer à `summarize` ; sinon rendu brut borné (comportement Plan 1).

```python
def make_recall(provider, *, summarize=None, threshold=5) -> ToolSpec:
    def run(args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            raise ToolError("argument 'query' : décris ce que tu cherches en mémoire")
        k = max(1, min(int(args.get("k", 5) or 5), 10))
        hits = provider.recall(query, k=k)
        if not hits:
            return "(aucun souvenir pertinent)"
        if summarize is not None and len(hits) >= threshold:
            return summarize(query, hits)
        out, total = [], 0
        for h in hits:
            line = f"- {h.text}" + (f"  [{h.source}]" if h.source else "")
            if total + len(line) > _MAX_RECALL_CHARS:
                break
            out.append(line); total += len(line)
        return "Souvenirs pertinents :\n" + "\n".join(out)
    # ... (description/parameters inchangés)
```

- [ ] **Step 2 : fournir un résumeur basé client (dans le câblage, Task 6)**

Le résumeur vit côté `__main__`/`app` (il a le `client`). Forme :
```python
def _make_recall_summarizer(client, model):
    def summarize(query, hits):
        joined = "\n".join(f"- {h.text}" for h in hits)
        prompt = (f"Question : {query}\n\nSouvenirs bruts :\n{joined}\n\n"
                  "Condense ces souvenirs en une synthèse dense et fidèle (3-5 lignes max), "
                  "centrée sur la question. Cite les faits utiles, ignore le bruit.")
        txt = ""
        for kind, chunk in client.stream_chat(
            [{"role": "user", "content": prompt}], "Tu condenses des souvenirs en une note dense.",
            max_tokens=300, model=model, thinking=False):
            if kind == "content":
                txt += chunk
        return "Synthèse mémoire :\n" + txt.strip()
    return summarize
```
(Cette fonction sera définie en Task 6 et passée à `make_recall`.)

- [ ] **Step 3 : smoke — seuil respecté avec un résumeur factice (sans modèle)**

Crée `/tmp/smoke_recall_sum.py` :
```python
import tempfile, os
from loom.memory import get_provider
from loom.tools.memory import make_recall
d = tempfile.mkdtemp(); prov = get_provider("local", db_path=os.path.join(d,"m.db"))
for i in range(6): prov.remember(f"fait numero {i} sur le sujet alpha", source=f"t{i}")
calls = {"n": 0}
def fake_sum(q, hits): calls["n"] += 1; return f"RESUME de {len(hits)} hits"
rec = make_recall(prov, summarize=fake_sum, threshold=5)
out = rec.run({"query": "alpha sujet fait", "k": 6})
assert out.startswith("RESUME") and calls["n"] == 1, out
# sous le seuil -> brut
rec2 = make_recall(prov, summarize=fake_sum, threshold=50)
assert rec2.run({"query": "alpha", "k": 2}).startswith("Souvenirs")
print("OK recall summarize")
```
Run : `uv run python /tmp/smoke_recall_sum.py` — Attendu : `OK recall summarize`.

- [ ] **Step 4 : `ruff` + commit**
```bash
uv run ruff check loom/tools/memory.py
git add loom/tools/memory.py && git commit -m "feat(tools): recall avec summarization LLM au-dela d'un seuil (offline si modele local)"
```

---

## Task 6 : câblage — catalogue skills appris, résumeur recall, registre

**Files:**
- Modify: `loom/web/__main__.py`
- Modify: `loom/tools/__init__.py` (si `collect_skills`/`build_registry` y sont appelés avec `skills_dir`)

- [ ] **Step 1 : passer `learned_dir` à `collect_skills` partout**

Repère les appels `collect_skills(skills_dir, plugins_dir)` (dans `app.py` ~384, et dans `build_registry`/`make_use_skill` via `loom/tools/__init__.py`). Ajoute l'argument `learned_dir=cfg.chat.learned_skills_dir` à ces appels. Pour `build_registry`, ajoute un paramètre `learned_skills_dir=None` propagé au `collect_skills` interne de `make_use_skill`.

- [ ] **Step 2 : résumeur recall + passage à `make_recall`**

Dans `loom/web/__main__.py`, définis `_make_recall_summarizer(client, model)` (Task 5 Step 2). Dans `loom/tools/__init__.py`, à l'enregistrement de `recall` (Plan 1 Task 5 Step 3), passe `summarize` et `threshold` quand `memory` les porte :
```python
        if "recall" in enabled:
            specs.append(make_recall(memory.provider,
                                     summarize=getattr(memory, "summarize", None),
                                     threshold=getattr(memory, "threshold", 5)))
```
Et dans `__main__`, enrichis le `SimpleNamespace memory` (Plan 1 Task 6) avec `summarize=_make_recall_summarizer(client, cfg.default_model)` et `threshold=cfg.memory.recall_summarize_threshold` (seulement si `cfg.memory.recall_summarize`).

- [ ] **Step 3 : assembler `stores` pour reflect**

Dans `build_app`, construis l'objet passé à `create_app` pour reflect :
```python
    reflect_stores = SimpleNamespace(
        provider=mem_provider, paths=mem_paths, learned_dir=cfg.chat.learned_skills_dir,
    )
```
et passe à `create_app(...)` : `reflect_stores=reflect_stores`, `reflect_enabled=cfg.chat.reflect_enabled`, `reflect_min_actions=cfg.chat.reflect_min_actions`, `reflect_model=cfg.default_model`. Ajoute ces paramètres (mot-clé) à la signature de `create_app`.

- [ ] **Step 4 : smoke — l'app se construit avec le câblage complet**

Run : `uv run python -c "from pathlib import Path; from loom.config import load_config; from loom.web.__main__ import build_app; RT=Path('loom'); build_app(load_config(RT/'loom.config.toml', RT/'loom.config.personnel.toml')); print('OK build_app cable')"`
Attendu : `OK build_app cable` (aucune exception).

- [ ] **Step 5 : `ruff` + commit**
```bash
uv run ruff check loom/web/__main__.py loom/tools/__init__.py
git add loom/web/__main__.py loom/tools/__init__.py
git commit -m "feat(web): cable catalogue skills appris + resumeur recall + stores reflect"
```

---

## Task 7 : déclenchement de `reflect` post-tour (app.py)

**Files:**
- Modify: `loom/web/app.py` (après `_persist()`, ~ligne 539)

- [ ] **Step 1 : appeler `reflect` après `_persist()`, non bloquant**

Dans `generate()`, juste après `_persist()` (la ligne ~539, AVANT l'auto-titre), ajoute :
```python
                # Apprentissage post-tour (HORS de la loop d'action) : ne s'exécute que si
                # le tour a fait du vrai travail (>= reflect_min_actions). Toute défaillance
                # est avalée — la réponse utilisateur est déjà rendue (design §6, §11).
                if reflect_enabled and saved and len(actions) >= reflect_min_actions:
                    try:
                        from loom.agent.reflect import reflect as _reflect

                        _reflect(
                            conv.to_messages(), actions, answer,
                            client=client, model=conv.model or reflect_model,
                            provider=reflect_stores.provider,
                            paths=reflect_stores.paths,
                            learned_dir=reflect_stores.learned_dir,
                        )
                    except Exception:  # noqa: BLE001 - apprentissage best-effort, jamais bloquant
                        pass
```
(`actions` = la trace déjà construite dans `generate()` ; `answer` = la réponse finale ; `conv.to_messages()` = la trajectoire du tour. Tout est déjà en portée à cet endroit.)

- [ ] **Step 2 : smoke — la branche est inerte sur un tour trivial / sans erreur d'import**

Run : `uv run python -c "import loom.web.app, loom.agent.reflect; print('OK import reflect dans app')"`
Attendu : `OK import reflect dans app`. (Le déclenchement réel se prouve en E2E, Task 8.)

- [ ] **Step 3 : `ruff` + commit**
```bash
uv run ruff check loom/web/app.py
git add loom/web/app.py && git commit -m "feat(web): declenche reflect post-tour (non bloquant, garde reflect_min_actions)"
```

---

## Task 8 : présenter l'apprentissage au modèle + E2E (preuve runtime)

**Files:**
- Modify: `loom/prompts/chat.system.md`

- [ ] **Step 1 : mention des skills appris**

Dans `loom/prompts/chat.system.md`, près de la mention `use_skill` / du catalogue, ajoute une phrase :
```markdown
Certains skills du catalogue sont marqués `learned:` (suffixe ⟳) : tu te les es forgés lors de tours passés. Utilise-les comme les autres (use_skill), et fie-toi à ta mémoire (recall) quand une tâche ressemble à du déjà-vu.
```

- [ ] **Step 2 : smoke prompt**
Run : `uv run python -c "from loom.prompts import CHAT_SYSTEM as C; assert 'learned:' in C; print('OK')"`

- [ ] **Step 3 : commit**
```bash
git add loom/prompts/chat.system.md && git commit -m "docs(prompt): presente les skills appris (learned:) au modele"
```

- [ ] **Step 4 : E2E (utilisateur lance la stack)** — preuve runtime :
  1. Faire un tour avec ≥1 action mutante (ex. « crée un script X et lance-le »).
  2. Vérifier qu'APRÈS la réponse, `reflect` a tourné : inspecter `loom/data/memory.db` (un épisode ?), `loom/data/USER.md`/`MEMORY.md` (une ligne promue ?), `loom/skills_learned/` (un SKILL.md si procédure réutilisable ?). Note : `reflect` est volontairement sélectif — il peut légitimement ne RIEN écrire sur un tour peu généralisable.
  3. Tour suivant : vérifier que le bloc identité + un éventuel `learned:*` apparaissent au catalogue, et que `recall` retrouve l'épisode.
  4. Vérifier la non-régression : un tour de chat trivial (0 action) ne déclenche aucun appel `reflect` (pas de latence ajoutée).

- [ ] **Step 5 : rapporter le RÉSULTAT réel** (constaté, pas supposé) ; si échec, lire l'erreur, corriger, relancer.

---

## Self-Review (effectuée)

- **Couverture spec :** §4.1 stockage séparé (Task 1+2 ✅), §4.2 namespace+marqueur (Task 1 ✅), §4.3 frontmatter (Task 1 ✅), §4.4 cycle créer/améliorer (Task 4 `_write_learned_skill` improve ✅), §6.1 déclenchement post-persist + garde (Task 7 ✅), §6.2 appel borné + schéma (Task 3+4 ✅), §6.3 validation pure/anti-trivial/anti-doublon (Task 4 `validate_reflect_json` ✅), §6.4 pas de rail (greffe hors stream_chat_tools ✅), §6.5 barre de qualité (prompt Task 3 + validation Task 4 ✅), §6.6 recall summarisé (Task 5 ✅), §11 non bloquant (Task 7 try/except ✅).
- **Placeholders :** code complet pour `reflect.py`, `reflect.system.md`, modifs skills.py ; intégrations `__main__`/`app.py`/`__init__.py` pointent des lignes/fonctions réelles (collect_skills ~384, _persist ~539, build_registry) avec patron — l'exécutant lit le voisinage.
- **Cohérence des types :** `ReflectResult`(6 listes) ⇄ `validate_reflect_json` ⇄ `apply_reflect` ⇄ schéma du prompt (mêmes 6 clés) ; `reflect(messages, actions, answer, *, client, model, provider, paths, learned_dir)` ⇄ appel Task 7 ; `make_recall(provider, *, summarize, threshold)` ⇄ câblage Task 6 ; `_parse_skill_md` renvoie 4 valeurs partout (Task 1 Step 2 corrige les appelants).
- **Dépendance Plan 1 :** `loom/memory/{__init__,local,identity}.py`, outils `remember`/`recall`, `MemoryConfig`, injection identité — tous requis et supposés livrés.

## Execution Handoff

Plans 1 et 2 complets et sauvegardés. Ordre d'exécution : **Plan 1 (fondation) d'abord**, puis Plan 2 (il en dépend).
