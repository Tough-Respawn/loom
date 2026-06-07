# Profils par modèle — Implementation Plan (Phase A : valeur)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Appliquer des correctifs déterministes propres au modèle actif (1er : normaliser les guillemets typographiques de Qwen) au contenu écrit par les outils, activés par un `profile.md` par modèle.

**Architecture:** Un module `loom/models_profile.py` charge `loom/models/<id>/profile.md` (frontmatter de flags) et applique des fixes curatés au `content` des outils d'écriture, au choke point `ToolRegistry.run`. Phase A ne touche PAS le chemin de lancement (la def des modèles reste dans `loom.config.toml`).

**Tech Stack:** Python stdlib (parsing frontmatter manuel, pas de dépendance YAML). Vérif par smokes inline (`uv run python -c …`), pas de pytest (règle projet).

---

## File Structure
- **Create** `loom/models_profile.py` — chargement du profil + registre de fixes + application.
- **Create** `loom/models/qwen3.5-4b-abliterated/profile.md`, `loom/models/gemma-uncensored/profile.md`.
- **Modify** `loom/tools/base.py` — `ToolRegistry` porte un profil et l'applique dans `run`.
- **Modify** `loom/tools/__init__.py` — `build_registry(active_model=…)` charge le profil.
- **Modify** `loom/web/__main__.py` — passe `conversation.model` à `make_registry`/`build_registry`.

---

## Task 1: `loom/models_profile.py`

**Files:** Create `loom/models_profile.py`

- [ ] **Step 1: Failing smoke**

```bash
uv run python -c "from loom.models_profile import load_profile, Profile, FIXES; print('OK')"
```
Expected: `ModuleNotFoundError: No module named 'loom.models_profile'`

- [ ] **Step 2: Create the module**

```python
# loom/models_profile.py
"""Profils par modèle : correctifs DÉTERMINISTES propres à chaque modèle, activés par un
profile.md dans loom/models/<id>/. Le .md ne contient PAS de logique : son frontmatter
ACTIVE des fixes déjà codés ici (registre curaté). Chaque modèle a ses travers ; on les
corrige sans dépendre du prompt (ex. Qwen3.5 ré-émet des guillemets typographiques)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Fichiers de PROSE : on n'y touche pas (les guillemets typographiques peuvent y être voulus).
_PROSE_EXT = frozenset({".md", ".markdown", ".txt", ".rst"})
_SMART_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"'}


def _normalize_quotes(content: str, suffix: str) -> str:
    """Remplace ’ ‘ ” “ par ' et " — sauf dans les fichiers de prose."""
    if suffix.lower() in _PROSE_EXT:
        return content
    for bad, good in _SMART_QUOTES.items():
        content = content.replace(bad, good)
    return content


# Registre curaté : nom de fix -> fonction (content, suffix) -> content.
FIXES = {
    "normalize_quotes": _normalize_quotes,
}

# Outils d'écriture et les clés d'arguments qui portent du contenu à corriger.
_CONTENT_KEYS = {
    "write_file": ("content",),
    "append_file": ("content",),
    "replace_lines": ("content",),
    "insert_lines": ("content",),
    "edit_file": ("old_string", "new_string"),
}


@dataclass(frozen=True)
class Profile:
    model_id: str
    fixes: tuple[str, ...]

    def apply(self, tool_name: str, args: dict, suffix: str) -> dict:
        """Applique les fixes actifs au contenu des arguments d'un outil d'écriture."""
        keys = _CONTENT_KEYS.get(tool_name)
        if not keys or not self.fixes:
            return args
        out = dict(args)
        for k in keys:
            v = out.get(k)
            if isinstance(v, str):
                for name in self.fixes:
                    fn = FIXES.get(name)
                    if fn:
                        v = fn(v, suffix)
                out[k] = v
        return out


_EMPTY = Profile("", ())


def _parse_frontmatter_fixes(text: str) -> list[str]:
    """Lit les fixes ACTIFS (valeur vraie) sous 'fixes:' dans le frontmatter YAML simple
    d'un profile.md. Parsing minimal (sans dépendance YAML)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    fm: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        fm.append(line)
    active: list[str] = []
    in_fixes = False
    for line in fm:
        stripped = line.strip()
        if stripped.startswith("fixes:"):
            in_fixes = True
            continue
        if in_fixes:
            if line and not line[0].isspace():  # fin du bloc indenté
                in_fixes = False
                continue
            if ":" in stripped:
                name, _, val = stripped.partition(":")
                if name.strip() in FIXES and val.strip().lower() in {"true", "yes", "on", "1"}:
                    active.append(name.strip())
    return active


def load_profile(model_id: str, models_dir: Path | None = None) -> Profile:
    """Charge le profil d'un modèle depuis loom/models/<id>/profile.md. Absent/illisible
    -> profil vide (aucun fix). Ne lève jamais."""
    if not model_id:
        return _EMPTY
    path = (models_dir or MODELS_DIR) / model_id / "profile.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return Profile(model_id, ())
    return Profile(model_id, tuple(_parse_frontmatter_fixes(text)))
```

- [ ] **Step 3: Behaviour smoke**

```bash
PYTHONUTF8=1 uv run python -c "
from loom.models_profile import Profile, load_profile
p = Profile('x', ('normalize_quotes',))
# code file: smart quotes normalized
a = p.apply('write_file', {'path':'a.py','content':'d[’k’]'}, '.py')
assert a['content'] == \"d['k']\", a['content']
# prose file: untouched
b = p.apply('write_file', {'path':'a.md','content':'mot ’ok’'}, '.md')
assert b['content'] == 'mot ’ok’'
# edit_file: both strings normalized
c = p.apply('edit_file', {'path':'x.py','old_string':'a’','new_string':'b’'}, '.py')
assert c['old_string']==\"a'\" and c['new_string']==\"b'\"
# empty profile: passthrough
e = Profile('y', ()).apply('write_file', {'content':'z’'}, '.py')
assert e['content']=='z’'
print('behaviour OK')
"
```
Expected: `behaviour OK`

- [ ] **Step 4: Lint + commit**

```bash
uvx ruff check loom/models_profile.py && uvx ruff format loom/models_profile.py
git add loom/models_profile.py
git commit -m "feat(profils): models_profile.py (fixes deterministes par modele)"
```

---

## Task 2: Brancher le profil dans le registre

**Files:** Modify `loom/tools/base.py`, `loom/tools/__init__.py`, `loom/web/__main__.py`

- [ ] **Step 1: `ToolRegistry` porte un profil (base.py)**

Dans `loom/tools/base.py`, remplacer la signature de `__init__` et `run` de `ToolRegistry`.

Remplacer :
```python
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {s.name: s for s in specs}
```
par :
```python
    def __init__(self, specs: list[ToolSpec], profile=None) -> None:
        self._specs = {s.name: s for s in specs}
        self._profile = profile  # loom.models_profile.Profile | None (duck-typed)
```

Dans `ToolRegistry.run`, remplacer :
```python
        try:
            args = validate_and_coerce(name, spec.parameters, args)
            return spec.run(args)
```
par :
```python
        try:
            args = validate_and_coerce(name, spec.parameters, args)
            if self._profile is not None:
                p = args.get("path")
                suffix = Path(p).suffix if isinstance(p, str) else ""
                args = self._profile.apply(name, args, suffix)
            return spec.run(args)
```
(`Path` est déjà importé dans base.py.)

- [ ] **Step 2: `build_registry` charge le profil (tools/__init__.py)**

Dans `loom/tools/__init__.py`, ajouter le paramètre `active_model` à `build_registry` (après `permission=None`) :
```python
    permission=None,
    active_model: str | None = None,
```

Le passer au sous-registre : dans `_build_sub_registry`, remplacer l'appel par :
```python
            return build_registry(
                workspace_dir,
                max_bytes,
                _SUBAGENT_TOOLS,
                web_cfg=web_cfg,
                active_model=active_model,
            )
```

À la fin de `build_registry`, remplacer `return ToolRegistry(specs)` par :
```python
    from loom.models_profile import load_profile

    profile = load_profile(active_model) if active_model else None
    return ToolRegistry(specs, profile=profile)
```

- [ ] **Step 3: passer le modèle actif (web/__main__.py)**

Dans `loom/web/__main__.py`, fonction `make_registry`, remplacer l'appel `build_registry(...)` pour ajouter :
```python
            active_model=(conversation.model if conversation else cfg.default_model),
```
(en plus des arguments existants `permission=permission`, etc.)

- [ ] **Step 4: Smoke d'intégration (registre applique le profil)**

```bash
PYTHONUTF8=1 uv run python -c "
import tempfile
from pathlib import Path
from loom.models_profile import Profile
from loom.tools.base import ToolRegistry, ToolSpec
seen = {}
spec = ToolSpec(name='write_file', description='', parameters={'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content']} if False else {'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}}}, run=lambda a: seen.update(a) or 'ok')
reg = ToolRegistry([spec], profile=Profile('q', ('normalize_quotes',)))
reg.run('write_file', {'path':'x.py','content':'d[’k’]'})
assert seen['content']==\"d['k']\", seen
print('integration OK')
"
```
Expected: `integration OK`

- [ ] **Step 5: Lint + commit**

```bash
uvx ruff check loom/tools/base.py loom/tools/__init__.py loom/web/__main__.py
git add loom/tools/base.py loom/tools/__init__.py loom/web/__main__.py
git commit -m "feat(profils): registre applique le profil du modele actif"
```

---

## Task 3: Les deux `profile.md` + end-to-end

**Files:** Create `loom/models/qwen3.5-4b-abliterated/profile.md`, `loom/models/gemma-uncensored/profile.md`

- [ ] **Step 1: profile Qwen**

Créer `loom/models/qwen3.5-4b-abliterated/profile.md` :
```markdown
---
fixes:
  normalize_quotes: true
---
Qwen3.5-4B émet des guillemets typographiques (’ ‘ ” “) au lieu d'ASCII -> casse la
syntaxe Python, et il les ré-émet malgré le prompt. On normalise le contenu écrit dans
les fichiers de code (les fichiers de prose .md/.txt sont épargnés).
```

- [ ] **Step 2: profile Gemma (doc, aucun fix)**

Créer `loom/models/gemma-uncensored/profile.md` :
```markdown
---
fixes: {}
---
Gemma E4B n'a pas de travers de sortie nécessitant un correctif déterministe. Profil
volontairement vide.
```

- [ ] **Step 3: End-to-end smoke (chargement réel + application)**

```bash
PYTHONUTF8=1 uv run python -c "
from loom.models_profile import load_profile
q = load_profile('qwen3.5-4b-abliterated')
g = load_profile('gemma-uncensored')
print('qwen fixes:', q.fixes, '| gemma fixes:', g.fixes)
assert q.fixes == ('normalize_quotes',)
assert g.fixes == ()
out = q.apply('replace_lines', {'path':'m.py','content':'if game_state[’revealed’]:\n    pass'}, '.py')
import ast; ast.parse(out['content'])
print('end-to-end OK')
"
```
Expected: `qwen fixes: ('normalize_quotes',) | gemma fixes: ()` then `end-to-end OK`

- [ ] **Step 4: Commit**

```bash
git add loom/models/qwen3.5-4b-abliterated/profile.md loom/models/gemma-uncensored/profile.md
git commit -m "feat(profils): profile.md Qwen (normalize_quotes) + Gemma (vide)"
```

---

## Self-Review
- **Spec coverage:** §5 profile.md format → Task 3. §6 normalize_quotes fix (code-only) → Task 1. §7 application au choke point ToolRegistry.run + build_registry(active_model) + web passe conv.model → Task 2. §8 smokes → chaque tâche. §10 Phase A (sans toucher le lancement) → ce plan. Phase B (refonte config) = plan séparé ultérieur.
- **Placeholder scan:** aucun — code complet partout.
- **Type consistency:** `Profile(model_id, fixes)`, `load_profile(model_id)`, `Profile.apply(tool_name, args, suffix)`, `ToolRegistry(specs, profile)`, `build_registry(..., active_model)` — cohérents Tasks 1→2→3.
