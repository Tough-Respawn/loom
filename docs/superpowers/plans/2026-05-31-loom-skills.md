# Loom v3.2 — Skills / Contexte — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permettre d'injecter de la connaissance (format `SKILL.md` façon Claude Code) dans le contexte du modèle, via une sélection manuelle de skills par conversation.

**Architecture:** Un dossier `loom/skills/<nom>/SKILL.md`. `loom/skills.py` découvre/charge les skills et compose le system prompt. `Conversation` mémorise les skills actifs (persisté). `web/app.py` expose la sélection et injecte le contenu des skills actifs dans le system prompt envoyé au modèle. L'UI a un panneau de cases à cocher.

**Tech Stack:** Python 3.12+, `uv`, `pytest`, Flask, HTMX. Aucune nouvelle dépendance (mini-parseur frontmatter maison).

**Spec de référence:** [docs/superpowers/specs/2026-05-31-loom-skills-design.md](../specs/2026-05-31-loom-skills-design.md)

> Projet **hors git** : sauter les étapes `git`. Toolchain `uv`. Tests sans serveur/modèle réel.

---

## Structure des fichiers

| Fichier | Changement |
|---|---|
| `loom/skills.py` | 🆕 `Skill`, `list_skills`, `load_skill`, `compose_system_prompt` |
| `loom/skills/exemple/SKILL.md` | 🆕 skill d'exemple |
| `loom/conversation.py` | `active_skills` + `set_skills` + persistance |
| `loom/loom.config.toml` + `loom/config.py` | `[chat] skills_dir` |
| `loom/web/app.py` | `create_app(..., skills_dir)`, `POST /skills`, compose dans `/chat` |
| `loom/web/__main__.py` | passe `cfg.chat.skills_dir` |
| `loom/web/templates/index.html` | panneau Skills (cases à cocher) |
| tests | `test_skills.py` 🆕, `test_conversation.py`, `test_config.py`, `test_web.py` |

---

## Task 1: Module `skills.py` + skill d'exemple

**Files:**
- Create: `loom/skills.py`
- Create: `loom/skills/exemple/SKILL.md`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Créer le skill d'exemple `loom/skills/exemple/SKILL.md`**

```markdown
---
name: exemple
description: Skill d'exemple — montre le format. Remplace-le par ta propre connaissance.
---

Ceci est un skill d'exemple. Un skill est un fichier markdown de connaissance que tu
injectes dans le contexte du modèle en le cochant dans l'interface.

Pour créer le tien : crée `loom/skills/<nom>/SKILL.md` avec un frontmatter `name` et
`description`, puis ton contenu (ton archi, tes conventions, ta doc…).
```

- [ ] **Step 2: Écrire le test qui échoue**

```python
# tests/test_skills.py
from loom.skills import Skill, list_skills, load_skill, compose_system_prompt


def _write_skill(root, folder, text):
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def test_list_skills_parses_frontmatter(tmp_path):
    _write_skill(tmp_path, "dagster",
                 "---\nname: dagster\ndescription: Mon archi\n---\nCorps de la connaissance.")
    skills = list_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "dagster"
    assert skills[0].description == "Mon archi"
    assert "Corps de la connaissance." in skills[0].body
    assert "---" not in skills[0].body


def test_list_skills_fallback_name_without_frontmatter(tmp_path):
    _write_skill(tmp_path, "brut", "Juste du texte sans frontmatter.")
    skills = list_skills(tmp_path)
    assert skills[0].name == "brut"
    assert skills[0].description == ""
    assert skills[0].body.strip() == "Juste du texte sans frontmatter."


def test_list_skills_ignores_dir_without_skill_md(tmp_path):
    (tmp_path / "vide").mkdir()
    assert list_skills(tmp_path) == []


def test_list_skills_missing_dir_returns_empty(tmp_path):
    assert list_skills(tmp_path / "absent") == []


def test_compose_system_prompt_appends_active_bodies(tmp_path):
    base = "Tu es utile."
    s = Skill(name="dagster", description="d", body="ARCHI_XYZ")
    out = compose_system_prompt(base, [s])
    assert out.startswith(base)
    assert "ARCHI_XYZ" in out
    assert "# Skill : dagster" in out


def test_load_skill_by_name(tmp_path):
    _write_skill(tmp_path, "a", "---\nname: a\ndescription: x\n---\nAAA")
    assert load_skill(tmp_path, "a").body.strip() == "AAA"
    assert load_skill(tmp_path, "absent") is None
```

- [ ] **Step 3: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_skills.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'loom.skills'`.

- [ ] **Step 4: Implémenter `loom/skills.py`**

```python
# loom/skills.py
"""Skills : fichiers markdown de connaissance injectables dans le contexte (format Claude Code)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str


def _parse_skill_md(text: str, fallback_name: str) -> tuple[str, str, str]:
    """Parse frontmatter (name/description) + corps. Renvoie (name, description, body)."""
    name, description, body = fallback_name, "", text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            for line in front.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key, val = key.strip().lower(), val.strip()
                    if key == "name" and val:
                        name = val
                    elif key == "description":
                        description = val
    return name, description, body


def list_skills(skills_dir: str | Path) -> list[Skill]:
    """Scanne <skills_dir>/<nom>/SKILL.md et renvoie les skills trouvés (triés par nom de dossier)."""
    skills_dir = Path(skills_dir)
    out: list[Skill] = []
    if not skills_dir.exists():
        return out
    for sub in sorted(skills_dir.iterdir()):
        md = sub / "SKILL.md"
        if sub.is_dir() and md.exists():
            try:
                text = md.read_text(encoding="utf-8")
            except OSError:
                continue
            name, desc, body = _parse_skill_md(text, sub.name)
            out.append(Skill(name=name, description=desc, body=body))
    return out


def load_skill(skills_dir: str | Path, name: str) -> Skill | None:
    for skill in list_skills(skills_dir):
        if skill.name == name:
            return skill
    return None


def compose_system_prompt(base: str, active: list[Skill]) -> str:
    """Concatène le prompt de base et le corps de chaque skill actif."""
    parts = [base]
    for skill in active:
        parts.append(f"# Skill : {skill.name}\n{skill.body}")
    return "\n\n".join(parts)
```

- [ ] **Step 5: Lancer les tests**

Run: `uv run pytest tests/test_skills.py -v`
Expected: PASS (6 tests verts).

- [ ] **Step 6: Commit**

```powershell
git add loom/skills.py loom/skills/exemple/SKILL.md tests/test_skills.py
git commit -m "feat(skills): module skills (format SKILL.md) + exemple"
```

---

## Task 2: `Conversation.active_skills` (persisté)

**Files:**
- Modify: `loom/conversation.py`
- Test: `tests/test_conversation.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_conversation.py (ajouter)
def test_active_skills_roundtrip(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    conv.set_skills(["dagster", "conventions"])
    conv.save(path)
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.active_skills == ["dagster", "conventions"]


def test_load_old_json_without_active_skills(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text('{"system_prompt": "s", "messages": []}', encoding="utf-8")
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.active_skills == []
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_conversation.py::test_active_skills_roundtrip -v`
Expected: FAIL — `AttributeError: 'Conversation' object has no attribute 'set_skills'`.

- [ ] **Step 3: Modifier `loom/conversation.py`**

Ajouter le champ au dataclass (après `messages`) :
```python
    active_skills: list[str] = field(default_factory=list)
```

Ajouter la méthode (après `reset`) :
```python
    def set_skills(self, names: list[str]) -> None:
        self.active_skills = list(names)
```

Mettre à jour `save` pour inclure `active_skills` :
```python
        data = {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "active_skills": self.active_skills,
        }
```

Mettre à jour `load` pour le lire (tolérant) :
```python
            return cls(
                system_prompt=data.get("system_prompt", default_system_prompt),
                messages=list(data.get("messages", [])),
                active_skills=list(data.get("active_skills", [])),
            )
```

- [ ] **Step 4: Lancer les tests conversation**

Run: `uv run pytest tests/test_conversation.py -v`
Expected: PASS (anciens + 2 nouveaux).

- [ ] **Step 5: Commit**

```powershell
git add loom/conversation.py tests/test_conversation.py
git commit -m "feat(conversation): active_skills persiste"
```

---

## Task 3: Config `skills_dir`

**Files:**
- Modify: `loom/loom.config.toml`, `loom/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Ajouter `skills_dir` dans `[chat]` de `loom/loom.config.toml`**

Sous `web_port = 8000`, ajouter :
```toml
skills_dir = "loom/skills"
```

- [ ] **Step 2: Écrire le test qui échoue**

```python
# tests/test_config.py (ajouter)
def test_chat_skills_dir_default(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert cfg.chat.skills_dir == "loom/skills"
```

- [ ] **Step 3: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_config.py::test_chat_skills_dir_default -v`
Expected: FAIL — `AttributeError: 'ChatConfig' object has no attribute 'skills_dir'`.

- [ ] **Step 4: Modifier `loom/config.py`**

Ajouter le champ au dataclass `ChatConfig` (après `web_port`) :
```python
    skills_dir: str = "loom/skills"
```
Et dans `load_config`, dans la construction de `ChatConfig(...)`, ajouter :
```python
        skills_dir=ch.get("skills_dir", "loom/skills"),
```

- [ ] **Step 5: Lancer les tests config**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (anciens + nouveau).

- [ ] **Step 6: Commit**

```powershell
git add loom/loom.config.toml loom/config.py tests/test_config.py
git commit -m "feat(config): [chat] skills_dir"
```

---

## Task 4: `web/app.py` — sélection des skills + injection dans `/chat`

**Files:**
- Modify: `loom/web/app.py`
- Modify: `loom/web/__main__.py`
- Test: `tests/test_web.py`

- [ ] **Step 1: Mettre à jour `_make` et ajouter les tests dans `tests/test_web.py`**

Modifier `_make` pour fournir un `skills_dir` de test, et ajouter les tests :
```python
def _make(tmp_path, events=(("content", "Hel"), ("content", "lo"))):
    conv = Conversation(system_prompt="sys")
    history = tmp_path / "conv.json"
    skills_dir = tmp_path / "skills"
    (skills_dir / "dagster").mkdir(parents=True)
    (skills_dir / "dagster" / "SKILL.md").write_text(
        "---\nname: dagster\ndescription: archi\n---\nARCHI_DAGSTER_XYZ",
        encoding="utf-8",
    )
    app = create_app(conv, FakeClient(list(events)), history, skills_dir)
    return app, conv, history


def test_post_skills_updates_conversation(tmp_path):
    app, conv, _ = _make(tmp_path)
    resp = app.test_client().post("/skills", data={"skill": ["dagster"]})
    assert resp.status_code == 200
    assert conv.active_skills == ["dagster"]


def test_chat_injects_active_skill_into_system_prompt(tmp_path):
    app, conv, _ = _make(tmp_path)
    conv.set_skills(["dagster"])
    client = app.test_client()
    client.post("/chat", data={"message": "salut"})
    # FakeClient mémorise le system_prompt reçu
    assert "ARCHI_DAGSTER_XYZ" in app.config["_fake_client"].last_system_prompt
```

Mettre à jour `FakeClient` pour mémoriser le `system_prompt` reçu et l'exposer via `app.config` :
```python
class FakeClient:
    def __init__(self, events):
        self._events = events
        self.last_system_prompt = None

    def stream_chat(self, messages, system_prompt):
        self.last_system_prompt = system_prompt
        yield from self._events
```
Et dans `_make`, après `create_app`, exposer le client : `app.config["_fake_client"] = <le FakeClient>`.
(Adapter : créer le `FakeClient` dans une variable, le passer à `create_app`, puis le stocker.)

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_web.py::test_post_skills_updates_conversation -v`
Expected: FAIL — `TypeError: create_app() takes 3 positional arguments but 4 were given` (ou 404 sur `/skills`).

- [ ] **Step 3: Modifier `loom/web/app.py`**

Ajouter l'import en haut :
```python
from loom.skills import list_skills, load_skill, compose_system_prompt
```

Changer la signature de `create_app` :
```python
def create_app(conversation, client, history_path, skills_dir) -> Flask:
    app = Flask(__name__)
    history_path = str(history_path)
    skills_dir = str(skills_dir)
```

Mettre à jour la route index pour passer les skills :
```python
    @app.get("/")
    def index() -> str:
        skills = list_skills(skills_dir)
        return render_template(
            "index.html",
            messages=conversation.messages,
            skills=skills,
            active_skills=conversation.active_skills,
        )
```

Ajouter la route de sélection (avant `return app`) :
```python
    @app.post("/skills")
    def skills_update():
        selected = request.form.getlist("skill")
        conversation.set_skills(selected)
        conversation.save(history_path)
        skills = list_skills(skills_dir)
        return render_template(
            "_skills.html", skills=skills, active_skills=conversation.active_skills
        )
```

Dans la route `/chat`, juste avant `def generate():`, composer le system prompt effectif :
```python
        active = [s for s in (load_skill(skills_dir, n) for n in conversation.active_skills) if s]
        system_prompt = compose_system_prompt(conversation.system_prompt, active)
```
Et dans `generate()`, remplacer `conversation.system_prompt` par `system_prompt` dans l'appel :
```python
                for kind, text in client.stream_chat(
                    conversation.to_messages(), system_prompt
                ):
```

- [ ] **Step 4: Créer le fragment `loom/web/templates/_skills.html`**

```html
<div id="skills-panel">
  {% for s in skills %}
    <label class="skill" title="{{ s.description }}">
      <input type="checkbox" name="skill" value="{{ s.name }}"
             hx-post="/skills" hx-target="#skills-panel" hx-swap="outerHTML"
             hx-include="#skills-panel"
             {% if s.name in active_skills %}checked{% endif %}>
      {{ s.name }}
    </label>
  {% else %}
    <span class="muted">Aucun skill. Ajoute loom/skills/&lt;nom&gt;/SKILL.md</span>
  {% endfor %}
</div>
```

- [ ] **Step 5: Mettre à jour `loom/web/__main__.py`**

Passer `skills_dir` à `create_app` :
```python
    app = create_app(conversation, client, cfg.chat.history_path, cfg.chat.skills_dir)
```

- [ ] **Step 6: Lancer les tests web**

Run: `uv run pytest tests/test_web.py -v`
Expected: PASS (anciens adaptés + 3 nouveaux). *(Le template index.html inclura `_skills.html` en Task 5 ; pour que `GET /` rende sans erreur dès maintenant, voir Task 5 — si besoin, exécuter Task 5 Step 1 avant de relancer.)*

- [ ] **Step 7: Commit**

```powershell
git add loom/web/app.py loom/web/__main__.py loom/web/templates/_skills.html tests/test_web.py
git commit -m "feat(web): selection skills + injection dans /chat"
```

---

## Task 5: UI — panneau Skills dans la page

**Files:**
- Modify: `loom/web/templates/index.html`

- [ ] **Step 1: Insérer le panneau Skills dans `loom/web/templates/index.html`**

Juste sous le `<h1>…</h1>` (avant `<div id="messages">`), ajouter un panneau repliable :
```html
  <details class="skills-box">
    <summary>🧩 Skills (contexte)</summary>
    {% include "_skills.html" %}
  </details>
```

Et ajouter le style correspondant dans le bloc `<style>` :
```css
    .skills-box { margin: 8px 0; font-size: 13px; }
    .skills-box summary { cursor: pointer; color: #9aa0b4; }
    #skills-panel { display: flex; flex-wrap: wrap; gap: 10px; padding: 8px 0; }
    .skill { display: inline-flex; align-items: center; gap: 4px; background: #161922;
             border: 1px solid #2a2d3e; border-radius: 8px; padding: 4px 8px; }
```

- [ ] **Step 2: Ajouter le test de rendu du panneau dans `tests/test_web.py`**

```python
def test_index_lists_skills(tmp_path):
    app, _, _ = _make(tmp_path)
    body = app.test_client().get("/").get_data(as_text=True)
    assert "dagster" in body
```

- [ ] **Step 3: Lancer les tests web (le rendu complet doit passer)**

Run: `uv run pytest tests/test_web.py -v`
Expected: PASS — `GET /` rend la page avec le panneau skills (le test `test_index_lists_skills` voit "dagster").

- [ ] **Step 3: Commit**

```powershell
git add loom/web/templates/index.html
git commit -m "feat(web): panneau Skills dans l'UI"
```

---

## Task 6: Vérification finale

- [ ] **Step 1: Suite complète**

Run: `uv run pytest -q`
Expected: PASS — tous les tests verts (skills, conversation, config, web + existants).

---

## Definition of Done (v3.2)

- [ ] `uv run pytest` : tout vert (skills, active_skills, config skills_dir, web sélection+injection).
- [ ] `loom/skills/exemple/SKILL.md` existe ; l'UI liste les skills disponibles.
- [ ] Cocher un skill → `conversation.active_skills` mis à jour + persisté.
- [ ] Un message envoyé avec un skill actif → le corps du skill est dans le system prompt reçu par le client.
- [ ] (Manuel) Déposer `loom/skills/dagster/SKILL.md`, le cocher, demander qqch sur l'archi → le modèle répond en connaissant la stack.
