# UI unifiée Loom — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Une seule entrée qui route automatiquement chat vs build (classifieur LLM + override), un sélecteur de dossier natif, et tous les réglages dans un drawer — centre épuré.

**Architecture:** On garde `/chat` et `/run` intacts ; on ajoute `loom/classify.py` (pur), deux endpoints Flask (`/classify`, `/pick-folder`), et on restructure `index.html` + `app.js` (drawer + barre unifiée). Le front oriente l'appel SSE selon le mode (auto→classify, sinon forcé).

**Tech Stack:** Python/Flask, `uv`/`uvx ruff`/`pytest`, Preact+htm (front sans build), tkinter (sous-processus). Spec : [2026-06-03-loom-ui-unifiee-design.md](../specs/2026-06-03-loom-ui-unifiee-design.md).

---

## File Structure

| Fichier | Responsabilité | Tâches |
|---|---|---|
| `loom/classify.py` | `classify_intent()` pur : message → "build"/"chat" (défaut chat) | 1 |
| `loom/web/app.py` | endpoints `/classify` + `/pick-folder` (import `subprocess` au top) | 2,3 |
| `loom/web/templates/index.html` | header + ⚙️, drawer réglages, barre unifiée + puce workdir | 4 |
| `loom/web/static/app.js` | submit unifié (classify+route), drawer toggle, pick-folder, mode segmented | 5 |
| `tests/test_classify.py` | tests `classify_intent` | 1 |
| `tests/test_web.py` | tests `/classify` + `/pick-folder` (mocks) | 2,3 |

---

## Task 1 : `classify_intent` (pur)

**Files:** Create `loom/classify.py` ; Test `tests/test_classify.py`

- [ ] **Step 1 : test qui échoue** — `tests/test_classify.py`
```python
from loom.classify import classify_intent


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, messages, system_prompt, max_tokens=8, model=None,
                 thinking=False, temperature=None):
        self.calls.append({"messages": messages, "thinking": thinking})
        return self.reply


def test_classify_build():
    c = FakeClient("BUILD")
    assert classify_intent(c, "crée un démineur en html", model="m") == "build"


def test_classify_chat():
    c = FakeClient("CHAT")
    assert classify_intent(c, "explique-moi les closures", model="m") == "chat"


def test_classify_defaults_to_chat_when_ambiguous():
    assert classify_intent(FakeClient(""), "?", model="m") == "chat"
    assert classify_intent(FakeClient("bla bla"), "x", model="m") == "chat"


def test_classify_runs_thinking_off():
    c = FakeClient("CHAT")
    classify_intent(c, "x", model="m")
    assert c.calls[0]["thinking"] is False
```

- [ ] **Step 2 : rouge** — `uv run pytest tests/test_classify.py -v` → FAIL (no module).

- [ ] **Step 3 : implémenter** — `loom/classify.py`
```python
# loom/classify.py
"""Routage d'intention : décide si une demande relève du BUILD (créer/modifier des
fichiers de code) ou du CHAT (question/discussion). Un seul appel court ; défaut sûr = chat."""

from __future__ import annotations

_CLASSIFY_SYS = (
    "Tu es un routeur d'intention. Tu lis la demande et tu réponds UN SEUL mot : "
    "BUILD si elle consiste à CRÉER ou MODIFIER des fichiers de code dans un projet "
    "(faire une app/un jeu/un script, corriger ou refactorer du code) ; sinon CHAT "
    "(question, explication, discussion). Réponds uniquement BUILD ou CHAT."
)


def classify_intent(client, message: str, *, model: str | None) -> str:
    """Renvoie 'build' ou 'chat'. Défaut 'chat' (sûr) si la réponse n'est pas 'build'."""
    raw = client.complete(
        [{"role": "user", "content": message}],
        _CLASSIFY_SYS,
        max_tokens=4,
        model=model,
        thinking=False,
        temperature=0.0,
    )
    return "build" if "build" in (raw or "").strip().lower() else "chat"
```

- [ ] **Step 4 : vert** — `uv run pytest tests/test_classify.py -v` → 4 PASS.

- [ ] **Step 5 : lint + commit**
```bash
uvx ruff check loom/classify.py tests/test_classify.py
git add loom/classify.py tests/test_classify.py
git commit -m "feat(ui): classify_intent (routage build/chat, défaut chat)"
```

---

## Task 2 : endpoint `POST /classify`

**Files:** Modify `loom/web/app.py` ; Test `tests/test_web.py`

- [ ] **Step 1 : test qui échoue** — ajouter à `tests/test_web.py` (suivre le pattern d'app de test existant : il y a déjà une fixture/factory `create_app` mockée ; réutilise-la). Le client mock doit avoir un `complete()` renvoyant "BUILD".
```python
def test_classify_endpoint_returns_mode(client_app):
    # client_app = test client Flask construit via create_app(..., models=["m"], client=fake)
    # où fake.complete(...) renvoie "BUILD"
    resp = client_app.post("/classify", data={"message": "crée un jeu"})
    assert resp.status_code == 200
    assert resp.get_json()["mode"] == "build"


def test_classify_empty_message_is_chat(client_app):
    resp = client_app.post("/classify", data={"message": "  "})
    assert resp.get_json()["mode"] == "chat"
```
Si la suite n'a pas de fixture réutilisable, construire l'app dans le test avec un `FakeClient` minimal (`complete` → "BUILD") et `models=["m"]`. Lire `tests/test_web.py` pour le pattern exact de construction d'app.

- [ ] **Step 2 : rouge** — `uv run pytest tests/test_web.py -k classify -v` → FAIL (404).

- [ ] **Step 3 : implémenter** — dans `loom/web/app.py`, à côté des autres routes de `create_app`. (Le `chat_lock = threading.Lock()` et `client`/`models` existent déjà dans la closure ; vérifier leurs noms en lisant le fichier.)
```python
    @app.route("/classify", methods=["POST"])
    def classify():
        message = (request.form.get("message") or "").strip()
        if not message:
            return {"mode": "chat"}
        if not chat_lock.acquire(blocking=False):
            return {"mode": "chat"}  # occupé -> défaut sûr, ne bloque pas l'utilisateur
        try:
            from loom.classify import classify_intent

            mode = classify_intent(
                client, message, model=(models[0] if models else None)
            )
        finally:
            chat_lock.release()
        return {"mode": mode}
```

- [ ] **Step 4 : vert** — `uv run pytest tests/test_web.py -k classify -v` → PASS.

- [ ] **Step 5 : lint + commit**
```bash
uvx ruff check loom/web/app.py tests/test_web.py
git add loom/web/app.py tests/test_web.py
git commit -m "feat(ui): endpoint /classify (route build/chat sous chat_lock)"
```

---

## Task 3 : endpoint `POST /pick-folder` (dialogue natif)

**Files:** Modify `loom/web/app.py` (ajouter `import subprocess` + `import sys` au top du module pour rendre le mock simple) ; Test `tests/test_web.py`

- [ ] **Step 1 : test qui échoue** — ajouter à `tests/test_web.py` :
```python
def test_pick_folder_returns_selected_path(client_app, monkeypatch):
    import loom.web.app as appmod

    class _Proc:
        returncode = 0
        stdout = "C:/Users/Amine/projet\n"
        stderr = ""

    monkeypatch.setattr(appmod.subprocess, "run", lambda *a, **k: _Proc())
    resp = client_app.post("/pick-folder")
    assert resp.get_json()["path"] == "C:/Users/Amine/projet"


def test_pick_folder_cancel_returns_empty(client_app, monkeypatch):
    import loom.web.app as appmod

    class _Proc:
        returncode = 0
        stdout = "\n"
        stderr = ""

    monkeypatch.setattr(appmod.subprocess, "run", lambda *a, **k: _Proc())
    resp = client_app.post("/pick-folder")
    assert resp.get_json()["path"] == ""
```

- [ ] **Step 2 : rouge** — `uv run pytest tests/test_web.py -k pick_folder -v` → FAIL (404).

- [ ] **Step 3 : implémenter** — au top de `loom/web/app.py`, ajouter `import subprocess` et `import sys`. Puis dans `create_app` :
```python
    @app.route("/pick-folder", methods=["POST"])
    def pick_folder():
        # Sous-processus : évite les soucis tkinter hors thread principal de Flask.
        script = (
            "import tkinter, tkinter.filedialog as fd;"
            "r=tkinter.Tk(); r.withdraw(); r.attributes('-topmost', True);"
            "p=fd.askdirectory(); print(p if p else '')"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
            )
        except Exception as exc:  # noqa: BLE001 - tkinter absent / timeout
            return {"path": "", "error": str(exc)[:200]}
        path = (proc.stdout or "").strip()
        if proc.returncode != 0 and not path:
            return {"path": "", "error": (proc.stderr or "sélecteur indisponible")[:200]}
        return {"path": path}
```

- [ ] **Step 4 : vert** — `uv run pytest tests/test_web.py -k pick_folder -v` → PASS.

- [ ] **Step 5 : lint + suite + commit**
```bash
uvx ruff check loom/web/app.py tests/test_web.py && uv run pytest -q
git add loom/web/app.py tests/test_web.py
git commit -m "feat(ui): endpoint /pick-folder (dialogue natif tkinter en sous-processus)"
```

---

## Task 4 : restructurer `index.html` (drawer + barre unifiée)

**Files:** Modify `loom/web/templates/index.html`. Pas de test Python (vérif manuelle).

- [ ] **Step 1 : header minimal + bouton ⚙️**
Remplacer le `<h1>` actuel (qui contient thinking-toggle, models, reset) par :
```html
  <header class="topbar">
    <span class="brand">Loom</span>
    <button class="gear" id="settings-btn" type="button" title="Réglages">⚙️</button>
  </header>
```

- [ ] **Step 2 : drawer réglages** (déplacer DANS le drawer : modèle, thinking, skills, outils, reset)
Après `<header>`, ajouter :
```html
  <div class="drawer-scrim" id="drawer-scrim" hidden></div>
  <aside class="drawer" id="settings-drawer" hidden aria-label="Réglages">
    <div class="drawer-head">
      <strong>Réglages</strong>
      <button class="gear" id="drawer-close" type="button" title="Fermer">✕</button>
    </div>
    <label id="thinking-toggle" title="Réflexion préalable (décoché = réponse directe)">
      <input type="checkbox" id="thinking-cb" {% if thinking %}checked{% endif %}> 🧠 Réflexion
    </label>
    <div class="drawer-section"><span class="drawer-label">Modèle</span>{% include "_models.html" %}</div>
    <details class="skills-box" open><summary>🧩 Skills (contexte)</summary>{% include "_skills.html" %}</details>
    <details class="skills-box" open><summary>🛠️ Outils & permissions</summary>{% include "_tools.html" %}
      <div class="hint">Les outils ⚠️ (écriture/shell) modifient ton système — gardés par le mode permission.</div>
    </details>
    <button class="reset" id="reset-btn" type="button">Reset conversation</button>
  </aside>
```
SUPPRIMER l'ancien panneau `<details class="agents-box"> … #run-form … </details>` (l'entrée unique le remplace).

- [ ] **Step 3 : barre de saisie unifiée** (remplacer le `<form id="chat">` actuel)
```html
  <div id="previewWrap"><img id="preview" alt="aperçu"><button type="button" id="clearImgBtn">✕</button></div>
  <form id="chat">
    <div class="bar-controls">
      <div class="seg" id="mode-seg" role="group" aria-label="Mode">
        <button type="button" data-mode="auto" class="on">Auto</button>
        <button type="button" data-mode="chat">Chat</button>
        <button type="button" data-mode="build">Build</button>
      </div>
      <span class="workdir-chip" id="workdir-chip" title="Dossier cible des runs build">
        📁 <span id="workdir-path">{{ workspace_dir }}</span>
        <button type="button" id="pick-folder-btn">Choisir…</button>
      </span>
    </div>
    <div class="bar-input">
      <textarea id="input" rows="2" placeholder="Écris… (Entrée = envoyer, Maj+Entrée = ligne)" autofocus></textarea>
      <input type="file" id="file" accept="image/*" style="display:none">
      <button type="button" id="fileBtn" title="Joindre une image">🖼️</button>
      <button id="sendBtn" type="submit">Envoyer</button>
    </div>
  </form>
```

- [ ] **Step 4 : CSS** — ajouter dans le `<style>` (drawer overlay glissant + segmented + chip). Code minimal :
```css
    .topbar { display:flex; justify-content:space-between; align-items:center; }
    .brand { font-size:18px; font-weight:600; }
    .gear { background:#1b1f2b; padding:6px 10px; }
    .drawer-scrim { position:fixed; inset:0; background:rgba(0,0,0,.5); z-index:9; }
    .drawer { position:fixed; top:0; right:0; height:100vh; width:320px; max-width:88vw;
              background:#11141d; border-left:1px solid #2a2d3e; z-index:10; padding:16px;
              overflow:auto; display:flex; flex-direction:column; gap:12px;
              box-shadow:-8px 0 24px rgba(0,0,0,.4); }
    .drawer[hidden], .drawer-scrim[hidden] { display:none; }
    .drawer-head { display:flex; justify-content:space-between; align-items:center; }
    .drawer-section { display:flex; flex-direction:column; gap:4px; }
    .drawer-label { font-size:12px; color:#8a93a6; }
    .bar-controls { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:6px; }
    .seg { display:inline-flex; border:1px solid #2a2d3e; border-radius:8px; overflow:hidden; }
    .seg button { background:#161922; color:#9aa0b4; padding:4px 12px; border-radius:0; }
    .seg button.on { background:#2f6f4f; color:#fff; }
    .workdir-chip { display:inline-flex; align-items:center; gap:6px; font-size:12px;
                    color:#cfe6da; background:#131a22; border:1px solid #28323e;
                    border-radius:8px; padding:3px 8px; font-family:ui-monospace,monospace; }
    .workdir-chip #workdir-path { max-width:240px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .workdir-chip button { background:#1d2a22; color:#9fd9bd; padding:2px 8px; font-size:11px; }
    .bar-input { display:flex; gap:8px; }
```

- [ ] **Step 5 : vérif manuelle + commit**
Lancer `uv run python -m loom.web`, ouvrir `:8000` : header + ⚙️, drawer s'ouvre/ferme, barre unifiée visible, puce workdir présente. (Le câblage JS vient en Task 5 ; ici on valide le rendu statique.)
```bash
git add loom/web/templates/index.html
git commit -m "feat(ui): drawer réglages + barre de saisie unifiée (markup/CSS)"
```

---

## Task 5 : câbler `app.js` (submit unifié, drawer, pick-folder, mode)

**Files:** Modify `loom/web/static/app.js`. Pas de test Python (vérif manuelle). LIRE `app.js` d'abord pour reprendre les helpers existants (envoi `/chat`, envoi `/run`, gestion SSE, `init_json`).

- [ ] **Step 1 : état mode + workdir (localStorage)**
En tête de l'init : lire `localStorage.loomMode` (défaut `"auto"`) et `localStorage.loomWorkdir` (défaut = `workspace_dir` d'`init_json`). Refléter le mode actif sur `#mode-seg button.on` et le workdir sur `#workdir-path`.

- [ ] **Step 2 : segmented control**
Sur clic d'un `#mode-seg button` : retirer `.on` des autres, l'ajouter au cliqué, persister `localStorage.loomMode = data-mode`.

- [ ] **Step 3 : drawer toggle**
`#settings-btn` → ouvrir (`drawer.hidden=false; scrim.hidden=false`). `#drawer-close`, clic sur `#drawer-scrim`, touche `Escape` → fermer. (Les contrôles déplacés — model select, thinking, skills, reset — gardent leurs handlers existants ; vérifier qu'ils ciblent toujours les bons ids, inchangés.)

- [ ] **Step 4 : pick-folder**
`#pick-folder-btn` → `fetch('/pick-folder', {method:'POST'})` → si `json.path` non vide : `#workdir-path` ← path, `localStorage.loomWorkdir = path`. Si `json.error` : petit message (console + chip en rouge bref). Le `#workdir-path` doit aussi rester éditable en secours (double-clic → prompt, optionnel — sinon garder la saisie via le champ existant).

- [ ] **Step 5 : submit unifié**
Sur submit de `#chat` :
  1. lire `message`, `mode = localStorage.loomMode`, `workdir = localStorage.loomWorkdir`.
  2. si `mode === "auto"` : `const r = await fetch('/classify', {method:'POST', body: form(message)}); mode = (await r.json()).mode;` puis afficher un badge éphémère « → Build/Chat » avec un bouton « plutôt {autre} » qui relance en forçant l'autre mode.
  3. router : `mode === "build"` → POST `/run` avec `mode=build`, `workspace=workdir`, `task=message` (réutiliser le handler `/run` existant) ; sinon → POST `/chat` (handler existant, multimodal/image inchangé).
  - Réutiliser les fonctions de streaming SSE déjà en place pour `/chat` et `/run` ; ne PAS réécrire le rendu d'events.

- [ ] **Step 6 : vérif manuelle + commit**
`uv run python -m loom.web` : « crée un démineur » en Auto → run build dans le workdir ; « explique les closures » → chat ; forcer Chat/Build court-circuite ; ⚙️ ouvre les réglages ; « Choisir… » ouvre le dialogue Windows.
```bash
git add loom/web/static/app.js
git commit -m "feat(ui): submit unifié (classify+route), drawer, pick-folder, mode"
```

---

## Self-Review (effectuée)
- **Couverture spec** : §2.1 entrée unique→T4/T5 ; §2.2 classifieur→T1/T2 ; §2.3 garde-fou segmented→T5.S2/S5 ; §2.4 workdir natif→T3/T5.S4 ; §2.5 drawer→T4/T5.S3 ; §2.6 /chat /run inchangés→réutilisés en T5. Critères §9 couverts. ✅
- **Placeholders** : code réel pour T1-T3 (testables) ; T4-T5 (front, non unit-testables) donnent markup/CSS/logique concrets + vérif manuelle. ✅
- **Cohérence** : `classify_intent(client, message, *, model)` identique T1→T2 ; endpoints renvoient `{mode}` / `{path}` cohérents avec ce qu'`app.js` consomme (T5). ✅
