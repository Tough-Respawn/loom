# Spec — UI unifiée Loom (entrée unique + drawer réglages + workdir natif)

> Issu d'un brainstorming. Objectif : supprimer la couture chat / multi-agent côté UI, router
> automatiquement selon l'intention, et sortir tous les réglages du flux principal vers une sidebar.

## 1. Objectif & contexte

Aujourd'hui l'UI Loom (`loom/web/templates/index.html` + `loom/web/static/app.js`) sépare deux mondes :
un **chat** (form `#chat` → route SSE `/chat`) et un **multi-agent** (form `#run-form` dans un panneau
`<details>`, champ dossier tapé à la main → route SSE `/run`). Réglages (modèle, thinking, skills,
outils/permissions, reset) sont empilés en haut du flux principal. C'est confus.

But : **une seule entrée** qui route automatiquement chat vs build, un **sélecteur de dossier natif**,
et un **centre épuré** (réglages déplacés dans un drawer).

## 2. Décisions verrouillées

1. **Entrée unique** : suppression du panneau `#run-form` séparé. Un seul champ de saisie.
2. **Routage par classifieur LLM** : à l'envoi en mode Auto, un appel court décide `build` ou `chat`.
   **Défaut sûr = chat** (le chat n'écrit pas de fichiers).
3. **Garde-fou** : segmented control `[Auto | Chat | Build]`. Auto par défaut ; Chat/Build forcés
   court-circuitent le classifieur. En Auto, le mode retenu est affiché avec un recours pour basculer.
4. **Sélecteur de dossier natif** : bouton « Choisir… » → `POST /pick-folder` → sous-processus tkinter
   `askdirectory` → chemin absolu. Puce workdir **toujours visible**. Le workdir ne sert qu'au build.
5. **Drawer réglages** (glisse depuis la droite, bouton ⚙️) contenant : sélection modèle, toggle 🧠
   thinking, Skills, Outils & permissions, bouton Reset. Le centre ne garde que titre + messages +
   barre de saisie.
6. **Réutilisation** : `/chat` et `/run` restent **inchangés** (streaming). On ajoute seulement
   `/classify` et `/pick-folder`. Le client oriente l'appel.

## 3. Architecture & flux

### Flux d'envoi (client, `app.js`)
```
submit(message, mode_selectionné, workdir)
 │
 ├─ mode == "chat"  ─────────────► POST /chat   (SSE, inchangé)
 ├─ mode == "build" ─────────────► POST /run mode=build workspace=workdir (SSE, inchangé)
 └─ mode == "auto"
        │ POST /classify {message}  → {mode: "build"|"chat"}
        │ afficher badge « → Build » / « → Chat » (+ recours : relancer forcé dans l'autre mode)
        └─ router vers /chat ou /run comme ci-dessus
```

### Endpoints serveur (`app.py`)
- **`POST /classify`** → `{ "mode": "build" | "chat" }`. Acquiert `chat_lock` (non bloquant ;
  si occupé → 429 comme les autres routes), appelle `classify_intent(client, message, model=...)`,
  relâche. Le classifieur fait **un** appel court (`max_tokens≈3`, thinking off, température basse).
- **`POST /pick-folder`** → `{ "path": "<abs>" }` (ou `{ "path": "" }` si annulé). Lance un
  **sous-processus** `sys.executable -c "<script tkinter>"` qui ouvre `askdirectory` (fenêtre
  `-topmost`), imprime le chemin choisi sur stdout. Timeout borné ; si tkinter indisponible/échec →
  HTTP 200 `{ "path": "", "error": "<msg court>" }` (le champ reste éditable en secours).

### Module `loom/classify.py`
```python
def classify_intent(client, message: str, *, model: str | None) -> str:
    """Renvoie 'build' ou 'chat'. Défaut 'chat' (sûr) si la réponse est ambiguë.
    Un seul appel court : le modèle répond un mot. 'build' = créer/modifier des FICHIERS de code
    dans un projet ; 'chat' = question / discussion / explication."""
```
Pur, sans I/O, testable avec un FakeClient. Détection : la réponse contient « build » (insensible à
la casse) → `build` ; sinon `chat`.

## 4. Unités & frontières

| Unité | Rôle | Statut |
|---|---|---|
| `loom/classify.py` · `classify_intent()` | message → "build"/"chat" (défaut chat) | **nouveau**, pur |
| `app.py` · `POST /classify` | classifie sous `chat_lock` | **nouveau** |
| `app.py` · `POST /pick-folder` | sous-processus tkinter `askdirectory` | **nouveau** |
| `app.py` · `/chat`, `/run` | streaming SSE | **inchangés** |
| `index.html` | drawer réglages + barre unifiée + puce workdir | **restructuré** |
| `app.js` | submit unifié (classify+route), drawer toggle, pick-folder, mode segmented | **modifié** |
| `templates/_models.html`, `_skills.html`, `_tools.html` | déplacés DANS le drawer (mêmes includes) | **déplacés** |

### Layout cible (`index.html`)
- **Header minimal** : titre « Loom » + bouton ⚙️ (ouvre le drawer).
- **Drawer** (`<aside id="settings-drawer">`, overlay glissant depuis la droite, fermé par défaut) :
  modèle (`_models.html`), toggle 🧠, Skills (`_skills.html`), Outils & permissions (`_tools.html`),
  bouton Reset. Fermeture par clic hors drawer / bouton ✕ / Échap.
- **`#messages`** : inchangé (monté par Preact).
- **Barre de saisie** (sticky bas) : ligne 1 = segmented `[Auto|Chat|Build]` + puce
  `📁 <workdir> [Choisir…]` ; ligne 2 = textarea + 🖼️ + Envoyer.

## 5. Gestion d'état (front)
- `mode` ∈ {auto, chat, build} (défaut auto), persisté en `localStorage`.
- `workdir` (chaîne, défaut = `workspace_dir` du serveur via `init_json`), persisté en `localStorage`.
- Le badge de route Auto est éphémère (par message), avec un bouton « plutôt {autre mode} » qui relance.

## 6. Cas limites
- **Build sans workdir choisi** → utilise le workspace par défaut serveur (comportement actuel de
  `/run` quand `workspace` est vide) ; affiché dans `run_info`.
- **`/pick-folder` annulé** → `path: ""` → on ne change pas la puce.
- **tkinter absent / pas d'affichage** → `{path:"", error}` → toast/erreur courte, champ workdir reste
  éditable à la main (secours).
- **`/classify` pendant un run** → 429 (lock occupé) → le front retombe sur `chat` (défaut sûr) OU
  affiche « occupé » ; choix retenu : défaut `chat` silencieux pour ne pas bloquer.
- **Classifieur ambigu / réponse vide** → `chat` (défaut sûr).

## 7. Tests
- `classify_intent` : « crée un démineur » → build ; « explique-moi les closures » → chat ; réponse
  ambiguë/vide → chat (FakeClient).
- `POST /classify` : renvoie `{mode}` ; 429 si lock occupé (test Flask, client mocké).
- `POST /pick-folder` : `askdirectory` mocké (monkeypatch du subprocess) → `{path}` ; annulation →
  `{path:""}` ; échec tkinter → `{path:"", error}`.
- Front : non couvert par les tests Python (vérif manuelle) — garder la logique de routing minimale.

## 8. Hors périmètre (YAGNI)
- Pas de classifieur déterministe de secours (le défaut chat suffit).
- Pas de navigateur de dossiers in-browser (dialogue natif retenu).
- Pas de persistance serveur des workdirs récents (localStorage front suffit).

## 9. Critères d'acceptation
1. Un seul champ de saisie ; le panneau `#run-form` séparé n'existe plus.
2. En Auto, « crée un jeu… » lance le build ; « explique X » reste en chat ; le mode retenu est affiché
   et basculable.
3. Le bouton « Choisir… » ouvre le vrai sélecteur de dossier Windows et remplit la puce workdir.
4. Modèle, thinking, skills, permissions, reset sont dans le drawer ; le centre ne contient que
   titre + messages + barre de saisie.
5. `/chat` et `/run` inchangés (tests existants verts) ; `classify_intent` testé ; `/classify` et
   `/pick-folder` testés (tkinter mocké).
