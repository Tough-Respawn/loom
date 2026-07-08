# Modèles IMAGE dans Loom — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sélectionner un modèle image dans le sélecteur de l'UI Loom ; chaque message user devient un prompt d'image ; l'image générée par ComfyUI s'affiche dans la conversation.

**Architecture:** Troisième type de modèle (après local/distant), déclaré par dossier `loom/models/_IMAGE/<id>/` (model.toml + workflow.json au format API ComfyUI). Une branche dédiée dans `/chat` court-circuite la boucle tool-use : unload du LLM, ComfyUI démarré/géré par Loom (Job Object kill-on-close), `POST /prompt` + poll `/history`, PNG servi par un endpoint et affiché en markdown.

**Tech Stack:** Python (stdlib urllib/json), Flask (app existante), ComfyUI en processus externe (`C:\tools\ComfyUI`, venv privé). AUCUNE dépendance ajoutée à Loom.

## Global Constraints

- **Pas de suite pytest** (choix produit) : vérification par smokes `uv run python -c "..."` + ruff. Chaque tâche remplace « test » par son smoke.
- Branche dédiée : `feat/image-models`. Commits Conventional Commits courts.
- `loom/models/*/` est gitignoré : les dossiers `_IMAGE/<id>/` ne sont PAS commités (comme les modèles LLM perso). Le code, lui, est commité.
- Aucun import torch/comfy côté Loom ; dialogue exclusivement HTTP (`http://127.0.0.1:8188`).
- Messages d'erreur : lisibles dans le chat, jamais de stacktrace brute (patron des messages `[génération interrompue : …]`).
- Réutiliser : `_win_job_kill_on_close`/`_terminate_tree` (loom/runtime/manager.py), `unload_local()` (LoomClient), `stay_awake` (déjà pris par la génération), `_local_gen_lock` (app.py).

---

### Task 1: Découverte des modèles image (`loom/runtime/image_models.py`)

**Files:**
- Create: `loom/runtime/image_models.py`

**Interfaces:**
- Produces: `ImageModel` (dataclass : `id, label, dir, width, height, comfy_dir, comfy_port, workflow_path`) et `discover_image_models(models_dir: Path|None) -> list[ImageModel]`.

- [ ] **Step 1: Écrire le module**

```python
# loom/runtime/image_models.py
"""Découverte des modèles IMAGE : loom/models/_IMAGE/<id>/ (model.toml + workflow.json).

Troisième type de modèle (après local llama-swap et distant API) : un dossier par
modèle, patron des LLM. Le préfixe '_' du parent exclut ces dossiers de la découverte
llama-swap (convention _TEMPLATE/_REMOTE). Le workflow.json est un graphe ComfyUI au
FORMAT API avec les placeholders {PROMPT} et {SEED} (remplacés à la soumission)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class ImageModel:
    id: str
    label: str
    dir: str
    width: int
    height: int
    comfy_dir: str  # racine de l'install ComfyUI (pour la démarrer nous-mêmes)
    comfy_port: int
    workflow_path: str


def discover_image_models(models_dir: Path | None = None) -> list[ImageModel]:
    """Scanne loom/models/_IMAGE/*/ ; dossier sans model.toml OU sans workflow.json
    -> ignoré (message console, pas d'exception : un dossier cassé ne bloque pas l'app)."""
    base = (models_dir or MODELS_DIR) / "_IMAGE"
    out: list[ImageModel] = []
    if not base.is_dir():
        return out
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        toml_p, wf_p = d / "model.toml", d / "workflow.json"
        if not (toml_p.is_file() and wf_p.is_file()):
            print(f"[loom] modèle image ignoré (fichier manquant) : {d.name}")
            continue
        try:
            data = tomllib.loads(toml_p.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"[loom] modèle image illisible ({d.name}) : {exc}")
            continue
        out.append(
            ImageModel(
                id=d.name,
                label=str(data.get("label") or d.name),
                dir=str(d),
                width=int(data.get("width") or 1024),
                height=int(data.get("height") or 1024),
                comfy_dir=str(data.get("comfy_dir") or "C:/tools/ComfyUI"),
                comfy_port=int(data.get("comfy_port") or 8188),
                workflow_path=str(wf_p),
            )
        )
    return out
```

- [ ] **Step 2: Smoke (dossier absent → liste vide ; dossier semé → découvert)**

```bash
uv run python -c "
import tempfile, pathlib
from loom.runtime.image_models import discover_image_models
with tempfile.TemporaryDirectory() as t:
    root = pathlib.Path(t)
    assert discover_image_models(root) == []
    d = root / '_IMAGE' / 'demo'
    d.mkdir(parents=True)
    (d / 'model.toml').write_text('label = \"Demo\"\nwidth = 832\n', encoding='utf-8')
    (d / 'workflow.json').write_text('{}', encoding='utf-8')
    (root / '_IMAGE' / 'casse').mkdir()  # sans fichiers -> ignoré
    ms = discover_image_models(root)
    assert [m.id for m in ms] == ['demo'] and ms[0].width == 832 and ms[0].height == 1024
    print('SMOKE OK')"
```
Expected: `SMOKE OK` (+ ligne « modèle image ignoré : casse »).

- [ ] **Step 3: ruff + commit**

```bash
uv run ruff check loom/runtime/image_models.py
git add loom/runtime/image_models.py
git commit -m "feat(image): decouverte des modeles image (models/_IMAGE/<id>/)"
```

---

### Task 2: Semer `loom/models/_IMAGE/krea2-turbo/` (non commité) + doc README modèles

**Files:**
- Create (non commité, gitignoré) : `loom/models/_IMAGE/krea2-turbo/model.toml`, `loom/models/_IMAGE/krea2-turbo/workflow.json`
- Modify (commité) : `loom/models/README.md`

**Interfaces:**
- Produces: le workflow API avec `{PROMPT}`/`{SEED}` que `ComfyClient.submit` (Task 3) consommera ; le nœud de sortie DOIT être `SaveImage` avec `filename_prefix = "loom"`.

- [ ] **Step 1: model.toml**

```toml
# Krea-2-Turbo GGUF Q4_K_M + LoRA Krea2-realism-V2 (0.8) — chaîne validée le 2026-07-08.
label = "Krea 2 Turbo (image)"
width = 1024
height = 1024
comfy_dir = "C:/tools/ComfyUI"
comfy_port = 8188
```

- [ ] **Step 2: workflow.json (format API ComfyUI, sémantique du template officiel : 8 steps, cfg 1, euler/simple, négatif = ConditioningZeroOut)**

```json
{
  "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "krea2_turbo-Q4_K_M.gguf"}},
  "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": "Krea2-realism-V2.safetensors", "strength_model": 0.8}},
  "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
  "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": "{PROMPT}"}},
  "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
  "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
  "sampler": {"class_type": "KSampler", "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0], "seed": "{SEED}", "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
  "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
  "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
  "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "loom"}}
}
```

- [ ] **Step 3: README des modèles — section « Modèles image »**

Dans `loom/models/README.md`, après la section « Local vs distant, d'un coup d'œil » :

```markdown
## Modèles IMAGE (`_IMAGE/<id>/`)
Un modèle image = un dossier sous `_IMAGE/` : `model.toml` (label, taille par défaut,
racine/port ComfyUI) + `workflow.json` (graphe ComfyUI **format API** avec `{PROMPT}`
et `{SEED}`). Sélectionnable dans l'UI comme un LLM : un message = une image. Le moteur
est ComfyUI (install séparée, venv privé) ; Loom le démarre et lui parle en HTTP.
Ajouter un modèle image = copier un dossier, éditer deux fichiers — comme les GGUF.
```

- [ ] **Step 4: Smoke découverte réelle + commit README**

```bash
uv run python -c "
from loom.runtime.image_models import discover_image_models
ms = discover_image_models()
assert any(m.id == 'krea2-turbo' for m in ms), ms
print('SMOKE OK', [m.id for m in ms])"
git add loom/models/README.md
git commit -m "docs(models): section modeles image (_IMAGE)"
```

---

### Task 3: Client + manager ComfyUI (`loom/runtime/comfy.py`)

**Files:**
- Create: `loom/runtime/comfy.py`

**Interfaces:**
- Consumes: `ImageModel` (Task 1) ; `_win_job_kill_on_close`, `_assign_to_job`, `_terminate_tree` de `loom/runtime/manager.py`.
- Produces: `ComfyEngine` avec `ensure_up(timeout=180) -> bool`, `generate(workflow_template: str, prompt: str, timeout=600) -> bytes` (PNG), `free() -> None`, `stop() -> None`. Exception `ComfyError(message)` au message montrable.

- [ ] **Step 1: Écrire le module**

```python
# loom/runtime/comfy.py
"""Moteur ComfyUI géré par Loom : démarrage (Job Object kill-on-close), soumission
d'un workflow API et récupération du PNG. HTTP uniquement (urllib), aucune dépendance.

VRAM (6 Go) : UN modèle à la fois — l'appelant décharge le LLM (unload_local) AVANT
generate(), et appelle free() quand on rebascule sur un modèle texte."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from loom.runtime.manager import _assign_to_job, _terminate_tree, _win_job_kill_on_close


class ComfyError(RuntimeError):
    """Erreur montrable dans le chat (jamais de stacktrace brute)."""


class ComfyEngine:
    def __init__(self, comfy_dir: str, port: int = 8188) -> None:
        self.dir = Path(comfy_dir)
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None
        self._job = None
        self._lock = threading.Lock()

    # --- process ---------------------------------------------------------
    def is_up(self, timeout: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/system_stats", timeout=timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def ensure_up(self, timeout: float = 180.0) -> bool:
        """Démarre ComfyUI si besoin (venv privé de son install) et attend le port.
        Ne tue jamais une instance lancée hors Loom (on ne gère que la nôtre)."""
        if self.is_up():
            return True
        py = self.dir / ".venv" / "Scripts" / "python.exe"
        if not py.is_file():
            raise ComfyError(
                f"ComfyUI introuvable ({py}) — vérifie comfy_dir dans model.toml."
            )
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                try:
                    self._proc = subprocess.Popen(
                        [str(py), "main.py", "--port", str(self.port)],
                        cwd=str(self.dir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=(sys.platform != "win32"),
                    )
                except OSError as exc:
                    raise ComfyError(f"démarrage ComfyUI impossible : {exc}") from exc
                if sys.platform == "win32":
                    if self._job is None:
                        self._job = _win_job_kill_on_close()
                    if self._job is not None:
                        _assign_to_job(self._job, self._proc.pid)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_up():
                return True
            time.sleep(1.0)
        raise ComfyError(
            f"ComfyUI ne répond pas sur :{self.port} après {int(timeout)} s "
            "(premier démarrage lent ? relance ; sinon lance-le à la main pour voir l'erreur)."
        )

    def stop(self) -> None:
        with self._lock:
            p, self._proc = self._proc, None
            if p is not None and p.poll() is None:
                _terminate_tree(p)

    # --- génération ------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float = 30.0) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else {}

    def generate(self, workflow_template: str, prompt: str, timeout: float = 600.0) -> bytes:
        """Injecte prompt+seed dans le template, soumet, attend, renvoie le PNG (bytes).
        Le prompt est injecté via json.dumps (jamais de collage brut : quotes/retours
        à la ligne sûrs). {SEED} : entier aléatoire 63 bits."""
        wf = workflow_template.replace('"{PROMPT}"', json.dumps(prompt, ensure_ascii=False))
        wf = wf.replace('"{SEED}"', str(random.getrandbits(63)))
        try:
            graph = json.loads(wf)
        except json.JSONDecodeError as exc:
            raise ComfyError(f"workflow.json invalide après injection : {exc}") from exc
        try:
            sub = self._post("/prompt", {"prompt": graph})
        except (urllib.error.URLError, OSError) as exc:
            raise ComfyError(f"soumission à ComfyUI échouée : {exc}") from exc
        if "error" in sub or "prompt_id" not in sub:
            # nœud manquant / entrée invalide : ComfyUI détaille dans node_errors
            detail = json.dumps(sub, ensure_ascii=False)[:300]
            raise ComfyError(f"workflow refusé par ComfyUI : {detail}")
        pid = sub["prompt_id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(1.5)
            try:
                with urllib.request.urlopen(
                    f"{self.base}/history/{pid}", timeout=10
                ) as resp:
                    hist = json.loads(resp.read().decode("utf-8", "replace"))
            except (urllib.error.URLError, OSError):
                continue  # transitoire : ComfyUI charge le modèle
            entry = hist.get(pid)
            if not entry:
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [
                    m[1].get("exception_message", "")
                    for m in status.get("messages", [])
                    if m and m[0] == "execution_error"
                ]
                raise ComfyError(
                    "génération échouée côté ComfyUI : " + (msgs[0] if msgs else "?")[:200]
                )
            for out in entry.get("outputs", {}).values():
                for im in out.get("images", []):
                    q = urllib.parse.urlencode(
                        {
                            "filename": im["filename"],
                            "subfolder": im.get("subfolder", ""),
                            "type": im.get("type", "output"),
                        }
                    )
                    with urllib.request.urlopen(
                        f"{self.base}/view?{q}", timeout=30
                    ) as resp:
                        return resp.read()
        raise ComfyError(f"génération sans réponse après {int(timeout)} s (timeout).")

    def free(self) -> None:
        """Rend la VRAM (déchargement des modèles image) — best-effort, jamais bloquant."""
        try:
            self._post("/free", {"unload_models": True, "free_memory": True}, timeout=10)
        except (urllib.error.URLError, OSError, ComfyError):
            pass
```

(ajouter `import urllib.parse` avec les imports.)

- [ ] **Step 2: Smoke hors-ligne (pas de serveur : is_up False, erreurs propres)**

```bash
uv run python -c "
from loom.runtime.comfy import ComfyEngine, ComfyError
e = ComfyEngine('C:/tools/ComfyUI', port=8199)  # port vide
assert e.is_up() is False
e.free()  # best-effort : ne lève pas
try:
    e.generate('{\"pos\":{\"inputs\":{\"text\":\"{PROMPT}\"}},\"sampler\":{\"inputs\":{\"seed\":\"{SEED}\"}}}', 'un chat')
    raise SystemExit('aurait dû lever')
except ComfyError as ex:
    print('SMOKE OK:', str(ex)[:60])"
```
Expected: `SMOKE OK: soumission à ComfyUI échouée : …`.

- [ ] **Step 3: Vérifier l'injection (unité pure)**

```bash
uv run python -c "
import json
from loom.runtime.comfy import ComfyEngine
tpl = json.dumps({'pos': {'inputs': {'text': '{PROMPT}'}}, 's': {'inputs': {'seed': '{SEED}'}}})
wf = tpl.replace('\"{PROMPT}\"', json.dumps('l\'été \"réel\"\nligne 2')).replace('\"{SEED}\"', '42')
g = json.loads(wf)
assert g['pos']['inputs']['text'].startswith('l') and g['s']['inputs']['seed'] == 42
print('SMOKE OK')"
```

- [ ] **Step 4: ruff + commit**

```bash
uv run ruff check loom/runtime/comfy.py
git add loom/runtime/comfy.py
git commit -m "feat(image): moteur ComfyUI gere (start kill-on-close, submit/poll, free)"
```

---

### Task 4: Câblage `__main__.py` → `create_app`

**Files:**
- Modify: `loom/web/__main__.py` (autour de la construction `create_app(...)`, ~l.217)

**Interfaces:**
- Consumes: `discover_image_models()` (Task 1).
- Produces: `create_app(..., image_models=[ImageModel, ...])` — Task 5 lira `image_models` (liste, défaut `[]`).

- [ ] **Step 1: Découvrir et passer les modèles image**

Dans `build_app`, avant l'appel `create_app` :

```python
    from loom.runtime.image_models import discover_image_models

    image_models = discover_image_models()
```

Et dans l'appel `create_app(...)`, ajouter l'argument :

```python
        image_models=image_models,
```

- [ ] **Step 2: Smoke import (l'app se construit encore)**

```bash
uv run python -c "import loom.web.__main__ as m; print('import OK')"
```
(la construction complète est vérifiée en Task 5, une fois `create_app` accepteur du paramètre.)

- [ ] **Step 3: Commit (avec Task 5 — même mouvement)**

---

### Task 5: Branche image dans l'app (`loom/web/app.py`) + sélecteur + endpoint image

**Files:**
- Modify: `loom/web/app.py` — signature `create_app`, endpoint `/models` (~l.1957), branche dans `/chat` (avant la construction du registry, ~l.1300), nouvel endpoint `GET /genimg/<name>`.

**Interfaces:**
- Consumes: `ComfyEngine`, `ComfyError` (Task 3) ; `image_models` (Task 4) ; `client.unload_local()`, `_local_gen_lock`, `_stay_awake`, `_sse` (existants dans app.py).
- Produces: SSE `content` portant `![prompt](/genimg/<fichier>.png)` ; PNG persistés sous `var/generated/` ET copiés dans `<workspace>/images/`.

- [ ] **Step 1: Signature + structures**

Dans `create_app(...)`, ajouter le paramètre `image_models=None` et, près des autres inits :

```python
    from loom.runtime.comfy import ComfyEngine, ComfyError

    image_models = list(image_models or [])
    image_model_ids = {m.id for m in image_models}
    _image_by_id = {m.id: m for m in image_models}
    # Un moteur par (dir, port) — en pratique un seul.
    _engines: dict[tuple, ComfyEngine] = {}

    def _engine_for(im) -> ComfyEngine:
        key = (im.comfy_dir, im.comfy_port)
        if key not in _engines:
            _engines[key] = ComfyEngine(im.comfy_dir, im.comfy_port)
        return _engines[key]

    _generated_dir = Path(cfg_data_dir) / "generated" if False else None  # voir Step 4
```

(le chemin exact de `var/generated` : `Path(remote_store_path).parent / "generated"` — même racine `var/` que le store distant ; créer au 1er usage.)

- [ ] **Step 2: Sélecteur — inclure les modèles image**

Là où la liste `models` est servie (endpoint qui renvoie `[{"id": m, "remote": ...}]`, ~l.1957) :

```python
        return [
            {"id": m, "remote": m in remote_model_ids, "image": False}
            for m in models
        ] + [
            {"id": im.id, "remote": False, "image": True} for im in image_models
        ]
```

Et côté validation du modèle sélectionné (si une liste blanche existe), accepter les ids image. Front (`loom/web/static/app.js`) : afficher le badge : dans le rendu du sélecteur, si `m.image` → suffixe ` 🖼` ou `(image)` selon le style existant du fichier (suivre le rendu du badge distant existant).

- [ ] **Step 3: Branche `/chat` (AVANT la construction du registry/boucle)**

Au début de `generate()` (après `_stay_awake.acquire()` et la prise de `chat_lock`, au point où `conv.model` est connu, avant `tool_factory`) :

```python
            if conv.model in image_model_ids:
                im = _image_by_id[conv.model]
                try:
                    # Même GPU que le LLM local -> même sérialisation.
                    _local_gen_lock.acquire()
                    _img_held = True
                    yield _sse("status", label="préparation du moteur image…")
                    client.unload_local()  # VRAM libre pour la diffusion
                    eng = _engine_for(im)
                    eng.ensure_up()
                    yield _sse("status", label="génération de l'image…")
                    png = eng.generate(
                        Path(im.workflow_path).read_text(encoding="utf-8"), message
                    )
                    name = f"loom_{int(time.time())}.png"
                    gen_dir = Path(remote_store_path).parent / "generated"
                    gen_dir.mkdir(parents=True, exist_ok=True)
                    (gen_dir / name).write_bytes(png)
                    ws_dir = Path(_session().workspace) / "images"
                    ws_dir.mkdir(parents=True, exist_ok=True)
                    (ws_dir / name).write_bytes(png)
                    md = (
                        f"![{(message or 'image')[:80]}](/genimg/{name})\n\n"
                        f"Image écrite : `{(ws_dir / name)}`"
                    )
                    conv.add_assistant(md)
                    session_store.save(sess)
                    yield _sse("content", text=md)
                    yield _sse("done")
                except ComfyError as exc:
                    err = f"\n[génération d'image interrompue : {exc}]"
                    conv.add_assistant(err)
                    session_store.save(sess)
                    yield _sse("content", text=err)
                    yield _sse("done")
                finally:
                    if _img_held:
                        _local_gen_lock.release()
                    chat_lock.release()
                    _stay_awake.release()
                    yield _sse("status", label="")
                return
```

**Adapter à l'existant** : reprendre les noms réels des helpers de `generate()`
(`_sse`, ajout du message assistant, sauvegarde de session, event de fin) en LISANT
le voisinage — la structure ci-dessus est le contrat, les noms exacts font foi sur place.
Si `add_assistant` n'existe pas, utiliser le même mécanisme que la fin de génération
normale (append au `conv.messages` + save).

- [ ] **Step 4: Endpoint de service des PNG**

```python
    @app.get("/genimg/<path:name>")
    def genimg(name: str):
        gen_dir = Path(remote_store_path).parent / "generated"
        # send_from_directory refuse les traversées de chemin.
        return send_from_directory(gen_dir, name, mimetype="image/png")
```

(`send_from_directory` : déjà importé par Flask dans app.py ? sinon l'ajouter à l'import Flask.)

- [ ] **Step 5: Retour texte — rendre la VRAM**

Dans la branche LOCALE existante de `generate()` (là où `_local_gen_lock` est pris pour un modèle local, ~l.1391), avant l'appel modèle :

```python
                    # Un moteur image chargé tiendrait la VRAM -> le vider (best-effort).
                    for _eng in _engines.values():
                        if _eng.is_up(timeout=0.5):
                            _eng.free()
```

- [ ] **Step 6: Smoke construction d'app**

```bash
uv run python -c "
from loom.config import load_config
from loom.web.__main__ import build_app, CONFIG_PATH, PERSONAL_CONFIG_PATH
app = build_app(load_config(CONFIG_PATH, PERSONAL_CONFIG_PATH))
print('app OK', [str(r) for r in app.url_map.iter_rules() if 'genimg' in str(r)])"
```
Expected: `app OK ['/genimg/<path:name>']`.

- [ ] **Step 7: ruff + commit Tasks 4+5**

```bash
uv run ruff check loom/web
git add loom/web/__main__.py loom/web/app.py loom/web/static/app.js
git commit -m "feat(image): modeles image selectionnables dans l'UI (branche /chat dediee)"
```

---

### Task 6: Vérification E2E réelle (OBLIGATOIRE avant tout « ça marche »)

**Files:** aucun (runtime).

- [ ] **Step 1: Stack complète**

```bash
uv run python -m loom.web   # (le serveur modèle peut être éteint : la branche image ne l'exige pas)
```

- [ ] **Step 2: Depuis l'UI (http://127.0.0.1:8000)**
  1. Sélecteur → `krea2-turbo` (badge image visible).
  2. Message : `Casual smartphone selfie of a young woman at her gaming desk, triple monitor glow, soft indoor light, natural skin texture.`
  3. Attendre (label « génération de l'image… », 1er run lent : démarrage ComfyUI + chargement).
  4. CONSTATER : l'image s'affiche dans le chat ; le PNG existe dans `<workspace>/images/` et `var/generated/`.

- [ ] **Step 3: Bascule retour texte**
  1. Sélecteur → `qwen3.6-35b-a3b-abliterated`, message « ping » → réponse normale (VRAM rendue, LLM rechargé).

- [ ] **Step 4: Cas d'erreur propre**
  1. Renommer temporairement `workflow.json` → message d'erreur lisible dans le chat (pas de stacktrace), app vivante. Remettre le fichier.

- [ ] **Step 5: Merge + push (après OK user)**

```bash
git checkout master && git merge feat/image-models
# push via URL tokenisée (jamais origin nu)
```

## Self-review (fait à l'écriture)
- Couverture spec : découverte par dossier (T1/T2), moteur géré kill-on-close (T3), branche /chat + unload + sérialisation + SSE (T5), bascule texte↔image (T5 step 5), affichage inline via markdown + endpoint (T5), erreurs lisibles (T3/T5), E2E (T6). Multi-onglets : sérialisation via `_local_gen_lock` (T5).
- Placeholders : les noms de helpers exacts d'app.py sont à confirmer sur place (signalé dans T5 Step 3) — assumé : app.py fait 2000+ lignes, le contrat est complet, les noms font foi.
- Types : `ImageModel` produit en T1, consommé T4/T5 ; `ComfyEngine.generate(template: str, prompt: str) -> bytes` produit T3, consommé T5.
