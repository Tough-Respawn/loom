# Loom v4 — Model-agnostic (llama-swap) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Faire du modèle un paramètre : registre de modèles + llama-swap (swap à chaud) + sélecteur UI, le tout ciblable par appel (fondation multi-agent).

**Architecture:** `[[models]]` dans la config ; `swap.py` génère un `llama-swap.yaml` ; `serve.py` télécharge les GGUF et lance llama-swap ; `client`/`conversation`/`web` portent le `model` comme paramètre. Migration **rétro-compatible** : l'ancien `[model]` reste supporté.

**Tech Stack:** Python 3.12+, `uv`, `pytest`, openai, flask, llama-swap (binaire externe). Pas de nouvelle dép Python (mini-sérialiseur YAML maison).

**Spec:** [docs/superpowers/specs/2026-06-01-loom-model-agnostic-design.md](../specs/2026-06-01-loom-model-agnostic-design.md)

> Projet hors git : sauter les étapes git. Tests sans serveur/modèle réel. NE PAS télécharger de
> modèle ni installer llama-swap dans le workflow (étapes machine, hors workflow).

---

## Task 1: Registre de modèles (config rétro-compatible)

**Files:** `loom/config.py`, `loom/loom.config.toml`, `tests/test_config.py`

- [ ] **Step 1: Test qui échoue** — ajouter dans `tests/test_config.py`

```python
MULTI = """
[[models]]
id = "gemma"
repo = "r/gemma"
filename = "gemma.gguf"
mmproj_filename = "mmproj.gguf"
n_layers = 35
size_mb = 5340

[[models]]
id = "qwen"
repo = "r/qwen"
filename = "qwen.gguf"
n_layers = 48
size_mb = 21000

[server]
context = 8192
port = 8080
bin = "llama-server"

[chat]
default_model = "qwen"
"""


def test_models_registry_and_default(tmp_path):
    cfg = load_config(_write(tmp_path, "loom.config.toml", MULTI))
    assert [m.id for m in cfg.models] == ["gemma", "qwen"]
    assert cfg.default_model == "qwen"
    assert cfg.model_by_id("gemma").filename == "gemma.gguf"
    assert cfg.model_by_id("gemma").mmproj_filename == "mmproj.gguf"
    assert cfg.model_by_id("qwen").mmproj_filename == ""
    # propriété de compat `model` = modèle par défaut
    assert cfg.model.id == "qwen"


def test_legacy_single_model_still_works(tmp_path):
    # BASE utilise [model] (ancien format) -> 1 modèle id="default"
    cfg = load_config(_write(tmp_path, "loom.config.toml", BASE))
    assert len(cfg.models) == 1
    assert cfg.models[0].id == "default"
    assert cfg.default_model == "default"
    assert cfg.model.repo.endswith("Qwen2.5-Coder-7B-Instruct-GGUF")
```

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_config.py::test_models_registry_and_default -v` → FAIL (AttributeError `models`).

- [ ] **Step 3: Modifier `loom/config.py`**

Ajouter `id` et `n_gpu_layers` à `ModelConfig` :
```python
@dataclass
class ModelConfig:
    repo: str
    filename: str
    n_layers: int
    size_mb: int
    mmproj_filename: str = ""
    id: str = ""
    n_gpu_layers: int | None = None
```

Modifier `RuntimeConfig` : remplacer le champ `model: ModelConfig` par la liste + le défaut, et
ajouter une **propriété de compat** `model` :
```python
@dataclass
class RuntimeConfig:
    models: list[ModelConfig]
    default_model: str
    context: int
    port: int
    server_bin: str
    override_n_gpu_layers: int | None
    override_threads: int | None
    chat: ChatConfig

    def model_by_id(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        return self.models[0]

    @property
    def model(self) -> ModelConfig:
        return self.model_by_id(self.default_model)
```

Dans `load_config`, remplacer la construction du modèle unique par le parsing du registre
(rétro-compatible). Remplacer le bloc `m = data["model"]` ... `model=ModelConfig(...)` par :
```python
    raw_models = data.get("models")
    if raw_models:
        models = [
            ModelConfig(
                repo=rm["repo"], filename=rm["filename"],
                n_layers=int(rm["n_layers"]), size_mb=int(rm["size_mb"]),
                mmproj_filename=rm.get("mmproj_filename", ""),
                id=rm.get("id", "") or rm["filename"],
                n_gpu_layers=rm.get("n_gpu_layers"),
            )
            for rm in raw_models
        ]
    else:
        m = data["model"]
        models = [
            ModelConfig(
                repo=m["repo"], filename=m["filename"],
                n_layers=int(m["n_layers"]), size_mb=int(m["size_mb"]),
                mmproj_filename=m.get("mmproj_filename", ""), id="default",
            )
        ]
    default_model = ch.get("default_model") or models[0].id
```
Et le `return RuntimeConfig(...)` devient :
```python
    return RuntimeConfig(
        models=models,
        default_model=default_model,
        context=int(s["context"]),
        port=int(s["port"]),
        server_bin=s["bin"],
        override_n_gpu_layers=o.get("n_gpu_layers"),
        override_threads=o.get("threads"),
        chat=chat,
    )
```
(`ch` doit donc être calculé avant ce bloc — il l'est déjà pour `ChatConfig`.)

- [ ] **Step 4: Migrer `loom/loom.config.toml`** — remplacer la section `[model]` par :
```toml
[[models]]
id = "gemma-uncensored"
repo = "mradermacher/gemma-4-E4B-it-uncensored-GGUF"
filename = "gemma-4-E4B-it-uncensored.Q4_K_M.gguf"
mmproj_filename = "mmproj-F16.gguf"
n_layers = 35
size_mb = 5340
```
Et ajouter dans `[chat]` : `default_model = "gemma-uncensored"`.

- [ ] **Step 5: Vérifier** — `uv run pytest tests/test_config.py -v` → PASS (anciens via compat + nouveaux). Puis `uv run pytest -q` (la propriété `model` garde serve.py/tests compatibles).

- [ ] **Step 6: Commit** — `git commit -m "feat(config): registre [[models]] + default_model (compat [model])"`

---

## Task 2: `swap.py` — génération du llama-swap.yaml

**Files:** `loom/swap.py` 🆕, `tests/test_swap.py` 🆕

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_swap.py
from loom.hardware import HardwareProfile
from loom.config import ModelConfig
from loom.swap import build_swap_config, dump_yaml


def _models():
    return [
        ModelConfig(repo="r", filename="gemma.gguf", n_layers=35, size_mb=5000,
                    mmproj_filename="mmproj.gguf", id="gemma"),
        ModelConfig(repo="r", filename="qwen.gguf", n_layers=48, size_mb=21000, id="qwen"),
    ]


def test_build_swap_config_structure():
    prof = HardwareProfile(True, "GPU", 6000, 12)
    cfg = build_swap_config(_models(), prof, llama_bin="llama-server",
                            models_dir="/m", context=8192)
    assert set(cfg["models"]) == {"gemma", "qwen"}
    gemma_cmd = cfg["models"]["gemma"]["cmd"]
    assert "/m/gemma.gguf" in gemma_cmd
    assert "--mmproj" in gemma_cmd and "/m/mmproj.gguf" in gemma_cmd
    assert "${PORT}" in gemma_cmd
    assert "-c 8192" in gemma_cmd
    # qwen n'a pas de mmproj
    assert "--mmproj" not in cfg["models"]["qwen"]["cmd"]


def test_dump_yaml_simple():
    y = dump_yaml({"models": {"a": {"cmd": "x -m /p/a.gguf --port ${PORT}"}}})
    assert "models:" in y
    assert '"a":' in y
    assert "cmd:" in y
    assert "${PORT}" in y
```

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_swap.py -v` → FAIL (module absent).

- [ ] **Step 3: Implémenter `loom/swap.py`**

```python
# loom/swap.py
"""Génération de la config llama-swap (un modèle = une commande llama-server)."""
from __future__ import annotations

from pathlib import Path

from loom.config import ModelConfig
from loom.hardware import HardwareProfile, recommend_gpu_layers
from loom.server_args import build_server_args


def _model_cmd(
    model: ModelConfig, profile: HardwareProfile, llama_bin: str,
    models_dir: str, context: int,
) -> str:
    model_path = f"{models_dir}/{model.filename}"
    if model.n_gpu_layers is not None:
        ngl = model.n_gpu_layers
    elif profile.has_gpu:
        ngl = recommend_gpu_layers(profile.vram_free_mb, model.size_mb, model.n_layers)
    else:
        ngl = 0
    mmproj = f"{models_dir}/{model.mmproj_filename}" if model.mmproj_filename else None
    args = build_server_args(
        server_bin=llama_bin, model_path=model_path, port="${PORT}",
        context=context, n_gpu_layers=ngl, threads=profile.cpu_threads,
        mmproj_path=mmproj,
    )
    return " ".join(str(a) for a in args).replace("\\", "/")


def build_swap_config(
    models: list[ModelConfig], profile: HardwareProfile, llama_bin: str,
    models_dir: str, context: int,
) -> dict:
    return {
        "models": {
            m.id: {"cmd": _model_cmd(m, profile, llama_bin, models_dir, context)}
            for m in models
        }
    }


def dump_yaml(config: dict) -> str:
    """Sérialise la structure simple {models: {id: {cmd: str}}} en YAML."""
    lines = ["models:"]
    for model_id, entry in config["models"].items():
        lines.append(f'  "{model_id}":')
        cmd = entry["cmd"].replace('"', '\\"')
        lines.append(f'    cmd: "{cmd}"')
    return "\n".join(lines) + "\n"


def write_swap_yaml(config: dict, path: str | Path) -> None:
    Path(path).write_text(dump_yaml(config), encoding="utf-8")
```

> Note : `build_server_args` reçoit `port="${PORT}"` (string) ; il fait `str(port)` donc le macro
> llama-swap passe tel quel.

- [ ] **Step 4: Vérifier** — `uv run pytest tests/test_swap.py -v` → PASS (2 tests).

- [ ] **Step 5: Commit** — `git commit -m "feat(swap): generation llama-swap.yaml depuis le registre"`

---

## Task 3: `serve.py` — télécharge le registre + lance llama-swap

**Files:** `loom/serve.py`, `loom/loom.config.toml` (déjà migré), `tests/test_serve.py`

- [ ] **Step 1: Test qui échoue** — ajouter dans `tests/test_serve.py`

```python
from loom.serve import ensure_all_models
from loom.config import ModelConfig
from unittest.mock import patch


def test_ensure_all_models_downloads_each(tmp_path):
    models = [
        ModelConfig(repo="r1", filename="a.gguf", n_layers=1, size_mb=1, id="a"),
        ModelConfig(repo="r2", filename="b.gguf", n_layers=1, size_mb=1,
                    mmproj_filename="mm.gguf", id="b"),
    ]
    with patch("loom.serve.ensure_model") as ens:
        ens.return_value = tmp_path / "x"
        ensure_all_models(models, tmp_path)
    # a.gguf, b.gguf, et le mmproj de b => 3 appels
    assert ens.call_count == 3
```

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_serve.py::test_ensure_all_models_downloads_each -v` → FAIL (import).

- [ ] **Step 3: Modifier `loom/serve.py`**

Ajouter les imports en haut :
```python
from loom.swap import build_swap_config, write_swap_yaml
```

Ajouter la fonction de téléchargement groupé (après `resolve_mmproj_path`) :
```python
def ensure_all_models(models, models_dir: Path) -> None:
    """Télécharge le GGUF (et le mmproj) de chaque modèle du registre s'il manque."""
    for m in models:
        ensure_model(m.repo, m.filename, models_dir)
        if m.mmproj_filename:
            ensure_model(m.repo, m.mmproj_filename, models_dir)
```

Réécrire `main()` pour générer le yaml et lancer llama-swap :
```python
SWAP_YAML = MODELS_DIR.parent / "llama-swap.yaml"


def main() -> int:
    cfg = load_config(CONFIG_PATH, LOCAL_CONFIG_PATH)
    profile = detect_hardware()
    print(f"[loom] Profil détecté : {profile}", file=sys.stderr)

    ensure_all_models(cfg.models, MODELS_DIR)
    swap = build_swap_config(
        cfg.models, profile, llama_bin=cfg.server_bin,
        models_dir=str(MODELS_DIR), context=cfg.context,
    )
    write_swap_yaml(swap, SWAP_YAML)
    print(f"[loom] {len(cfg.models)} modèle(s), défaut={cfg.default_model}", file=sys.stderr)

    swap_bin = getattr(cfg, "swap_bin", None) or "llama-swap"
    args = [swap_bin, "--config", str(SWAP_YAML), "--listen", f"127.0.0.1:{cfg.port}"]
    print(f"[loom] Lancement llama-swap : {' '.join(args)}", file=sys.stderr)
    try:
        return subprocess.run(args).returncode
    except FileNotFoundError:
        print(
            f"[loom] ERREUR : binaire '{swap_bin}' introuvable. "
            "Télécharge llama-swap (voir docs/install-windows.md).",
            file=sys.stderr,
        )
        return 1
```
(Supprimer l'ancienne logique `model_path = ensure_model(...)` / `build_launch(...)` du `main`.
On garde `resolve_n_gpu_layers`/`build_launch`/`build_server_args` : ils servent à `swap.py`.)

Ajouter le support de `swap_bin` dans la config serveur : dans `loom.config.toml` `[server]`,
ajouter `swap_bin = "llama-swap"`, et dans `config.py` `RuntimeConfig` ajouter `swap_bin: str` +
parsing `s.get("swap_bin", "llama-swap")`. (Mettre la valeur après `server_bin`.)

- [ ] **Step 4: Vérifier** — `uv run pytest tests/test_serve.py -v` → PASS. Puis `uv run pytest -q`.

- [ ] **Step 5: Commit** — `git commit -m "feat(serve): telecharge le registre + lance llama-swap"`

---

## Task 4: `client.py` — `model` paramètre d'appel

**Files:** `loom/client.py`, `tests/test_client.py`

- [ ] **Step 1: Test qui échoue** — ajouter dans `tests/test_client.py`

```python
def test_build_create_kwargs_uses_given_model():
    kw = build_create_kwargs(model="qwen", messages=[{"role": "user", "content": "x"}],
                             system_prompt="s", max_tokens=100)
    assert kw["model"] == "qwen"
```

- [ ] **Step 2: Vérifier l'échec** — c'est déjà le cas si `build_create_kwargs` prend `model` ;
  sinon adapter. `uv run pytest tests/test_client.py::test_build_create_kwargs_uses_given_model -v`.

- [ ] **Step 3: Modifier `stream_chat` pour accepter un `model` optionnel**

```python
    def stream_chat(
        self, messages: list[dict], system_prompt: str,
        max_tokens: int = 2048, model: str | None = None,
    ) -> Iterator[tuple[str, str]]:
        kwargs = build_create_kwargs(model or self.model, messages, system_prompt, max_tokens)
        stream = self._client.chat.completions.create(**kwargs)
        yield from _iter_events(stream)
```

- [ ] **Step 4: Vérifier** — `uv run pytest tests/test_client.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(client): model parametre d'appel"`

---

## Task 5: `conversation.py` — modèle de la conversation

**Files:** `loom/conversation.py`, `tests/test_conversation.py`

- [ ] **Step 1: Test qui échoue** — ajouter dans `tests/test_conversation.py`

```python
def test_model_roundtrip_and_default(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys", model="gemma")
    conv.set_model("qwen")
    conv.save(path)
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.model == "qwen"


def test_load_old_json_without_model(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text('{"system_prompt": "s", "messages": []}', encoding="utf-8")
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.model == ""
```

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_conversation.py::test_model_roundtrip_and_default -v` → FAIL.

- [ ] **Step 3: Modifier `loom/conversation.py`**

Ajouter le champ (après `active_skills`) :
```python
    model: str = ""
```
Ajouter la méthode :
```python
    def set_model(self, model: str) -> None:
        self.model = model
```
Inclure dans `save` (dict `data`) : `"model": self.model,`
Et dans `load` : `model=data.get("model", ""),`

- [ ] **Step 4: Vérifier** — `uv run pytest tests/test_conversation.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(conversation): modele de la conversation persiste"`

---

## Task 6: `web` — sélection de modèle + transmission

**Files:** `loom/web/app.py`, `loom/web/__main__.py`, `loom/web/templates/index.html`, `loom/web/templates/_models.html` 🆕, `tests/test_web.py`

- [ ] **Step 1: MAJ tests** — dans `tests/test_web.py`, adapter `_make` (passer `models` + `default_model`)
  et ajouter les tests :

```python
def _make(tmp_path, events=(("content", "Hel"), ("content", "lo")), budget=100000):
    conv = Conversation(system_prompt="sys", model="gemma")
    history = tmp_path / "conv.json"
    skills_dir = tmp_path / "skills"
    (skills_dir / "dagster").mkdir(parents=True)
    (skills_dir / "dagster" / "SKILL.md").write_text(
        "---\nname: dagster\ndescription: archi\n---\nARCHI_DAGSTER_XYZ", encoding="utf-8"
    )
    fake = FakeClient(list(events))
    app = create_app(conv, fake, history, skills_dir, max_tokens=2048,
                     context_budget=budget, keep_recent=3,
                     models=["gemma", "qwen"])
    app.config["_fake_client"] = fake
    return app, conv, history


def test_index_lists_models(tmp_path):
    app, _, _ = _make(tmp_path)
    body = app.test_client().get("/").get_data(as_text=True)
    assert "gemma" in body and "qwen" in body


def test_post_model_updates_conversation(tmp_path):
    app, conv, _ = _make(tmp_path)
    resp = app.test_client().post("/model", data={"model": "qwen"})
    assert resp.status_code == 200
    assert conv.model == "qwen"


def test_chat_sends_conversation_model(tmp_path):
    app, conv, _ = _make(tmp_path)
    conv.set_model("qwen")
    app.test_client().post("/chat", data={"message": "salut"})
    assert app.config["_fake_client"].last_model == "qwen"
```

Mettre à jour `FakeClient.stream_chat` pour mémoriser le `model` :
```python
    def stream_chat(self, messages, system_prompt, max_tokens=2048, model=None):
        self.last_system_prompt = system_prompt
        self.last_model = model
        yield from self._events
```

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_web.py::test_post_model_updates_conversation -v` → FAIL.

- [ ] **Step 3: Modifier `loom/web/app.py`**

Signature : ajouter `models` (liste d'ids) en keyword-only :
```python
def create_app(
    conversation, client, history_path, skills_dir, *,
    max_tokens=2048, context_budget=3000, keep_recent=6, models=None,
) -> Flask:
    ...
    models = list(models or [])
```

`GET /` passe les modèles :
```python
    @app.get("/")
    def index() -> str:
        skills = list_skills(skills_dir)
        return render_template(
            "index.html", messages=conversation.messages, skills=skills,
            active_skills=conversation.active_skills,
            models=models, current_model=conversation.model,
        )
```

Route `POST /model` (avant `return app`) :
```python
    @app.post("/model")
    def model_update():
        conversation.set_model(request.form.get("model", ""))
        conversation.save(history_path)
        return render_template("_models.html", models=models,
                               current_model=conversation.model)
```

Dans `/chat`, passer le modèle à `stream_chat` :
```python
                for kind, text in client.stream_chat(
                    conversation.to_messages(), system_prompt, max_tokens,
                    model=conversation.model or None,
                ):
```

- [ ] **Step 4: Créer `loom/web/templates/_models.html`**

```html
<select id="model-select" name="model" hx-post="/model" hx-target="#model-select"
        hx-swap="outerHTML">
  {% for mid in models %}
    <option value="{{ mid }}"{% if mid == current_model %} selected{% endif %}>{{ mid }}</option>
  {% endfor %}
</select>
```

- [ ] **Step 5: Insérer le sélecteur dans `index.html`** — dans le `<h1>`, avant le bouton Reset :
```html
    {% include "_models.html" %}
```
(et un petit style : `#model-select { background:#161922; color:#e6e6e6; border:1px solid #2a2d3e;
border-radius:8px; padding:4px 8px; font-size:13px; }`)

- [ ] **Step 6: MAJ `loom/web/__main__.py`** — passer la liste des ids :
```python
    app = create_app(
        conversation, client, cfg.chat.history_path, cfg.chat.skills_dir,
        max_tokens=cfg.chat.max_tokens, context_budget=budget,
        keep_recent=cfg.chat.keep_recent_messages,
        models=[m.id for m in cfg.models],
    )
```
Et initialiser le modèle de la conversation s'il est vide :
```python
    if not conversation.model:
        conversation.set_model(cfg.default_model)
```

- [ ] **Step 7: Vérifier** — `uv run pytest tests/test_web.py -v` puis `uv run pytest -q` → tout vert.

- [ ] **Step 8: Commit** — `git commit -m "feat(web): selecteur de modele + transmission au client"`

---

## Task 7: Vérification finale

- [ ] **Step 1:** `uv run pytest -q` → tous verts.

## Definition of Done (v4)
- [ ] Registre `[[models]]` + `default_model` chargé (compat `[model]`).
- [ ] `swap.py` génère le yaml (cmd `${PORT}`, `--mmproj` sur modèles vision).
- [ ] `serve.py` télécharge le registre + lance llama-swap.
- [ ] `client`/`conversation`/`web` portent le `model` ; sélecteur UI fonctionnel.
- [ ] `uv run pytest` tout vert.
- [ ] (Manuel) Installer le binaire llama-swap, `uv run loom/serve.py`, choisir le modèle dans l'UI.
