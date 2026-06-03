# Fondation Runtime (Sous-projet A) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire tourner un modèle open-source local servi via API OpenAI-compatible, qui auto-détecte le hardware (GPU sinon CPU), est benchmarké, et démarre à l'identique sur une autre machine.

**Architecture:** `llama-server` (binaire llama.cpp, spécifique plateforme) sert le modèle. Un lanceur Python cross-platform (`runtime/serve.py`) auto-détecte les ressources, résout la config, télécharge le GGUF si absent, et démarre le serveur. Tout le code au-dessus ne parle qu'à l'API OpenAI-compatible (`/v1`). Le code Python est découpé en modules à responsabilité unique et testables (`hardware`, `config`, `models_fetch`, `server_args`), orchestrés par `serve.py`.

**Tech Stack:** Python 3.12, `uv` (gestion env + run), `pytest`, `huggingface_hub` (download GGUF), `requests` (benchmark), `tomllib` (stdlib), llama.cpp (`llama-server`).

**Spec de référence:** [docs/superpowers/specs/2026-05-31-runtime-foundation-design.md](../specs/2026-05-31-runtime-foundation-design.md)

---

## Structure des fichiers

| Fichier | Responsabilité |
|---|---|
| `pyproject.toml` | Déclare les deps (uv) : pytest, huggingface_hub, requests |
| `.gitignore` | Exclut `runtime/models/`, `*.local.toml`, `.venv`, caches |
| `runtime/__init__.py` | Marque le package |
| `runtime/hardware.py` | Détection hardware (GPU/VRAM/threads) + recommandation `n_gpu_layers` |
| `runtime/config.py` | Chargement + fusion `runtime.config.toml` (+ override local) |
| `runtime/models_fetch.py` | Garantit la présence du GGUF (download si absent) |
| `runtime/server_args.py` | Fonction pure : construit la ligne de commande `llama-server` |
| `runtime/serve.py` | Orchestrateur : assemble tout et lance le serveur |
| `runtime/benchmark.py` | Test d'intégration : tok/s, TTFT, validité JSON contraint |
| `runtime/runtime.config.toml` | Config versionnée (modèle épinglé, contexte, port) |
| `tests/test_hardware.py` | Tests de `hardware.py` |
| `tests/test_config.py` | Tests de `config.py` |
| `tests/test_models_fetch.py` | Tests de `models_fetch.py` |
| `tests/test_server_args.py` | Tests de `server_args.py` |
| `docs/adr/0001-llamacpp-vs-ollama.md` | ADR du choix de runtime |
| `docs/install-windows.md` | Install `llama-server` CUDA sur le laptop |
| `docs/install-linux.md` | Install `llama-server` CPU sur le VPS |

---

## Task 0: Scaffolding du projet (uv + git)

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `runtime/__init__.py`

- [ ] **Step 1: Initialiser git (recommandé pour commits incrémentaux)**

Run:
```powershell
git init
```
Expected: `Initialized empty Git repository ...`. *(Si tu refuses git, saute cette étape ; les `git commit` suivants deviennent optionnels.)*

- [ ] **Step 2: Créer `pyproject.toml`**

```toml
[project]
name = "local-llm-runtime"
version = "0.1.0"
description = "Fondation runtime pour LLM local (sous-projet A)"
requires-python = ">=3.12"
dependencies = [
    "huggingface-hub>=0.25",
    "requests>=2.32",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Créer `.gitignore`**

```gitignore
# Modèles (trop lourds)
runtime/models/
# Surcharges machine
*.local.toml
# Python / uv
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
*.pyc
```

- [ ] **Step 4: Créer le package `runtime/__init__.py`**

```python
"""Fondation runtime pour LLM local servi via API OpenAI-compatible."""
```

- [ ] **Step 5: Synchroniser l'environnement uv et vérifier pytest**

Run:
```powershell
uv sync; uv run pytest --version
```
Expected: une version `pytest 8.x` s'affiche (aucun test pour l'instant).

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml .gitignore runtime/__init__.py
git commit -m "chore: scaffolding uv + structure runtime"
```

---

## Task 1: Recommandation de `n_gpu_layers` (fonction pure)

On commence par la logique pure et testable, avant la détection qui dépend du système.

**Files:**
- Create: `runtime/hardware.py`
- Test: `tests/test_hardware.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_hardware.py
from runtime.hardware import recommend_gpu_layers


def test_cpu_only_when_no_vram_budget():
    # VRAM dispo <= marge KV -> tout sur CPU
    assert recommend_gpu_layers(vram_free_mb=512, model_size_mb=4700,
                                total_layers=28, kv_headroom_mb=1024) == 0


def test_all_layers_when_model_fits():
    # 6 Go libres, modèle 4.7 Go + marge -> toutes les couches sur GPU
    assert recommend_gpu_layers(vram_free_mb=6000, model_size_mb=4700,
                                total_layers=28, kv_headroom_mb=1024) == 28


def test_partial_offload_when_tight():
    # budget = 4000-1000 = 3000 ; 3000/6000 * 32 = 16 couches
    assert recommend_gpu_layers(vram_free_mb=4000, model_size_mb=6000,
                                total_layers=32, kv_headroom_mb=1024) == 16
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_hardware.py -v`
Expected: FAIL — `ImportError: cannot import name 'recommend_gpu_layers'`.

- [ ] **Step 3: Implémenter la fonction pure**

```python
# runtime/hardware.py
"""Détection hardware et recommandation de réglages d'offload GPU."""
from __future__ import annotations


def recommend_gpu_layers(
    vram_free_mb: int,
    model_size_mb: int,
    total_layers: int,
    kv_headroom_mb: int = 1024,
) -> int:
    """Nombre de couches à offloader sur GPU selon la VRAM libre.

    Renvoie 0 (CPU-only) si le budget VRAM ne dépasse pas la marge réservée
    au cache KV. Sinon offload proportionnel, plafonné à toutes les couches.
    """
    budget_mb = vram_free_mb - kv_headroom_mb
    if budget_mb <= 0:
        return 0
    if budget_mb >= model_size_mb:
        return total_layers
    return max(0, round(total_layers * budget_mb / model_size_mb))
```

- [ ] **Step 4: Lancer le test pour vérifier le succès**

Run: `uv run pytest tests/test_hardware.py -v`
Expected: PASS (3 tests verts).

- [ ] **Step 5: Commit**

```powershell
git add runtime/hardware.py tests/test_hardware.py
git commit -m "feat(hardware): recommandation n_gpu_layers selon VRAM"
```

---

## Task 2: Parsing de la sortie `nvidia-smi`

**Files:**
- Modify: `runtime/hardware.py`
- Test: `tests/test_hardware.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_hardware.py  (ajouter en haut l'import)
from runtime.hardware import recommend_gpu_layers, parse_nvidia_smi


def test_parse_nvidia_smi_valid():
    # format CSV sans header : "name, memory.free [MiB]"
    out = "NVIDIA GeForce RTX 2060, 5800\n"
    name, free_mb = parse_nvidia_smi(out)
    assert name == "NVIDIA GeForce RTX 2060"
    assert free_mb == 5800


def test_parse_nvidia_smi_empty_returns_none():
    assert parse_nvidia_smi("") is None
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_hardware.py::test_parse_nvidia_smi_valid -v`
Expected: FAIL — `ImportError: cannot import name 'parse_nvidia_smi'`.

- [ ] **Step 3: Implémenter le parseur (ajout dans `runtime/hardware.py`)**

```python
def parse_nvidia_smi(output: str) -> tuple[str, int] | None:
    """Parse la 1re ligne de `nvidia-smi --query-gpu=name,memory.free
    --format=csv,noheader,nounits`. Renvoie (nom, vram_libre_mb) ou None.
    """
    line = output.strip().splitlines()[0] if output.strip() else ""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1])
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `uv run pytest tests/test_hardware.py -v`
Expected: PASS (5 tests verts).

- [ ] **Step 5: Commit**

```powershell
git add runtime/hardware.py tests/test_hardware.py
git commit -m "feat(hardware): parsing sortie nvidia-smi"
```

---

## Task 3: Détection hardware complète (`detect_hardware`)

**Files:**
- Modify: `runtime/hardware.py`
- Test: `tests/test_hardware.py`

- [ ] **Step 1: Écrire le test qui échoue (avec mocks)**

```python
# tests/test_hardware.py (ajouter ces imports en haut)
from unittest.mock import patch
from runtime.hardware import detect_hardware, HardwareProfile


def test_detect_hardware_with_gpu():
    fake = "NVIDIA GeForce RTX 2060, 5800\n"
    with patch("runtime.hardware._run_nvidia_smi", return_value=fake), \
         patch("runtime.hardware.os.cpu_count", return_value=12):
        prof = detect_hardware()
    assert isinstance(prof, HardwareProfile)
    assert prof.has_gpu is True
    assert prof.gpu_name == "NVIDIA GeForce RTX 2060"
    assert prof.vram_free_mb == 5800
    assert prof.cpu_threads == 12


def test_detect_hardware_cpu_only():
    with patch("runtime.hardware._run_nvidia_smi", return_value=None), \
         patch("runtime.hardware.os.cpu_count", return_value=16):
        prof = detect_hardware()
    assert prof.has_gpu is False
    assert prof.gpu_name is None
    assert prof.vram_free_mb == 0
    assert prof.cpu_threads == 16
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_hardware.py::test_detect_hardware_with_gpu -v`
Expected: FAIL — `ImportError: cannot import name 'detect_hardware'`.

- [ ] **Step 3: Implémenter `detect_hardware` + `HardwareProfile` (ajout dans `runtime/hardware.py`)**

Ajouter les imports en haut du fichier :
```python
import os
import shutil
import subprocess
from dataclasses import dataclass
```

Puis le dataclass et les fonctions :
```python
@dataclass
class HardwareProfile:
    has_gpu: bool
    gpu_name: str | None
    vram_free_mb: int
    cpu_threads: int


def _run_nvidia_smi() -> str | None:
    """Exécute nvidia-smi si présent, renvoie sa sortie CSV ou None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return res.stdout
    except (subprocess.SubprocessError, OSError):
        return None


def detect_hardware() -> HardwareProfile:
    """Détecte le meilleur profil disponible : GPU NVIDIA si présent, sinon CPU."""
    threads = os.cpu_count() or 4
    raw = _run_nvidia_smi()
    parsed = parse_nvidia_smi(raw) if raw else None
    if parsed is None:
        return HardwareProfile(False, None, 0, threads)
    name, free_mb = parsed
    return HardwareProfile(True, name, free_mb, threads)
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `uv run pytest tests/test_hardware.py -v`
Expected: PASS (7 tests verts).

- [ ] **Step 5: Vérifier la détection réelle sur le laptop**

Run: `uv run python -c "from runtime.hardware import detect_hardware; print(detect_hardware())"`
Expected: affiche `HardwareProfile(has_gpu=True, gpu_name='NVIDIA GeForce RTX 2060', vram_free_mb=<~6000>, cpu_threads=12)`.

- [ ] **Step 6: Commit**

```powershell
git add runtime/hardware.py tests/test_hardware.py
git commit -m "feat(hardware): detect_hardware (GPU NVIDIA sinon CPU)"
```

---

## Task 4: Chargement de configuration

**Files:**
- Create: `runtime/config.py`
- Create: `runtime/runtime.config.toml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Créer la config versionnée `runtime/runtime.config.toml`**

```toml
# Config runtime — paramètres NON liés au hardware.
# Les valeurs hardware (n_gpu_layers, threads) sont auto-détectées par serve.py ;
# on ne les met ici (section [override]) que pour forcer/plafonner.

[model]
# Qwen2.5-Coder-7B-Instruct, quant Q4_K_M (~4.7 Go), 28 couches.
repo = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
n_layers = 28
size_mb = 4700

[server]
context = 8192
port = 8080
# Chemin du binaire llama-server (à adapter par machine via le fichier .local.toml).
bin = "llama-server"

[override]
# n_gpu_layers = 20   # décommenter pour forcer
# threads = 8
```

- [ ] **Step 2: Écrire le test qui échoue**

```python
# tests/test_config.py
from pathlib import Path
from runtime.config import load_config, RuntimeConfig


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


BASE = """
[model]
repo = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
n_layers = 28
size_mb = 4700

[server]
context = 8192
port = 8080
bin = "llama-server"

[override]
"""


def test_load_base_config(tmp_path):
    cfg_path = _write(tmp_path, "runtime.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.model.repo.endswith("Qwen2.5-Coder-7B-Instruct-GGUF")
    assert cfg.model.n_layers == 28
    assert cfg.model.size_mb == 4700
    assert cfg.context == 8192
    assert cfg.port == 8080
    assert cfg.server_bin == "llama-server"
    assert cfg.override_n_gpu_layers is None
    assert cfg.override_threads is None


def test_local_override_merges(tmp_path):
    cfg_path = _write(tmp_path, "runtime.config.toml", BASE)
    local = _write(tmp_path, "runtime.config.local.toml",
                   '[server]\nbin = "C:/tools/llama/llama-server.exe"\n'
                   '[override]\nn_gpu_layers = 12\n')
    cfg = load_config(cfg_path, local_path=local)
    assert cfg.server_bin == "C:/tools/llama/llama-server.exe"
    assert cfg.override_n_gpu_layers == 12
```

- [ ] **Step 3: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.config'`.

- [ ] **Step 4: Implémenter `runtime/config.py`**

```python
# runtime/config.py
"""Chargement et fusion de la configuration runtime (TOML)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    repo: str
    filename: str
    n_layers: int
    size_mb: int


@dataclass
class RuntimeConfig:
    model: ModelConfig
    context: int
    port: int
    server_bin: str
    override_n_gpu_layers: int | None
    override_threads: int | None


def _deep_merge(base: dict, over: dict) -> dict:
    """Fusionne récursivement `over` dans une copie de `base`."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path, local_path: str | Path | None = None) -> RuntimeConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if local_path is not None and Path(local_path).exists():
        local = tomllib.loads(Path(local_path).read_text(encoding="utf-8"))
        data = _deep_merge(data, local)

    m = data["model"]
    s = data["server"]
    o = data.get("override", {})
    return RuntimeConfig(
        model=ModelConfig(m["repo"], m["filename"], int(m["n_layers"]), int(m["size_mb"])),
        context=int(s["context"]),
        port=int(s["port"]),
        server_bin=s["bin"],
        override_n_gpu_layers=o.get("n_gpu_layers"),
        override_threads=o.get("threads"),
    )
```

- [ ] **Step 5: Lancer les tests pour vérifier le succès**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 tests verts).

- [ ] **Step 6: Commit**

```powershell
git add runtime/config.py runtime/runtime.config.toml tests/test_config.py
git commit -m "feat(config): chargement TOML + override local"
```

---

## Task 5: Garantie de présence du modèle (`ensure_model`)

**Files:**
- Create: `runtime/models_fetch.py`
- Test: `tests/test_models_fetch.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_models_fetch.py
from pathlib import Path
from unittest.mock import patch
from runtime.models_fetch import ensure_model


def test_returns_existing_without_download(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    existing = models_dir / "model.gguf"
    existing.write_bytes(b"GGUF")
    with patch("runtime.models_fetch.hf_hub_download") as dl:
        result = ensure_model("some/repo", "model.gguf", models_dir)
    dl.assert_not_called()
    assert result == existing


def test_downloads_when_absent(tmp_path):
    models_dir = tmp_path / "models"
    target = models_dir / "model.gguf"

    def fake_download(repo_id, filename, local_dir, **kwargs):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / filename).write_bytes(b"GGUF")
        return str(Path(local_dir) / filename)

    with patch("runtime.models_fetch.hf_hub_download", side_effect=fake_download) as dl:
        result = ensure_model("some/repo", "model.gguf", models_dir)
    dl.assert_called_once()
    assert result == target
    assert target.exists()
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_models_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.models_fetch'`.

- [ ] **Step 3: Implémenter `runtime/models_fetch.py`**

```python
# runtime/models_fetch.py
"""Garantit la présence locale d'un fichier GGUF (download si absent)."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


def ensure_model(repo: str, filename: str, models_dir: str | Path) -> Path:
    """Renvoie le chemin local du GGUF, en le téléchargeant depuis HF si absent."""
    models_dir = Path(models_dir)
    target = models_dir / filename
    if target.exists():
        return target
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(models_dir),
    )
    return target
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `uv run pytest tests/test_models_fetch.py -v`
Expected: PASS (2 tests verts).

- [ ] **Step 5: Commit**

```powershell
git add runtime/models_fetch.py tests/test_models_fetch.py
git commit -m "feat(models): ensure_model télécharge le GGUF si absent"
```

---

## Task 6: Construction de la commande `llama-server` (fonction pure)

**Files:**
- Create: `runtime/server_args.py`
- Test: `tests/test_server_args.py`

- [ ] **Step 1: Écrire le test qui échoue**

```python
# tests/test_server_args.py
from runtime.server_args import build_server_args


def test_build_server_args_gpu():
    args = build_server_args(
        server_bin="llama-server", model_path="/m/model.gguf",
        port=8080, context=8192, n_gpu_layers=28, threads=12,
    )
    assert args[0] == "llama-server"
    assert "-m" in args and "/m/model.gguf" in args
    assert "--port" in args and "8080" in args
    assert "-c" in args and "8192" in args
    assert "-ngl" in args and "28" in args
    assert "-t" in args and "12" in args
    assert "--host" in args and "127.0.0.1" in args


def test_build_server_args_cpu_only():
    args = build_server_args(
        server_bin="llama-server", model_path="/m/model.gguf",
        port=9000, context=4096, n_gpu_layers=0, threads=16,
    )
    i = args.index("-ngl")
    assert args[i + 1] == "0"
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_server_args.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.server_args'`.

- [ ] **Step 3: Implémenter `runtime/server_args.py`**

```python
# runtime/server_args.py
"""Construction (pure) de la ligne de commande llama-server."""
from __future__ import annotations


def build_server_args(
    server_bin: str,
    model_path: str,
    port: int,
    context: int,
    n_gpu_layers: int,
    threads: int,
) -> list[str]:
    """Liste d'arguments pour lancer llama-server en API OpenAI-compatible local."""
    return [
        server_bin,
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(context),
        "-ngl", str(n_gpu_layers),
        "-t", str(threads),
    ]
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `uv run pytest tests/test_server_args.py -v`
Expected: PASS (2 tests verts).

- [ ] **Step 5: Commit**

```powershell
git add runtime/server_args.py tests/test_server_args.py
git commit -m "feat(server): build_server_args (commande llama-server)"
```

---

## Task 7: Orchestrateur `serve.py`

Assemble les modules : config → détection hardware → résolution `n_gpu_layers` → modèle → commande → lancement. On extrait la résolution dans une fonction pure testée, le `main()` reste fin.

**Files:**
- Create: `runtime/serve.py`
- Test: `tests/test_serve.py`

- [ ] **Step 1: Écrire le test qui échoue (résolution de l'offload)**

```python
# tests/test_serve.py
from runtime.serve import resolve_n_gpu_layers
from runtime.hardware import HardwareProfile


def test_resolve_uses_override_when_present():
    prof = HardwareProfile(True, "GPU", 6000, 12)
    assert resolve_n_gpu_layers(prof, override=10,
                                model_size_mb=4700, total_layers=28) == 10


def test_resolve_cpu_profile_gives_zero():
    prof = HardwareProfile(False, None, 0, 16)
    assert resolve_n_gpu_layers(prof, override=None,
                                model_size_mb=4700, total_layers=28) == 0


def test_resolve_gpu_auto_recommends_all_when_fits():
    prof = HardwareProfile(True, "GPU", 6000, 12)
    assert resolve_n_gpu_layers(prof, override=None,
                                model_size_mb=4700, total_layers=28) == 28
```

- [ ] **Step 2: Lancer le test pour vérifier l'échec**

Run: `uv run pytest tests/test_serve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.serve'`.

- [ ] **Step 3: Implémenter `runtime/serve.py`**

```python
# runtime/serve.py
"""Lanceur cross-platform et auto-adaptatif de llama-server.

Usage : uv run runtime/serve.py
Auto-détecte le hardware (GPU NVIDIA sinon CPU), résout la config,
télécharge le GGUF si absent, et démarre le serveur en API OpenAI-compatible.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from runtime.config import RuntimeConfig, load_config
from runtime.hardware import HardwareProfile, detect_hardware, recommend_gpu_layers
from runtime.models_fetch import ensure_model
from runtime.server_args import build_server_args

RUNTIME_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUNTIME_DIR / "runtime.config.toml"
LOCAL_CONFIG_PATH = RUNTIME_DIR / "runtime.config.local.toml"
MODELS_DIR = RUNTIME_DIR / "models"


def resolve_n_gpu_layers(
    profile: HardwareProfile,
    override: int | None,
    model_size_mb: int,
    total_layers: int,
) -> int:
    """Override prioritaire ; sinon 0 en CPU, sinon recommandation auto."""
    if override is not None:
        return override
    if not profile.has_gpu:
        return 0
    return recommend_gpu_layers(profile.vram_free_mb, model_size_mb, total_layers)


def build_launch(cfg: RuntimeConfig, profile: HardwareProfile, model_path: Path) -> list[str]:
    n_gpu = resolve_n_gpu_layers(
        profile, cfg.override_n_gpu_layers, cfg.model.size_mb, cfg.model.n_layers
    )
    threads = cfg.override_threads or profile.cpu_threads
    return build_server_args(
        server_bin=cfg.server_bin,
        model_path=str(model_path),
        port=cfg.port,
        context=cfg.context,
        n_gpu_layers=n_gpu,
        threads=threads,
    )


def main() -> int:
    cfg = load_config(CONFIG_PATH, LOCAL_CONFIG_PATH)
    profile = detect_hardware()
    print(f"[runtime] Profil détecté : {profile}", file=sys.stderr)

    model_path = ensure_model(cfg.model.repo, cfg.model.filename, MODELS_DIR)
    args = build_launch(cfg, profile, model_path)
    print(f"[runtime] Lancement : {' '.join(args)}", file=sys.stderr)

    try:
        return subprocess.run(args).returncode
    except FileNotFoundError:
        print(
            f"[runtime] ERREUR : binaire '{cfg.server_bin}' introuvable. "
            "Voir docs/install-windows.md ou docs/install-linux.md.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Lancer les tests pour vérifier le succès**

Run: `uv run pytest tests/test_serve.py -v`
Expected: PASS (3 tests verts).

- [ ] **Step 5: Vérifier la suite complète**

Run: `uv run pytest -v`
Expected: PASS — tous les tests des Tasks 1-7 verts.

- [ ] **Step 6: Commit**

```powershell
git add runtime/serve.py tests/test_serve.py
git commit -m "feat(serve): orchestrateur auto-adaptatif (config+hw+modele+lancement)"
```

---

## Task 8: Installer `llama-server` et démarrer pour de vrai (laptop)

Étape d'install manuelle (un binaire ne se TDD pas) — vérifiée par des commandes.

**Files:** aucun fichier de code (étape d'environnement).

- [ ] **Step 1: Télécharger le binaire llama.cpp CUDA pour Windows**

Aller sur les releases GitHub de llama.cpp et récupérer l'archive Windows CUDA (ex. `llama-<version>-bin-win-cuda-x64.zip`). L'extraire dans `C:\tools\llama\`.

Run (vérif) :
```powershell
& "C:\tools\llama\llama-server.exe" --version
```
Expected: une ligne de version llama.cpp s'affiche.

- [ ] **Step 2: Pointer la config locale vers ce binaire**

Créer `runtime/runtime.config.local.toml` (gitignoré) :
```toml
[server]
bin = "C:/tools/llama/llama-server.exe"
```

- [ ] **Step 3: Lancer le serveur (télécharge le GGUF au 1er run)**

Run:
```powershell
uv run runtime/serve.py
```
Expected (stderr) : `Profil détecté : HardwareProfile(has_gpu=True, ...)` puis téléchargement du GGUF (~4.7 Go au 1er lancement), puis llama-server démarre et écoute sur `127.0.0.1:8080`.

- [ ] **Step 4: Vérifier l'endpoint, wifi coupé**

Couper le wifi, puis dans un autre terminal :
```powershell
curl http://127.0.0.1:8080/v1/models
```
Expected: réponse JSON listant le modèle chargé (preuve que ça marche 100 % offline).

- [ ] **Step 5: Commit (la config locale est gitignorée — rien à committer ici)**

Aucune action git. Noter le bon fonctionnement pour la mise à jour de `ETAT_PROJET.md` (Task 11).

---

## Task 9: Benchmark + smoke-test GBNF (`benchmark.py`)

**Files:**
- Create: `runtime/benchmark.py`

Test d'intégration : nécessite le serveur de la Task 8 en cours d'exécution.

- [ ] **Step 1: Implémenter `runtime/benchmark.py`**

```python
# runtime/benchmark.py
"""Benchmark du runtime local : débit, latence, validité JSON contraint.

Prérequis : serve.py doit tourner. Usage : uv run runtime/benchmark.py
Sort en code != 0 si l'endpoint ne répond pas, si le débit est sous le seuil,
ou si la sortie JSON contrainte n'est pas un JSON valide.
"""
from __future__ import annotations

import json
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8080"
MIN_TOKENS_PER_SEC = 5.0  # plancher (CPU-friendly) ; relever sur GPU


def _chat(messages: list[dict], **extra) -> tuple[dict, float]:
    payload = {"model": "local", "messages": messages, "stream": False, **extra}
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json(), elapsed


def bench_throughput() -> float:
    data, elapsed = _chat(
        [{"role": "user", "content": "Écris une fonction Python qui inverse une liste."}],
        max_tokens=128,
    )
    completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    print(f"[bench] {completion_tokens} tokens en {elapsed:.2f}s -> {tps:.1f} tok/s")
    return tps


def bench_json_grammar() -> bool:
    """Force une sortie JSON via response_format et vérifie qu'elle parse."""
    data, _ = _chat(
        [{"role": "user", "content": 'Donne un objet JSON {"langage": ..., "annee": ...}.'}],
        response_format={"type": "json_object"},
        max_tokens=128,
    )
    content = data["choices"][0]["message"]["content"]
    try:
        json.loads(content)
        print("[bench] sortie JSON contrainte : VALIDE")
        return True
    except json.JSONDecodeError:
        print(f"[bench] sortie JSON contrainte : INVALIDE -> {content!r}")
        return False


def main() -> int:
    try:
        tps = bench_throughput()
        json_ok = bench_json_grammar()
    except requests.RequestException as exc:
        print(f"[bench] ERREUR : endpoint injoignable ({exc}). serve.py tourne ?")
        return 1

    ok = tps >= MIN_TOKENS_PER_SEC and json_ok
    print(f"[bench] RÉSULTAT : {'OK' if ok else 'ÉCHEC'} "
          f"(seuil {MIN_TOKENS_PER_SEC} tok/s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lancer le benchmark contre le serveur en cours**

Avec `serve.py` actif dans un autre terminal :
```powershell
uv run runtime/benchmark.py
```
Expected: affiche le débit (tok/s) sur le 7B en GPU (viser ≥ ~15-20 tok/s), `sortie JSON contrainte : VALIDE`, et `RÉSULTAT : OK` (exit 0).

- [ ] **Step 3: Commit**

```powershell
git add runtime/benchmark.py
git commit -m "feat(benchmark): debit + TTFT + validite JSON contraint"
```

---

## Task 10: ADR + guides d'installation

**Files:**
- Create: `docs/adr/0001-llamacpp-vs-ollama.md`
- Create: `docs/install-windows.md`
- Create: `docs/install-linux.md`

- [ ] **Step 1: Écrire l'ADR `docs/adr/0001-llamacpp-vs-ollama.md`**

```markdown
# ADR 0001 — Runtime d'inférence : llama.cpp plutôt qu'Ollama

- Statut : Accepté
- Date : 2026-05-31

## Contexte
Besoin d'un runtime local servant un modèle GGUF via API OpenAI-compatible, sur
laptop Windows (RTX 2060 6 Go) ET VPS Linux CPU 64 Go. Contraintes de l'utilisateur :
agnosticisme (pas de techno « installe un modèle » type LM Studio), ne pas
réinventer la roue.

## Options
- **llama.cpp (`llama-server`)** : moteur fondamental, GGUF brut, GBNF, contrôle
  total des flags, API OpenAI-compatible native. Setup manuel un peu plus long.
- **Ollama** : installeur simple, mais registre/format propriétaire, wrapper
  par-dessus llama.cpp, sortie contrainte limitée au JSON-schema.

## Décision
llama.cpp. Un GGUF est un fichier nu (agnostique) ; llama.cpp EST la roue de
référence (Ollama en est un wrapper) ; GBNF est nécessaire pour fiabiliser les
petits modèles. L'abstraction API OpenAI-compatible garde Ollama swappable plus tard.

## Conséquences
- + Contrôle total (offload GPU, contexte), GBNF, zéro lock-in.
- − ~15-20 min de setup binaire en plus, à documenter (install-windows/linux.md).
```

- [ ] **Step 2: Écrire `docs/install-windows.md`**

```markdown
# Install llama-server (Windows, laptop GPU)

1. Récupérer la release Windows CUDA de llama.cpp (`...-bin-win-cuda-x64.zip`).
2. Extraire dans `C:\tools\llama\`.
3. Vérifier : `& "C:\tools\llama\llama-server.exe" --version`.
4. Créer `runtime/runtime.config.local.toml` :
   ```toml
   [server]
   bin = "C:/tools/llama/llama-server.exe"
   ```
5. Lancer : `uv run runtime/serve.py` (télécharge le GGUF au 1er run).
```

- [ ] **Step 3: Écrire `docs/install-linux.md`**

```markdown
# Install llama-server (Linux, VPS CPU)

1. Build CPU ou release Linux de llama.cpp ; placer `llama-server` dans le PATH
   ou un dossier connu.
2. Vérifier : `llama-server --version`.
3. (Optionnel) `runtime/runtime.config.local.toml` si le binaire n'est pas dans le PATH :
   ```toml
   [server]
   bin = "/opt/llama/llama-server"
   ```
4. Lancer : `uv run runtime/serve.py`. Pas de GPU -> `serve.py` bascule
   automatiquement en CPU (`n_gpu_layers = 0`) et règle les threads.
```

- [ ] **Step 4: Commit**

```powershell
git add docs/adr/0001-llamacpp-vs-ollama.md docs/install-windows.md docs/install-linux.md
git commit -m "docs: ADR llama.cpp vs Ollama + guides d'install"
```

---

## Task 11: Mise à jour de `ETAT_PROJET.md`

**Files:**
- Modify: `ETAT_PROJET.md`

- [ ] **Step 1: Ajouter une section sur le sous-projet A livré**

Ajouter à la fin de `ETAT_PROJET.md` :
```markdown
## Sous-projet A — Fondation runtime (livré)

- Runtime : llama.cpp (`llama-server`), API OpenAI-compatible sur `127.0.0.1:8080`.
- Lanceur auto-adaptatif : `uv run runtime/serve.py` (GPU détecté sinon CPU).
- Modèle par défaut : Qwen2.5-Coder-7B-Instruct (Q4_K_M), épinglé dans `runtime.config.toml`.
- Benchmark : `uv run runtime/benchmark.py` (débit + validité JSON contraint).
- Reproductibilité : config locale par machine via `runtime.config.local.toml`.
- Décision documentée : `docs/adr/0001-llamacpp-vs-ollama.md`.
- Suite : sous-projet B (boucle agentique).
```

- [ ] **Step 2: Commit**

```powershell
git add ETAT_PROJET.md
git commit -m "docs: etat projet - sous-projet A livre"
```

---

## Definition of Done (sous-projet A)

- [ ] `uv run pytest` : tous les tests verts (hardware, config, models_fetch, server_args, serve).
- [ ] `uv run runtime/serve.py` démarre llama-server et auto-détecte le hardware (log du profil).
- [ ] `curl http://127.0.0.1:8080/v1/models` répond **wifi coupé**.
- [ ] `uv run runtime/benchmark.py` : exit 0, débit affiché, JSON contraint VALIDE.
- [ ] Sur une 2ᵉ machine sans GPU, le même dépôt bascule CPU automatiquement (aucune édition de code).
- [ ] ADR + guides d'install committés.
