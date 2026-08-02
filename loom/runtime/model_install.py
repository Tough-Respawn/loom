"""Installation d'un modèle LOCAL choisi via /add-model : id dérivé du repo,
recommandation de quant selon le matériel, écriture du model.toml, téléchargement
en arrière-plan et finalisation post-download (métadonnées GGUF -> model.toml)."""

from __future__ import annotations

import re
import threading
from pathlib import Path

from loom.runtime.gguf_meta import read_gguf_meta
from loom.runtime.models_fetch import ModelUnavailable, ensure_model

_STRIP_SUFFIX = re.compile(r"[-_.](gguf)$", re.IGNORECASE)


def derive_model_id(repo_id: str) -> str:
    """Id proposé (nom du dossier + sélecteur UI) depuis le nom du repo HF."""
    name = repo_id.split("/")[-1].lower()
    name = _STRIP_SUFFIX.sub("", name)
    name = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-.")
    return name or "modele"


def recommend_quant(
    files: list[dict], vram_budget_mb: int, ram_total_mb: int, margin_mb: int = 6144
) -> list[dict]:
    """Annote chaque quant : `fits` (tient dans le budget mémoire) et `recommended`
    (le plus gros qui tient). Heuristique SIMPLE : nos modèles tournent experts en
    RAM (--cpu-moe).

    Budget = `vram_budget_mb` + `ram_total_mb` − marge. DEUX invariants, appris à
    la dure (2026-07-23) :
    - RAM TOTALE, jamais la dispo du moment : un modèle déjà chargé (déchargé par
      llama-swap avant le nouveau) ou Loom/navigateur ne doit pas rétrécir le
      budget, sinon on masque des quants qui tiennent (ornith Q8 34 Go tourne,
      mais un Q4 19 Go était marqué « ne tiendra pas ») ;
    - `vram_budget_mb` = 0 sur mémoire UNIFIÉE (iGPU) : la VRAM y EST la RAM
      (même LPDDR5) — l'additionner double-compterait. Seul un GPU DISCRET ajoute
      sa VRAM propre (cf. HardwareProfile.vram_is_discrete côté appelant).
    L'utilisateur reste roi (il peut choisir un quant marqué « ne tiendra pas »)."""
    budget = max(0, vram_budget_mb + ram_total_mb - margin_mb)
    out = [dict(f, fits=f["size_mb"] <= budget, recommended=False) for f in files]
    fitting = [f for f in out if f["fits"]]
    if fitting:
        max(fitting, key=lambda f: f["size_mb"])["recommended"] = True
    return out


def write_model_toml(
    model_dir: str | Path,
    repo: str,
    filename: str,
    size_mb: int,
    mmproj_filename: str | None = None,
) -> Path:
    """Crée <model_dir>/model.toml (+ profile.md stub). Écrit AVANT le download :
    si celui-ci coupe, ensure_model() reprend au premier serve (filet existant)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Généré par /add-model — ajuste librement (cf. loom/models/_TEMPLATE/model.toml).",
        f'repo = "{repo}"',
        f'filename = "{filename}"',
        f"size_mb = {int(size_mb)}",
    ]
    if mmproj_filename:
        lines.append(f'mmproj_filename = "{mmproj_filename}"')
    p = model_dir / "model.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stub = model_dir / "profile.md"
    if not stub.exists():
        stub.touch()
    return p


def finalize_model_toml(model_dir: str | Path, gguf_path: str | Path) -> dict:
    """Complète model.toml depuis le header GGUF téléchargé : n_layers, et
    cpu_moe = true si le modèle est un MoE (experts en RAM = notre défaut, cf.
    règle du parc). Best-effort : un GGUF illisible laisse le toml tel quel."""
    try:
        meta = read_gguf_meta(gguf_path)
    except (OSError, ValueError):
        return {}
    import tomlkit

    p = Path(model_dir) / "model.toml"
    doc = tomlkit.parse(p.read_text(encoding="utf-8"))
    if meta.get("n_layers"):
        doc["n_layers"] = meta["n_layers"]
    if meta.get("expert_count"):
        doc["cpu_moe"] = True
    p.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return meta


class DownloadJob:
    """Téléchargement en ARRIÈRE-PLAN (thread daemon) des fichiers d'un modèle.

    La progression se lit sur le DISQUE (fichiers finis + *.incomplete du cache
    HF sous dest_dir) : pas de callback dans hf_hub_download, et ça survit à un
    onglet fermé. `on_done(job)` est appelé à la fin, succès OU échec — c'est lui
    qui finalise (toml, montage, message de fin dans la conversation)."""

    def __init__(self, repo: str, filenames: list[str], dest_dir, total_mb: int):
        self.repo = repo
        self.filenames = list(filenames)
        self.dest_dir = Path(dest_dir)
        self.total_mb = int(total_mb)
        self.done = False
        self.error: str | None = None
        self.final_message = ""
        self._on_done = None
        self._thread: threading.Thread | None = None

    def start(self, on_done=None) -> "DownloadJob":
        self._on_done = on_done
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="loom-model-download"
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for fn in self.filenames:
                ensure_model(self.repo, fn, self.dest_dir)
        except ModelUnavailable as exc:
            self.error = str(exc)
        finally:
            # Publier `done` seulement après le montage évite un sélecteur UI incomplet.
            if self._on_done is not None:
                try:
                    self._on_done(self)
                except Exception as exc:  # noqa: BLE001 - jamais planter le thread
                    print(f"[loom] add-model finalisation : {exc}", flush=True)
            self.done = True

    def progress_mb(self) -> int:
        total = 0
        for p in self.dest_dir.rglob("*"):
            if p.is_file() and (p.suffix == ".gguf" or p.name.endswith(".incomplete")):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total // (1024 * 1024)


def start_download(
    repo: str, filenames: list[str], dest_dir, total_mb: int, on_done=None
) -> DownloadJob:
    return DownloadJob(repo, filenames, dest_dir, total_mb).start(on_done)
