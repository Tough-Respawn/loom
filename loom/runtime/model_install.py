"""Installation d'un modèle LOCAL choisi via /add-model : id dérivé du repo,
recommandation de quant selon le matériel, écriture du model.toml, téléchargement
en arrière-plan et finalisation post-download (métadonnées GGUF -> model.toml)."""

from __future__ import annotations

import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from loom.runtime import gguf_remote
from loom.runtime.gguf_meta import read_gguf_meta
from loom.runtime.models_fetch import ModelUnavailable, ensure_model
from loom.utils import atomic_write_text

_STRIP_SUFFIX = re.compile(r"[-_.](gguf)$", re.IGNORECASE)


def derive_model_id(repo_id: str) -> str:
    """Id proposé (nom du dossier + sélecteur UI) depuis le nom du repo HF."""
    name = repo_id.split("/")[-1].lower()
    name = _STRIP_SUFFIX.sub("", name)
    name = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-.")
    return name or "modele"


def recommend_quant(
    files: list[dict],
    vram_budget_mb: int,
    ram_total_mb: int,
    margin_mb: int = 6144,
    analysis: dict | None = None,
    supported=None,
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
    L'utilisateur reste roi (il peut choisir un quant marqué « ne tiendra pas »).

    Avec `analysis` (en-tête GGUF lu à distance, cf. gguf_remote.analyze_repo) et
    `supported` (formats du llama.cpp installé), la règle devient (2026-10-17 : l'ancienne
    « le plus gros qui tient » recommandait le F16 de 50 Go d'un dense 27B) :
    - jamais un format inconnu du llama.cpp installé (quant d'un fork), ni pleine
      précision (F16/BF16/F32 : 2 à 4 fois plus d'octets par token pour un gain nul) ;
    - DENSE (tous les poids relus à chaque token) : le plus gros quant entièrement en
      VRAM ; sinon le moins lu en RAM parmi ceux d'au moins 2 bits par poids ;
    - MoE (seuls les experts actifs relus) : le plus gros qui tient, experts en RAM.
    Chaque quant reçoit `ram_share` : part des octets lus par token qui viennent de la
    RAM (0 = tout en VRAM), un FAIT de placement et non une promesse de vitesse."""
    budget = max(0, vram_budget_mb + ram_total_mb - margin_mb)
    out = []
    for f in files:
        label = gguf_remote.quant_label(f["filename"])
        if label in gguf_remote.FULL_PRECISION:
            status = "full"
        elif (
            supported and label and not gguf_remote.label_supported(label, supported)
        ) or (
            analysis
            and analysis.get("file") == f["filename"]
            and analysis["unknown_types"]
        ):
            status = "unsupported"
        else:
            status = "ok"
        out.append(
            dict(f, fits=f["size_mb"] <= budget, recommended=False, label=label,
                 status=status, ram_share=None, bpw=None)
        )  # fmt: skip
    eligible = [f for f in out if f["fits"] and f["status"] == "ok"]
    if analysis and vram_budget_mb > 0:
        gpu_mb = max(0, vram_budget_mb - VRAM_RESERVE_MB)
        for f in out:
            f["ram_share"] = _ram_share(f["size_mb"], analysis, gpu_mb)
            if analysis["n_params"] and not analysis.get("split"):
                f["bpw"] = f["size_mb"] * 1024 * 1024 * 8 / analysis["n_params"]
        if analysis["moe"]:
            pick = max(eligible, key=lambda f: f["size_mb"], default=None)
        else:
            in_vram = [f for f in eligible if f["ram_share"] == 0]
            # Classe 1 bit sur un modèle classique : trop dégradé, même quand unsloth
            # garde des tenseurs plus précis (UD-IQ1_M ≈ 2,0 bits effectifs). Les formats
            # ternaires (Q2_0, TQ1_0), conçus pour ces débits, restent admis.
            usable = [
                f for f in eligible
                if (f["bpw"] is None or f["bpw"] >= 2.0)
                and not (f["label"] or "").startswith(("IQ1", "Q1_"))
            ]  # fmt: skip
            pick = (
                max(in_vram, key=lambda f: f["size_mb"])
                if in_vram
                else min(usable, key=lambda f: f["ram_share"], default=None)
            )
    else:  # mémoire unifiée ou en-tête illisible : la taille seule
        pick = max(eligible, key=lambda f: f["size_mb"], default=None)
    if pick is not None:
        pick["recommended"] = True
    return out


# VRAM laissée au contexte (KV) et aux tampons de calcul ; le bench tranche ensuite.
VRAM_RESERVE_MB = 1536


def _ram_share(size_mb: int, a: dict, gpu_mb: int) -> float:
    """Part des octets lus par token qui viennent de la RAM, pour un quant de `size_mb`
    aux proportions de l'en-tête `a`. Dense : tout est relu, ce qui déborde de la VRAM
    vient de la RAM. MoE : experts en RAM (--cpu-moe), seuls les actifs sont relus."""
    if size_mb <= gpu_mb:
        return 0.0
    emb = size_mb * a["f_emb"]  # embedding d'entrée : une ligne par token, négligeable
    if not a["moe"]:
        read = size_mb - emb
        return min(1.0, (size_mb - gpu_mb) / read) if read > 0 else 0.0
    exp = size_mb * a["f_exp"]
    common = size_mb - exp - emb
    active = exp * a["expert_used"] / max(1, a["expert_count"])
    common_ram = max(0.0, common - gpu_mb)
    read = common + active
    return min(1.0, (active + common_ram) / read) if read > 0 else 0.0


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
    atomic_write_text(p, "\n".join(lines) + "\n")
    stub = model_dir / "profile.md"
    if not stub.exists():
        stub.touch()
    return p


# Marqueur -lv 4 émis une fois l'en-tête, l'architecture et les hyperparamètres
# validés, AVANT la lecture des poids : le test ne coûte qu'environ 1 s.
_PROBE_OK = "load_tensors: loading model tensors"
_PROBE_ERR = re.compile(r"^\S+ E |exiting due to model loading error")


def probe_loadable(
    server_bin: str | Path, gguf_path: str | Path, timeout_s: float = 120.0
) -> tuple[bool | None, str]:
    """Le build llama-server configuré sait-il ouvrir ce GGUF ? Lance le serveur,
    s'arrête dès que l'en-tête est validé (ou refusé), sans charger les poids.

    (True, "") = lisible ; (False, cause) = refusé (ex. « invalid ggml type 142 »,
    quant d'un fork) ; (None, raison) = test impossible ou non concluant. Ne détecte
    pas une erreur survenant pendant la lecture des tenseurs eux-mêmes."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    try:
        proc = subprocess.Popen(
            [str(server_bin), "-m", str(gguf_path), "--port", str(port), "-c", "512"]
            + ["-ngl", "0", "-lv", "4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return None, f"llama-server non lancé ({exc})"
    errors: list[str] = []
    deadline = time.monotonic() + timeout_s
    try:
        for line in proc.stdout:
            line = line.strip()
            if _PROBE_OK in line:
                return True, ""
            if _PROBE_ERR.search(line):
                # Garder le message sans l'horodatage llama.cpp (« 0.00.41 E »).
                errors.append(re.sub(r"^\S+ E\s+", "", line))
                if "exiting" in line or len(errors) >= 3:
                    break
            if time.monotonic() > deadline:
                return None, "test de chargement trop long, non concluant"
    finally:
        proc.kill()
        proc.wait(timeout=10)
    if errors:
        return False, errors[0]
    return None, "llama-server s'est arrêté sans diagnostic"


def finalize_model_toml(model_dir: str | Path, gguf_path: str | Path) -> dict:
    """Complète model.toml depuis le header GGUF téléchargé : n_layers,
    cpu_moe = true si le modèle est un MoE (experts en RAM = notre défaut, cf.
    règle du parc), cache_isolation = true si sa mémoire est récurrente.
    Best-effort : un GGUF illisible laisse le toml tel quel."""
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
        # Défaut SÛR (tient toujours si les denses tiennent), mais NON mesuré : la
        # sonde de placement (loom-setup, /rebench) compare experts-CPU et tout-GPU
        # et peut l'inverser (Ornith 2026-10-09 : tout-GPU +19 % de génération).
        doc.add(
            tomlkit.comment("placement NON mesuré : experts en RAM par défaut (sûr)")
        )
        doc.add(tomlkit.comment("— /rebench compare avec tout-GPU et écrit l'élu."))
        doc["cpu_moe"] = True
    if meta.get("recurrent") and "cache_isolation" not in doc:
        # Deux slots d'emblée : sans ça reflect/titre écrasent la conversation, et
        # un modèle ajouté par /add-model n'a jamais eu de sonde d'isolation.
        doc.add(tomlkit.comment("Mémoire récurrente (en-tête GGUF) : appels annexes"))
        doc.add(tomlkit.comment("sur le slot 1, la conversation garde le slot 0."))
        doc["cache_isolation"] = True
    atomic_write_text(p, tomlkit.dumps(doc))
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

    def start(self, on_done=None) -> DownloadJob:
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
