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
    """Annote chaque quant : `fits` (tient dans le budget mémoire) et `recommended`.

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
    `supported` (formats du llama.cpp installé), la VRAM passe d'abord (2026-10-17) :
    - jamais un format inconnu du llama.cpp installé (quant d'un fork), ni la classe
      1 bit d'un modèle classique ;
    - le plus gros quant d'au moins 4 bits par poids qui tient ENTIER en VRAM avec son
      contexte, pleine précision comprise ;
    - sinon le plus gros dont la génération estimée atteint TARGET_TPS, à défaut MIN_TPS
      (dense : les couches qui débordent sont lues en RAM ; MoE : experts en RAM) ;
    - sinon rien : le dépôt ne convient pas à cette machine (cf. suggest_alternatives).
    Chaque quant reçoit `ram_share` (part des octets lus par token venant de la RAM) et,
    s'il en lit, `tps` : une ESTIMATION physique (débits de VRAM_GBPS/RAM_GBPS), que le
    bench remplace par une mesure après l'installation."""
    budget = max(0, vram_budget_mb + ram_total_mb - margin_mb)
    out = []
    for f in files:
        label = gguf_remote.quant_label(f["filename"])
        unsupported = (
            supported and label and not gguf_remote.label_supported(label, supported)
        ) or (
            analysis
            and analysis.get("file") == f["filename"]
            and analysis["unknown_types"]
        )
        out.append(
            dict(f, fits=f["size_mb"] <= budget, recommended=False, label=label,
                 status="unsupported" if unsupported else "ok", ram_share=None,
                 bpw=None, tps=None)
        )  # fmt: skip
    eligible = [f for f in out if f["fits"] and f["status"] == "ok"]
    if analysis and vram_budget_mb > 0:
        gpu_mb = max(0, vram_budget_mb - vram_reserve_mb(analysis))
        for f in out:
            ram, vram = _reads_mb(f["size_mb"], analysis, gpu_mb)
            f["ram_share"] = ram / (ram + vram) if ram + vram else 0.0
            if ram:
                f["tps"] = 1 / (ram / 1024 / RAM_GBPS + vram / 1024 / VRAM_GBPS)
            if analysis["n_params"] and not analysis.get("split"):
                f["bpw"] = f["size_mb"] * 1024 * 1024 * 8 / analysis["n_params"]
            # Classe 1 bit sur un modèle classique : trop dégradé, même quand unsloth
            # garde des tenseurs plus précis (UD-IQ1_M ≈ 2,0 bits effectifs). Les formats
            # ternaires (Q2_0, TQ1_0), conçus pour ces débits, restent admis.
            if f["status"] == "ok" and (
                (f["bpw"] is not None and f["bpw"] < 2.0)
                or (f["label"] or "").startswith(("IQ1", "Q1_"))
            ):
                f["status"] = "degraded"
        usable = [f for f in eligible if f["status"] == "ok"]
        in_vram = [
            f for f in usable
            if f["ram_share"] == 0 and (f["bpw"] is None or f["bpw"] >= DECENT_BPW)
        ]  # fmt: skip
        fast = [f for f in usable if f["ram_share"] == 0 or f["tps"] >= TARGET_TPS] or [
            f for f in usable if f["tps"] >= MIN_TPS
        ]  # fmt: skip
        pick = max(in_vram or fast, key=lambda f: f["size_mb"], default=None)
    else:
        # Mémoire unifiée ou en-tête illisible : aucune vitesse estimable, donc pas de pari
        # sur la pleine précision (2 à 4 fois plus d'octets par token) hors de la VRAM.
        in_vram = [
            f for f in eligible
            if f["size_mb"] + VRAM_OVERHEAD_MB + KV_UNKNOWN_MB <= vram_budget_mb
        ]  # fmt: skip
        quant = [f for f in eligible if f["label"] not in gguf_remote.FULL_PRECISION]
        pick = max(
            in_vram or quant or eligible, key=lambda f: f["size_mb"], default=None
        )
    if pick is not None:
        pick["recommended"] = True
    return out


# Hypothèses de placement et de vitesse ; /rebench mesure ensuite le vrai réglage.
VRAM_OVERHEAD_MB = 1024  # contexte CUDA et tampons de calcul
RESERVE_CTX = 16384  # contexte de travail minimal dont le KV doit tenir en VRAM
KV_UNKNOWN_MB = 512  # KV réservé quand l'en-tête ne permet pas de l'estimer
MIN_TPS = 8.0  # vitesse de lecture confortable, même seuil que le verdict de loom-setup
# Visée : au-dessus, plus de bits coûtent de la vitesse pour peu de qualité ; sans palier,
# le plus gros quant tombait toujours à la limite des 8 t/s, dans la marge d'erreur.
TARGET_TPS = 12.0
DECENT_BPW = 4.0  # à partir du Q4, la perte de qualité devient faible
# ponytail: débits fixes, calés sur ornith Q8 (i7-10750H, DDR4-2933, RTX 2060 : 15,1 t/s
# mesurés pour 1,02 Gio lus en RAM et 2,02 Gio en VRAM par token). Une autre machine
# s'en écarte : les dériver d'une génération mesurée par /rebench quand elle sera stockée.
RAM_GBPS = 18.0
VRAM_GBPS = 250.0


def vram_reserve_mb(a: dict) -> float:
    """VRAM à laisser hors des poids : tampons + cache KV (q8_0, 1,0625 octet par
    élément) de RESERVE_CTX tokens, les couches à fenêtre glissante bornées à leur
    fenêtre."""
    full, swa = a.get("kv_per_token") or 0, a.get("kv_swa_per_token") or 0
    if not (full or swa):
        return VRAM_OVERHEAD_MB + KV_UNKNOWN_MB
    elems = full * RESERVE_CTX + swa * min(RESERVE_CTX, a.get("swa_window") or 0)
    return VRAM_OVERHEAD_MB + elems * 1.0625 / 2**20


def _reads_mb(size_mb: int, a: dict, gpu_mb: float) -> tuple[float, float]:
    """(RAM, VRAM) en Mo lus par token pour un quant de `size_mb` aux proportions de
    l'en-tête `a`. Ce qui déborde de la VRAM est lu en RAM ; un MoE qui ne tient pas
    entier garde ses experts en RAM (--cpu-moe) et n'en relit que les actifs.
    L'embedding d'entrée (une ligne par token) est négligé."""
    exp = size_mb * a["f_exp"]
    common = size_mb - exp - size_mb * a["f_emb"]
    active = exp * a["expert_used"] / max(1, a["expert_count"])
    if size_mb <= gpu_mb:
        return 0.0, common + active
    vram = min(common, gpu_mb)
    return common - vram + active, vram


# Taille annoncée dans un nom de repo (« -27B », « -E4B », « -0.6b ») et paramètres actifs
# d'un MoE (« -A3B »).
_SIZE_RE = re.compile(r"[-_.][Ee]?(\d+(?:\.\d+)?)[Bb](?![A-Za-z0-9])")
_ACTIVE_RE = re.compile(r"[-_]A(\d+(?:\.\d+)?)B(?![A-Za-z0-9])", re.IGNORECASE)


def family_stem(repo_id: str | None) -> str | None:
    """Nom de famille d'un modèle : ce qui précède sa taille (« Qwen3.8-27B-GGUF » ->
    « Qwen3.8 »). None si le nom n'annonce pas de taille."""
    name = (repo_id or "").split("/")[-1]
    m = _SIZE_RE.search(name)
    stem = name[: m.start()] if m else ""
    return stem if len(stem) >= 3 else None


def _hit_size(hit: dict) -> tuple[float, float | None]:
    """(paramètres, paramètres actifs si MoE) d'un résultat de search_family. Hugging Face
    lit parfois un GGUF auxiliaire du repo (drafter, projecteur) : le nom fait alors foi."""
    name = hit["repo_id"].split("/")[-1]
    params = hit.get("params") or 0
    if hit.get("arch") in (None, "", "clip", "dflash") or params < 5e8:
        m = _SIZE_RE.search(name)
        params = float(m.group(1)) * 1e9 if m else 0.0
    a = _ACTIVE_RE.search(name)
    return params, float(a.group(1)) * 1e9 if a else None


def _assumed(params: float, active: float | None) -> dict:
    """Proportions supposées d'un modèle dont on n'a pas lu l'en-tête. MoE : un tiers des
    paramètres actifs sont des experts, le reste est relu à chaque token (mesuré sur
    Qwen3.6-35B-A3B : 1,0 B d'experts actifs, 2,2 B communs)."""
    a = {"moe": False, "expert_count": 0, "expert_used": 0, "f_exp": 0.0,
         "f_emb": 0.02, "n_params": params, "kv_per_token": 0, "unknown_types": [],
         "split": False, "file": None}  # fmt: skip
    if active:
        experts = params - 2 * active / 3
        a.update(moe=True, expert_count=experts, expert_used=active / 3,
                 f_exp=experts / params)  # fmt: skip
    return a


def suggest_alternatives(
    repo: str,
    analysis: dict | None,
    hits: list[dict],
    list_files,
    vram_budget_mb: int,
    ram_total_mb: int,
    supported=None,
    limit: int = 3,
    faster_than: float = 0.0,
) -> list[dict]:
    """Repos de `hits` (search_family, par téléchargements décroissants) dont un quant
    passe recommend_quant sur cette machine (tout en VRAM, ou estimé à plus de
    `faster_than` t/s), les plus gros modèles d'abord.

    Même taille que le modèle analysé : même modèle chez un autre publieur, ses
    proportions lues s'appliquent (un format chargeable, un quant plus petit). Sinon
    elles sont supposées d'après le nombre de paramètres et le nom. Au plus 8 repos
    inventoriés (en parallèle), un par taille (deux pour celle du modèle analysé)."""
    from concurrent.futures import ThreadPoolExecutor

    counts: dict = {}
    cands = []
    for h in hits:
        params, active = _hit_size(h)
        if h["repo_id"] == repo or params < 5e8:
            continue
        same = bool(analysis) and (
            abs(params - analysis["n_params"]) <= 0.05 * analysis["n_params"]
            and bool(active) == analysis["moe"]
        )
        key = (round(params / 1e9), active)
        if counts.get(key, 0) >= (2 if same else 1):
            continue
        counts[key] = counts.get(key, 0) + 1
        a = dict(analysis, file=None) if same else _assumed(params, active)
        cands.append((h["repo_id"], params, active, a))
        if len(cands) == 8:
            break

    def evaluate(c):
        repo_id, params, active, a = c
        try:
            files = [f for f in list_files(repo_id) if not f.get("is_aux")]
        except Exception:  # noqa: BLE001 - un repo illisible n'est juste pas proposé
            return None
        out = recommend_quant(
            files, vram_budget_mb, ram_total_mb, analysis=a, supported=supported
        )
        pick = next((f for f in out if f["recommended"]), None)
        if pick and pick["tps"] is not None and pick["tps"] <= faster_than:
            return None
        return pick and {
            "repo_id": repo_id,
            "params": params,
            "active": active,
            "pick": pick,
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        found = [r for r in ex.map(evaluate, cands) if r]
    return sorted(found, key=lambda r: -r["params"])[:limit]


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
