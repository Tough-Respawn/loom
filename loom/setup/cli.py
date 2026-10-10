"""Installeur interactif console : `uv run loom-setup`.

Quatre étapes, chacune sur le même contrat HITL : état constaté → proposition
EXPLIQUÉE (quoi, où, quelle taille) → confirmation → action → résultat.
1. Détection (OS/GPU/RAM) · 2. Binaire llama.cpp + routeur llama-swap +
   outillage agent (Playwright/rg/npx/docker — constaté, installé si possible) ·
3. Modèle qui fit · 4. Bench du matériel → réglages écrits dans config/local.toml.
PRINCIPE (vécu 2026-07-22, llama-swap jamais provisionné -> crash au 2e modèle) :
tout ce dont Loom a besoin pour fonctionner est installé ou constaté ICI —
jamais découvert par une panne.
Tout ce qui s'affiche part aussi dans var/logs/setup.log ; le bilan final
récapitule. Relançable : ne refait que ce qui manque."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loom.runtime.hardware import detect_hardware, ram_available_mb, top_ram_processes
from loom.runtime.model_install import (
    derive_model_id,
    finalize_model_toml,
    recommend_quant,
    start_download,
    write_model_toml,
)
from loom.runtime.gguf_meta import read_gguf_meta
from loom.runtime.hf_catalog import HfCatalogError
from loom.runtime.platform_info import detect as detect_platform
from loom.runtime.term import colorize, supports_color
from loom.setup import bench as bench_mod
from loom.setup import llama_release
from loom.setup import tooling as tooling_mod
from loom.setup import topology as topo_mod
from loom.setup.catalog import (
    budget_mb,
    filter_by_budget,
    fitting_entries,
    parse_hf_repo,
    pick_mmproj,
    probe_repo,
    resolve_entry,
)
from loom.setup.llama_release import (
    find_llama_swap,
    local_arch,
    select_assets,
    select_swap_asset,
)
from loom.setup.report import SetupReport
from loom.setup.steps import (
    first_model_file,
    incomplete_models,
    installed_model_ids,
    models_roots,
    read_raw_config,
    resolve_bin,
    server_bin_status,
    set_default_model,
    set_local_values,
    set_server_bin,
    set_swap_bin,
    swap_bin_status,
)
from loom.utils import atomic_write_text

LOOM_DIR = Path(__file__).resolve().parent.parent  # = loom/ (le package)
REPO_ROOT = LOOM_DIR.parent
CONFIG_PATH = REPO_ROOT / "config" / "defaults.toml"
PERSONAL_CONFIG_PATH = REPO_ROOT / "config" / "local.toml"
PACKAGE_MODELS = LOOM_DIR / "models"
RUNTIME_DIR = REPO_ROOT / "var" / "runtime" / "llama"
SETUP_LOG = REPO_ROOT / "var" / "logs" / "setup.log"

_YES = {"o", "oui", "y", "yes"}


class Console:
    """I/O console injectable (fake dans les tests). Chaque `say` part aussi
    dans le log (en TEXTE BRUT — jamais de codes ANSI dans un fichier) ;
    `progress` réécrit la même ligne (\\r) et n'est PAS loggé. Couleur auto :
    seulement sur un vrai terminal (NO_COLOR respecté)."""

    def __init__(
        self,
        log_path: Path | None = None,
        assume_yes: bool = False,
        input_fn=input,
        print_fn=print,
        color: bool | None = None,
    ):
        self.log_path = log_path
        self.assume_yes = assume_yes
        self._input = input_fn
        self._print = print_fn
        self.color = supports_color(sys.stdout) if color is None else color
        # Afficher un chrono sur TTY rend les longues sondes visibles sans perturber les tests.
        self._prog_lock = threading.Lock()
        self._prog_msg: str | None = None
        self._prog_t0 = 0.0
        self._prog_len = 0
        self._ticker: threading.Thread | None = None
        self._live = print_fn is print and sys.stdout.isatty()

    def _log(self, msg: str) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass

    def say(self, msg: str = "") -> None:
        with self._prog_lock:
            interrupted = self._live and self._prog_msg is not None
        if interrupted:
            self._print()  # clôt la ligne de progression, le chrono repart dessous
        # Colorer ligne par ligne car les règles sont ancrées en début de ligne.
        if self.color:
            self._print("\n".join(colorize(line) for line in msg.split("\n")))
        else:
            self._print(msg)
        self._log(msg)

    def progress(self, msg: str) -> None:
        with self._prog_lock:
            self._prog_msg = msg
            self._prog_t0 = time.monotonic()
        self._draw_progress()
        if self._live:
            self._start_ticker()

    def _draw_progress(self) -> None:
        with self._prog_lock:
            msg = self._prog_msg
            if msg is None:
                return
            secs = int(time.monotonic() - self._prog_t0)
            if secs >= 60:
                stamp = f" — {secs // 60} min {secs % 60:02d} s"
            elif secs >= 5:
                stamp = f" — {secs} s"
            else:
                stamp = ""
            line = f"  {msg}{stamp}"
            # Effacer le reliquat d'une ligne précédente plus longue.
            pad = " " * max(0, self._prog_len - len(line))
            self._prog_len = len(line)
            self._print(f"\r{line}{pad}", end="", flush=True)

    def _start_ticker(self) -> None:
        if self._ticker is not None and self._ticker.is_alive():
            return

        def _tick() -> None:
            while True:
                time.sleep(1.0)
                with self._prog_lock:
                    if self._prog_msg is None:
                        return
                self._draw_progress()

        self._ticker = threading.Thread(
            target=_tick, daemon=True, name="setup-progress-chrono"
        )
        self._ticker.start()

    def progress_end(self) -> None:
        with self._prog_lock:
            self._prog_msg = None
            self._prog_len = 0
        self._print()

    def _prompt(self, text: str) -> str:
        """Question en GRAS (point d'interaction) — seulement sur un vrai terminal."""
        from loom.runtime.term import BOLD, paint

        return paint(text, BOLD) if self.color else text

    def ask(self, prompt: str, default: str = "") -> str:
        if self.assume_yes:
            return default
        # Retirer le BOM ajouté au premier stdin pipé par PowerShell.
        raw = self._input(self._prompt(f"  {prompt} ")).strip().strip("﻿").strip()
        return raw or default

    def confirm(self, question: str, default: bool = True) -> bool:
        if self.assume_yes:
            return True
        suffix = "[O/n]" if default else "[o/N]"
        raw = (
            self._input(self._prompt(f"  {question} {suffix} "))
            .strip()
            .strip("﻿")
            .strip()
        )
        if not raw:
            return default
        return raw.lower() in _YES


@dataclass
class Deps:
    """Effets de bord injectables — les défauts sont les implémentations réelles.
    Les tests substituent des fakes (release JSON figée, download no-op…)."""

    detect_hardware: object = detect_hardware
    ram_available_mb: object = ram_available_mb
    detect_platform: object = detect_platform
    fetch_release: object = None  # () -> dict (client httpx construit ici)
    fetch_swap_release: object = None  # () -> dict (release llama-swap)
    tool_checks: object = tooling_mod.tool_checks
    install_playwright: object = tooling_mod.install_playwright_browser
    download_and_extract: object = None  # (plan, dest_root, progress_cb) -> Path
    find_llama_server: object = llama_release.find_llama_server
    verify_binary: object = llama_release.verify_binary
    probe_repo: object = probe_repo
    search_models: object = None  # (query) -> [{repo_id,…}]
    start_download: object = start_download
    top_ram_processes: object = top_ram_processes
    run_bench: object = bench_mod.run_llama_bench
    find_llama_bench: object = bench_mod.find_llama_bench
    has_gpu_backend: object = bench_mod.has_gpu_backend
    cpu_physical: object = None  # () -> int|None (cœurs physiques)
    sleep: object = time.sleep
    gpu_vram_total_mb: object = topo_mod.gpu_vram_total_mb
    make_probe: object = topo_mod.ServerProbe  # (**kw) -> objet avec .run(ctx, depth)
    ram_total_mb: object = None  # () -> int (RAM totale, Mo) ; défaut psutil

    def __post_init__(self):
        if self.ram_total_mb is None:
            self.ram_total_mb = _real_ram_total_mb
        if self.fetch_release is None:
            self.fetch_release = _real_fetch_release
        if self.fetch_swap_release is None:
            self.fetch_swap_release = _real_fetch_swap_release
        if self.download_and_extract is None:
            self.download_and_extract = _real_download_and_extract
        if self.search_models is None:
            from loom.runtime.hf_catalog import search_models

            self.search_models = search_models
        if self.cpu_physical is None:
            self.cpu_physical = _real_cpu_physical


def _real_ram_total_mb() -> int:
    import psutil

    return int(psutil.virtual_memory().total // (1024 * 1024))


def _real_fetch_release() -> dict:
    import httpx

    with httpx.Client(timeout=30) as client:
        return llama_release.fetch_latest_release(client)


def _real_fetch_swap_release() -> dict:
    import httpx

    with httpx.Client(timeout=30) as client:
        return llama_release.fetch_latest_release(
            client, url=llama_release.SWAP_RELEASES_URL
        )


def _real_download_and_extract(plan, dest_root, progress_cb):
    import httpx

    with httpx.Client(timeout=None) as client:
        return llama_release.download_and_extract(plan, dest_root, client, progress_cb)


def _real_cpu_physical() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


def step_detection(con: Console, report: SetupReport, deps: Deps):
    con.say("[1/4] Détection du système")
    plat = deps.detect_platform()
    hw = deps.detect_hardware()
    ram = deps.ram_available_mb()
    # Avant le binaire, seule la détection NVIDIA peut choisir le build CUDA.
    gpu = (
        f"{hw.gpu_name} ({hw.vram_free_mb} Mo VRAM libre)"
        if hw.has_gpu
        else "pas de NVIDIA — détection complète après l'étape binaire"
    )
    con.say(f"  OS  : {plat.label}")
    con.say(f"  GPU : {gpu}")
    con.say(f"  RAM : {ram} Mo disponibles")
    detail = f"{plat.label} · GPU {gpu} · {ram} Mo RAM"
    report.add("detection", "ok", detail)
    return plat, hw, ram


def step_binary(con: Console, report: SetupReport, deps: Deps, plat, hw, raw_cfg):
    con.say("")
    con.say("[2/4] Binaire llama-server")
    present, bin_name = server_bin_status(raw_cfg)
    if present:
        con.say(f"  Trouvé : {bin_name} → rien à faire.")
        report.add("binaire", "ok", f"déjà en place ({bin_name})")
        return

    con.say(
        f'  Config [server] bin = "{bin_name}" → introuvable (ni fichier, ni PATH).'
    )

    existing = deps.find_llama_server(RUNTIME_DIR)
    if existing is not None:
        version = deps.verify_binary(existing)
        if version and con.confirm(
            f"Binaire déjà présent ({existing.parent.name}, {version}) — le réutiliser ?"
        ):
            set_server_bin(PERSONAL_CONFIG_PATH, existing)
            con.say(f"  [ok] config/local.toml : [server] bin = {existing}")
            report.add("binaire", "fait", f"réutilisé ({version}) → config/local.toml")
            return

    try:
        release = deps.fetch_release()
    except RuntimeError as exc:
        con.say(f"  [échec] {exc}")
        report.add("binaire", "echec", str(exc))
        return

    plan = select_assets(release, plat.key, local_arch(), hw.has_gpu)
    if plan is None:
        con.say("  Aucun asset précompilé ne convient à cette machine.")
        con.say(f"  Release : {release.get('html_url', '?')}")
        for a in release.get("assets", []):
            con.say(f"    - {a['name']}")
        con.say(
            "  → installe llama.cpp à la main (télécharge ou compile), puis mets le "
            'chemin dans config/local.toml : [server] bin = "…/llama-server".'
        )
        report.add("binaire", "manuel", "aucun asset compatible — guidage donné")
        return

    con.say(
        f"  Proposition : release {plan.tag} de llama.cpp (ggml-org), {plan.reason} :"
    )
    for a in plan.assets:
        con.say(f"    - {a['name']} ({a['size_mb']} Mo)")
    dest_dir = RUNTIME_DIR / plan.tag
    con.say(f"  Installation dans {dest_dir} puis écriture dans config/local.toml.")
    if not con.confirm(f"Télécharger et installer ({plan.total_mb} Mo) ?"):
        con.say("  [passé] Ignoré — tu peux relancer loom-setup plus tard.")
        report.add("binaire", "ignore", "téléchargement refusé")
        return

    def _cb(name, done_mb, total_mb):
        con.progress(f"{name} : {done_mb}/{total_mb or '?'} Mo")

    try:
        extracted = deps.download_and_extract(plan, RUNTIME_DIR, _cb)
    except (RuntimeError, OSError) as exc:
        con.progress_end()
        con.say(f"  [échec] Téléchargement/extraction : {exc}")
        report.add("binaire", "echec", f"téléchargement : {exc}")
        return
    con.progress_end()

    binary = deps.find_llama_server(extracted)
    version = deps.verify_binary(binary) if binary else None
    if binary is None or version is None:
        con.say(
            "  [échec] Binaire extrait mais inutilisable (--version muet) — config/local.toml "
            "laissé intact. Vérifie l'archive ou installe à la main."
        )
        report.add("binaire", "echec", "binaire extrait mais --version muet")
        return
    set_server_bin(PERSONAL_CONFIG_PATH, binary)
    con.say(f"  Vérification --version : OK ({version})")
    con.say(f"  [ok] config/local.toml : [server] bin = {binary}")
    report.add(
        "binaire", "fait", f"installé ({plan.tag}, {plan.backend}) → config/local.toml"
    )


def _refresh_gpu(con: Console, deps: Deps, hw, raw_cfg):
    """Re-détection AGNOSTIQUE une fois le binaire en place : `--list-devices` du
    binaire installé est la source de vérité (Vulkan AMD/Intel/NVIDIA, CUDA…) —
    s'il liste un device, on le prend ; s'il n'en liste aucun, ce build ne sait
    pas offloader et le profil devient CPU. Sans binaire : profil étape 1 gardé."""
    _, bin_name = server_bin_status(raw_cfg)
    server_bin = resolve_bin(bin_name)
    if server_bin is None:
        return hw
    fresh = deps.detect_hardware(server_bin)
    if fresh.has_gpu and (not hw.has_gpu or fresh.gpu_name != hw.gpu_name):
        con.say(
            f"  → GPU confirmé par le binaire : {fresh.gpu_name} "
            f"({fresh.vram_free_mb} Mo libres, backend {fresh.backend or '?'})"
        )
    return fresh


def step_swap(con: Console, report: SetupReport, deps: Deps, plat, raw_cfg):
    """Routeur multi-modèles (llama-swap), provisionné D'OFFICE avec le binaire.
    Vécu 2026-07-22 : jamais installé par le setup, la bascule mono->multi au
    2e /add-model plantait le serve au démarrage suivant (« binaire llama-swap
    introuvable »). Best-effort : un échec n'empêche pas le mono-modèle."""
    # Sans llama-server, ne pas installer un routeur inutilisable.
    server_present, _ = server_bin_status(raw_cfg)
    if not server_present:
        report.add("swap", "ignore", "reporté (llama-server absent)")
        return
    present, bin_name = swap_bin_status(raw_cfg)
    if present:
        con.say(f"  Routeur multi-modèles : {bin_name} → rien à faire.")
        report.add("swap", "ok", f"déjà en place ({bin_name})")
        return
    try:
        release = deps.fetch_swap_release()
    except RuntimeError as exc:
        con.say(
            f"  [attention] llama-swap non installé ({exc}) — le multi-modèles "
            "sera indisponible (un seul modèle local à la fois)."
        )
        report.add("swap", "manuel", "téléchargement impossible")
        return
    plan = select_swap_asset(release, plat.key, local_arch())
    if plan is None:
        con.say("  [attention] aucun asset llama-swap pour cette plateforme.")
        report.add("swap", "manuel", "aucun asset compatible")
        return
    con.say(
        f"  Routeur multi-modèles : installation de llama-swap ({plan.total_mb} Mo)…"
    )
    try:
        extracted = deps.download_and_extract(
            plan, RUNTIME_DIR.parent / "llama-swap", lambda *a: None
        )
    except (RuntimeError, OSError) as exc:
        con.say(f"  [échec] téléchargement llama-swap : {exc}")
        report.add("swap", "echec", f"téléchargement : {exc}")
        return
    binary = find_llama_swap(extracted)
    if binary is None:
        con.say("  [échec] archive llama-swap extraite mais binaire introuvable.")
        report.add("swap", "echec", "binaire absent de l'archive")
        return
    set_swap_bin(PERSONAL_CONFIG_PATH, binary)
    con.say(f"  [ok] config/local.toml : [server] swap_bin = {binary}")
    report.add("swap", "fait", f"installé ({plan.tag}) → config/local.toml")


def step_tooling(con: Console, report: SetupReport, deps: Deps):
    """Outillage des outils de l'agent (dégradable, jamais bloquant) : constat
    de chaque dépendance externe + installation de ce qui l'est (navigateur
    Playwright). Même philosophie que llama-swap : tout ce dont Loom a besoin
    est provisionné/constaté par le setup, pas découvert par une panne."""
    checks = deps.tool_checks()
    missing = [c for c in checks if not c["present"]]
    if not missing:
        con.say("  Outillage agent : complet → rien à faire.")
        report.add("outillage", "ok", "complet")
        return
    for c in checks:
        if c["present"]:
            continue
        if c.get("autofix") == "playwright":
            if con.confirm(
                "Navigateur Playwright absent (check_page/check_interactive). "
                "L'installer (~130 Mo) ?"
            ):
                con.progress("playwright install chromium…")
                ok, detail = deps.install_playwright()
                con.progress_end()
                if ok:
                    con.say("  [ok] navigateur Playwright installé.")
                    continue
                con.say(f"  [échec] installation Playwright : {detail}")
            else:
                con.say("  [passé] navigateur Playwright — check_page dégradé.")
            continue
        con.say(f"  [attention] {c['name']} absent — {c['role']}.\n    -> {c['hint']}")
    still = [c["name"] for c in deps.tool_checks() if not c["present"]]
    if still:
        report.add("outillage", "manuel", f"manquant : {', '.join(still)}")
    else:
        report.add("outillage", "fait", "complété")


def _offer_free_ram(con: Console, deps: Deps, hw, ram: int) -> int:
    """Avant de choisir un modèle : montre les gros consommateurs de RAM et
    laisse l'utilisateur en FERMER lui-même (on ne tue jamais rien nous-mêmes),
    puis re-mesure. Renvoie la RAM disponible finale. Le budget se recalcule à
    chaque tour : plus de RAM = meilleur modèle proposé."""
    if con.assume_yes:  # non-interactif : personne pour fermer quoi que ce soit
        return ram
    budget = budget_mb(hw.budget_vram_mb, ram)
    tight = budget < 6_000  # en dessous, on rate les ~4B/8B confortables
    hint = (
        "Ta RAM est serrée : en libérer débloquerait un meilleur modèle."
        if tight
        else "Plus de RAM libre = un modèle plus costaud proposé."
    )
    if not con.confirm(
        f"{hint} Voir ce qui consomme (tu fermes toi-même, rien n'est tué) ?",
        default=tight,
    ):
        return ram
    for _ in range(5):  # plafond : jamais d'attente infinie sur un stdin épuisé
        rows = deps.top_ram_processes()
        if not rows:
            con.say("  (liste des processus indisponible)")
            return ram
        for r in rows:
            proc = f"({r['count']} processus)" if r["count"] > 1 else ""
            con.say(f"    {r['name']:<28} {r['mb']:>7} Mo {proc}")
        ans = con.ask(
            "Ferme ce que tu n'utilises pas (gestionnaire de tâches), puis Entrée "
            "pour re-mesurer — ou « c » pour continuer :"
        )
        ram = deps.ram_available_mb()
        budget = budget_mb(hw.budget_vram_mb, ram)
        con.say(f"  → RAM disponible : {ram} Mo · budget : {budget} Mo")
        if ans.lower().startswith("c"):
            break
    return ram


def _download_model(
    con: Console,
    report: SetupReport,
    deps: Deps,
    *,
    repo: str,
    dest: Path,
    filename: str,
    size_mb: int,
    model_id: str,
    mmproj: str | None = None,
    part_files: list[str] | None = None,
) -> None:
    """Télécharge (ou REPREND) les fichiers d'un modèle, puis finalise : model.toml
    complété depuis le header GGUF + défaut de la machine. Partagé entre
    l'installation fraîche et la reprise d'un téléchargement interrompu."""
    filenames = list(part_files or [filename])
    if mmproj and not (Path(dest) / mmproj).is_file():
        filenames.append(mmproj)

    # Empêcher la veille pendant les gros téléchargements sans activité utilisateur.
    from loom.runtime.stay_awake import StayAwake

    awake = StayAwake()
    awake.acquire()
    try:
        job = deps.start_download(repo, filenames, dest, size_mb)
        while not job.done:
            con.progress(f"téléchargement… {job.progress_mb()}/{size_mb} Mo")
            deps.sleep(2)
    finally:
        awake.release()
    con.progress_end()

    if job.error:
        con.say(f"  [échec] {job.error}")
        con.say(
            "  (model.toml déjà écrit : relance loom-setup pour REPRENDRE le "
            "téléchargement — il reprend aussi au premier serve.)"
        )
        report.add("modele", "echec", f"téléchargement : {job.error}")
        return
    meta = finalize_model_toml(dest, Path(dest) / filename)
    # Le premier modèle présent sur cette machine devient son défaut local.
    set_default_model(PERSONAL_CONFIG_PATH, model_id)
    extra = " (MoE → cpu_moe = true)" if meta.get("expert_count") else ""
    con.say(f"  [ok] Modèle « {model_id} » installé{extra} — défaut de cette machine.")
    report.add("modele", "fait", f"{model_id} ({filename}, {size_mb} Mo)")


def step_model(con: Console, report: SetupReport, deps: Deps, hw, ram, raw_cfg):
    con.say("")
    con.say("[3/4] Modèle")
    # Un profil sans GGUF reste une installation incomplète à reprendre.
    missing = incomplete_models(raw_cfg, PACKAGE_MODELS)
    if missing:
        mid, folder, data = missing[0]
        con.say(
            f"  [attention] « {mid} » : model.toml présent mais GGUF absent — "
            "téléchargement interrompu."
        )
        if con.confirm(
            f"Reprendre le téléchargement ({data.get('size_mb', '?')} Mo) ?"
        ):
            _download_model(
                con,
                report,
                deps,
                repo=data.get("repo", ""),
                dest=folder,
                filename=data["filename"],
                size_mb=int(data.get("size_mb", 0)),
                model_id=mid,
                mmproj=data.get("mmproj_filename"),
            )
        else:
            con.say(
                "  [passé] Reprise refusée — relance loom-setup quand tu veux "
                "(le download reprend là où il s'était arrêté)."
            )
            report.add("modele", "ignore", f"reprise refusée ({mid})")
        return
    ids = installed_model_ids(raw_cfg, PACKAGE_MODELS)
    if ids:
        con.say(f"  {len(ids)} modèle(s) branché(s) ({', '.join(ids)}) → rien à faire.")
        report.add("modele", "ok", f"{len(ids)} branché(s) : {', '.join(ids)}")
        return

    budget = budget_mb(hw.budget_vram_mb, ram)
    con.say("  Aucun modèle branché (<racine>/local/text/ vide).")
    con.say(
        f"  Budget estimé : {budget} Mo (VRAM discrète libre + RAM − 4 Go de marge ; "
        "la mémoire d'un iGPU EST la RAM, on ne la compte pas deux fois)."
    )
    ram = _offer_free_ram(con, deps, hw, ram)
    budget = budget_mb(hw.budget_vram_mb, ram)
    entries = fitting_entries(budget)

    con.say("  Recommandé pour ta machine :")
    for i, e in enumerate(entries, start=1):
        con.say(f"    {i}. {e['label']}")
    con.say(
        f"    {len(entries) + 1}. Recherche libre Hugging Face (nom, URL ou id de repo)"
    )
    con.say("    0. Passer (tu pourras taper /add-model dans le chat)")
    default = "1" if entries else "0"
    choice = con.ask(f"Ton choix [{default}] :", default=default)

    if choice == "0":
        con.say("  [passé] Passé — /add-model dans le chat quand tu veux.")
        report.add("modele", "ignore", "reporté (/add-model dans le chat)")
        return

    repo = None
    if choice == str(len(entries) + 1):
        query = con.ask("Recherche Hugging Face (nom du modèle, ou URL/id du repo) :")
        if not query:
            report.add("modele", "ignore", "recherche vide")
            return
        # Une URL ou un id explicite contourne la recherche, pas la validation du repo.
        repo = parse_hf_repo(query)
        if repo is not None:
            con.say(f"  → repo repéré : {repo}")
        else:
            try:
                hits = deps.search_models(query)
            except Exception as exc:  # noqa: BLE001 - HfCatalogError montrable
                con.say(f"  [échec] {exc}")
                report.add("modele", "ignore", "recherche impossible (hors-ligne ?)")
                return
            # Filtrer grossièrement les familles impossibles avant de charger leurs quants.
            hits, hidden = filter_by_budget(hits, budget)
            if hidden:
                con.say(
                    f"  ({hidden} résultat(s) masqué(s) : trop gros pour ton "
                    f"budget de {budget} Mo)"
                )
            if not hits:
                con.say(
                    "  Aucun repo jouable sur cette machine pour cette recherche — "
                    "libère de la RAM (ferme des applis) ou vise plus petit (3-4B)."
                )
                report.add("modele", "ignore", "recherche sans résultat jouable")
                return
            for i, h in enumerate(hits, start=1):
                est = f", ~{h['est_mb']} Mo mini" if h.get("est_mb") else ""
                con.say(
                    f"    {i}. {h['repo_id']} ({h['downloads']} téléchargements{est})"
                )
            pick = con.ask("Quel repo [1] :", default="1")
            try:
                repo = hits[int(pick) - 1]["repo_id"]
            except (ValueError, IndexError):
                report.add("modele", "ignore", "choix de repo invalide")
                return
    else:
        try:
            entry = entries[int(choice) - 1]
        except (ValueError, IndexError):
            report.add("modele", "ignore", "choix invalide")
            return
        # Distinguer une panne réseau d'une famille réellement sans modèle jouable.
        try:
            repo = resolve_entry(entry, deps.search_models, budget)
        except HfCatalogError as exc:
            con.say(f"  [échec] {exc}")
            report.add("modele", "ignore", f"entrée non résolue ({entry['label']})")
            return
        if repo is None:
            con.say(
                f"  [échec] « {entry['label']} » introuvable sur Hugging Face "
                "(famille disparue ?) — réessaie, ou recherche libre."
            )
            report.add("modele", "ignore", f"entrée non résolue ({entry['label']})")
            return
        con.say(f"  → repo retenu : {repo}")

    try:
        files = deps.probe_repo(repo)
    except HfCatalogError as exc:
        con.say(f"  [échec] {exc}")
        report.add("modele", "ignore", f"repo injoignable ({repo})")
        return
    if files is None:
        con.say(
            f"  [échec] Repo « {repo} » injoignable (hors-ligne, renommé ?) — réessaie plus "
            "tard ou passe par /add-model dans le chat."
        )
        report.add("modele", "ignore", f"repo injoignable ({repo})")
        return

    quants = [f for f in files if not f.get("is_aux", f["is_mmproj"])]
    if not quants:
        con.say(f"  [échec] Aucun GGUF exploitable dans {repo}.")
        report.add("modele", "ignore", f"aucun GGUF dans {repo}")
        return
    annotated = recommend_quant(quants, hw.budget_vram_mb, ram)
    rec = next((f for f in annotated if f["recommended"]), annotated[0])
    fit_txt = "tient dans le budget" if rec["fits"] else "NE TIENDRA PAS (trop gros)"
    if not con.confirm(
        f"Quant recommandé : {rec['filename']} ({rec['size_mb']} Mo) — {fit_txt}. "
        "Télécharger ?"
    ):
        con.say(
            "  [passé] Passé — /add-model dans le chat pour choisir un autre quant."
        )
        report.add("modele", "ignore", "quant refusé")
        return

    mmproj = pick_mmproj(files)
    model_id = derive_model_id(repo)
    dest = models_roots(raw_cfg, PACKAGE_MODELS)[0] / "local" / "text" / model_id
    write_model_toml(
        dest, repo, rec["filename"], rec["size_mb"], mmproj_filename=mmproj
    )
    _download_model(
        con,
        report,
        deps,
        repo=repo,
        dest=dest,
        filename=rec["filename"],
        size_mb=rec["size_mb"],
        model_id=model_id,
        mmproj=mmproj,
        part_files=rec.get("part_files"),
    )


def _read_model_toml(gguf_path: Path) -> dict:
    """model.toml voisin du GGUF (cpu_moe, n_cpu_moe, mmproj…) — {} s'il manque.
    C'est LUI qui porte les flags que l'exécutant utilisera : la sonde doit les lire."""
    p = Path(gguf_path).parent / "model.toml"
    if not p.is_file():
        return {}
    import tomllib

    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _set_model_context(gguf_path: Path, context: int, mecanisme: str) -> None:
    """Écrit le contexte CALIBRÉ dans le model.toml du modèle benché (la vérité est
    par modèle : la pente KV dépend de l'architecture). Remplace la ligne `context =`
    existante ou l'ajoute, sans toucher au reste du fichier (commentaires compris)."""
    p = Path(gguf_path).parent / "model.toml"
    if not p.is_file():
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    stamp = f"# context calibré par loom-setup (pente mesurée) — {mecanisme}"
    new_line = f"context = {context}"
    for i, line in enumerate(lines):
        code = line.split("#")[0].strip()
        if code.startswith("context") and code.replace(" ", "").startswith("context="):
            lines[i] = new_line
            if i == 0 or not lines[i - 1].strip().startswith("# context calibré"):
                lines.insert(i, stamp)
            else:
                lines[i - 1] = stamp
            break
    else:
        lines += ["", stamp, new_line]
    atomic_write_text(p, "\n".join(lines) + "\n")


def _set_model_threads(gguf_path: Path, threads: int, detail: str) -> None:
    """Écrit les threads MESURÉS sur le placement élu dans le model.toml (option « par
    modèle » : prioritaires sur [override] threads de la machine). Remplace la ligne
    `threads =` existante ou l'ajoute, un seul tampon, le reste du fichier intact."""
    p = Path(gguf_path).parent / "model.toml"
    if not p.is_file():
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    stamp = f"# threads élus par la sonde (placement élu) — {detail}"
    new_line = f"threads = {int(threads)}"
    for i, line in enumerate(lines):
        code = line.split("#")[0].strip().replace(" ", "")
        if code.startswith("threads="):
            lines[i] = new_line
            if i == 0 or not lines[i - 1].strip().startswith("# threads élus"):
                lines.insert(i, stamp)
            else:
                lines[i - 1] = stamp
            break
    else:
        lines += ["", stamp, new_line]
    atomic_write_text(p, "\n".join(lines) + "\n")


def _set_model_cache_isolation(gguf_path: Path, needed: bool, detail: str) -> None:
    """Écrit le verdict MESURÉ de la sonde d'isolation dans le model.toml (vérité
    par modèle) : true = le cache ne survit pas à la pollution du slot -> serve
    monte --parallel à 2 pour ce modèle. false documente « sondé, pas nécessaire »
    (différent de « jamais sondé » = ligne absente)."""
    p = Path(gguf_path).parent / "model.toml"
    if not p.is_file():
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    stamp = f"# cache_isolation sondé par le bench (A -> pollution -> A) — {detail}"
    new_line = f"cache_isolation = {'true' if needed else 'false'}"
    for i, line in enumerate(lines):
        code = line.split("#")[0].strip()
        if code.replace(" ", "").startswith("cache_isolation="):
            lines[i] = new_line
            if i == 0 or not lines[i - 1].strip().startswith("# cache_isolation sondé"):
                lines.insert(i, stamp)
            else:
                lines[i - 1] = stamp
            break
    else:
        lines += ["", stamp, new_line]
    atomic_write_text(p, "\n".join(lines) + "\n")


def _archive_setup(
    con, trace: dict, *, echec: dict | None = None, applied: dict | None = None
):
    """Archive le compte rendu progressif d'un bench loom-setup (archive.bench_payload),
    en échec (étape + erreur) comme en succès (+ application notée). Un échec d'écriture
    est DIT, jamais silencieux, et n'empêche rien."""
    from loom.setup import archive as archive_mod

    model_id = Path(str(trace.get("gguf") or "modele")).parent.name or "modele"
    try:
        path = archive_mod.archive_bench(
            model_id, archive_mod.bench_payload(**trace, echec=echec)
        )
        con.say(f"  archive : {path}")
        if applied is not None and not archive_mod.note_application(path, applied):
            con.say(
                "  [attention] configuration appliquée, annotation de l'archive échouée."
            )
    except Exception as exc:  # noqa: BLE001 - l'archive n'empêche jamais le bench
        con.say(f"  [attention] archive du bench non écrite ({exc}).")


def _sans_none(obj):
    """Copie récursive sans valeurs None : TOML n'a pas de null, tomlkit refuse."""
    if isinstance(obj, dict):
        return {k: _sans_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_sans_none(v) for v in obj if v is not None]
    return obj


def _set_model_placement(gguf_path: Path, placement, detail: str) -> None:
    """Écrit le placement MESURÉ des poids (loom.setup.placement) dans le model.toml :
    `cpu_moe`, `n_cpu_moe` et `n_gpu_layers` posés ou RETIRÉS selon le candidat élu
    (tout GPU -> n_gpu_layers = 999 explicite, pour ne pas dépendre de la VRAM libre
    au lancement ; CPU seul -> 0 ; experts partiels -> n_cpu_moe). Un seul tampon,
    remplacé à chaque mesure ; le reste du fichier est intact."""
    p = Path(gguf_path).parent / "model.toml"
    if not p.is_file():
        return
    wanted: dict[str, str | None] = {
        "cpu_moe": "true" if placement.cpu_moe else "false",
        "n_cpu_moe": (
            str(placement.n_cpu_moe) if placement.n_cpu_moe is not None else None
        ),
        # Partiel dense : le -ngl EXACT validé, pas une recommandation recalculée au
        # lancement sur la VRAM libre du moment.
        "n_gpu_layers": {
            "gpu_total": "999",
            "cpu": "0",
            "gpu_partiel": str(placement.ngl),
        }.get(placement.label),
    }
    # Identité du placement seul : les batchs ont leur propre tampon (ubatch/batch).
    key = str(getattr(placement, "key", placement.label)).split("@")[0]
    stamp = f"# placement élu par la sonde — {key} : {detail}"
    out: list[str] = []
    done: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("# placement élu par la sonde"):
            continue  # ancien tampon : réécrit ci-dessous
        code = line.split("#")[0].strip().replace(" ", "")
        key = next((k for k in wanted if code.startswith(f"{k}=")), None)
        if key is None:
            out.append(line)
            continue
        val = wanted[key]
        if val is None or key in done:
            continue  # clé retirée pour ce placement (ou doublon)
        out.append(f"{key} = {val}")
        done.add(key)
    missing = [
        f"{k} = {v}" for k, v in wanted.items() if v is not None and k not in done
    ]
    first = next(
        (
            i
            for i, line in enumerate(out)
            if any(
                line.split("#")[0].strip().replace(" ", "").startswith(f"{k}=")
                for k in done
            )
        ),
        None,
    )
    if first is None:
        out += ["", stamp, *missing]
    else:
        out[first + 1 : first + 1] = missing
        out.insert(first, stamp)
    atomic_write_text(p, "\n".join(out) + "\n")


def _set_model_ubatch(gguf_path: Path, ubatch: int, batch: int, detail: str) -> None:
    """Écrit les batchs de prefill MESURÉS par la sonde d'ubatch dans le model.toml
    (vérité par modèle : l'optimum dépend de l'architecture et du quant). Remplace
    les lignes existantes ou les ajoute, sans toucher au reste du fichier."""
    p = Path(gguf_path).parent / "model.toml"
    if not p.is_file():
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    stamp = f"# ubatch/batch élus par la sonde de prefill — {detail}"
    wanted = {"ubatch": f"ubatch = {ubatch}", "batch": f"batch = {batch}"}
    done: set[str] = set()
    for i, line in enumerate(lines):
        code = line.split("#")[0].strip().replace(" ", "")
        for key, new_line in wanted.items():
            if code.startswith(f"{key}="):
                lines[i] = new_line
                done.add(key)
    missing = [wanted[k] for k in ("ubatch", "batch") if k not in done]
    if missing:
        lines += ["", stamp] + missing
    else:
        # Les deux lignes existaient : poser (ou rafraîchir) le tampon au-dessus
        # de la première d'entre elles.
        for i, line in enumerate(lines):
            if line.split("#")[0].strip().replace(" ", "").startswith("ubatch="):
                if i > 0 and lines[i - 1].strip().startswith("# ubatch/batch élus"):
                    lines[i - 1] = stamp
                else:
                    lines.insert(i, stamp)
                break
    atomic_write_text(p, "\n".join(lines) + "\n")


def step_bench(con: Console, report: SetupReport, deps: Deps, raw_cfg):
    """[4/4] Bench du matériel avec le VRAI modèle : mesure -t (et -ngl si backend
    GPU), calcule le contexte qui tient en RAM, écrit le tout dans local.toml."""
    con.say("")
    con.say("[4/4] Réglages machine (bench)")
    if raw_cfg.get("bench"):
        con.say(
            "  Déjà calibré (table [bench] dans config/local.toml — supprime-la "
            "pour re-mesurer)."
        )
        report.add("bench", "ok", "déjà calibré")
        return

    _, bin_name = server_bin_status(raw_cfg)
    server_bin = resolve_bin(bin_name)
    model = first_model_file(raw_cfg, PACKAGE_MODELS)
    if server_bin is None or model is None:
        # Nommer précisément l'élément manquant dans le diagnostic.
        manque = []
        if server_bin is None:
            manque.append("le binaire llama-server")
        if model is None:
            manque.append("le GGUF du modèle (téléchargement incomplet ?)")
        quoi = " et ".join(manque)
        con.say(f"  [passé] Il manque {quoi} — bench sauté.")
        report.add("bench", "ignore", f"manque {quoi}")
        return
    bench_bin = deps.find_llama_bench(server_bin)
    if bench_bin is None:
        con.say(
            "  [manuel] llama-bench introuvable à côté du binaire — réglages par défaut "
            "conservés (réinstalle via loom-setup pour l'avoir)."
        )
        report.add("bench", "manuel", "llama-bench absent de la release")
        return
    gguf_path, model_size_mb = model

    import os

    # Le binaire et les métadonnées GGUF dimensionnent les candidats d'offload.
    # Le binaire que l'EXÉCUTANT lancera pour CE modèle (model.toml server_bin — un build
    # qui porte une PR —, sinon le global) porte la détection matérielle, la sonde et le
    # build tracé. llama-bench, lui, reste celui livré à côté du binaire global.
    model_toml = _read_model_toml(gguf_path)
    probe_bin = topo_mod.model_server_bin(model_toml, str(server_bin))
    hw = deps.detect_hardware(probe_bin)
    try:
        meta = read_gguf_meta(gguf_path)
    except ValueError:
        meta = {}

    threads = bench_mod.thread_candidates(os.cpu_count() or 4, deps.cpu_physical())
    # GPU exploitable : le profil `--list-devices` du binaire fait foi (un build statique
    # n'a aucune DLL à côté de l'exe), cf. bench.gpu_backend_available.
    gpu_ok = bench_mod.gpu_backend_available(
        hw, probe_bin, has_dll=deps.has_gpu_backend
    )
    # Pour un MoE, mesurer l'offload réel avec experts en RAM plutôt qu'un impossible tout-GPU.
    moe = bool(meta.get("expert_count"))
    ngl, ncmoe = bench_mod.ngl_candidates(
        gpu_ok,
        hw.vram_free_mb,
        model_size_mb,
        meta.get("n_layers"),
        moe=moe,
    )
    combos = len(threads) * len(ngl)
    con.say(
        f"  On mesure la vitesse réelle sur TON modèle ({gguf_path.name}) : "
        f"{combos} combinaisons de threads{' et offload GPU' if len(ngl) > 1 else ''}."
    )
    con.say("  Durée : ~2-10 min selon la machine (CPU à fond, c'est normal).")
    if not con.confirm("Lancer le bench maintenant ?"):
        con.say("  [passé] Sauté — relançable à tout moment : uv run loom-setup.")
        report.add("bench", "ignore", "refusé (relançable)")
        return

    con.progress("bench en cours… (llama-bench, plusieurs minutes)")
    try:
        # progress -> chaque mesure terminée s'affiche (ligne \r + chrono) au lieu
        # d'un silence de plusieurs minutes.
        rows = deps.run_bench(
            bench_bin, gguf_path, threads, ngl, n_cpu_moe=ncmoe, progress=con.progress
        )
    except RuntimeError as exc:
        con.progress_end()
        con.say(f"  [échec] {exc}")
        report.add("bench", "echec", str(exc))
        return
    con.progress_end()
    best = bench_mod.pick_best(rows)
    if best is None:
        con.say("  [échec] Aucune mesure de génération exploitable.")
        report.add("bench", "echec", "sortie llama-bench vide")
        return

    # Mesurer pente et débit avec les vrais flags évite les erreurs d'une formule KV théorique.
    # La VRAM vient d'abord du profil de l'exécutant (`--list-devices`, Vulkan compris) ;
    # nvidia-smi n'est qu'un repli — sinon une AMD passe en topologie « ram » et la
    # sonde mesure sans profil GPU (vécu 2026-10-09).
    vram_total = int(hw.vram_total_mb or deps.gpu_vram_total_mb() or 0)
    topo = topo_mod.discover_topology(meta, gpu_ok, vram_total)
    server_cfg = raw_cfg.get("server") or {}
    headroom = int(server_cfg.get("gpu_kv_headroom_mb", 640) or 640)
    # Utiliser la RAM totale rend la recommandation reproductible. Mémoire unifiée :
    # le device est la RAM, comptée une fois (même quantité que la sonde mesure).
    ram_total_mb = int(deps.ram_total_mb())
    uma = bool(hw.has_gpu and not hw.vram_is_discrete)
    budget = topo_mod.memory_budget_mb(
        topo, vram_total, ram_total_mb, headroom, uma=uma
    )
    is_moe = bool(meta.get("expert_count"))
    mmproj_name = model_toml.get("mmproj_filename")
    from loom.setup import placement as place_mod

    # Configuration ACTUELLE résolue comme l'exécutant (resolve_ngl) : réglages du
    # model.toml, sinon l'override machine — celui que ce bench va écrire pour un
    # dense —, sinon la VRAM libre. C'est la référence (ligne de base) du placement
    # ET le point de départ de la sonde.
    override_ngl = (
        best["ngl"]
        if (len(ngl) > 1 and not moe)
        else (raw_cfg.get("override") or {}).get("n_gpu_layers")
    )
    cur_pl = place_mod.current_placement(
        model_toml,
        n_layers=meta.get("n_layers"),
        size_mb=model_size_mb,
        profile=hw,
        override_ngl=override_ngl,
        headroom=headroom,
    )
    probe = deps.make_probe(
        server_bin=probe_bin,
        model_path=str(gguf_path),
        threads=best["threads"],
        # Slots de l'exécutant : [server] n_parallel global monté par l'isolation
        # ACTUELLE (model.toml) — un nouveau verdict seul la remplacera. Sans ça, une
        # sonde d'isolation en échec faisait mesurer à 1 slot un modèle qui tourne à 2.
        n_parallel=topo_mod.probe_slots(
            server_cfg, bool(model_toml.get("cache_isolation"))
        ),
        ngl=(
            cur_pl.ngl
            if (cur_pl is not None and topo != topo_mod.TOPO_RAM)
            else (0 if topo == topo_mod.TOPO_RAM else best["ngl"])
        ),
        topology=topo,
        mmproj_path=str(gguf_path.parent / mmproj_name) if mmproj_name else None,
        cpu_moe=bool(model_toml.get("cpu_moe", is_moe)),
        n_cpu_moe=model_toml.get("n_cpu_moe"),
        # Batchs de l'exécutant : modèle, sinon repli machine [server] ubatch/batch.
        ubatch=topo_mod.probe_batches(model_toml, server_cfg)[0],
        batch=topo_mod.probe_batches(model_toml, server_cfg)[1],
        # Checkpoints des hybrides : mesurer la mémoire que l'exécutant prendra.
        checkpoint_min_step=(
            model_toml.get("checkpoint_min_step")
            or server_cfg.get("checkpoint_min_step")
        ),
        ctx_checkpoints=model_toml.get("ctx_checkpoints"),
        # Flags machine et mode de mesure mémoire dérivés du profil de l'exécutant.
        profile=hw,
    )
    from dataclasses import replace as _dc_replace

    from loom.runtime.model_profile import ModelProfile
    from loom.setup import placement as place_mod

    # Isolation D'ABORD (sur la configuration actuelle) : le placement se compare
    # ensuite avec les slots FINAUX — le KV du second slot compte dans la faisabilité
    # et dans la mesure (revue du 2026-10-10 : « mêmes slots »).
    # Compte rendu PROGRESSIF (archive.BENCH_SCHEMA) : archivé même si une étape échoue.
    trace: dict = {
        "source": "loom-setup",
        "etape": "isolation",
        "gguf": str(gguf_path),
        "server_bin": probe_bin,
        "materiel": hw,
        "llama_bench": best,
    }
    con.progress("sonde d'isolation du cache (A -> pollution -> A)…")
    isolation: bool | None = None
    iso_detail = ""
    first = back = 0
    try:
        first, back = probe.probe_isolation()
        isolation = topo_mod.isolation_needed(first, back, meta.get("recurrent"))
        iso_detail = f"retour {back}/{first} tokens retraités"
        if meta.get("recurrent"):
            iso_detail += ", mémoire récurrente"
    except Exception as exc:  # noqa: BLE001 - sonde best-effort : sans verdict, rien d'écrit
        con.progress_end()
        iso_detail = f"sonde illisible ({exc}) — isolation actuelle conservée"
        con.say(
            f"  [attention] sonde d'isolation illisible ({exc}) — verdict non écrit, "
            f"isolation actuelle conservée ({probe.n_parallel} slot(s))."
        )
    else:
        con.progress_end()
        marque = "[attention]" if isolation else "[ok]"
        # Nouveau verdict : il remplace l'isolation actuelle (dans les deux sens).
        probe.n_parallel = topo_mod.probe_slots(server_cfg, isolation)
        # Libellé honnête : ce que la mesure a montré, et pourquoi on isole quand même.
        con.say(f"  {marque} {topo_mod.isolation_text(isolation, first, back)}")
    trace["isolation"] = {
        "necessaire": isolation,
        "first": first,
        "back": back,
        "detail": iso_detail,
        "avant": bool(model_toml.get("cache_isolation", False)),
        "slots_mesure": probe.n_parallel,
    }
    # Placement MESURÉ des poids (où vivent denses et experts) x couples de batchs,
    # AVANT la calibration : elle mesure ainsi la configuration qui servira vraiment.
    # Cf. loom/setup/placement.py (Ornith, 2026-10-09).
    # Profil GGUF : couches à cache KV, poids par famille — chaque donnée avec sa
    # provenance. Le KV est estimé au contexte UTILE avec le type de cache de
    # l'exécutant (q8_0 sous profil GPU) et les slots FINAUX.
    profile = ModelProfile.from_meta(meta, model_size_mb=model_size_mb)
    for ligne in profile.describe():
        con.say(f"  profil : {ligne}")
    ctx_utile = place_mod.useful_context(
        model_toml.get("context"), server_cfg.get("context"), meta.get("context_length")
    )
    pl_slots = int(getattr(probe, "n_parallel", 1) or 1)
    # Mémoire par contexte au-delà des poids : KV au contexte utile + état récurrent
    # (état vivant + checkpoints par slot) — la faisabilité des hybrides ne dépend plus
    # de la seule pente mesurée (Bonsai 2 : 32 x 149 Mio par slot).
    estimation = place_mod.memory_estimate_mb(
        profile,
        ctx_utile,
        gpu_tuning=hw.has_gpu,
        slots=pl_slots,
        checkpoints=model_toml.get("ctx_checkpoints"),
    )
    # Ventilée : KV + état vivant côté DEVICE, checkpoints côté HÔTE (tableaux RAM du
    # serveur) — les imputer à la VRAM faisait perdre des candidats tout-GPU.
    kv_mb = estimation["device_mb"]
    host_mb = estimation["host_mb"]
    trace["memoire_estimee"] = estimation
    if estimation["recurrent_mb"]:
        con.say(
            f"  mémoire estimée au contexte {ctx_utile} ({pl_slots} slot(s)) : device "
            f"{kv_mb} Mio (KV {estimation['kv_mb']} + état vivant "
            f"{estimation['recurrent_live_mb']}), hôte {host_mb} Mio "
            f"({estimation['checkpoints']} checkpoints par slot, en RAM)"
        )
    # Candidats par faisabilité (profil GGUF), la configuration ACTUELLE en base ; ce
    # qu'on ne mesure pas est tracé « non exploré ». Contraintes de prefill optionnelles
    # ([placement] dans local.toml) : explicite (N tokens en T s) ou plancher de confort.
    plan = place_mod.plan_placements(
        moe=is_moe,
        n_layers=meta.get("n_layers"),
        model_size_mb=model_size_mb,
        kv_mb=kv_mb,
        gpu_backend=gpu_ok,
        vram_total_mb=vram_total,
        ram_total_mb=ram_total_mb,
        uma=not hw.vram_is_discrete,
        headroom_mb=headroom,
        current=cur_pl,
        profile=profile,
        host_extra_mb=host_mb,
    )
    prefill_c, pp_floor = place_mod.constraints_from_config(raw_cfg)
    # Les finalistes sont comparés x deux couples (ubatch, batch) : celui de
    # l'exécutant (la base) et l'alternative du parc — la sonde ubatch séparée disparaît.
    couples = place_mod.batch_couples(
        (getattr(probe, "ubatch", None), getattr(probe, "batch", None))
    )
    trace.update(
        etape="placement",
        profil=profile.describe(),
        contexte_utile=ctx_utile,
        kv_estime_mb=kv_mb,
        hote_estime_mb=host_mb,
        plan=plan,
        couples=couples,
        contrainte_prefill=prefill_c,
        prefill_floor_ratio=pp_floor,
        flags={
            "threads": int(best["threads"]),
            "gpu_tuning": bool(hw.has_gpu),
            "unified_memory": bool(not hw.vram_is_discrete),
            "ubatch": getattr(probe, "ubatch", None),
            "batch": getattr(probe, "batch", None),
            "slots": int(getattr(probe, "n_parallel", 1) or 1),
            "checkpoint_min_step": getattr(probe, "checkpoint_min_step", None),
        },
    )
    con.progress("sonde de placement (où vivent les poids, x batchs)…")
    try:
        pl_res = place_mod.probe_placement(
            lambda pl: _dc_replace(
                probe, ngl=pl.ngl, cpu_moe=pl.cpu_moe, n_cpu_moe=pl.n_cpu_moe
            ),
            plan.candidates,
            progress=lambda m: con.progress(f"placement : {m}"),
            useful_ctx=ctx_utile,
            non_explores=plan.non_explores,
            prefill=prefill_c,
            pp_floor_ratio=pp_floor,
            batch_couples=couples,
        )
    except Exception:  # noqa: BLE001 - sonde best-effort : sans verdict, rien d'écrit
        pl_res = None
    con.progress_end()
    trace["placement"] = pl_res
    trace["etape"] = "calibration"
    if pl_res and pl_res.get("placement") is None:
        con.say(
            f"  [attention] placement : {pl_res['mecanisme']} — flags actuels conservés."
        )
    if pl_res and pl_res.get("placement") is not None and pl_res["mesures"]:
        # Élu (comparé, ou seul candidat validé) avec ses batchs : la suite
        # (calibration, validation finale) mesure cette configuration-là.
        pl = pl_res["placement"]
        probe = _dc_replace(
            probe,
            ngl=pl.ngl,
            cpu_moe=pl.cpu_moe,
            n_cpu_moe=pl.n_cpu_moe,
            ubatch=pl.ubatch or getattr(probe, "ubatch", None),
            batch=pl.batch or getattr(probe, "batch", None),
        )
        con.say(f"  [ok] placement : {pl.describe()} — {pl_res['mecanisme']}")
    # Threads sur le placement ÉLU (option par modèle, 2026-10-10) : un placement avec
    # du calcul CPU se mesure avec les candidats du parc, au contexte et à la
    # profondeur de la finale ; tout GPU : non exploré, et la trace le dit.
    th_res = None
    pl_elu_th = (pl_res or {}).get("placement")
    if pl_elu_th is not None and not place_mod.needs_cpu_compute(pl_elu_th):
        th_res = {
            "non_explore": (
                f"non exploré : {pl_elu_th.key.split('@')[0]} sans calcul CPU attendu "
                f"(threads machine {best['threads']} conservés)"
            )
        }
        con.say(f"  threads : {th_res['non_explore']}")
    elif pl_elu_th is not None and pl_res.get("mesures"):
        th_options = place_mod.thread_options(
            best["threads"], os.cpu_count() or 4, deps.cpu_physical()
        )
        con.progress("sonde de threads sur le placement élu…")
        try:
            th_res = place_mod.probe_threads(
                lambda o: _dc_replace(probe, threads=o.threads),
                th_options,
                ctx=pl_res["ctx_final"],
                depth=pl_res["depth_final"],
                progress=lambda m: con.progress(f"threads : {m}"),
                prefill=prefill_c,
                pp_floor_ratio=pp_floor,
            )
        except Exception:  # noqa: BLE001 - sonde best-effort : sans verdict, rien d'écrit
            th_res = None
        con.progress_end()
        if th_res:
            th_res["placement"] = pl_elu_th.key.split("@")[0]
            if th_res["threads"] != int(getattr(probe, "threads", best["threads"])):
                probe = _dc_replace(probe, threads=th_res["threads"])
            con.say(
                f"  [ok] threads sur {th_res['placement']} : {th_res['threads']} — "
                f"{th_res['mecanisme']}"
            )
        else:
            con.say(
                "  [attention] sonde de threads illisible — threads machine conservés."
            )
    trace["threads"] = th_res
    con.say(
        f"  Topologie découverte : {topo} (budget {budget} Mo). Calibration du "
        "contexte par PENTE MESURÉE + vitesse en profondeur (~5-15 min)…"
    )
    con.progress("calibration du contexte…")
    try:
        calib = topo_mod.calibrate(
            probe,
            meta,
            topology=topo,
            budget_mb=budget,
            progress=lambda m: con.progress(f"calibration : {m}"),
        )
    except Exception as exc:  # noqa: BLE001 - erreurs opérationnelles comprises (OSError…)
        con.progress_end()
        con.say(
            f"  [échec] calibration échouée ({type(exc).__name__}: {exc}) — context "
            "inchangé, relance loom-setup."
        )
        report.add("bench", "echec", f"calibration contexte : {exc}")
        _archive_setup(con, trace, echec={"etape": "calibration", "erreur": str(exc)})
        return
    con.progress_end()
    trace["calibration"] = calib
    trace["etape"] = "batchs"
    context = calib["context"]

    # Sonde d'ubatch sur la MÊME sonde serveur que la calibration (flags exacts,
    # aucune dépendance à llama-bench) : un seul levier à la fois, ajouté au couple
    # threads/ngl déjà gagnant, sur un prompt assez long pour que le levier existe
    # (à 128 tokens tout tient dans un micro-batch : aucun effet mesurable).
    pl_elu = (pl_res or {}).get("placement")
    if pl_elu is not None and pl_elu.ubatch:
        # Les batchs viennent du 2x2 des finalistes (même contexte, même profondeur,
        # mêmes slots que le placement) : la sonde ubatch séparée est obsolète.
        ub_res = {
            "ubatch": int(pl_elu.ubatch),
            "batch": int(pl_elu.batch or pl_elu.ubatch),
            "pp_ts": float(pl_res.get("pp_ts") or 0.0),
            "gain_pct": None,
            "mesures": {
                k: v.get("pp_ts")
                for k, v in (pl_res.get("mesures") or {}).items()
                if "pp_ts" in v
            },
            "detail": (
                f"{pl_res.get('pp_ts')} t/s à profondeur {pl_res.get('depth_final')} "
                f"(finalistes x batchs, ctx {pl_res.get('ctx_final')})"
            ),
        }
    else:
        con.progress("sonde ubatch (prefill sur prompt long)…")
        try:
            ub_res = bench_mod.probe_ubatch(
                lambda ub, b: _dc_replace(probe, ubatch=ub, batch=b),
                progress=lambda m: con.progress(f"sonde ubatch : {m}"),
            )
        except Exception:  # noqa: BLE001 - sonde best-effort, jamais fatale
            ub_res = None
        con.progress_end()

    # Vérifier le CACHE avec la configuration FINALE (placement élu, slots décidés,
    # batchs mesurés) : conversation sur le slot 0, appel annexe routé comme Loom le
    # fera, retour — le cache doit être réutilisé. C'est la preuve qui justifie de
    # traiter le gros prefill comme un coût amorti.
    trace["ubatch"] = ub_res
    trace["etape"] = "réglage final"
    # Sonde FINALE = placement élu + slots décidés + batchs mesurés. D'abord valider ce
    # réglage complet au contexte CALIBRÉ, à la profondeur de la comparaison : le
    # verdict n'assemble plus des mesures prises avec des paramètres différents.
    probe_final = (
        _dc_replace(probe, ubatch=ub_res["ubatch"], batch=ub_res["batch"])
        if ub_res
        else probe
    )
    con.progress("validation du réglage final au contexte calibré…")
    try:
        final = place_mod.validate_final(
            probe_final,
            ctx=context,
            depth=place_mod.final_depth(ctx_utile),
            n_layers=meta.get("n_layers"),
            reference_tg=(pl_res or {}).get("tg_ts"),
            prefill=prefill_c,
            progress=lambda m: con.progress(f"réglage final : {m}"),
        )
    except Exception as exc:  # noqa: BLE001 - validation best-effort, nommée
        final = {"echec": f"{type(exc).__name__}: {exc}", "ctx": context}
    con.progress_end()
    if "echec" in final:
        # Une baisse de vitesse avertit ; un échec de FONCTIONNEMENT empêche : rien
        # n'est écrit, la configuration actuelle reste en place.
        con.say(
            f"  [échec] réglage final non validé ({final['echec']}) — réglages NON "
            "écrits, configuration actuelle conservée. Relance loom-setup une fois la "
            "cause levée."
        )
        report.add("bench", "echec", f"réglage final non validé : {final['echec']}")
        trace["final"] = final
        _archive_setup(
            con, trace, echec={"etape": "réglage final", "erreur": final["echec"]}
        )
        return
    else:
        if final.get("coherent") is None:
            coh = ""
        elif final["coherent"]:
            coh = (
                f" — cohérent avec la mesure de placement ({final['ecart_pct']:+.1f} %)"
            )
        else:
            coh = (
                f" — ne reproduit pas la mesure de placement ({final['ecart_pct']:+.1f} %),"
                " à appliquer avec prudence"
            )
        marque = "attention" if final.get("coherent") is False else "ok"
        con.say(
            f"  [{marque}] réglage final {final['placement']} (ctx {final['ctx']}, "
            f"{final['slots']} slot(s), ub {final['ubatch']}/b {final['batch']}) : "
            f"génération {final['tg_ts']} t/s, prefill {final['pp_ts']} t/s à profondeur "
            f"{final['depth']}{coh}"
        )
    cache_v = None
    con.progress("vérification du cache avec la configuration finale…")
    try:
        # Avec le contexte réellement alloué, pas un 4 096 de confort.
        cache_v = probe_final.verify_cache(ctx=context)
    except Exception as exc:  # noqa: BLE001 - vérification best-effort, jamais fatale
        con.progress_end()
        con.say(f"  [attention] vérification du cache impossible ({exc}).")
    else:
        con.progress_end()
        if cache_v.get("reused") is True:
            con.say(
                "  [ok] cache réutilisé après routage des appels annexes "
                f"({topo_mod.cache_check_text(cache_v)})."
            )
        elif cache_v.get("reused") is False:
            con.say(
                "  [attention] cache NON réutilisé avec la configuration finale "
                f"({topo_mod.cache_check_text(cache_v)})."
            )
        else:
            con.say("  [attention] vérification du cache illisible.")
    build = deps.verify_binary(probe_bin) or "build ?"
    trace.update(final=final, cache=cache_v, build=build, etape="écriture")

    values = {
        "server": {"context": context},
        "override": {"threads": best["threads"]},
        "bench": {
            "threads": best["threads"],
            "ngl": best["ngl"],
            "tg_ts": round(best["tg_ts"], 2),
            "pp_ts": round(best["pp_ts"], 2),
            "context": context,
            # Conserver le mécanisme rend la recommandation explicable.
            "context_mode": calib["mode"],
            "context_mecanisme": calib["mecanisme"],
            "context_pente_kb_tok": calib["slope_kb_tok"],
            "context_valide_jusqua": calib["valide_jusqua"],
            # False = plancher de repli, aucun barreau de vitesse validé.
            "context_valide": bool(calib.get("valide", True)),
            "context_utile_estime": ctx_utile,
            "kv_estime_mb": estimation["kv_mb"],
            "recurrent_estime_mb": estimation["recurrent_mb"],
            "memoire_estimee_mb": estimation["total_mb"],
            "checkpoints_estimes": estimation["checkpoints"],
        },
    }
    if cache_v and cache_v.get("reused") is not None:
        values["bench"]["cache_verifie"] = bool(cache_v["reused"])
        values["bench"]["cache_verifie_detail"] = topo_mod.cache_check_text(cache_v)
    if th_res and "non_explore" in th_res:
        values["bench"]["threads_non_explore"] = th_res["non_explore"]
    elif th_res:
        values["bench"]["threads_modele"] = th_res["threads"]
        values["bench"]["threads_placement"] = th_res.get("placement")
        values["bench"]["threads_mecanisme"] = th_res["mecanisme"]
        values["bench"]["threads_mesures"] = _sans_none(th_res["mesures"])
    if "echec" in final:
        values["bench"]["final_echec"] = final["echec"]
    else:
        values["bench"]["final_placement"] = final["placement"]
        values["bench"]["final_ctx"] = final["ctx"]
        values["bench"]["final_depth"] = final["depth"]
        values["bench"]["final_slots"] = final["slots"]
        values["bench"]["final_tg_ts"] = final["tg_ts"]
        values["bench"]["final_pp_ts"] = final["pp_ts"]
        values["bench"]["final_tg_disp_pct"] = final["tg_disp_pct"]
        values["bench"]["final_echantillons"] = _sans_none(final["echantillons"])
        for k in ("ubatch", "batch", "ecart_pct", "coherent"):
            if final.get(k) is not None:
                values["bench"][f"final_{k}"] = final[k]
    if pl_res:
        pl_elu = pl_res.get("placement")
        # `placement` = l'identité du placement ; le couple de batchs est tracé à part
        # (`placement_config`, ubatch/batch).
        values["bench"]["placement"] = (
            pl_elu.key.split("@")[0] if pl_elu else "aucun (échec)"
        )
        values["bench"]["placement_config"] = pl_elu.key if pl_elu else "aucun"
        values["bench"]["placement_couples"] = [
            f"ub {ub}/b {b}" for ub, b in (pl_res.get("couples") or [])
        ]
        values["bench"]["placement_mecanisme"] = pl_res["mecanisme"]
        values["bench"]["placement_compare"] = bool(pl_res.get("compare"))
        # Avec quoi le placement a été mesuré : moteur, slots, flags machine.
        values["bench"]["placement_build"] = build
        values["bench"]["placement_slots"] = pl_slots
        values["bench"]["placement_flags"] = {
            "threads": int(best["threads"]),
            "gpu_tuning": bool(hw.has_gpu),
            "unified_memory": bool(not hw.vram_is_discrete),
        }
        values["bench"]["placement_ctx_final"] = pl_res["ctx_final"]
        values["bench"]["placement_depth_final"] = pl_res["depth_final"]
        values["bench"]["placement_finalistes"] = list(pl_res.get("finalistes") or [])
        values["bench"]["placement_non_explores"] = [
            f"{n['key']} : {n['raison']}" for n in pl_res.get("non_explores") or []
        ]
        if pl_res["tg_ts"] is not None:
            values["bench"]["placement_tg_ts"] = pl_res["tg_ts"]
            values["bench"]["placement_pp_ts"] = pl_res["pp_ts"]
        if pl_res["gain_pct"] is not None:
            values["bench"]["placement_gain_pct"] = pl_res["gain_pct"]
        if pl_res["mesures"]:
            values["bench"]["placement_mesures"] = _sans_none(pl_res["mesures"])
        if (
            pl_res.get("preselection")
            and pl_res["preselection"] is not pl_res["mesures"]
        ):
            values["bench"]["placement_preselection"] = _sans_none(
                pl_res["preselection"]
            )
    # Repli MACHINE : un modèle ajouté plus tard n'est jamais benché et tombait sur les
    # constantes aveugles de llama-server. On n'écrit QUE ce qui a été mesuré.
    if ub_res:
        values["server"]["ubatch"] = ub_res["ubatch"]
        values["server"]["batch"] = ub_res["batch"]
        values["bench"]["ubatch"] = ub_res["ubatch"]
        values["bench"]["batch"] = ub_res["batch"]
        values["bench"]["ubatch_pp_ts"] = ub_res["pp_ts"]
        values["bench"]["ubatch_mesures"] = ub_res["mesures"]
    # `checkpoint_min_step` n'est PAS mesurable par llama-bench (il gouverne la
    # compaction en session, pas un débit). Défaut RAISONNÉ, réservé aux modèles à
    # mémoire hybride que la sonde d'isolation vient de détecter : le défaut serveur
    # (8192) y laisse des déserts -> compaction profonde à ~8k tokens retraités
    # (39 s mesurés) contre ~2k à 2048 (13,3 s). Étiqueté comme non mesuré.
    if isolation is not None:
        values["server"]["checkpoint_min_step"] = 2048
        values["bench"]["checkpoint_min_step_origine"] = (
            "défaut raisonné (NON mesuré) — modèle à mémoire hybride détecté"
        )
    # Persister même un zéro mesuré, sauf pour un MoE dont l'override global serait trompeur.
    if len(ngl) > 1 and not moe:
        values["override"]["n_gpu_layers"] = best["ngl"]
    set_local_values(PERSONAL_CONFIG_PATH, values)
    # La pente dépend de l'architecture; persister donc le contexte par modèle.
    _set_model_context(gguf_path, context, calib["mecanisme"])
    if pl_res and pl_res.get("placement") is not None and pl_res["mesures"]:
        # Écrit ce qui a été VALIDÉ (comparé, ou candidat unique validé au contexte utile).
        import datetime as _dt

        _set_model_placement(
            gguf_path,
            pl_res["placement"],
            f"{_dt.date.today().isoformat()}, {build} — {pl_res['mecanisme']}",
        )
    if ub_res:
        _set_model_ubatch(
            gguf_path,
            ub_res["ubatch"],
            ub_res["batch"],
            ub_res.get("detail")
            or f"{ub_res['pp_ts']} t/s sur {bench_mod.UBATCH_PROBE_PROMPT} tokens",
        )
    if th_res and th_res.get("compare"):
        # Vérité PAR MODÈLE : les threads mesurés sur son placement élu, prioritaires sur
        # l'override machine (qui reste le repli des modèles non benchés).
        import datetime as _dt

        _set_model_threads(
            gguf_path,
            th_res["threads"],
            f"{_dt.date.today().isoformat()} — {th_res['mecanisme']} (sur "
            f"{th_res.get('placement')} à ctx {th_res['ctx']}, profondeur {th_res['depth']})",
        )
    if isolation is not None:
        cache_txt = ""
        if cache_v and cache_v.get("reused") is not None:
            cache_txt = (
                " ; cache réutilisé après routage"
                if cache_v["reused"]
                else " ; cache NON réutilisé avec la configuration finale"
            ) + f" ({topo_mod.cache_check_text(cache_v)})"
        _set_model_cache_isolation(gguf_path, isolation, iso_detail + cache_txt)
    # Archive DURABLE du bench (var/bench/<modèle>/<horodatage>.json) : le compte rendu
    # progressif complet, puis la trace de l'application (immédiate ici).
    trace.update(ecrit=values, etape="fin")
    _archive_setup(
        con,
        trace,
        applied={
            "context": context,
            "placement": (pl_res or {}).get("placement"),
            "ubatch": (ub_res or {}).get("ubatch"),
            "batch": (ub_res or {}).get("batch"),
            "cache_isolation": isolation,
            "threads": (th_res or {}).get("threads"),
        },
    )
    gpu_txt = f", offload GPU -ngl {best['ngl']}" if best["ngl"] > 0 else ""
    con.say(
        f"  Mesuré : génération {best['tg_ts']:.1f} t/s · prefill "
        f"{best['pp_ts']:.1f} t/s (threads={best['threads']}{gpu_txt})"
    )
    if calib.get("valide", True):
        con.say(
            f"  [ok] context={context} ({topo}, pente {calib['slope_kb_tok']} Ko/token "
            f"mesurée, vitesse validée jusqu'à {calib['valide_jusqua']} tokens)"
        )
    else:
        con.say(
            f"  [attention] context={context} : repli NON validé — aucun barreau de "
            f"vitesse n'a pu être mesuré ({topo}, pente {calib['slope_kb_tok']} Ko/token)."
        )
    if pl_res and pl_res["tg_ts"] is not None:
        gain_pl = (
            f", {pl_res['gain_pct']:+.0f} % de génération vs {pl_res['baseline']}"
            if pl_res["gain_pct"] is not None
            else ""
        )
        con.say(
            f"  [ok] placement={pl_res['placement'].key} ({pl_res['tg_ts']} t/s gén., "
            f"{pl_res['pp_ts']} t/s prefill à ctx {pl_res['ctx_final']}, profondeur "
            f"{pl_res['depth_final']} tokens{gain_pl})"
        )
        for n in pl_res.get("non_explores") or []:
            con.say(f"       {n['key']} : {n['raison']}")
    if ub_res:
        gain = (
            f", +{ub_res['gain_pct']:.0f} % de prefill"
            if ub_res.get("gain_pct")
            else ""
        )
        con.say(
            f"  [ok] ubatch={ub_res['ubatch']} / batch={ub_res['batch']} "
            f"({ub_res['pp_ts']:.1f} t/s sur {bench_mod.UBATCH_PROBE_PROMPT} tokens{gain}) "
            "— défaut machine pour les modèles ajoutés ensuite"
        )
    con.say(f"     mécanisme : {calib['mecanisme']}")
    for line in _usage_verdict(best["tg_ts"], best["pp_ts"]):
        con.say(line)
    report.add(
        "bench",
        "fait",
        f"threads={best['threads']}{gpu_txt} · ctx={context} ({topo}) · "
        f"{best['tg_ts']:.1f} t/s gén.",
    )


def _fmt_duration(seconds: float) -> str:
    if seconds >= 90:
        return f"{round(seconds / 60)} min"
    return f"{int(seconds)} s"


def _usage_verdict(tg_ts: float, pp_ts: float) -> list[str]:
    """Traduit les vitesses MESURÉES en verdict d'usage franc. Liste vide si RAS.

    Seuils d'expérience (pas de spec machine, que du ressenti) : décode < 8 t/s =
    sous la vitesse de lecture confortable ; prefill tel qu'un prompt de 4 000
    tokens (démarrage de session Loom réaliste : system prompt + outils + fiche
    projet) dépasse ~2 min = attente sensible avant le premier mot."""
    if tg_ts <= 0 or pp_ts <= 0:
        return []
    warmup_s = 4000 / pp_ts
    slow_read = tg_ts < 8
    slow_warm = warmup_s > 120
    if not (slow_read or slow_warm):
        return []
    lines = ["  [attention] Verdict d'usage (mesuré, pas supposé) : ce sera lent."]
    if slow_warm:
        lines.append(
            f"    · prefill {pp_ts:.1f} t/s → un prompt de 4 000 tokens met "
            f"~{_fmt_duration(warmup_s)} avant le premier mot d'une session ;"
        )
    if slow_read:
        lines.append(
            f"    · génération {tg_ts:.1f} t/s → sous la vitesse de lecture "
            f"confortable (~8 t/s)."
        )
    lines.append(
        "    Précos : un quant plus léger (ex. Q4_K_M) ou un modèle plus petit "
        "ira nettement plus vite sur ce poste ;"
    )
    lines.append(
        "    garde ce modèle pour les échanges courts / le hors-ligne, et un "
        "modèle distant ([[remote_models]]) pour les gros chantiers."
    )
    return lines


def run(con: Console, deps: Deps) -> int:
    report = SetupReport()
    con.say("── Loom setup ────────────────────────────────────────")
    try:
        plat, hw, ram = step_detection(con, report, deps)
        raw_cfg = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
        step_binary(con, report, deps, plat, hw, raw_cfg)
        # Relire la config car chaque étape peut modifier la suivante.
        raw_cfg = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
        step_swap(con, report, deps, plat, raw_cfg)
        step_tooling(con, report, deps)
        raw_cfg = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
        hw = _refresh_gpu(con, deps, hw, raw_cfg)
        step_model(con, report, deps, hw, ram, raw_cfg)
        raw_cfg = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
        step_bench(con, report, deps, raw_cfg)
    except KeyboardInterrupt:
        con.say("")
        con.say("  Interrompu (Ctrl+C) — bilan de ce qui a été fait :")
    con.say(report.render())
    con.say(f"  Journal : {SETUP_LOG}")
    con.say("  Prochaine étape : uv run python -m loom.web   →   http://127.0.0.1:8000")
    con.say("  (l'interface démarre le serveur modèle toute seule, à la demande)")
    return 1 if report.failed else 0


def ensure_utf8_stdio() -> None:
    """Console Windows héritée (cp1252) : nos écrans utilisent ─/→/accents — on force
    UTF-8 avec repli, sinon UnicodeEncodeError dès la bannière quand la sortie
    est redirigée. stdin en utf-8-sig : un pipe PowerShell préfixe la 1re ligne
    du BOM UTF-8 (0xEF 0xBB 0xBF) qu'un décodage cp1252 transforme en « ï»¿1 »
    — utf-8-sig l'avale, et reste correct au clavier. Best-effort : reconfigure
    existe depuis 3.7, jamais bloquant."""
    targets = (
        (sys.stdout, "utf-8"),
        (sys.stderr, "utf-8"),
        (sys.stdin, "utf-8-sig"),
    )
    for stream, enc in targets:
        try:
            stream.reconfigure(encoding=enc, errors="replace")
        except (AttributeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loom-setup",
        description="Installeur Loom : binaire llama.cpp + premier modèle, guidé.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="accepte toutes les propositions par défaut"
    )
    args = parser.parse_args(argv)
    ensure_utf8_stdio()
    # Le journal ne couvre que l'exécution courante.
    try:
        SETUP_LOG.parent.mkdir(parents=True, exist_ok=True)
        SETUP_LOG.write_text(
            f"# loom-setup — {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    con = Console(log_path=SETUP_LOG, assume_yes=args.yes)
    return run(con, Deps())


if __name__ == "__main__":
    raise SystemExit(main())
