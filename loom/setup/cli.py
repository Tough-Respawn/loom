# loom/setup/cli.py
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

# Mêmes repères que serve.py : ce fichier vit dans loom/setup/, la racine du
# repo est deux niveaux au-dessus du package.
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
        # Chrono de progression : la ligne se ré-affiche chaque seconde avec le
        # temps écoulé dans l'étape courante (une sonde de calibration peut rester
        # muette plusieurs minutes — sans chrono, ça ressemble à un gel).
        # Ticker seulement sur un vrai terminal : les fakes de tests restent
        # synchrones et déterministes.
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
        # Coloration LIGNE PAR LIGNE : le bilan arrive en un bloc multi-lignes,
        # les règles (^…) ne matcheraient que la première sinon.
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
            # Efface la queue de l'ancienne ligne si la nouvelle est plus courte.
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
        # strip du BOM : stdin pipé depuis PowerShell préfixe la 1re ligne de
        # ﻿ — invisible mais "﻿1" != "1". Sans effet au clavier.
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
    # Calibration topologique du contexte (topology.py) — injectables pour tests.
    gpu_vram_total_mb: object = topo_mod.gpu_vram_total_mb
    make_probe: object = topo_mod.ServerProbe  # (**kw) -> objet avec .run(ctx, depth)

    def __post_init__(self):
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


# ─────────────────────────────── Étapes ────────────────────────────────


def step_detection(con: Console, report: SetupReport, deps: Deps):
    con.say("[1/4] Détection du système")
    plat = deps.detect_platform()
    hw = deps.detect_hardware()
    ram = deps.ram_available_mb()
    # Avant le binaire, seule la sonde NVIDIA existe (elle décide du build CUDA à
    # l'étape 2). Les autres GPU (AMD/Intel — Vulkan) sont confirmés juste après,
    # par le binaire lui-même (--list-devices) : affichage honnête, pas « aucun ».
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

    # Un binaire d'un run précédent traîne peut-être déjà sous var/runtime/llama/.
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
    # Pas de routeur sans serveur : si llama-server a été refusé/raté, on ne
    # télécharge rien et on n'écrit pas de config.
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

    # Garde-éveil pendant le téléchargement (même filet que serve.py : un GGUF de
    # 15+ Go sans activité utilisateur ne doit pas être coupé par la veille).
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
    # Premier modèle de CETTE machine = son défaut local (sinon defaults.toml
    # peut pointer un modèle du parc absent d'ici -> sélecteur UI fantôme).
    set_default_model(PERSONAL_CONFIG_PATH, model_id)
    extra = " (MoE → cpu_moe = true)" if meta.get("expert_count") else ""
    con.say(f"  [ok] Modèle « {model_id} » installé{extra} — défaut de cette machine.")
    report.add("modele", "fait", f"{model_id} ({filename}, {size_mb} Mo)")


def step_model(con: Console, report: SetupReport, deps: Deps, hw, ram, raw_cfg):
    con.say("")
    con.say("[3/4] Modèle")
    # Un model.toml SANS son GGUF (Ctrl+C pendant le download) n'est PAS un
    # modèle branché : le dire et proposer de finir, plutôt qu'un « rien à
    # faire » mensonger contredit par le bench deux lignes plus bas.
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
        # URL huggingface.co ou id org/repo collé tel quel : on court-circuite la
        # recherche, l'inventaire des quants (probe_repo) validera le repo.
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
            # Filtre par le budget de CETTE machine : inutile de proposer un 397B
            # à 1,6 Go de budget. Estimation depuis le nom (± large) ; la taille
            # réelle des quants tranche après le choix.
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
        # Le catalogue porte des FAMILLES (queries), pas des repos figés : le
        # repo réel se résout en live (top téléchargements qui fit le budget).
        # Erreur réseau/HF distinguée de « rien de jouable » : le message HF
        # (diagnostic proxy inclus) remplace un « famille disparue ? » trompeur.
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
        # `context = N` exactement — pas context_length ni un commentaire.
        if code.startswith("context") and code.replace(" ", "").startswith("context="):
            lines[i] = new_line
            if i == 0 or not lines[i - 1].strip().startswith("# context calibré"):
                lines.insert(i, stamp)
            else:
                lines[i - 1] = stamp
            break
    else:
        lines += ["", stamp, new_line]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        # Nommer CE qui manque : « binaire ou modèle » accusait le binaire même
        # quand seul le GGUF manquait (téléchargement interrompu).
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

    # Profil AGNOSTIQUE (binaire = source de vérité) + métadonnées GGUF : ils
    # dimensionnent les candidats -ngl (et servent à la topologie plus bas).
    hw = deps.detect_hardware(server_bin)
    try:
        meta = read_gguf_meta(gguf_path)
    except ValueError:
        meta = {}

    threads = bench_mod.thread_candidates(os.cpu_count() or 4, deps.cpu_physical())
    # MoE (expert_count dans le header) : le bench mesure la config RUNTIME
    # (denses sur GPU, experts en RAM via -ncmoe) — offloader tous les poids
    # d'un 35B en VRAM OOMait le device (vécu Ornith Q8).
    moe = bool(meta.get("expert_count"))
    ngl, ncmoe = bench_mod.ngl_candidates(
        deps.has_gpu_backend(server_bin) and hw.has_gpu,
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
        rows = deps.run_bench(bench_bin, gguf_path, threads, ngl, n_cpu_moe=ncmoe)
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

    # ── Contexte : calibration TOPOLOGIQUE (topology.py) — pente MESURÉE entre
    # deux chargements + échelle de vitesse en profondeur, avec les flags EXACTS
    # de l'exécutant. Remplace l'ex-formule « KV théorique vs RAM », fausse d'un
    # facteur 2 (q8_0 vs f16) à 5 (sliding-window) sur le parc réel (audit et
    # sondes du 2026-07-18).
    import psutil

    vram_total = deps.gpu_vram_total_mb()
    topo = topo_mod.discover_topology(
        meta, deps.has_gpu_backend(server_bin), vram_total
    )
    headroom = int((raw_cfg.get("server") or {}).get("gpu_kv_headroom_mb", 640) or 640)
    # RAM TOTALE (déterministe), jamais la dispo du moment — audit P3.
    ram_total_mb = int(psutil.virtual_memory().total // (1024 * 1024))
    budget = topo_mod.memory_budget_mb(topo, vram_total, ram_total_mb, headroom)
    model_toml = _read_model_toml(gguf_path)
    is_moe = bool(meta.get("expert_count"))
    mmproj_name = model_toml.get("mmproj_filename")
    probe = deps.make_probe(
        server_bin=str(server_bin),
        model_path=str(gguf_path),
        threads=best["threads"],
        # MoE + GPU : doctrine mesurée du parc — attention sur GPU, experts en RAM.
        ngl=99 if (is_moe and topo != topo_mod.TOPO_RAM) else best["ngl"],
        topology=topo,
        mmproj_path=str(gguf_path.parent / mmproj_name) if mmproj_name else None,
        cpu_moe=bool(model_toml.get("cpu_moe", is_moe)),
        n_cpu_moe=model_toml.get("n_cpu_moe"),
    )
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
    except (RuntimeError, ValueError) as exc:
        con.progress_end()
        con.say(
            f"  [échec] calibration échouée ({exc}) — context inchangé, relance loom-setup."
        )
        report.add("bench", "echec", f"calibration contexte : {exc}")
        return
    con.progress_end()
    context = calib["context"]

    values = {
        "server": {"context": context},
        "override": {"threads": best["threads"]},
        "bench": {
            "threads": best["threads"],
            "ngl": best["ngl"],
            "tg_ts": round(best["tg_ts"], 2),
            "pp_ts": round(best["pp_ts"], 2),
            "context": context,
            # La décision porte son MÉCANISME (audit P6) : on saura toujours
            # pourquoi ce chiffre, et jusqu'où la vitesse a été vérifiée.
            "context_mode": calib["mode"],
            "context_mecanisme": calib["mecanisme"],
            "context_pente_kb_tok": calib["slope_kb_tok"],
            "context_valide_jusqua": calib["valide_jusqua"],
        },
    }
    # Dès que le GPU a été TESTÉ, la mesure a le dernier mot — 0 compris (un
    # iGPU peut perdre contre le CPU) : sans l'écrire, l'auto-offload runtime
    # (resolve_ngl) re-prendrait un GPU mesuré plus lent. SAUF pour un MoE :
    # resolve_ngl (cpu_moe) ignore l'override, et un override GLOBAL issu d'une
    # mesure MoE (999/0) polluerait les modèles denses installés ensuite.
    if len(ngl) > 1 and not moe:
        values["override"]["n_gpu_layers"] = best["ngl"]
    set_local_values(PERSONAL_CONFIG_PATH, values)
    # Vérité PAR MODÈLE : la pente est propre à chaque architecture — le contexte
    # calibré s'écrit aussi dans le model.toml du modèle benché.
    _set_model_context(gguf_path, context, calib["mecanisme"])
    gpu_txt = f", offload GPU -ngl {best['ngl']}" if best["ngl"] > 0 else ""
    con.say(
        f"  Mesuré : génération {best['tg_ts']:.1f} t/s · prefill "
        f"{best['pp_ts']:.1f} t/s (threads={best['threads']}{gpu_txt})"
    )
    con.say(
        f"  [ok] context={context} ({topo}, pente {calib['slope_kb_tok']} Ko/token "
        f"mesurée, vitesse validée jusqu'à {calib['valide_jusqua']} tokens)"
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


# ─────────────────────────────── main ────────────────────────────────


def run(con: Console, deps: Deps) -> int:
    report = SetupReport()
    con.say("── Loom setup ────────────────────────────────────────")
    try:
        plat, hw, ram = step_detection(con, report, deps)
        raw_cfg = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
        step_binary(con, report, deps, plat, hw, raw_cfg)
        # Relire la config entre chaque étape : la précédente a pu la modifier.
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
    # Log frais à chaque run (on veut la session courante, pas l'historique).
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
