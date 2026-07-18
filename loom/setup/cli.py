# loom/setup/cli.py
"""Installeur interactif console : `uv run loom-setup`.

Quatre étapes, chacune sur le même contrat HITL : état constaté → proposition
EXPLIQUÉE (quoi, où, quelle taille) → confirmation → action → résultat.
1. Détection (OS/GPU/RAM) · 2. Binaire llama.cpp · 3. Modèle qui fit ·
4. Bench du matériel → meilleurs réglages écrits dans config/local.toml.
Tout ce qui s'affiche part aussi dans var/logs/setup.log ; le bilan final
récapitule. Relançable : ne refait que ce qui manque."""

from __future__ import annotations

import argparse
import sys
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
from loom.runtime.platform_info import detect as detect_platform
from loom.runtime.term import colorize, supports_color
from loom.setup import bench as bench_mod
from loom.setup import llama_release
from loom.setup.catalog import (
    budget_mb,
    filter_by_budget,
    fitting_entries,
    pick_mmproj,
    probe_repo,
    resolve_entry,
)
from loom.setup.llama_release import local_arch, select_assets
from loom.setup.report import SetupReport
from loom.setup.steps import (
    first_model_file,
    installed_model_ids,
    models_roots,
    read_raw_config,
    resolve_bin,
    server_bin_status,
    set_default_model,
    set_local_values,
    set_server_bin,
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
        self._print(colorize(msg) if self.color else msg)
        self._log(msg)

    def progress(self, msg: str) -> None:
        self._print(f"\r  {msg}", end="", flush=True)

    def progress_end(self) -> None:
        self._print()

    def ask(self, prompt: str, default: str = "") -> str:
        if self.assume_yes:
            return default
        # strip du BOM : stdin pipé depuis PowerShell préfixe la 1re ligne de
        # ﻿ — invisible mais "﻿1" != "1". Sans effet au clavier.
        raw = self._input(f"  {prompt} ").strip().strip("﻿").strip()
        return raw or default

    def confirm(self, question: str, default: bool = True) -> bool:
        if self.assume_yes:
            return True
        suffix = "[O/n]" if default else "[o/N]"
        raw = self._input(f"  {question} {suffix} ").strip().strip("﻿").strip()
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

    def __post_init__(self):
        if self.fetch_release is None:
            self.fetch_release = _real_fetch_release
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
    gpu = f"{hw.gpu_name} ({hw.vram_free_mb} Mo VRAM libre)" if hw.has_gpu else "aucun"
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
            con.say(f"  ✅ config/local.toml : [server] bin = {existing}")
            report.add("binaire", "fait", f"réutilisé ({version}) → config/local.toml")
            return

    try:
        release = deps.fetch_release()
    except RuntimeError as exc:
        con.say(f"  ❌ {exc}")
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
        con.say("  ⏭️ Ignoré — tu peux relancer loom-setup plus tard.")
        report.add("binaire", "ignore", "téléchargement refusé")
        return

    def _cb(name, done_mb, total_mb):
        con.progress(f"{name} : {done_mb}/{total_mb or '?'} Mo")

    try:
        extracted = deps.download_and_extract(plan, RUNTIME_DIR, _cb)
    except (RuntimeError, OSError) as exc:
        con.progress_end()
        con.say(f"  ❌ Téléchargement/extraction : {exc}")
        report.add("binaire", "echec", f"téléchargement : {exc}")
        return
    con.progress_end()

    binary = deps.find_llama_server(extracted)
    version = deps.verify_binary(binary) if binary else None
    if binary is None or version is None:
        con.say(
            "  ❌ Binaire extrait mais inutilisable (--version muet) — config/local.toml "
            "laissé intact. Vérifie l'archive ou installe à la main."
        )
        report.add("binaire", "echec", "binaire extrait mais --version muet")
        return
    set_server_bin(PERSONAL_CONFIG_PATH, binary)
    con.say(f"  Vérification --version : OK ({version})")
    con.say(f"  ✅ config/local.toml : [server] bin = {binary}")
    report.add(
        "binaire", "fait", f"installé ({plan.tag}, {plan.backend}) → config/local.toml"
    )


def _offer_free_ram(con: Console, deps: Deps, hw, ram: int) -> int:
    """Avant de choisir un modèle : montre les gros consommateurs de RAM et
    laisse l'utilisateur en FERMER lui-même (on ne tue jamais rien nous-mêmes),
    puis re-mesure. Renvoie la RAM disponible finale. Le budget se recalcule à
    chaque tour : plus de RAM = meilleur modèle proposé."""
    if con.assume_yes:  # non-interactif : personne pour fermer quoi que ce soit
        return ram
    budget = budget_mb(hw.vram_free_mb, ram)
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
        budget = budget_mb(hw.vram_free_mb, ram)
        con.say(f"  → RAM disponible : {ram} Mo · budget : {budget} Mo")
        if ans.lower().startswith("c"):
            break
    return ram


def step_model(con: Console, report: SetupReport, deps: Deps, hw, ram, raw_cfg):
    con.say("")
    con.say("[3/4] Modèle")
    ids = installed_model_ids(raw_cfg, PACKAGE_MODELS)
    if ids:
        con.say(f"  {len(ids)} modèle(s) branché(s) ({', '.join(ids)}) → rien à faire.")
        report.add("modele", "ok", f"{len(ids)} branché(s) : {', '.join(ids)}")
        return

    budget = budget_mb(hw.vram_free_mb, ram)
    con.say("  Aucun modèle branché (<racine>/local/text/ vide).")
    con.say(f"  Budget estimé : {budget} Mo (VRAM libre + RAM − 4 Go de marge).")
    ram = _offer_free_ram(con, deps, hw, ram)
    budget = budget_mb(hw.vram_free_mb, ram)
    entries = fitting_entries(budget)

    con.say("  Recommandé pour ta machine :")
    for i, e in enumerate(entries, start=1):
        con.say(f"    {i}. {e['label']}")
    con.say(f"    {len(entries) + 1}. Recherche libre Hugging Face")
    con.say("    0. Passer (tu pourras taper /add-model dans le chat)")
    default = "1" if entries else "0"
    choice = con.ask(f"Ton choix [{default}] :", default=default)

    if choice == "0":
        con.say("  ⏭️ Passé — /add-model dans le chat quand tu veux.")
        report.add("modele", "ignore", "reporté (/add-model dans le chat)")
        return

    repo = None
    if choice == str(len(entries) + 1):
        query = con.ask("Recherche Hugging Face (nom du modèle) :")
        if not query:
            report.add("modele", "ignore", "recherche vide")
            return
        try:
            hits = deps.search_models(query)
        except Exception as exc:  # noqa: BLE001 - HfCatalogError au message montrable
            con.say(f"  ❌ {exc}")
            report.add("modele", "ignore", "recherche impossible (hors-ligne ?)")
            return
        # Filtre par le budget de CETTE machine : inutile de proposer un 397B à
        # 1,6 Go de budget. Estimation depuis le nom (± large) ; la taille
        # réelle des quants tranche après le choix.
        hits, hidden = filter_by_budget(hits, budget)
        if hidden:
            con.say(
                f"  ({hidden} résultat(s) masqué(s) : trop gros pour ton budget "
                f"de {budget} Mo)"
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
            con.say(f"    {i}. {h['repo_id']} ({h['downloads']} téléchargements{est})")
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
        repo = resolve_entry(entry, deps.search_models, budget)
        if repo is None:
            con.say(
                f"  ❌ « {entry['label']} » introuvable sur Hugging Face (hors-ligne, "
                "famille disparue ?) — réessaie, ou recherche libre."
            )
            report.add("modele", "ignore", f"entrée non résolue ({entry['label']})")
            return
        con.say(f"  → repo retenu : {repo}")

    files = deps.probe_repo(repo)
    if files is None:
        con.say(
            f"  ❌ Repo « {repo} » injoignable (hors-ligne, renommé ?) — réessaie plus "
            "tard ou passe par /add-model dans le chat."
        )
        report.add("modele", "ignore", f"repo injoignable ({repo})")
        return

    quants = [f for f in files if not f.get("is_aux", f["is_mmproj"])]
    if not quants:
        con.say(f"  ❌ Aucun GGUF exploitable dans {repo}.")
        report.add("modele", "ignore", f"aucun GGUF dans {repo}")
        return
    annotated = recommend_quant(quants, hw.vram_free_mb, ram)
    rec = next((f for f in annotated if f["recommended"]), annotated[0])
    fit_txt = "tient dans le budget" if rec["fits"] else "NE TIENDRA PAS (trop gros)"
    if not con.confirm(
        f"Quant recommandé : {rec['filename']} ({rec['size_mb']} Mo) — {fit_txt}. "
        "Télécharger ?"
    ):
        con.say("  ⏭️ Passé — /add-model dans le chat pour choisir un autre quant.")
        report.add("modele", "ignore", "quant refusé")
        return

    mmproj = pick_mmproj(files)
    model_id = derive_model_id(repo)
    dest = models_roots(raw_cfg, PACKAGE_MODELS)[0] / "local" / "text" / model_id
    write_model_toml(
        dest, repo, rec["filename"], rec["size_mb"], mmproj_filename=mmproj
    )
    filenames = list(rec.get("part_files") or [rec["filename"]])
    if mmproj:
        filenames.append(mmproj)

    # Garde-éveil pendant le téléchargement (même filet que serve.py : un GGUF de
    # 15+ Go sans activité utilisateur ne doit pas être coupé par la veille).
    from loom.runtime.stay_awake import StayAwake

    awake = StayAwake()
    awake.acquire()
    try:
        job = deps.start_download(repo, filenames, dest, rec["size_mb"])
        while not job.done:
            con.progress(f"téléchargement… {job.progress_mb()}/{rec['size_mb']} Mo")
            deps.sleep(2)
    finally:
        awake.release()
    con.progress_end()

    if job.error:
        con.say(f"  ❌ {job.error}")
        con.say(
            "  (model.toml déjà écrit : le téléchargement REPRENDRA au premier serve.)"
        )
        report.add("modele", "echec", f"téléchargement : {job.error}")
        return
    meta = finalize_model_toml(dest, dest / rec["filename"])
    # Premier modèle de CETTE machine = son défaut local (sinon defaults.toml
    # peut pointer un modèle du parc absent d'ici -> sélecteur UI fantôme).
    set_default_model(PERSONAL_CONFIG_PATH, model_id)
    extra = " (MoE → cpu_moe = true)" if meta.get("expert_count") else ""
    con.say(f"  ✅ Modèle « {model_id} » installé{extra} — défaut de cette machine.")
    report.add("modele", "fait", f"{model_id} ({rec['filename']}, {rec['size_mb']} Mo)")


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
        con.say("  ⏭️ Binaire ou modèle pas encore en place — bench sauté.")
        report.add("bench", "ignore", "binaire ou modèle manquant")
        return
    bench_bin = deps.find_llama_bench(server_bin)
    if bench_bin is None:
        con.say(
            "  🔧 llama-bench introuvable à côté du binaire — réglages par défaut "
            "conservés (réinstalle via loom-setup pour l'avoir)."
        )
        report.add("bench", "manuel", "llama-bench absent de la release")
        return
    gguf_path, model_size_mb = model

    import os

    threads = bench_mod.thread_candidates(os.cpu_count() or 4, deps.cpu_physical())
    ngl = [0] + ([99] if deps.has_gpu_backend(server_bin) else [])
    combos = len(threads) * len(ngl)
    con.say(
        f"  On mesure la vitesse réelle sur TON modèle ({gguf_path.name}) : "
        f"{combos} combinaisons de threads{' et offload GPU' if len(ngl) > 1 else ''}."
    )
    con.say("  Durée : ~2-10 min selon la machine (CPU à fond, c'est normal).")
    if not con.confirm("Lancer le bench maintenant ?"):
        con.say("  ⏭️ Sauté — relançable à tout moment : uv run loom-setup.")
        report.add("bench", "ignore", "refusé (relançable)")
        return

    con.progress("bench en cours… (llama-bench, plusieurs minutes)")
    try:
        rows = deps.run_bench(bench_bin, gguf_path, threads, ngl)
    except RuntimeError as exc:
        con.progress_end()
        con.say(f"  ❌ {exc}")
        report.add("bench", "echec", str(exc))
        return
    con.progress_end()
    best = bench_mod.pick_best(rows)
    if best is None:
        con.say("  ❌ Aucune mesure de génération exploitable.")
        report.add("bench", "echec", "sortie llama-bench vide")
        return

    try:
        meta = read_gguf_meta(gguf_path)
    except ValueError:
        meta = {}
    context = bench_mod.compute_context(
        deps.ram_available_mb(), model_size_mb, bench_mod.kv_bytes_per_token(meta)
    )
    values = {
        "server": {"context": context},
        "override": {"threads": best["threads"]},
        "bench": {
            "threads": best["threads"],
            "ngl": best["ngl"],
            "tg_ts": round(best["tg_ts"], 2),
            "pp_ts": round(best["pp_ts"], 2),
            "context": context,
        },
    }
    if best["ngl"] > 0:
        values["override"]["n_gpu_layers"] = best["ngl"]
    set_local_values(PERSONAL_CONFIG_PATH, values)
    gpu_txt = f", offload GPU -ngl {best['ngl']}" if best["ngl"] > 0 else ""
    con.say(
        f"  Mesuré : génération {best['tg_ts']:.1f} t/s · prefill "
        f"{best['pp_ts']:.1f} t/s (threads={best['threads']}{gpu_txt})"
    )
    con.say(
        f"  ✅ config/local.toml : threads={best['threads']}{gpu_txt}, "
        f"context={context} (KV qui tient en RAM, sans swap)"
    )
    report.add(
        "bench",
        "fait",
        f"threads={best['threads']}{gpu_txt} · ctx={context} · "
        f"{best['tg_ts']:.1f} t/s gén.",
    )


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
    """Console Windows héritée (cp1252) : nos écrans utilisent ─/✅/⏭️ — on force
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
