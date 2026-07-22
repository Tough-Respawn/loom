# loom/runtime/serve.py
"""Lanceur cross-platform et auto-adaptatif de llama-server.

Usage : uv run loom/runtime/serve.py
Auto-détecte le hardware (GPU NVIDIA sinon CPU), résout la config,
télécharge les GGUF du registre si absents, génère le llama-swap.yaml
et démarre llama-swap (routeur multi-modèles, API OpenAI-compatible).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from loom.config import RuntimeConfig, load_config
from loom.runtime.hardware import (
    HardwareProfile,
    detect_hardware,
)
from loom.runtime.models_fetch import ModelUnavailable, ensure_model
from loom.runtime.ngl import resolve_ngl
from loom.runtime.server_args import build_server_args, resolve_parallel
from loom.runtime.swap import build_swap_config, write_swap_yaml

# serve.py vit dans loom/runtime/ : on remonte de DEUX niveaux pour pointer la racine
# du package loom/ (où vivent config, models, data) — pas le dossier runtime/.
LOOM_DIR = Path(__file__).resolve().parent.parent  # = loom/ (le package)
REPO_ROOT = LOOM_DIR.parent
# Config à la racine du repo (config/), modèles dans le package (loom/models), état machine
# sous var/ (gitignored : llama-swap.yaml généré + logs).
CONFIG_PATH = REPO_ROOT / "config" / "defaults.toml"
PERSONAL_CONFIG_PATH = REPO_ROOT / "config" / "local.toml"
MODELS_DIR = LOOM_DIR / "models"
SWAP_YAML = REPO_ROOT / "var" / "cache" / "llama-swap.yaml"
# Log PERSISTANT du serveur modèle (le terminal est éphémère / illisible à distance).
# La web app en recopie une vue dans chaque session active. Repart à neuf à chaque lancement.
SERVE_LOG = REPO_ROOT / "var" / "logs" / "serve.log"
# Dossier des sauvegardes de cache KV (--slot-save-path) : fichiers .kv transitoires,
# noms réutilisés (turnend/dispatch) -> taille bornée. Gitignored avec le reste de var/.
SLOTS_DIR = REPO_ROOT / "var" / "cache" / "slots"


def slots_dir() -> str:
    """Chemin (créé) du dossier des sauvegardes de slot KV, en séparateurs POSIX
    (le yaml des cmd llama-server est normalisé ainsi)."""
    SLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(SLOTS_DIR).replace("\\", "/")


def _log(msg: str) -> None:
    """Écrit une ligne sur stderr (terminal, colorée si TTY) ET dans serve.log
    (toujours en texte brut), sans jamais lever."""
    from loom.runtime.term import colorize, supports_color

    shown = colorize(msg) if supports_color(sys.stderr) else msg
    print(shown, file=sys.stderr)
    try:
        SERVE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SERVE_LOG, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def resolve_mmproj_path(
    mmproj_filename: str, models_dir: Path, repo: str = ""
) -> str | None:
    """Télécharge le mmproj si configuré, renvoie son chemin local (ou None)."""
    if not mmproj_filename:
        return None
    path = ensure_model(repo, mmproj_filename, models_dir)
    return str(path)


def ensure_all_models(models, models_dir: Path) -> None:
    """Télécharge le GGUF (et le mmproj) de chaque modèle s'il manque, DANS le dossier du
    modèle (loom/models/<id>/) quand il est connu, sinon dans la racine partagée.

    Garde-éveil le temps des téléchargements : un GGUF de 20+ Go prend des dizaines de
    minutes sans aucune activité utilisateur — la veille par inactivité couperait le
    transfert en plein vol (même garde que les générations de loom.web). Relâché dès la
    fin : servir des requêtes ne doit PAS bloquer la veille en permanence."""
    from loom.runtime.stay_awake import StayAwake

    awake = StayAwake()
    awake.acquire()
    try:
        for m in models:
            dest = Path(m.dir) if m.dir else models_dir
            ensure_model(m.repo, m.filename, dest)
            if m.mmproj_filename:
                ensure_model(m.repo, m.mmproj_filename, dest)
    finally:
        awake.release()


def build_launch(
    cfg: RuntimeConfig,
    profile: HardwareProfile,
    model_path: Path,
    mmproj_path: str | None = None,
) -> list[str]:
    n_gpu = resolve_ngl(
        cfg.model,
        profile,
        cfg.override_n_gpu_layers,
        cfg.gpu_kv_headroom_mb,
    )
    # En mode GPU, threads = cœurs PHYSIQUES (≈ logiques/2 si HyperThreading) : au-delà,
    # la contention HT ralentit la passe CPU (PLE de Gemma 3n). En CPU-only, tous les
    # threads.
    if cfg.override_threads:
        threads = cfg.override_threads
    elif profile.has_gpu:
        threads = max(1, profile.cpu_threads // 2)
    else:
        threads = profile.cpu_threads
    return build_server_args(
        server_bin=cfg.server_bin,
        model_path=str(model_path),
        port=cfg.port,
        context=cfg.context,
        n_gpu_layers=n_gpu,
        threads=threads,
        mmproj_path=mmproj_path,
        gpu_tuning=profile.has_gpu,
        unified_memory=not profile.vram_is_discrete,
        n_parallel=resolve_parallel(cfg.n_parallel, cfg.model.cache_isolation),
        cpu_moe=cfg.model.cpu_moe,
        n_cpu_moe=cfg.model.n_cpu_moe,
        slot_save_dir=slots_dir(),
        ubatch=cfg.model.ubatch,
        batch=cfg.model.batch,
    )


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Tue le process lancé ET tous ses enfants. llama-swap engendre llama-server comme
    PETIT-enfant : un simple terminate() le laisserait orphelin (15+ Go de RAM + la VRAM
    bloqués). Sur Windows on s'appuie sur `taskkill /T` (tue l'arbre par PID) ; sur POSIX
    on tue le groupe de session (l'enfant a été lancé avec start_new_session)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        import os
        import signal as _signal

        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run(args: list[str], bin_name: str, hint: str) -> int:
    """Lance un binaire externe ; sa sortie (stdout+stderr) va dans serve.log. Message clair
    s'il est introuvable. Sur Ctrl+C (ou toute sortie), termine PROPREMENT l'arbre de process
    pour ne JAMAIS laisser llama-server zombie tenir la RAM/VRAM."""
    _log(f"[loom] Lancement : {' '.join(args)}")
    SERVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SERVE_LOG, "a", encoding="utf-8", errors="replace") as fh:
        try:
            # start_new_session : sur POSIX, isole l'enfant dans son groupe -> killpg propre,
            # et Ctrl+C ne le frappe pas avant nous. Sans effet sur Windows (on a taskkill /T).
            proc = subprocess.Popen(
                args,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=(sys.platform != "win32"),
            )
        except FileNotFoundError:
            _log(f"[loom] ERREUR : binaire '{bin_name}' introuvable. {hint}")
            return 1
        try:
            return proc.wait()
        except KeyboardInterrupt:
            _log(
                "[loom] Arrêt (Ctrl+C) — terminaison de l'arbre (llama-swap + llama-server)…"
            )
            _terminate_tree(proc)
            return 0
        finally:
            # Filet de sécurité : quelle que soit la cause de sortie, pas d'orphelin.
            if proc.poll() is None:
                _terminate_tree(proc)


def launch_direct(cfg: RuntimeConfig, profile: HardwareProfile) -> int:
    """Un seul modèle : llama-server directement, pas besoin de llama-swap."""
    model = cfg.model  # = modèle par défaut
    base = Path(model.dir) if model.dir else cfg.models_dir
    model_path = base / model.filename
    mmproj_path = resolve_mmproj_path(model.mmproj_filename, base, repo=model.repo)
    args = build_launch(cfg, profile, model_path, mmproj_path)
    return _run(
        args,
        cfg.server_bin,
        "Lance 'uv run loom-setup' pour installer llama.cpp, ou renseigne "
        "'bin' dans config/local.toml.",
    )


def launch_swap(cfg: RuntimeConfig, profile: HardwareProfile) -> int:
    """Plusieurs modèles : llama-swap route vers le bon selon le champ 'model'."""
    swap = build_swap_config(
        cfg.models,
        profile,
        llama_bin=cfg.server_bin,
        models_dir=str(cfg.models_dir),
        context=cfg.context,
        override_n_gpu_layers=cfg.override_n_gpu_layers,
        slot_save_dir=slots_dir(),
        n_parallel=cfg.n_parallel,
    )
    write_swap_yaml(swap, SWAP_YAML)
    args = [
        cfg.swap_bin,
        "--config",
        str(SWAP_YAML),
        "--listen",
        f"127.0.0.1:{cfg.port}",
        # Recharge la config à la volée quand le fichier change : loom.web régénère ce yaml
        # après une édition de modèle local / param serveur -> effet SANS relancer llama-swap.
        "--watch-config",
    ]
    return _run(
        args,
        cfg.swap_bin,
        "Télécharge llama-swap et renseigne 'swap_bin' dans config/local.toml, "
        "ou garde un seul modèle pour lancer llama-server en direct.",
    )


def regenerate_swap_yaml(
    defaults_path=CONFIG_PATH, local_path=PERSONAL_CONFIG_PATH, out_path=SWAP_YAML
) -> Path | None:
    """Régénère le llama-swap.yaml depuis la config COURANTE (sans rien télécharger).

    Appelé par loom.web après une édition de modèle local (offload/contexte) ou d'un param
    serveur : llama-swap lancé avec --watch-config recharge alors le fichier tout seul, et le
    modèle rechargé (après déchargement) démarre avec les nouveaux args. Ferme la boucle
    « customiser un modèle local depuis l'UI » sans toucher au TOML ni tout relancer.
    Best-effort : renvoie le chemin écrit, ou None si la config est illisible."""
    try:
        cfg = load_config(defaults_path, local_path)
        # Profil AGNOSTIQUE : le binaire configuré fait foi (--list-devices) —
        # sans lui, l'auto-offload de resolve_ngl ignorait tout GPU non-NVIDIA.
        profile = detect_hardware(cfg.server_bin)
        swap = build_swap_config(
            cfg.models,
            profile,
            llama_bin=cfg.server_bin,
            models_dir=str(cfg.models_dir),
            context=cfg.context,
            override_n_gpu_layers=cfg.override_n_gpu_layers,
            slot_save_dir=slots_dir(),
            n_parallel=cfg.n_parallel,
        )
        write_swap_yaml(swap, out_path)
        return Path(out_path)
    except Exception:  # noqa: BLE001 - régénération best-effort, jamais fatale pour l'UI
        return None


def maybe_bootstrap(remote_ok: bool = False) -> int | None:
    """Premier run sur machine vierge (binaire ou modèle manquant) : lance
    l'installeur guidé loom-setup DANS ce terminal. Renvoie None si la machine
    est (devenue) prête à servir, sinon un code de sortie.

    remote_ok (loom.web) : un modèle DISTANT ([[remote_models]] ou store UI)
    suffit pour discuter -> pas d'installeur forcé s'il en existe un (boot
    « remote-only »). serve.py, moteur llama.cpp, garde remote_ok=False : un
    modèle local y reste obligatoire.

    Terminal non interactif (service, CI) : pas de questions possibles — on
    guide vers la commande dédiée et on sort proprement."""
    from loom.setup.steps import has_remote_models, needs_setup, read_raw_config

    raw = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
    if not needs_setup(raw, MODELS_DIR):
        return None
    if remote_ok and has_remote_models(raw):
        return None
    if not (sys.stdin and sys.stdin.isatty()):
        _log(
            "[loom] Binaire llama-server ou modèle manquant — lance "
            "'uv run loom-setup' pour l'installation guidée."
        )
        return 1
    from loom.setup.cli import SETUP_LOG, Console, Deps, ensure_utf8_stdio, run

    _log("[loom] Premier lancement : installation guidée (loom-setup)…")
    ensure_utf8_stdio()
    run(Console(log_path=SETUP_LOG), Deps())
    # L'installeur a pu être refusé ou échouer : on ne sert que si tout est là.
    raw = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
    if needs_setup(raw, MODELS_DIR):
        _log(
            "[loom] Configuration incomplète — relance 'uv run loom-setup' "
            "(ou complète config/local.toml à la main) puis relance serve."
        )
        return 1
    return None


def main() -> int:
    # Log serveur frais à chaque lancement (on veut la session courante, pas l'historique).
    try:
        SERVE_LOG.parent.mkdir(parents=True, exist_ok=True)
        SERVE_LOG.write_text("", encoding="utf-8")
    except OSError:
        pass
    code = maybe_bootstrap()
    if code is not None:
        return code
    cfg = load_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
    if not cfg.models:
        # load_config tolère un parc 100 % distant (boot remote-only de loom.web) ;
        # serve, lui, EST le moteur des modèles locaux -> rien à servir ici.
        _log(
            "[loom] Aucun modèle LOCAL à servir (parc distant uniquement) — "
            "lance 'uv run loom-setup' pour en installer un."
        )
        return 1
    profile = detect_hardware(cfg.server_bin)
    _log(f"[loom] Profil détecté : {profile}")

    try:
        ensure_all_models(cfg.models, cfg.models_dir)
    except ModelUnavailable as exc:
        # Modèle absent et non téléchargeable : on guide l'utilisateur (quoi poser, où)
        # et on sort proprement (code 1, sans stacktrace).
        _log(f"[loom] {exc}")
        return 1
    _log(f"[loom] {len(cfg.models)} modèle(s), défaut={cfg.default_model}")

    # Un seul modèle : pas de routeur, llama-server direct (zéro dépendance externe).
    # Plusieurs : llama-swap pour le hot-swap par le champ 'model' de la requête.
    if len(cfg.models) <= 1:
        return launch_direct(cfg, profile)
    return launch_swap(cfg, profile)


if __name__ == "__main__":
    raise SystemExit(main())
