# loom/web/app.py

"""Application Flask de Loom Chat : page + endpoints (chat SSE, reset, skills,

thinking, model)."""

from __future__ import annotations


import base64

import json

import os

import re

import shutil

import subprocess

import sys

import threading

import time

import traceback

from pathlib import Path


from flask import Flask, Response, render_template, request


from loom.agent import context
from loom.agent.client import _msg_chars, set_debug_log_path

from loom.extend.skills import (
    collect_skills,
    effective_skills,
    read_skill_source,
    render_catalog,
    write_skill_source,
)

from loom.prompts import CHAT_SYSTEM_STRONG, IMAGE_REFINE_SYSTEM
from loom.runtime import model_store
from loom.runtime.comfy import ComfyEngine, ComfyError
from loom.runtime.hardware import ram_available_mb
from loom.runtime.manager import ModelServerManager
from loom.runtime.platform_info import detect as platform_detect

from loom.runtime.models_profile import load_profile
from loom.runtime.stay_awake import StayAwake


MAX_IMAGE_BYTES = 10 * 1024 * 1024

MAX_IMAGES = 6  # nb max d'images jointes à un message (au-delà : ignorées)


# Détection d'un chemin ABSOLU dans un message (Windows `C:\...` / `C:/...` ou POSIX

# `/...`). Sert à l'auto-adoption du dossier de travail : si l'utilisateur désigne un

# dossier existant, la session l'adopte -> run_shell tourne dedans, les chemins relatifs

# s'y résolvent, et il n'a PAS à pointer le dossier dans l'UI.

_PATH_RE = re.compile(r"""(?:[A-Za-z]:[\\/]|[\\/])[^\s"'`<>|*?]*""")


def _detect_workspace(message: str, root: str | None = None) -> str | None:
    """Renvoie le dossier EXISTANT le plus spécifique cité dans `message` (résolu absolu),

    ou None. Un fichier existant -> son dossier parent. N'adopte QUE du réel (isdir/isfile),

    donc un chemin de référence faux n'a aucun effet.



    Si `root` est fourni, on accepte aussi un PROJET cité par son seul NOM quand c'est un

    sous-dossier direct de `root` (ex. « ... pour energy-data-platform » sans le chemin

    complet). Match EXACT sur un sous-dossier réel -> pas de faux positif sur un mot courant.

    """

    found: list[str] = []

    for raw in _PATH_RE.findall(message):
        p = raw.rstrip(".,;:!?)]}»\"'`").strip()

        if len(p) < 3:
            continue

        try:
            if os.path.isdir(p):
                found.append(p)

            elif os.path.isfile(p):
                found.append(os.path.dirname(p))

        except OSError:
            continue

    if root:
        try:
            subdirs = {
                e.name.lower(): os.path.join(root, e.name)
                for e in os.scandir(root)
                if e.is_dir()
            }

        except OSError:
            subdirs = {}

        for tok in re.findall(r"[A-Za-z0-9][\w.-]{2,}", message):
            hit = subdirs.get(tok.lower())

            if hit:
                found.append(hit)

    if not found:
        return None

    best = max(found, key=len)  # le chemin le plus long = le plus spécifique

    try:
        return str(Path(best).resolve())

    except OSError:
        return None


def _sse(event_type: str, **fields) -> str:
    """Sérialise un événement Server-Sent Events (UTF-8, accents préservés)."""

    return f"data: {json.dumps({'type': event_type, **fields}, ensure_ascii=False)}\n\n"


def _infer_title(client, model, message: str) -> str:
    """Titre court (3-5 mots) inféré par le modèle depuis la 1re demande, pour ne pas laisser
    la session « Nouvelle session ». La logique (thinking coupé + tentatives) vit dans
    client.infer_title ; ici on ne garde QUE le repli sur le début du message si le modèle ne
    renvoie rien d'exploitable."""
    try:
        title = client.infer_title(model, message)
    except Exception:  # noqa: BLE001 - un titre est cosmétique, jamais bloquant
        title = ""
    if not title:
        title = message.strip().splitlines()[0][:48].strip() or "Session"
    return title


def _build_user_content(message, images, *, is_vision, stash_dir) -> str | list:
    """Construit le contenu du message user à partir de N images jointes (max MAX_IMAGES).



    - Modèle VISION : images EMBARQUÉES (data URI) dans un message multimodal — il les VOIT.

    - Modèle TEXTE-ONLY : images ENREGISTRÉES sur disque (stash_dir) ; le message reste du

      texte, avec les chemins + consigne d'inspecter via read_image (qui route vers un VLM).



    Lève ValueError (-> 400) si une image est trop grande ou n'est pas une image.

    """

    imgs = [im for im in (images or []) if im and im.filename][:MAX_IMAGES]

    if not imgs:
        return message

    read: list[tuple[str, str, bytes]] = []  # (nom, mime, octets)

    for im in imgs:
        blob = im.read()

        if len(blob) > MAX_IMAGE_BYTES:
            raise ValueError(f"image trop grande : {im.filename}")

        mime = im.mimetype or "image/png"

        if not mime.startswith("image/"):
            raise ValueError(f"fichier non-image : {im.filename}")

        read.append((im.filename, mime, blob))

    if is_vision:
        # EMBARQUÉES : le modèle multimodal les VOIT déjà. Sans ce rappel, il croit devoir

        # les rouvrir via read_image, devine un chemin, échoue.

        note = (
            f"[{len(read)} image(s) jointe(s) à ce message — tu les VOIS déjà directement "
            "ci-dessous. N'utilise PAS read_image pour elles, ne devine aucun chemin.]\n"
        )

        parts: list = [{"type": "text", "text": note + message}]

        for _name, mime, blob in read:
            b64 = base64.b64encode(blob).decode("ascii")

            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

        return parts

    # TEXTE-ONLY : on enregistre sur disque et on donne les chemins ; read_image (routé vers

    # un VLM) décrira à la demande. Le modèle ne voit rien inline, inutile de l'embarquer.

    stash_dir = Path(stash_dir)

    stash_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []

    for i, (name, _mime, blob) in enumerate(read):
        safe = re.sub(r"[^\w.\-]", "_", name) or f"image{i}"

        p = stash_dir / f"{i:02d}_{safe}"

        p.write_bytes(blob)

        paths.append(str(p.resolve()))

    listing = "\n".join(f"- {p}" for p in paths)

    note = (
        f"[{len(paths)} image(s) jointe(s) à ce message, enregistrée(s) sur disque. Ton "
        "modèle ne les voit pas directement : inspecte-les avec read_image(path[, question]) "
        "— un modèle vision te les décrira. Chemins :\n" + listing + "]\n"
    )

    return note + message


# Mots qui EFFACENT l'objectif de session via « /goal <mot> » (façon /goal de Claude Code).

_GOAL_CLEAR_WORDS = {
    "clear",
    "stop",
    "off",
    "none",
    "reset",
    "cancel",
    "efface",
    "annule",
}


def _init_message(target_display: str) -> str:
    """Consigne dépliée par la commande /init : fait explorer le dossier `target_display`
    et écrire `target_display/loom.md` (fiche projet). Fonction pure (testable)."""
    return (
        f"Génère la fiche projet du dossier de travail « {target_display} ». Explore-le "
        "avec tes outils (list_dir, find_files, read_file) — n'invente RIEN, base-toi "
        "seulement sur ce que tu lis vraiment. Repère : le but du projet, la "
        "stack/langages/frameworks, l'arborescence importante, comment l'installer / le "
        "lancer / le tester, et les conventions (lint, gestionnaire de paquets, CI/CD). "
        "Puis ÉCRIS le fichier avec write_file au chemin EXACT "
        f"« {target_display}/loom.md » (à la racine de CE dossier, nulle part ailleurs), "
        "en markdown structuré : titre du projet, puis les sections `## But`, `## Stack`, "
        "`## Arborescence`, `## Lancer / Tester`, `## Conventions`, `## Points d'attention`. "
        "Concis et factuel ; si une info manque, dis-le plutôt que de la deviner. Termine "
        "en confirmant le chemin écrit."
    )


# Verbe compact par outil pour la TRACE D'ACTIONS persistée (anti-amnésie). Les outils

# de navigation (find/search/list) en sont absents : on mémorise les LECTURES et les

# CHANGEMENTS d'état, pas les allers-retours d'exploration.

_TRACE_VERB = {
    "read_file": "lu",
    "read_document": "lu",
    "read_image": "vu",
    "write_file": "créé",
    "append_file": "complété",
    "edit_file": "modifié",
    "run_shell": "exécuté",
    "dispatch_agent": "délégué",
}

_WRITE_NAMES = {
    "write_file",
    "append_file",
    "edit_file",
}


def _action_trace_line(evt: dict) -> str | None:
    """Rend un `tool_result` en ligne compacte pour la trace, ou None s'il ne mérite pas

    d'être mémorisé (navigation, écriture échouée/différée)."""

    name = evt.get("name") or ""

    verb = _TRACE_VERB.get(name)

    if verb is None:
        return None

    ok = bool(evt.get("ok"))

    # Une écriture échouée/différée n'est pas un changement d'état à retenir (si elle

    # réussit ensuite dans le même tour, c'est cette réussite-là qui sera tracée).

    if name in _WRITE_NAMES and not ok:
        return None

    mark = "" if ok else "✗ "

    if name == "run_shell":
        head = (evt.get("preview") or "").split("\n")[0][:60]

        return f"{mark}{verb} shell: {head}".strip()

    if name == "dispatch_agent":
        return f"{mark}{verb} une sous-tâche"

    return f"{mark}{verb} {evt.get('path') or '?'}"


def create_app(
    client,
    skills_dir,
    session_store,
    *,
    max_tokens=2048,
    context_budget=3000,
    keep_recent=6,
    context_window=8192,
    models=None,
    vision_models=None,
    interrupt_wait=15.0,
    tool_factory=None,
    available_tools=None,
    permission=None,
    permission_mode="ask",
    confirm_timeout=300.0,
    workspace_dir=".",
    plugins_dir="loom/plugins",
    keepwarm_enabled=True,
    keepwarm_interval=150.0,
    identity_paths=None,
    identity_max_tokens=400,
    project_memory_max_tokens=600,
    learned_skills_dir=None,
    reflect_stores=None,
    reflect_enabled=False,
    reflect_min_actions=1,
    reflect_model=None,
    model_contexts=None,
    model_max_tokens=None,
    remote_model_ids=None,
    remote_model_names=None,
    model_prices=None,
    model_descriptions=None,
    remote_store_path=None,
    config_defaults_path=None,
    config_local_path=None,
    local_models=None,
    image_models=None,
) -> Flask:

    app = Flask(__name__)

    # Recharge le template à chaque requête : éditer index.html ne nécessite pas de

    # redémarrer le serveur (sinon Jinja sert la version compilée au démarrage).

    app.config["TEMPLATES_AUTO_RELOAD"] = True

    app.jinja_env.auto_reload = True

    # Pas de cache navigateur sur les statiques (app.js/css) : éditer le frontend prend effet

    # au simple rechargement, sans hard-refresh. Sinon un app.js mis à jour reste servi depuis

    # le cache et diverge du template rechargé côté serveur (bugs fantômes).

    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Windows sert les .js/.css statiques avec un mimetype tiré du registre, souvent SANS
    # `charset` -> le navigateur les décode en Windows-1252 et les glyphes UTF-8 deviennent
    # du mojibake (é -> Ã©, fleche -> â†', check -> âœ"). On force `charset=utf-8` sur toute
    # réponse textuelle qui n'en déclare pas, pour un décodage client correct quelle que
    # soit la source du mimetype.
    @app.after_request
    def _force_utf8_charset(response):
        ctype = response.headers.get("Content-Type", "")
        if "charset=" not in ctype.lower() and (
            ctype.startswith("text/")
            or "javascript" in ctype
            or "json" in ctype
            or "xml" in ctype
        ):
            response.headers["Content-Type"] = f"{ctype}; charset=utf-8"
        return response

    skills_dir = str(skills_dir)

    workspace_dir = str(workspace_dir)

    plugins_dir = str(plugins_dir)

    # Seuil de microcompact INTERNE à la boucle d'outils : on vide les vieux résultats

    # d'outils quand le contexte vivant approche la fenêtre du modèle (en réservant la place

    # de la réponse). Distinct du résumé inter-tours (context_budget) qui ne porte que sur

    # l'historique persisté. Calculé PAR MODÈLE (_model_limits) : un modèle distant à grande

    # fenêtre exploite SA fenêtre + son max_tokens au lieu du global, sans toucher au réglage

    # local. Repli sur le global pour tout id non listé.

    model_contexts = dict(model_contexts or {})

    model_max_tokens = dict(model_max_tokens or {})
    # Prix par modèle ($/M tokens) : id -> (input, output, cached). Local/absent -> (0,0,0).
    model_prices = dict(model_prices or {})
    # Rôle en une ligne par modèle -> infobulle (title) des options du sélecteur.
    model_descriptions = dict(model_descriptions or {})

    # Config VIVANTE de loom.web : au lieu de figer ces valeurs au démarrage, on les tient dans
    # un holder mutable que le runtime consulte À CHAQUE usage, et qu'on RECHARGE depuis le
    # disque après chaque édition (/config/set). Résultat : permissions, plafonds, budgets,
    # reflect et keep-warm prennent effet À CHAUD, sans redémarrer loom.web. (Les params du
    # SERVEUR MODÈLE restent hors de portée : autre process, cf. serve.py.)
    _settings = {
        "max_tokens": max_tokens,
        "context_budget": context_budget,
        "keep_recent": keep_recent,
        "identity_max_tokens": identity_max_tokens,
        "project_memory_max_tokens": project_memory_max_tokens,
        "reflect_enabled": reflect_enabled,
        "reflect_min_actions": reflect_min_actions,
        "keepwarm_enabled": keepwarm_enabled,
        "keepwarm_interval": keepwarm_interval,
        "permission_mode": permission_mode,
    }
    # La fonction de permission capture cfg.permissions ; pour appliquer un changement de mode
    # à chaud on la remplace dans ce holder (le runtime lit _perm["fn"]).
    _perm = {"fn": permission}

    def _reload_app_config():
        """Relit defaults.toml + local.toml et met à jour le holder + la permission (à chaud).
        Best-effort : une config invalide ne casse pas l'app en cours (on garde l'ancienne)."""
        if not (config_defaults_path and config_local_path):
            return
        try:
            from loom.agent.context import effective_context_budget
            from loom.config import load_config
            from loom.permissions import evaluate

            c = load_config(config_defaults_path, config_local_path)
        except Exception as e:  # noqa: BLE001 - reload best-effort, jamais fatal
            print(f"[loom] reload config échoué: {e}", flush=True)
            return
        _settings.update(
            max_tokens=c.chat.max_tokens,
            context_budget=effective_context_budget(
                c.chat.context_token_budget, c.context, c.chat.max_tokens
            ),
            keep_recent=c.chat.keep_recent_messages,
            identity_max_tokens=c.chat.identity_max_tokens,
            project_memory_max_tokens=c.chat.project_memory_max_tokens,
            reflect_enabled=c.chat.reflect_enabled,
            reflect_min_actions=c.chat.reflect_min_actions,
            keepwarm_enabled=c.chat.keepwarm_enabled,
            keepwarm_interval=c.chat.keepwarm_interval,
            permission_mode=c.permissions.mode,
        )
        _perm["fn"] = lambda name, args: evaluate(name, args, c.permissions)

    def _regen_swap_yaml():
        """Régénère le llama-swap.yaml depuis la config (llama-swap -watch-config le recharge).
        Best-effort, silencieux si serve indispo. Renvoie True si écrit."""
        if not (config_defaults_path and config_local_path):
            return False
        try:
            from loom.runtime.serve import regenerate_swap_yaml

            return bool(regenerate_swap_yaml(config_defaults_path, config_local_path))
        except Exception as e:  # noqa: BLE001
            print(f"[loom] regen swap yaml échoué: {e}", flush=True)
            return False

    def _apply_to_model_server(section):
        """Param SERVEUR/OVERRIDE (affecte le lancement de llama-server) : régénère le yaml et
        décharge les modèles locaux -> ils se relancent avec les nouveaux args au prochain usage
        (llama-swap -watch-config). Ne touche pas les autres process. Best-effort, en tâche de fond."""
        if section not in ("server", "override"):
            return
        if _regen_swap_yaml():
            threading.Thread(
                target=client.unload_local, daemon=True, name="loom-reload-models"
            ).start()

    def _price_of(model_id):
        return model_prices.get(model_id, (0.0, 0.0, 0.0))

    def _ctx_info(model_id):
        """(fenêtre de contexte, source) du modèle -> dénominateur de la jauge + provenance.

        Distant : on demande D'ABORD au PROVIDER (`client.remote_context`, mis en cache) —
        c'est le modèle lui-même qui fait autorité. S'il ne publie rien (Z.ai/OpenAI), repli
        sur la valeur déclarée en config. Local : la fenêtre est celle qu'on a ALLOUÉE au
        serveur (n_ctx) = notre limite volontaire, signalée comme telle. Sources possibles :
        `provider` (fait autorité), `config` (déclaré, non vérifiable), `local` (notre limite)."""
        declared = model_contexts.get(model_id) or context_window
        if model_id in remote_model_ids:
            provided = client.remote_context(model_id)
            if provided:
                return provided, "provider"
            return declared, "config"
        return declared, "local"

    def _model_limits(model_id):
        """(plafond de sortie, seuil de microcompact) pour `model_id`.

        Le max_tokens global est une contrainte LOCALE (calibrée pour la VRAM de la machine).
        Un modèle DISTANT ne l'hérite PAS : sa machine est plus puissante. Non défini -> None
        (plafond OMIS dans la requête, le provider applique SA limite). La réserve de
        microcompact reste modeste côté distant (leur fenêtre est large, le seuil compte peu)."""
        win = model_contexts.get(model_id) or context_window
        explicit = model_max_tokens.get(model_id)
        if model_id in remote_model_ids:
            cap = explicit  # None possible -> pas de cap imposé
            reserve = explicit or 8192
        else:
            cap = explicit or _settings["max_tokens"]  # local : plafond global
            reserve = cap
        return cap, max(1024, win - reserve - 1024)

    def _totals(conv):
        """Compteurs de session + fenêtre du modèle (jauge de remplissage du contexte).
        La fenêtre dépend du modèle (que l'app connaît), pas de la Conversation -> jointe ici,
        avec sa source (provider/config/local) pour que l'UI signale si le chiffre fait autorité."""
        win, src = _ctx_info(conv.model)
        return {**conv.usage_totals(), "context_window": win, "context_source": src}

    models = list(models or [])
    # Sérialise TOUTES les écritures de fichiers de config/modèles (model.toml, local.toml,
    # defaults.toml, store JSON) : Flask est threaded -> deux éditions concurrentes du même
    # fichier feraient une course read-modify-write (la dernière écrase l'autre).
    _toml_lock = threading.Lock()

    vision_models = set(vision_models or [])  # ids des modèles avec mmproj (vision)

    remote_model_ids = set(remote_model_ids or [])  # ids servis par une API distante
    # ids des modèles LOCAUX (servis par llama-swap sur la machine) = tout sauf les distants.
    # Sert à /machine_state (quel modèle machine est chargé).
    local_model_ids = [m for m in (models or []) if m not in remote_model_ids]

    # Modèles IMAGE (ComfyUI) : troisième type après local/distant. Sélectionnables comme
    # les autres ; un message user = un prompt d'image = une image dans le chat. Le moteur
    # ComfyUI est un processus externe géré (kill-on-close), un par (dir, port) — en
    # pratique un seul. AUCUNE dépendance Python côté Loom : HTTP seulement.
    image_models = list(image_models or [])
    image_model_ids = {m.id for m in image_models}
    # Sous-ensemble VIDÉO (découverts sous local/video) : même runtime, mais le
    # sélecteur UI les préfixe `video ·` au lieu de `image ·`.
    video_model_ids = {m.id for m in image_models if m.kind == "video"}
    _image_by_id = {m.id: m for m in image_models}
    models = list(models or []) + [m.id for m in image_models]
    _engines: dict[tuple, ComfyEngine] = {}
    _engines_lock = threading.Lock()

    def _engine_for(im) -> ComfyEngine:
        key = (im.comfy_dir, im.comfy_port)
        with _engines_lock:
            if key not in _engines:
                _engines[key] = ComfyEngine(im.comfy_dir, im.comfy_port)
            return _engines[key]

    # Marge RAM (Mo) gardée libre AU-DELÀ du LLM à charger : OS, cache KV, pics
    # transitoires. En dessous, on ne garde pas le cache image (jamais d'OOM pour
    # une optimisation de confort).
    _RAM_KEEP_MARGIN_MB = 4096

    def _free_image_engines(llm_size_mb: int = 0) -> bool | None:
        """Rend la VRAM tenue par un moteur image (best-effort, rapide) : appelé avant
        une génération LOCALE — 6 Go ne tiennent pas la diffusion ET le LLM.

        La RAM, elle, est arbitrée : si le LLM entrant tient À CÔTÉ du cache image
        (RAM disponible mesurée >= size_mb du LLM + marge), on garde le cache
        (keep_ram) — la prochaine image repart de la RAM, pas du disque. Machine
        étroite (ex. 32 Go) ou taille inconnue -> cache vidé, comportement historique.
        Renvoie True (cache gardé), False (cache vidé) ou None (aucun moteur actif)."""
        with _engines_lock:
            engines = list(_engines.values())
        up = [eng for eng in engines if eng.is_up(timeout=0.5)]
        if not up:
            return None
        keep = bool(llm_size_mb) and ram_available_mb() >= (
            llm_size_mb + _RAM_KEEP_MARGIN_MB
        )
        for eng in up:
            eng.free(keep_ram=keep)
        return keep

    def _local_size_mb(mid) -> int:
        """size_mb (model.toml) d'un modèle local, 0 si inconnu."""
        spec = next((m for m in local_model_specs if m.get("id") == mid), None)
        return int(spec.get("size_mb") or 0) if spec else 0

    _generated_dir = (
        Path(remote_store_path).resolve().parent / "generated"
        if remote_store_path
        else Path("var") / "generated"
    )
    # Détails des modèles LOCAUX (onglet Modèles locaux) : id/dir/offload/context. `dir` porte
    # le model.toml -> édition du tuning machine via tomlkit.
    local_model_specs = list(local_models or [])

    # Serveur modèle GÉRÉ par loom.web : démarré à la demande (sélection d'un modèle local,
    # /chat, bouton « démarrer ») comme ENFANT du process -> il meurt avec loom.web (Job
    # Object kill-on-close), et l'UI a un bouton « éteindre » pour libérer les ressources.
    server_manager = ModelServerManager()

    def _ensure_local_server(wait: float = 0.0) -> bool:
        """Serveur modèle joignable ? Sinon DÉMARRAGE AUTO, puis attente bornée à `wait` s.
        GGUF déjà présents -> llama-swap répond en ~1-2 s ; un premier téléchargement peut
        dépasser `wait` (pas grave : l'UI suit l'état via /machine_state)."""
        reachable, _ = client.running_local(timeout=2.0)
        if reachable:
            return True
        server_manager.start()
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            time.sleep(0.7)
            reachable, _ = client.running_local(timeout=2.0)
            if reachable:
                return True
        return False

    remote_model_names = dict(
        remote_model_names or {}
    )  # id Loom -> vrai modèle provider

    available_tools = list(available_tools or [])

    # Concurrence PAR SESSION : chaque session a son propre verrou de génération et son
    # signal d'annulation -> plusieurs sessions (onglets) génèrent EN PARALLÈLE. Une nouvelle
    # soumission n'interrompt QUE la génération de SA session (pas les autres). Verrou GLOBAL
    # `_local_gen_lock` en plus pour les modèles LOCAUX : llama-swap n'en sert qu'un à la fois
    # -> deux générations locales se sérialisent (limitation machine connue, signalée à l'UI).
    _gen_guard = threading.Lock()
    _sess_locks: dict[str, threading.Lock] = {}
    _sess_cancel: dict[str, threading.Event] = {}
    _local_gen_lock = threading.Lock()
    # Garde-éveil : tant qu'une génération tourne, on empêche la VEILLE du système (l'écran
    # peut s'éteindre) -> le travail continue en arrière-plan au lieu de geler à la mise en
    # veille (« network error »). No-op hors Windows.
    _stay_awake = StayAwake()
    # Événement d'annulation de la génération EN COURS sur CE thread (pour _confirm, qui tourne
    # dans le thread de génération) — posé au début de /chat.
    _confirm_local = threading.local()

    def _lock_for(sid: str) -> threading.Lock:
        with _gen_guard:
            return _sess_locks.setdefault(sid, threading.Lock())

    def _cancel_for(sid: str) -> threading.Event:
        with _gen_guard:
            return _sess_cancel.setdefault(sid, threading.Event())

    # Décisions de confirmation en attente : tool_call_id -> {event, approved}.

    # Renseignées par la route /tool_decision (autre thread), consommées par _confirm.

    pending: dict = {}

    # Horodatage de la dernière fin de génération (0 = jamais). Le keep-warm ne pinge

    # qu'après une vraie activité et seulement à l'idle (cf. thread plus bas).

    _last_activity = [0.0]

    # Session active : un fil persistant par projet. Tout passe par la session courante

    # (conversation + persistance) ; un seul mode, plus de legacy.

    _cur: dict = {"session": None}

    # Cache d'OBJETS session en mémoire : UNE instance par session, partagée entre requêtes
    # (onglets) -> pas de sauvegardes qui se clobberent quand plusieurs sessions tournent en
    # parallèle. `_cur` pointe la session FOCUS (défaut de l'index).
    _sessions_cache: dict = {}

    def _ensure_model(sess):
        # Une session neuve peut naître sans modèle -> requête model="" -> llama-swap renvoie
        # 404. On garantit un modèle valide (le 1er = défaut) ; corrige aussi les vides.
        if sess is not None and not sess.conversation.model and models:
            sess.conversation.set_model(models[0])
            session_store.save(sess)
        return sess

    def _get_session(sid: str):
        """Session par id, depuis le cache (une instance) ou chargée du disque. None si absente."""
        if not sid:
            return None
        with _gen_guard:
            s = _sessions_cache.get(sid)
        if s is None:
            s = session_store.load(sid)
            if s is not None:
                with _gen_guard:
                    s = _sessions_cache.setdefault(sid, s)
        return _ensure_model(s)

    def _session():
        cur = _cur["session"]
        if cur is None:
            cur = session_store.active() or session_store.create(
                workspace=workspace_dir
            )
            with _gen_guard:
                cur = _sessions_cache.setdefault(cur.id, cur)
            _cur["session"] = cur
        return _ensure_model(cur)

    def _ctx():
        """Renvoie (conversation, save) : la conversation de la session active et sa

        persistance. Point de vérité unique pour tous les endpoints."""

        sess = _session()

        return sess.conversation, (lambda: session_store.save(sess))

    def _confirm(tool_id: str, name: str, args: dict) -> bool:
        """Bloque jusqu'à la décision UI (OK/Refuser). Interruptible et borné.



        Renvoie False si refus, timeout, ou si une nouvelle soumission annule

        (cancel_event) — évite tout deadlock sur le verrou de chat.

        """

        ev = threading.Event()

        pending[tool_id] = {"event": ev, "approved": False}

        deadline = time.monotonic() + confirm_timeout

        # Annulation de LA session dont on exécute la génération (thread-local, posé par /chat).
        cancel_ev = getattr(_confirm_local, "ev", None)

        try:
            while not ev.wait(0.2):
                if (cancel_ev is not None and cancel_ev.is_set()) or (
                    time.monotonic() > deadline
                ):
                    return False

            return bool(pending[tool_id]["approved"])

        finally:
            pending.pop(tool_id, None)

    def _all_skills() -> list:
        return collect_skills(skills_dir, plugins_dir, learned_dir=learned_skills_dir)

    def _skills_ctx(conv) -> dict:
        """Contexte du panneau Skills : la liste COMPLÈTE (pour les cases), l'ensemble des
        skills ACTIFS (non désactivés) et ceux qui ont un override de session (badge UI)."""
        skills = _all_skills()
        disabled = set(conv.disabled_skills)
        return {
            "skills": skills,
            "active_skills": [s.name for s in skills if s.name not in disabled],
            "overridden_skills": [
                s.name for s in skills if s.name in conv.skill_overrides
            ],
        }

    def _index_context() -> dict:

        sess = _session()

        conv = sess.conversation

        ws = sess.workspace

        sessions = [
            {"id": m.id, "title": m.title, "workspace": m.workspace}
            for m in session_store.list()
        ]

        active_id = sess.id

        return {
            "messages": conv.messages,
            **_skills_ctx(conv),
            "usage_totals": _totals(conv),
            "models": models,
            "remote_model_ids": remote_model_ids,
            "image_model_ids": image_model_ids,
            "video_model_ids": video_model_ids,
            "model_descriptions": model_descriptions,
            "current_model": conv.model,
            "thinking": conv.thinking,
            "available_tools": available_tools,
            "active_tools": conv.active_tools,
            "workspace_dir": ws,
            "sessions": sessions,
            "active_session": active_id,
            "permission_mode": _settings["permission_mode"],
            # État initial pour l'hydratation côté client (Preact). On échappe '<'
            # pour ne pas pouvoir fermer la balise <script> depuis le contenu.
            "init_json": json.dumps(
                {
                    "messages": conv.messages,
                    "thinking": conv.thinking,
                    "usage_totals": _totals(conv),
                    # Onglet initial : la session active (id/titre/modèle/workspace) + toutes
                    # les sessions (pour la sidebar). Le multi-onglets s'hydrate là-dessus.
                    "active_session": active_id,
                    "title": sess.title,
                    "model": conv.model,
                    "workspace": ws,
                    "sessions": sessions,
                },
                ensure_ascii=False,
            ).replace("<", "\\u003c"),
        }

    # Garde CSRF : le serveur écoute sur 127.0.0.1 SANS auth, et tourne souvent en

    # mode=allow (outils exécutés sans confirmation). Une page web tierce OUVERTE dans le

    # navigateur de l'utilisateur peut POSTer en cross-origin vers 127.0.0.1 (requêtes

    # « simples », sans preflight) et piloter l'agent local -> exécution d'outils. Le

    # binding localhost NE protège PAS de ça. On refuse les POST dont l'en-tête

    # Sec-Fetch-Site (envoyé par tous les navigateurs modernes) trahit une origine tierce.

    # `same-origin`/`none` (notre propre page, barre d'adresse) passent ; un client non-

    # navigateur (curl, tests) n'envoie pas l'en-tête -> autorisé, on ne casse rien.

    @app.before_request
    def _csrf_guard():

        if request.method != "POST":
            return None

        if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            return Response("requête cross-origin refusée (CSRF)", status=403)

        return None

    @app.get("/")
    def index() -> str:

        return render_template("index.html", **_index_context())

    @app.get("/genimg/<sid>/<name>")
    def genimg_session(sid: str, name: str):
        # Sert les médias générés depuis le dossier de LA session (unique copie).
        # `sid` compose le CHEMIN de base : validé strictement (12 hex, format
        # uuid4.hex[:12] de SessionStore.create) sinon 404 — un sid forgé (`..`,
        # antislash Windows) ferait pointer la base hors de var/sessions.
        # send_from_directory protège `name` ; mimetype déduit du nom (png/webm).
        from flask import send_from_directory

        if not re.fullmatch(r"[0-9a-f]{12}", sid):
            return Response("session invalide", status=404)
        return send_from_directory(session_store.root / sid / "generated", name)

    @app.get("/genimg/<name>")
    def genimg(name: str):
        # LEGACY : messages d'avant 2026-07-09, servis depuis var/generated.
        from flask import send_from_directory

        return send_from_directory(_generated_dir, name)

    @app.get("/favicon.ico")
    def favicon():
        # Requête par défaut du navigateur (silence le 404) : on sert le SVG de la trame.
        # Le <link rel="icon" type="image/svg+xml"> reste la source primaire de l'onglet.
        from flask import send_from_directory

        return send_from_directory(
            app.static_folder, "favicon.svg", mimetype="image/svg+xml"
        )

    @app.post("/reset")
    def reset() -> str:

        conv, save = _ctx()

        conv.reset()

        save()

        # Le fil repart à neuf -> on efface aussi le journal d'affichage temps réel.
        session_store.clear_timeline(_session().id)

        return render_template("index.html", **_index_context())

    def _handle_goal_command(message, conv, save, chat_lock):
        """Traite la commande /goal : pose/statut/efface l'objectif de session.

        Retourne (message, response) : response non None = ack immédiat (return direct
        dans chat()) ; response None = continuer le flux normal (message éventuellement
        réécrit en consigne de démarrage)."""

        if message == "/goal" or message.startswith("/goal "):
            arg = message[len("/goal") :].strip()
            if arg and arg.lower() not in _GOAL_CLEAR_WORDS:
                # Pose l'objectif et AMORCE le travail : on remplace le message par une consigne
                # de démarrage et on laisse le flux normal tourner, objectif désormais actif.
                conv.set_goal(arg)
                save()
                message = (
                    f"Objectif à atteindre : {arg}\n"
                    "Commence MAINTENANT à agir pour l'atteindre, et PROUVE-le (exécute, montre "
                    "la sortie réelle). Ne t'arrête pas tant qu'il n'est pas démontré atteint."
                )
                # (pas de return : on tombe dans la génération normale ci-dessous)
            else:
                if not arg:
                    ack = (
                        f"Objectif courant : « {conv.goal} » (actif jusqu'à preuve d'atteinte, "
                        "/goal clear pour l'effacer)."
                        if conv.goal
                        else "Aucun objectif actif. Pose-en un : /goal <condition vérifiable>."
                    )
                else:
                    conv.set_goal("")
                    save()
                    ack = "Objectif effacé - retour au mode normal (arrêt au stop naturel)."
                chat_lock.release()

                def _goal_ack():
                    yield _sse("text", text=ack)
                    yield _sse("done")

                return message, Response(_goal_ack(), mimetype="text/event-stream")
        return message, None

    def _handle_init_command(message):
        """Traite /init : adopte un dossier cible si fourni, et réécrit le message en consigne
        de génération de fiche projet. Retourne le message (éventuellement réécrit)."""
        if message == "/init" or message.startswith("/init "):
            arg = message[len("/init") :].strip()
            _sess = _session()
            target_dir = _sess.workspace
            if arg:
                cand = Path(arg).expanduser()
                if cand.is_dir():
                    target_dir = str(cand.resolve())
                    if target_dir != _sess.workspace:
                        _sess.workspace = target_dir
                        session_store.save(_sess)
            target_display = str(Path(target_dir)).replace("\\", "/")
            message = _init_message(target_display)
            # (pas de return : le flux normal ci-dessous exécute la consigne)
        return message

    def _build_system_prompt(conv):
        """Construit le system prompt complet : identité always-on + base (strong/local) +
        catalogue des skills + déclaration du moteur + conventions OS + dossier de travail +
        objectif de session. Retourne (system_prompt, strong)."""

        skills = effective_skills(
            collect_skills(skills_dir, plugins_dir, learned_dir=learned_skills_dir),
            overrides=conv.skill_overrides,
            disabled=conv.disabled_skills,
        )

        catalog = render_catalog(skills)

        # Identité always-on (SOUL/USER/MEMORY) EN TÊTE : c'est la définition qui FAIT FOI
        # de qui est Loom (rôle, persona, style). Le mode d'emploi opérationnel (outils,
        # règles) de chat.system.md vient APRÈS et s'y conforme - on ne plante plus un
        # cadrage générique d'abord pour le corriger 12k caractères plus loin. Always-on =>
        # survit toujours à la microcompaction/summarization (qui ne touchent que
        # l'historique). Bornée par identity_max_tokens. Cf. design §5.6.
        _idblk = ""

        if identity_paths:
            from loom.memory.identity import identity_block

            _idblk = identity_block(
                identity_paths["soul_path"],
                identity_paths["user_path"],
                identity_paths["memory_md_path"],
                max_tokens=_settings["identity_max_tokens"],
            )

        # TIER du harnais : un modèle DISTANT (API, non quantifié) se pilote seul -> prompt
        # ALLÉGÉ (identité + outils + mémoire + sécurité), sans le scaffolding de comportement
        # de chat.system.md qui ne sert qu'à un petit modèle local. Le flag `strong` sert
        # aussi (plus bas) à couper les gardes de comportement dans la boucle d'outils.
        strong = bool(conv.model and conv.model in remote_model_ids)

        base_prompt = CHAT_SYSTEM_STRONG if strong else conv.system_prompt

        system_prompt = f"{_idblk}\n\n{base_prompt}" if _idblk else base_prompt

        # Distant (strong) : la machine du provider encaisse le parallélisme -> on incite à
        # GROUPER les sous-agents indépendants dans un même tour (ils tournent en parallèle).
        if strong:
            system_prompt += (
                "\n\nParallélisme : quand plusieurs sous-tâches sont INDÉPENDANTES (auditer/"
                "explorer des pans distincts), émets PLUSIEURS dispatch_agent dans le MÊME "
                "tour - ils s'exécutent EN PARALLÈLE, bien plus vite qu'un par tour. Un pan = "
                "un agent, lance-les ensemble."
            )

        if catalog:
            system_prompt += f"\n\n{catalog}"

        # Le modèle ignore par défaut sous quel backend il tourne (le prompt dit
        # "Tu es Loom") -> il baratine quand on lui demande "quel modèle ?". On lui
        # injecte son modèle courant pour qu'il réponde honnêtement. DISTANT vs LOCAL :
        # sans ça un modèle servi par une API répétait « je tourne en local/offline sur
        # llama.cpp » (la persona de Loom est « agent local ») -> confabulation d'infra.
        if conv.model:
            if conv.model in remote_model_ids:
                _pm = remote_model_names.get(conv.model)

                _label = (
                    f"« {_pm} » (route « {conv.model} »)"
                    if _pm
                    else f"« {conv.model} »"
                )

                system_prompt += (
                    f"\n\n# Ton moteur\nTon raisonnement est servi par le modèle DISTANT "
                    f"{_label}, via une API externe - PAS en local. Tes OUTILS, eux, "
                    "s'exécutent bien sur la machine de l'utilisateur, mais toi (le cerveau) "
                    "non. Ne prétends donc JAMAIS être offline, ni tourner sur llama.cpp / "
                    "llama-swap / une carte graphique locale : ce serait faux. Si on te "
                    "demande quel modèle/moteur tu utilises, donne ce nom honnêtement, sans "
                    "inventer de détails d'infrastructure."
                )

            else:
                system_prompt += (
                    f"\n\n# Ton moteur\nTu tournes sur le modèle local « {conv.model} ». "
                    "Si on te demande quel modèle/moteur tu utilises, réponds-le "
                    "honnêtement et directement (ce nom), sans esquiver."
                )

        # Système : Loom détecte SEUL l'OS et injecte ses conventions (shell, commandes,
        # chemins) -> le modèle produit du PowerShell sous Windows, du bash/unix sous
        # macOS/Linux, sans qu'on code l'OS en dur dans le prompt. Source unique partagée
        # avec run_shell (loom.runtime.platform_info) : jamais de divergence.
        system_prompt += "\n\n" + platform_detect().prompt_block()

        # Dossier de travail courant : le modèle l'IGNORE sinon et le devine en sondant
        # (git rev-parse à l'aveugle, list_dir…) -> tours gaspillés. On le lui dit, avec
        # le réflexe anti-tâtonnement quand ce dossier n'est pas un repo git. Reste EN BAS
        # (contexte volatil, près de l'action).
        _ws = _session().workspace

        system_prompt += (
            f"\n\n# Dossier de travail courant\nTes commandes (run_shell) tournent dans "
            f"`{_ws}` et les chemins relatifs s'y résolvent - n'y répète pas le nom de ce "
            "dossier dans tes chemins. Si une commande git échoue par « not a git "
            "repository », c'est que CE dossier n'est pas un repo : fais UN list_dir pour "
            "repérer le bon sous-dossier (puis `git -C <sous-dossier>`), ne relance pas la "
            "même commande à l'identique."
        )

        # Fiche projet auto-injectée : si `<workspace>/loom.md` existe (générée par /init),
        # le modèle la reçoit d'office au lieu de re-sonder le projet à chaque session.
        # Cache mtime (read_md) -> préfixe stable en session = prompt caching préservé ;
        # suit le workspace courant (adoption/changement en cours de session). Les DEUX
        # tiers la reçoivent (c'est de la mémoire, pas du scaffolding de comportement).
        # L'en-tête la cadre en CONTEXTE (pas instructions, possiblement périmée) : une
        # fiche est écrite en lisant le projet, un repo piégé ne doit pas pouvoir élever
        # ses consignes au rang de system prompt.
        from loom.memory.identity import project_block

        _pm_blk = project_block(_ws, max_tokens=_settings["project_memory_max_tokens"])

        if _pm_blk:
            system_prompt += f"\n\n{_pm_blk}"

        # Objectif de session (/goal), en DIRECTIVE DOUCE : pas de juge externe qui te
        # contredit (retiré - il recalait des preuves correctes). Tu restes seul maître de
        # ta propre vérification : ne te déclare pas fini tant que l'objectif n'est pas
        # ATTEINT ET PROUVÉ par tes exécutions (montre la sortie réelle) ; une fois prouvé,
        # dis-le et arrête-toi. L'utilisateur l'efface avec « /goal clear ».
        if conv.goal:
            system_prompt += (
                f"\n\n# Objectif de session\nTant qu'il est actif, oriente ton travail vers "
                f"cet objectif et ne le déclare atteint qu'une fois PROUVÉ par tes propres "
                f"exécutions (sortie réelle affichée) :\n{conv.goal}"
            )

        return system_prompt, strong

    @app.post("/chat")
    def chat():

        message = (request.form.get("message") or "").strip()

        if not message or len(message) > 5000:
            return Response("message invalide", status=400)

        # Session CIBLE : par `session_id` (onglet) sinon la session focus. Chaque session a
        # son verrou : une nouvelle soumission n'interrompt QUE la génération de SA session,
        # les autres onglets continuent en parallèle.
        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(req_sid) or _session()
        _cur["session"] = sess  # focus (défaut de l'index)
        sid = sess.id
        chat_lock = _lock_for(sid)
        cancel_event = _cancel_for(sid)

        if not chat_lock.acquire(blocking=False):
            # Une génération de CETTE session tourne déjà : on l'annule et on attend le verrou.

            cancel_event.set()

            if not chat_lock.acquire(timeout=interrupt_wait):
                return Response("occupé : cette session génère déjà", status=429)

        # On tient le verrou : repartir d'un signal d'annulation propre.

        cancel_event.clear()

        conv = sess.conversation
        save = lambda: session_store.save(sess)  # noqa: E731

        # Commande /goal : pilote l'OBJECTIF de complétion de la session. La logique
        # (pose/statut/efface) est factorisée dans _handle_goal_command, qui renvoie
        # (message, response) : response non None = ack immédiat à retourner directement.
        message, _goal_resp = _handle_goal_command(message, conv, save, chat_lock)
        if _goal_resp is not None:
            return _goal_resp

        # Commande /init : génère une fiche projet `loom.md` À LA RACINE DU DOSSIER
        # de TRAVAIL de la session. Factorisé dans _handle_init_command (adopte un
        # dossier cible si fourni, réécrit le message en consigne de génération).
        message = _handle_init_command(message)

        # Plus de garde bloquant : un modèle texte-only ne reçoit PAS l'image inline (qui

        # ferait planter un llama-server sans mmproj) — on la stocke sur disque et il l'inspecte

        # via read_image, routé vers un modèle vision (cf. _build_user_content plus bas).

        # Logs PAR SESSION (au même titre que session.json) : (1) trace des échanges modèle

        # routée vers sessions/<id>/debug.log ; (2) copie du log serveur modèle global

        # (var/logs/serve.log) dans la session — doublon assumé, pour tout avoir sous la main.

        _sdir = session_store.session_dir(_session().id)

        set_debug_log_path(_sdir / "debug.log")

        _serve_log = session_store.root.parent / "logs" / "serve.log"

        if _serve_log.exists():
            try:
                _sdir.mkdir(parents=True, exist_ok=True)

                shutil.copyfile(_serve_log, _sdir / "serve.log")

            except OSError:
                pass

        # Auto-adoption du dossier de travail : si le message désigne un dossier EXISTANT,

        # la session l'adopte avant le tour -> run_shell tourne dedans et les chemins

        # relatifs s'y résolvent, sans que l'utilisateur ait à pointer le dossier dans l'UI.

        adopted_ws = None

        detected = _detect_workspace(message, workspace_dir)

        if detected:
            sess = _session()

            if detected != sess.workspace:
                sess.workspace = detected

                session_store.save(sess)

                adopted_ws = detected

        try:
            content = _build_user_content(
                message,
                request.files.getlist("image"),
                is_vision=bool(conv.model and conv.model in vision_models),
                stash_dir=_sdir / "uploads",
            )

            conv.add("user", content)

            save()

            # Journal d'affichage temps réel : on y consigne le message user (le journal est la
            # source de RÉ-AFFICHAGE au rechargement -> il doit être complet, user inclus).
            session_store.append_event(sess.id, "user", {"content": message})

            # Résumé auto pré-tour : DÉPLACÉ dans generate() (plus bas) pour être VISIBLE
            # dans le stream (label d'activité « compaction… ») au lieu d'un blocage muet
            # avant le 1er octet. Le prompt système ne dépend pas de l'historique -> on le
            # construit ici sans attendre le résumé.
            system_prompt, strong = _build_system_prompt(conv)
        except ValueError as exc:
            chat_lock.release()

            return Response(str(exc), status=400)

        except Exception:  # noqa: BLE001
            chat_lock.release()
            traceback.print_exc()
            return Response("erreur interne", status=500)

        # --- Modèle IMAGE sélectionné : court-circuit de la boucle tool-use. Un message
        # user = un prompt d'image = une image dans la conversation. Même GPU que le LLM
        # local -> même sérialisation (_local_gen_lock) ; VRAM libérée (unload_local)
        # avant la diffusion ; erreurs TOUJOURS lisibles (patron « génération interrompue »).
        if conv.model in image_model_ids:
            _im = _image_by_id[conv.model]

            def generate_image():
                _stay_awake.acquire()
                _img_held = False
                _sess = _session()

                def _finish(md_text: str):
                    conv.add("assistant", md_text)
                    save()
                    session_store.append_event(_sess.id, "text", {"text": md_text})
                    return _sse("text", text=md_text)

                try:
                    yield _sse("status", label="préparation du moteur image…")
                    _local_gen_lock.acquire()
                    _img_held = True
                    # Photo d'ENTRÉE (modèles d'édition, ex. Kontext) : un chemin de
                    # fichier image dans le message est détecté, vérifié sur disque,
                    # retiré du texte (le prompt ne doit porter que l'instruction) et
                    # transmis au moteur ({IMAGE} du workflow). Chemins avec espaces :
                    # entre guillemets.
                    src_image, msg_text = None, message
                    for cand in re.findall(
                        r'"([^"]+\.(?:png|jpe?g|webp|bmp))"', message, re.IGNORECASE
                    ) + re.findall(
                        r"[A-Za-z]:[\\/][^\s\"']+\.(?:png|jpe?g|webp|bmp)",
                        message,
                        re.IGNORECASE,
                    ):
                        if Path(cand).is_file():
                            src_image = cand
                            msg_text = (
                                message.replace(f'"{cand}"', " ")
                                .replace(cand, " ")
                                .strip()
                            )
                            break
                    # Affinage du prompt (best-effort, JAMAIS bloquant) : le refiner
                    # déclaré par le modèle image (model.toml, id d'un modèle Loom)
                    # réécrit la demande — quelle que soit la langue — en prompt de
                    # diffusion anglais. Séquence VRAM sûre : le refiner est servi par
                    # llama-swap D'ABORD, puis déchargé (unload_local ci-dessous) —
                    # LLM et diffusion ne co-résident jamais.
                    prompt, refined = msg_text, False
                    if _im.refiner and _im.refiner in models:
                        yield _sse(
                            "status", label=f"affinage du prompt ({_im.refiner})…"
                        )
                        try:
                            if _im.refiner in remote_model_ids or _ensure_local_server(
                                wait=90.0
                            ):
                                # Édition d'une photo : le refiner doit produire une
                                # INSTRUCTION (quoi changer / quoi garder), pas une
                                # description de scène — on le lui dit dans le message.
                                _refine_in = (
                                    "[An input photo is attached; write an EDIT "
                                    "instruction: what to change, what must stay "
                                    f"identical.] {msg_text}"
                                    if src_image
                                    else msg_text
                                )
                                # Prompt système = règles générales + grammaire propre
                                # au générateur (refine_hints du model.toml) : chaque
                                # modèle a son style de prompt optimal.
                                _refine_sys = IMAGE_REFINE_SYSTEM
                                if _im.refine_hints:
                                    _refine_sys += (
                                        "\n\nTARGET-MODEL RULES (override the above "
                                        "when they conflict):\n" + _im.refine_hints
                                    )
                                out = ""
                                for kind, chunk in client.stream_chat(
                                    [{"role": "user", "content": _refine_in}],
                                    _refine_sys,
                                    max_tokens=512,
                                    model=_im.refiner,
                                    thinking=False,
                                ):
                                    if kind == "content":
                                        out += chunk
                                out = " ".join(out.split()).strip().strip('"')
                                if out:
                                    prompt, refined = out, True
                        except Exception:  # noqa: BLE001 - affinage best-effort
                            traceback.print_exc()
                        if not refined:
                            yield _sse(
                                "notice",
                                text="affinage indisponible — prompt envoyé tel quel.",
                            )
                    # FORMAT dynamique : le refiner termine par un tag
                    # [format: portrait|landscape|square] dérivé de la demande —
                    # extrait ici, retiré du prompt, converti en résolution. Absent
                    # -> dimensions du model.toml (les workflows sans {WIDTH}/{HEIGHT}
                    # ignorent simplement ces valeurs).
                    gen_w, gen_h = _im.width, _im.height
                    # Tag TOUJOURS retiré quelle que soit sa valeur (un petit modèle
                    # invente parfois la sienne, ex. "full-body" : elle ne doit JAMAIS
                    # fuir dans le prompt de diffusion) ; synonymes mappés, inconnu ->
                    # dimensions par défaut.
                    _fmt = re.search(
                        r"\[\s*format\s*:\s*([a-zà-ÿ -]+?)\s*\]\s*$",
                        prompt,
                        re.IGNORECASE,
                    )
                    if _fmt:
                        prompt = prompt[: _fmt.start()].strip()
                        _f = _fmt.group(1).lower()
                        if any(
                            k in _f
                            for k in ("portrait", "full-body", "full body", "vertical")
                        ):
                            gen_w, gen_h = 832, 1216
                        elif any(
                            k in _f
                            for k in ("landscape", "paysage", "wide", "horizontal")
                        ):
                            gen_w, gen_h = 1216, 832
                        elif any(k in _f for k in ("square", "carr")):
                            gen_w, gen_h = 1024, 1024
                    # Titre de session : une session image/vidéo mérite un nom comme
                    # les autres. Inféré par le REFINER (encore résident — coût nul en
                    # rechargement) ; sans refiner, _infer_title retombe sur le début
                    # du message. Fait AVANT unload_local.
                    if _sess.title == "Nouvelle session":
                        _title = _infer_title(client, _im.refiner or None, msg_text)
                        if _title:
                            _sess.title = _title
                            session_store.save(_sess)
                            yield _sse("session_title", id=_sess.id, title=_title)
                    client.unload_local()  # VRAM libre pour la diffusion
                    eng = _engine_for(_im)
                    eng.ensure_up()
                    yield _sse("status", label="génération de l'image…")
                    data, ext = eng.generate(
                        Path(_im.workflow_path).read_text(encoding="utf-8"),
                        prompt,
                        timeout=float(_im.timeout),
                        image_path=src_image,
                        width=gen_w,
                        height=gen_h,
                    )
                    # UNIQUE copie : dans le dossier de LA session (comme sa timeline).
                    # Le média suit le cycle de vie de la session — la supprimer emporte
                    # ses médias, aucun orphelin (décision user 2026-07-09 : fini les
                    # duplications var/generated + workspace + output ComfyUI).
                    name = f"loom_{int(time.time() * 1000)}{ext}"
                    media_dir = session_store.root / _sess.id / "generated"
                    media_dir.mkdir(parents=True, exist_ok=True)
                    (media_dir / name).write_bytes(data)
                    loc = str(media_dir / name)
                    if ext in (".png", ".jpg", ".jpeg", ".webp"):
                        md = (
                            f"![{(prompt or 'image')[:80]}](/genimg/{_sess.id}/{name})\n\n"
                            f"Image écrite : `{loc}`"
                        )
                    else:
                        # Vidéo (webm/mp4) : le markdown image ne la lit pas — lien
                        # cliquable, le navigateur la joue dans un onglet.
                        md = (
                            f"[vidéo générée — cliquer pour lire](/genimg/{_sess.id}/{name})\n\n"
                            f"Vidéo écrite : `{loc}`"
                        )
                    if refined:
                        # Le prompt réellement envoyé au diffuseur, visible dans le fil :
                        # l'utilisateur voit ce que l'affinage a fait de sa demande.
                        md += f"\n\nPrompt affiné ({_im.refiner}) : `{prompt}`"
                    yield _finish(md)
                except ComfyError as exc:
                    yield _finish(f"[génération d'image interrompue : {exc}]")
                except Exception as exc:  # noqa: BLE001 - jamais de stacktrace dans le chat
                    traceback.print_exc()
                    yield _finish(
                        f"[génération d'image interrompue : erreur interne — {str(exc)[:160]}]"
                    )
                finally:
                    if _img_held:
                        _local_gen_lock.release()
                    chat_lock.release()
                    _stay_awake.release()
                    yield _sse("status", label="")

            return Response(generate_image(), mimetype="text/event-stream")

        def generate():

            # Empêche la mise en veille du système tant que CE tour génère (release au
            # finally) : sans ça, une veille par inactivité gèle loom.web + llama.cpp et la
            # génération meurt (« connexion perdue »). L'écran peut s'éteindre, le travail
            # continue en arrière-plan.
            _stay_awake.acquire()
            # Annulation de CETTE session, lue par _confirm (même thread de génération).
            _confirm_local.ev = cancel_event
            # Verrou modèle LOCAL : pris dans le try ci-dessous (avant le 1er appel modèle),
            # libéré au finally. Distant -> jamais pris (vrai parallèle entre onglets).
            _local_held = False

            # Profil du modèle : correctifs déterministes (cadratins, guillemets

            # typographiques) appliqués au texte streamé du chat. Le profil existe

            # déjà pour les outils d'écriture (via tool_factory) ; on le recharge ici

            # pour l'appliquer AUSSI aux réponses du modèle, pas seulement aux fichiers.

            _profile = load_profile(conv.model) if conv.model else None

            if adopted_ws:  # informe l'UI que le dossier de travail a été adopté
                yield _sse("workspace", path=adopted_ws)

            # Démarrage AUTO du serveur modèle, RACONTÉ dans le fil : notices streamées
            # pendant le démarrage de la stack puis le chargement du modèle — l'utilisateur
            # voit que ça travaille au lieu de paniquer devant un silence. Fait DANS le
            # générateur (pas avant la Response) pour que ces étapes s'affichent en direct.
            if conv.model and conv.model not in remote_model_ids:
                _reachable, _running_txt = client.running_local(timeout=2.0)
                if not _reachable:
                    yield _sse(
                        "notice",
                        text="serveur modèle éteint — démarrage de la stack en cours…",
                    )
                    server_manager.start()
                    _deadline = time.monotonic() + 90.0
                    while time.monotonic() < _deadline and not cancel_event.is_set():
                        time.sleep(0.7)
                        _reachable, _running_txt = client.running_local(timeout=2.0)
                        if _reachable:
                            yield _sse("notice", text="serveur modèle démarré.")
                            break
                    if not _reachable and not cancel_event.is_set():
                        yield _sse(
                            "notice",
                            text=(
                                "le serveur modèle ne répond toujours pas (détails : "
                                "var/logs/serve.log) — la génération va échouer."
                            ),
                        )
                if _reachable and conv.model not in _running_txt:
                    yield _sse(
                        "notice",
                        text="chargement du modèle en mémoire — la première réponse met "
                        "plus de temps à démarrer…",
                    )

            answer = ""

            actions: list[str] = []  # trace compacte des outils (anti-amnésie)

            saved = False
            # Persistance AU FIL DE L'EAU : au lieu de tout sauver une seule fois EN FIN de tour
            # (un long audit interrompu/rechargé/relancé perdait TOUT), on met à jour EN PLACE
            # l'unique message assistant du tour (réponse en cours + trace compacte des actions)
            # et on sauve à CHAQUE étape marquante (outil terminé, flux de texte) + à la fin.
            _turn = {"idx": None, "last": 0.0}

            # Journal d'affichage TEMPS RÉEL : chaque événement visible est écrit à l'instant
            # dans timeline.jsonl (append, zéro batch) -> rejouable au rechargement. On y met
            # les événements qui reconstruisent la vue (raisonnement, texte, cartes d'outils) ;
            # pas les compteurs (metrics/totals) ni les décorations live (tool_stream/args).
            _TL = {
                "reasoning",
                "text",
                "tool_call",
                "tool_result",
                "phase",
                "notice",
                "parallel",
            }

            def _tl(event, **data):
                if event in _TL:
                    session_store.append_event(sess.id, event, data)
                return _sse(event, **data)

            def _persist(final=False):
                # On NE persiste pas les messages `tool` bruts (gonflerait le contexte + casserait
                # le résumeur) : seulement le texte + la trace des actions. Un même tour = UN seul
                # message assistant, mis à jour en place (pas de doublons).
                nonlocal saved

                body = answer

                if actions:
                    trace = "[Actions de ce tour : " + " · ".join(actions[:20]) + "]"

                    body = f"{body}\n\n{trace}" if body else trace

                if not body:  # rien à dire ET rien fait -> pas de bulle vide
                    return

                # Piloté par ÉVÉNEMENT (chaque outil terminé + fin), plus par un timer : le
                # temps réel de l'affichage vient du journal `timeline.jsonl`, pas d'ici. Ce
                # session.json ne porte que le contexte lean du modèle, inutile à chaque token.
                if _turn["idx"] is None:
                    conv.add("assistant", body)
                    _turn["idx"] = len(conv.messages) - 1
                else:
                    conv.messages[_turn["idx"]]["content"] = body

                save()
                saved = True

            # Registre construit selon les outils activés pour CETTE conversation

            # (toggles UI) ET le workspace de la session active : sans ça les outils

            # (write/edit/run_shell + sous-agent) retombent sur cfg.chat.workspace_dir

            # et écrivent à côté du dossier ciblé.

            # Résumé PRÉ-TOUR (proactif), DANS le stream pour être VISIBLE : si l'historique
            # dépasse le budget, on émet le label d'activité « compaction… », on résume, on
            # trace une carte, puis on efface le label. Gate `needs_summary` d'abord (sans
            # appel modèle) pour ne montrer le label QUE si un résumé va vraiment tourner.
            # Placé APRÈS le démarrage du serveur modèle (le résumé appelle le modèle).
            # LOCAL UNIQUEMENT : un modèle DISTANT a une grande fenêtre et gère lui-même son
            # contexte + son prefix-cache ; réécrire son historique casserait ce cache et
            # coûterait des tokens pour rien. On ne compacte donc que le local.
            _is_local_model = bool(conv.model) and conv.model not in remote_model_ids
            # SEUIL relatif à la FENÊTRE, pas le `context_budget` (3000) absolu : ce dernier
            # est comparé à system_prompt + messages, or le prompt système SEUL fait ~11k
            # tokens -> le seuil 3000 était TOUJOURS dépassé et la compaction partait à CHAQUE
            # message (même « poursuis »), en appelant le modèle (lent). On la déclenche
            # désormais seulement près de la saturation (même seuil que le microcompact).
            _, _pre_threshold = _model_limits(conv.model)
            if (
                _is_local_model
                and context.needs_summary(
                    conv.system_prompt, conv.messages, _pre_threshold
                )
                and len(conv.messages) > _settings["keep_recent"]
            ):
                yield _sse("status", label="compaction du contexte…")
                if context.summarize(
                    conv, client, _pre_threshold, _settings["keep_recent"]
                ):
                    save()
                    # Jauge à jour TOUT DE SUITE (estimation ~3 car./token), sans attendre
                    # l'usage réel du 1er appel du tour.
                    conv.context_tokens = (
                        len(conv.system_prompt)
                        + sum(_msg_chars(m.get("content")) for m in conv.messages)
                    ) // 3
                    yield _tl(
                        "tool_result",
                        name="(compaction)",
                        ok=True,
                        preview="Contexte résumé pour libérer de la place. Je reprends.",
                    )
                    yield _sse("totals", **_totals(conv))
                yield _sse("status", label="")  # efface le label d'activité

            ws = _session().workspace

            registry = tool_factory(conv.active_tools, ws, conv)

            use_tools = registry is not None and len(registry)

            # Limites du modèle courant (distant = sa grande fenêtre ; local = global).

            eff_max_tokens, eff_compact = _model_limits(conv.model)

            # `strong` (tier distant=fort) est calculé plus haut, à la construction du prompt :

            # il coupe ici les gardes de comportement (act_nudge, claim_audit, coupe non-progrès).

            # On ne garde que outils + mémoire + sécurité. Un modèle local garde le harnais complet.

            if use_tools:
                source = client.stream_chat_tools(
                    conv.to_messages(),
                    system_prompt,
                    eff_max_tokens,
                    model=conv.model or None,
                    registry=registry,
                    thinking=conv.thinking,
                    permission=_perm["fn"],
                    confirm=_confirm,
                    compact_after_tokens=eff_compact,
                    strong=strong,
                )

            else:
                source = client.stream_chat(
                    conv.to_messages(),
                    system_prompt,
                    eff_max_tokens,
                    model=conv.model or None,
                    thinking=conv.thinking,
                )

            interrupted = False

            recv_confirmed = 0  # reçus confirmés par l'usage (tool-calls inclus)

            cur_turn = 0  # reçus live du tour en cours (reset à chaque usage)

            sent_tokens = 0  # envoyés (prompt) cumulés via l'usage

            last_rate = 0.0  # dernier débit mesuré

            burst_start = None  # début de rafale (débit hors pauses outils)

            burst_tokens = 0

            last_tok = None

            # Auto-titre DÈS L'ENVOI (le titre dérive du MESSAGE, pas de la réponse). Pour un
            # modèle DISTANT : on l'infère en tâche de fond tout de suite et on le pousse au
            # client dès qu'il est prêt (interleavé), sans attendre la fin du tour -> l'onglet
            # prend son vrai nom en ~1-2s même sur une génération longue. Pour un modèle LOCAL,
            # on NE le fait PAS ici (llama-swap = 1 slot ; un appel concurrent contendrait avec
            # la génération) : on garde le titrage en fin de tour, quand le slot est libre.
            _titled = {"value": None, "emitted": False}
            _title_ready = threading.Event()
            _immediate_title = (
                sess.title == "Nouvelle session" and conv.model in remote_model_ids
            )
            if _immediate_title:

                def _do_title(_msg=message, _model=conv.model):
                    _t = ""
                    try:
                        _t = _infer_title(client, _model or None, _msg)
                    except Exception:  # noqa: BLE001 - titre best-effort, jamais bloquant
                        _t = ""
                    _titled["value"] = _t or ""
                    if _t:
                        sess.title = _t
                    _title_ready.set()

                threading.Thread(
                    target=_do_title, daemon=True, name="loom-title"
                ).start()

            try:
                # Modèle LOCAL : llama-swap n'en sert qu'UN à la fois -> on sérialise via le
                # verrou global (limitation machine connue, signalée à l'UI). Modèle DISTANT :
                # pas de verrou -> cette session génère EN PARALLÈLE des autres onglets.
                if conv.model and conv.model not in remote_model_ids:
                    if not _local_gen_lock.acquire(blocking=False):
                        yield _sse(
                            "notice",
                            text=(
                                "modèle local occupé : une autre session génère déjà sur la "
                                "machine — mise en file (le parallèle réel n'existe qu'avec un "
                                "modèle distant)."
                            ),
                        )
                        _local_gen_lock.acquire()
                    _local_held = True
                    # Un moteur image encore chargé tiendrait la VRAM que le LLM va
                    # réclamer : le vider d'abord (best-effort, rapide si rien à vider).
                    # Son cache RAM n'est gardé que si le LLM tient à côté (64 Go : oui ;
                    # machine étroite : non, et on le DIT — l'utilisateur comprend
                    # pourquoi la prochaine image rechargera depuis le disque).
                    if _free_image_engines(_local_size_mb(conv.model)) is False:
                        yield _sse(
                            "notice",
                            text=(
                                "moteur image déchargé de la RAM (insuffisante pour "
                                "garder LLM + cache image ensemble) : la prochaine "
                                "image repaiera le chargement disque."
                            ),
                        )

                for kind, payload in source:
                    # Titre distant prêt (thread de fond) -> on le pousse dès la 1re occasion.
                    if (
                        _immediate_title
                        and _title_ready.is_set()
                        and not _titled["emitted"]
                    ):
                        _titled["emitted"] = True
                        if _titled["value"]:
                            session_store.save(sess)
                            yield _sse(
                                "session_title", id=sess.id, title=_titled["value"]
                            )

                    if cancel_event.is_set():
                        # Une nouvelle soumission demande l'arrêt : on stoppe net

                        # et on persiste ce qui a déjà été généré.

                        interrupted = True

                        break

                    if kind == "status":
                        # Signal d'activité (ex. compaction en cours) : piloté vers le label
                        # animé au-dessus du composer, comme « le modèle tourne ».
                        yield _sse("status", **payload)

                    elif kind == "context_estimate":
                        # Compaction : la jauge de contexte est rafraîchie IMMÉDIATEMENT
                        # (estimation), sans attendre l'usage réel du prochain appel — sinon
                        # elle resterait au pic pendant tout l'appel suivant. L'usage réel du
                        # tour d'après la corrigera de toute façon.
                        conv.context_tokens = int(payload.get("tokens", 0) or 0)
                        yield _sse("totals", **_totals(conv))

                    elif kind == "reasoning":
                        yield _tl("reasoning", text=payload)

                    elif kind == "content":
                        if _profile is not None:
                            payload = _profile.apply_to_text(payload)
                        answer += payload

                        # Temps réel : le texte est journalisé à l'instant (rejouable). Le
                        # session.json (contexte) se met à jour aux frontières d'outils + fin.
                        yield _tl("text", text=payload)

                    elif kind == "parallel":
                        yield _tl("parallel", **payload)

                    elif kind == "tool_call":
                        yield _tl("tool_call", **payload)

                    elif kind == "tool_request":
                        yield _sse("tool_request", **payload)

                    elif kind == "tool_begin":
                        yield _sse("tool_begin", **payload)

                    elif kind == "tool_args":
                        yield _sse("tool_args", **payload)

                    elif kind == "tool_stream":
                        yield _sse("tool_stream", **payload)

                    elif kind == "tool_result":
                        line = _action_trace_line(payload)

                        if line and line not in actions:
                            actions.append(line)

                        yield _tl("tool_result", **payload)

                        _persist()  # checkpoint contexte (event-driven) : l'outil vient de finir

                    elif kind == "usage":
                        # Fin d'un tour : llama-server donne le prompt réel et le completion

                        # EXACT (tool-calls inclus) -> on cumule envoyés/reçus à travers les

                        # tours ET les outils, et on réconcilie le tour courant.

                        _p = payload.get("prompt_tokens", 0) or 0
                        _c = payload.get("completion_tokens", 0) or 0
                        _cached = payload.get("cached_tokens", 0) or 0

                        sent_tokens += _p

                        recv_confirmed += _c

                        cur_turn = 0

                        # Cumul RÉEL de la session : chaque appel refacture tout le contexte en
                        # INPUT -> on somme input/output/cache/coût sur TOUS les appels
                        # (persisté), pas seulement le tour. C'est LA vraie somme facturée, et
                        # `cached` mesure si le prompt caching du provider mord.
                        _pin, _pout, _pcached = _price_of(conv.model)
                        conv.add_usage(_p, _c, _cached, _pin, _pout, _pcached)

                        yield _sse("usage", **payload)

                        yield _sse(
                            "metrics",
                            sent=sent_tokens,
                            recv=recv_confirmed,
                            tok_s=last_rate,
                        )

                        yield _sse("totals", **_totals(conv))

                    elif kind == "sub_usage":
                        # Conso d'un SOUS-AGENT (dispatch_agent) : ses tokens sont RÉELS et
                        # facturés -> on les ajoute aux totaux de session (coût, N×, in/out/
                        # cache). `set_context=False` : son prompt n'est PAS le contexte du fil
                        # principal, on ne touche donc pas la jauge de remplissage ni les
                        # métriques per-tour (sent/recv) qui décrivent le tour principal.
                        _sp = payload.get("prompt_tokens", 0) or 0
                        _sc = payload.get("completion_tokens", 0) or 0
                        _scached = payload.get("cached_tokens", 0) or 0
                        _pin, _pout, _pcached = _price_of(conv.model)
                        conv.add_usage(
                            _sp,
                            _sc,
                            _scached,
                            _pin,
                            _pout,
                            _pcached,
                            set_context=False,
                        )
                        yield _sse("totals", **_totals(conv))

                    elif kind == "phase":
                        yield _tl("phase", **payload)

                    # Compteur live : chaque delta (texte OU arguments d'un tool_call) =

                    # 1 vrai token streamé par llama-server. On compte aussi tool_args

                    # pour que le compteur avance pendant la génération d'un appel (gros

                    # write_file inclus) au lieu de se figer. On affiche le cumul + un débit

                    # mesuré sur la rafale courante ; le timer se réinitialise après >1s sans

                    # token (pause d'exécution) pour que les tok/s reflètent la génération.

                    if kind in ("reasoning", "content", "tool_args"):
                        now = time.monotonic()

                        if last_tok is None or now - last_tok > 1.0:
                            burst_start = now

                            burst_tokens = 0

                        burst_tokens += 1

                        cur_turn += 1

                        last_tok = now

                        span = now - burst_start

                        tok_s = round(burst_tokens / span, 1) if span > 0 else 0.0

                        last_rate = tok_s

                        yield _sse(
                            "metrics",
                            sent=sent_tokens,
                            recv=recv_confirmed + cur_turn,
                            tok_s=tok_s,
                        )

                if interrupted:
                    _persist(
                        final=True
                    )  # interrompu : on force la sauvegarde du travail fait

                    return

                if not answer.strip():
                    answer = "(le modèle a seulement réfléchi — augmente max_tokens)"

                    yield _sse("text", text=answer)

                _persist(final=True)  # fin de tour : écriture finale garantie

                # Cache souverain : le slot local contient LA conversation à cet
                # instant précis -> on le sauve MAINTENANT (~ms), avant que le titre
                # (inline ci-dessous) et le reflect (maintenance) ne l'écrasent ; la
                # maintenance le RESTAURERA (~ms) au lieu de re-préfiller des minutes.
                _kv_saved = client.save_slot(conv.model, "turnend.kv")

                # Apprentissage post-tour + restauration du cache : DÉPORTÉS dans un
                # thread (cf. _post_turn_maintenance). Avant, reflect tournait ICI,
                # avant le `done` -> l'UI restait sur « le modèle travaille » pendant
                # un appel modèle entier, ET le cache KV de la conversation était
                # écrasé -> re-prefill INTÉGRAL au message suivant (bug 2026-07-10).
                # Le thread attend le verrou local (libéré à la fermeture du flux).
                _do_reflect = (
                    _settings["reflect_enabled"]
                    and reflect_stores is not None
                    and saved
                    and len(actions) >= _settings["reflect_min_actions"]
                )

                threading.Thread(
                    target=_post_turn_maintenance,
                    args=(
                        sess,
                        conv.to_messages(),
                        list(actions),
                        answer,
                        conv.model,
                        _do_reflect,
                        _kv_saved,
                    ),
                    daemon=True,
                    name="loom-post-turn",
                ).start()

                # Auto-titre : à la 1re vraie réponse, nommer la session (le modèle infère le
                # sujet). On titre LA session de CETTE génération (`sess`), pas la session
                # focus (_cur) — sinon, en multi-onglets concurrent, on titrerait la mauvaise.

                if _immediate_title:
                    # Distant : filet de secours si le thread de titre n'a pas fini avant la
                    # fin de la boucle (ou tour sans événement) -> on l'attend brièvement.
                    if not _titled["emitted"]:
                        _title_ready.wait(timeout=8)
                        _titled["emitted"] = True
                        if _titled["value"]:
                            sess.title = _titled["value"]
                            session_store.save(sess)
                            yield _sse(
                                "session_title", id=sess.id, title=_titled["value"]
                            )
                elif saved and sess.title == "Nouvelle session":
                    # Local : titrage en fin de tour (slot llama-swap libre, pas de contention).
                    _title = _infer_title(client, conv.model or None, message)
                    if _title:
                        sess.title = _title
                        session_store.save(sess)
                        yield _sse("session_title", id=sess.id, title=_title)

                yield _sse("done")

            except GeneratorExit:
                # L'utilisateur a soumis un nouveau message : le client a fermé le

                # flux. On persiste la réponse PARTIELLE déjà reçue, puis on relaie

                # l'interruption (re-raise obligatoire pour le protocole générateur).

                _persist(final=True)  # client parti : écriture finale garantie

                raise

            except Exception:  # noqa: BLE001 - on remonte l'erreur au client SSE
                traceback.print_exc()
                yield _sse("error", message="erreur interne")

            finally:
                _last_activity[0] = time.time()  # marque l'activité pour le keep-warm

                if _local_held:
                    _local_gen_lock.release()

                chat_lock.release()
                _stay_awake.release()  # plus de veille bloquée si plus aucune génération

        return Response(generate(), mimetype="text/event-stream")

    @app.post("/fork")
    def fork():
        """Repart d'un message utilisateur : tronque l'historique APRES ce message (exclus),

        renvoie son texte pour pre-remplir l'input. user_index = N-ieme message user (0-based)
        COMPTÉ SUR L'AFFICHAGE — or la compaction fusionne les vieux tours côté serveur, donc
        l'index peut avoir glissé : on retrouve alors le message par son CONTENU (`text`,
        envoyé par l'UI), et sinon on répond une erreur EXPLICITE (plus d'échec muet)."""

        user_index = int(request.form.get("user_index", "-1"))
        ui_text = (request.form.get("text") or "").strip()

        conv, save = _ctx()

        msgs = conv.messages

        def _as_text(content) -> str:
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                ).strip()
            return str(content).strip()

        # Trouve le N-ieme message user dans l'historique persiste

        user_msgs = [i for i, m in enumerate(msgs) if m.get("role") == "user"]

        target_idx = None
        if 0 <= user_index < len(user_msgs):
            cand = user_msgs[user_index]
            # L'index ne vaut que si le contenu correspond encore (pas de glissement).
            if not ui_text or _as_text(msgs[cand].get("content", "")) == ui_text:
                target_idx = cand
        if target_idx is None and ui_text:
            # Index périmé (compaction) : dernier message user au MÊME contenu.
            for i in reversed(user_msgs):
                if _as_text(msgs[i].get("content", "")) == ui_text:
                    target_idx = i
                    break
        if target_idx is None:
            return Response(
                "ce message a été résumé par la compaction : « repartir » n'est "
                "possible que depuis un message encore présent dans le contexte "
                "(les plus récents).",
                status=400,
            )

        content = msgs[target_idx].get("content", "")

        if isinstance(content, list):
            text = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )

        else:
            text = str(content)

        # Tronque : garde jusqu'au message user (inclus), efface la suite

        conv.messages = msgs[: target_idx + 1]

        save()

        return {"text": text}

    @app.post("/compact")
    def compact():
        """Compaction MANUELLE (bouton près de la jauge de contexte) : résume les vieux
        tours de la session ciblée en un bloc dense et remplace l'historique. session_id =
        onglet ciblé, sinon la session focus. Refuse si une génération tourne (le verrou de
        session est pris) — on ne mute pas l'historique sous une boucle active. Renvoie les
        compteurs d'usage à jour (jauge de contexte ré-estimée) + `collapsed`."""
        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(req_sid) if req_sid else _session()
        if sess is None:
            return Response("session introuvable", status=404)
        lock = _lock_for(sess.id)
        if not lock.acquire(blocking=False):
            return Response("occupé : cette session génère déjà", status=429)
        try:
            conv = sess.conversation
            # DÉTERMINISTE et INSTANTANÉ (aucun appel modèle -> pas de blocage de plusieurs
            # minutes). Cible = prompt système (INCOMPRESSIBLE) + ~4000 car. (~1,3k tokens)
            # de conversation : on clippe le reste de l'historique.
            target_chars = len(conv.system_prompt) + 4000
            new_msgs, freed = client.compact_conversation(
                conv.messages,
                system_prompt=conv.system_prompt,
                target_chars=target_chars,
            )
            if freed:
                conv.messages = new_msgs
                # `freed` = tokens réellement libérés (delta de la conversation, le prompt
                # système s'annule). On le RETRANCHE du ctx réel du dernier appel -> jauge à
                # jour immédiatement ET exacte (pas une ré-estimation qui oublierait les
                # schémas d'outils). L'usage réel du prochain tour confirmera au token près.
                conv.context_tokens = max(0, conv.context_tokens - freed)
                session_store.save(sess)
        finally:
            lock.release()
        return {**_totals(conv), "collapsed": freed}

    @app.post("/cancel")
    def cancel():

        # Bouton Stop : pose le signal d'annulation de LA session ciblée (par session_id, sinon
        # la session focus) -> SA boucle /chat s'arrête net et libère son verrou. Les AUTRES
        # sessions (onglets) ne sont PAS touchées. Sans effet si rien ne tourne pour elle.

        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(req_sid) if req_sid else _cur["session"]
        if sess is not None:
            _cancel_for(sess.id).set()

        return Response("", status=204)

    @app.post("/tool_decision")
    def tool_decision():

        pend = pending.get(request.form.get("id", ""))

        if pend is not None:
            pend["approved"] = request.form.get("approve") == "1"

            pend["event"].set()

        return Response("", status=204)

    @app.post("/tools")
    def tools_update():

        conv, save = _ctx()

        conv.set_tools(request.form.getlist("tool"))

        save()

        return render_template(
            "_tools.html",
            available_tools=available_tools,
            active_tools=conv.active_tools,
        )

    @app.post("/skills")
    def skills_update():
        # Toggle des skills (façon /tools) : le formulaire porte les skills COCHÉS. Les
        # décochés (tous les autres) deviennent `disabled_skills` de la session -> retirés
        # du catalogue et de use_skill. Re-render le panneau (case maître incluse).
        conv, save = _ctx()
        enabled = set(request.form.getlist("skill"))
        all_names = [s.name for s in _all_skills()]
        conv.set_disabled_skills([n for n in all_names if n not in enabled])
        save()
        return render_template("_skills.html", **_skills_ctx(conv))

    @app.get("/skill")
    def skill_get():
        # Source d'un skill pour l'éditeur : texte brut du SKILL.md, ou l'override de session
        # s'il existe (ce que le modèle voit réellement pour cette session).
        conv, _ = _ctx()
        name = request.args.get("name", "")
        skill = next((s for s in _all_skills() if s.name == name), None)
        if skill is None:
            return {"error": f"skill inconnu : {name}"}, 404
        override = conv.skill_overrides.get(name)
        return {
            "name": skill.name,
            "description": skill.description,
            "source": override if override is not None else read_skill_source(skill),
            "has_override": override is not None,
            "learned": bool(getattr(skill, "learned", False)),
            "editable_on_disk": skill.base_dir != "",
        }

    @app.post("/skill/save")
    def skill_save():
        # Enregistre l'édition d'un skill. scope=session -> override de session (n'écrit
        # PAS le disque) ; scope=global -> écrit le SKILL.md pour TOUTES les sessions et
        # lève l'override de session (le fichier fait désormais foi).
        conv, save = _ctx()
        name = request.form.get("name", "")
        body = request.form.get("body", "")
        scope = request.form.get("scope", "session")
        skill = next((s for s in _all_skills() if s.name == name), None)
        if skill is None:
            return {"error": f"skill inconnu : {name}"}, 404
        if scope == "global":
            try:
                write_skill_source(skill, body)
            except OSError as exc:
                return {"error": f"écriture impossible : {exc}"}, 400
            conv.set_skill_override(name, None)
            save()
            return {"ok": True, "scope": "global"}
        conv.set_skill_override(name, body)
        save()
        return {"ok": True, "scope": "session"}

    @app.post("/thinking")
    def thinking_update():

        conv, save = _ctx()

        conv.set_thinking(request.form.get("thinking") == "1")

        save()

        return Response(str(int(conv.thinking)), mimetype="text/plain")

    @app.route("/pick-folder", methods=["POST"])
    def pick_folder():

        # Sous-processus : évite les soucis tkinter hors du thread principal de Flask.

        script = (
            "import tkinter, tkinter.filedialog as fd;"
            "r=tkinter.Tk(); r.withdraw(); r.attributes('-topmost', True);"
            "p=fd.askdirectory(); print(p if p else '')"
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
            )

        except Exception as exc:  # noqa: BLE001 - tkinter absent / timeout
            return {"path": "", "error": str(exc)[:200]}

        path = (proc.stdout or "").strip()

        if proc.returncode != 0 and not path:
            return {
                "path": "",
                "error": (proc.stderr or "sélecteur indisponible")[:200],
            }

        return {"path": path}

    @app.post("/model")
    def model_update():

        conv, save = _ctx()

        model = request.form.get("model", "")

        conv.set_model(model)

        save()

        # Mémorise ce choix : il devient le défaut des prochaines sessions / lancements.

        session_store.set_default_model(model)

        # Cycle de vie du modèle SUR LA MACHINE. Sélectionner un modèle LOCAL le CHARGE
        # (warmup : llama-swap charge à la 1re requête, et swap l'ancien si besoin) ;
        # passer à un modèle DISTANT (API) DÉCHARGE le local pour LIBÉRER LA VRAM. Les deux
        # en tâche de fond (best-effort) : la réponse UI reste instantanée, l'indicateur
        # d'état (/machine_state) reflète ensuite le résultat réel via llama-swap.
        if model in remote_model_ids:
            threading.Thread(
                target=client.unload_local, daemon=True, name="loom-unload"
            ).start()
        elif model in image_model_ids:
            # Modèle IMAGE : libérer la VRAM du LLM et préchauffer ComfyUI en fond
            # (équivalent du warmup local : la 1re image n'attend pas le démarrage).
            def _prep_image(m=model):
                client.unload_local()
                try:
                    _engine_for(_image_by_id[m]).ensure_up()
                except ComfyError as exc:
                    print(f"[loom] préchauffage ComfyUI : {exc}", flush=True)

            threading.Thread(
                target=_prep_image, daemon=True, name="loom-image-warmup"
            ).start()
        elif model:
            # Modèle LOCAL : démarre le serveur s'il est éteint (démarrage auto), PUIS
            # warmup. Le tout en fond : la réponse UI reste instantanée, le chip suit.
            def _start_and_warm(m=model):
                _ensure_local_server(wait=90.0)
                client.warmup_local(m)

            threading.Thread(
                target=_start_and_warm,
                daemon=True,
                name="loom-warmup",
            ).start()

        return render_template(
            "_models.html",
            models=models,
            current_model=conv.model,
            remote_model_ids=remote_model_ids,
            image_model_ids=image_model_ids,
            video_model_ids=video_model_ids,
            model_descriptions=model_descriptions,
        )

    # ---- Gestionnaire de modèles (UI) : ajouter/tester/supprimer un modèle DISTANT à chaud,
    # sans redémarrer. Un distant = URL + clé (rien en VRAM) -> l'ajout monte une route et met
    # à jour les registres partagés en place. Persisté dans le store JSON (remote_store_path).
    def _models_payload():
        """Liste ordonnée pour reconstruire le <select> côté client (id + local/distant)."""
        return [
            {
                "id": m,
                "remote": m in remote_model_ids,
                "image": m in image_model_ids,
                "video": m in video_model_ids,
                "desc": model_descriptions.get(m, ""),
            }
            for m in models
        ]

    def _remote_list():
        """Modèles distants montés, pour le panneau de config. Jamais la clé en clair :
        seulement sa présence. `managed` = ajouté via l'UI (éditable/supprimable) vs déclaré
        dans local.toml (lecture seule ici)."""
        managed_ids = {m.get("id") for m in model_store.load(remote_store_path)}
        out = []
        for mid in remote_model_ids:
            info = client.remote_route_info(mid)
            key = client.remote_api_key(mid)
            out.append(
                {
                    "id": mid,
                    "base_url": info["base_url"],
                    "model": info["model"],
                    "context": model_contexts.get(mid),
                    "max_tokens": model_max_tokens.get(mid),
                    "vision": mid in vision_models,
                    "has_key": info["has_key"],
                    # Indice masqué (4 derniers car.) : l'utilisateur voit sa propre clé de
                    # façon partielle, jamais la clé entière renvoyée au client.
                    "key_hint": ("…" + key[-4:]) if key else "",
                    "managed": mid in managed_ids,
                }
            )
        return sorted(out, key=lambda x: x["id"])

    def _mount_remote(rec):
        """Monte à chaud un modèle distant `rec` (dict) dans TOUS les registres partagés."""
        mid = rec["id"]
        client.add_remote_route(
            mid,
            {
                "base_url": rec["base_url"],
                "api_key": rec.get("api_key", ""),
                "model": rec["model"],
                "enable_thinking_param": bool(rec.get("enable_thinking_param", False)),
            },
        )
        remote_model_ids.add(mid)
        remote_model_names[mid] = rec["model"]
        if rec.get("context"):
            model_contexts[mid] = int(rec["context"])
        if rec.get("max_tokens"):
            model_max_tokens[mid] = int(rec["max_tokens"])
        model_prices[mid] = (
            float(rec.get("price_in", 0.0) or 0.0),
            float(rec.get("price_out", 0.0) or 0.0),
            float(rec.get("price_cached", 0.0) or 0.0),
        )
        if rec.get("vision"):
            vision_models.add(mid)
        else:
            vision_models.discard(mid)
        if mid not in models:
            models.append(mid)

    @app.get("/models/config")
    def models_config():
        return {"remotes": _remote_list(), "models": _models_payload()}

    @app.post("/models/remote/test")
    def models_remote_test():
        b = request.get_json(silent=True) or {}
        base_url = (b.get("base_url") or "").strip().rstrip("/")
        model = (b.get("model") or "").strip()
        mid = (b.get("id") or "").strip()
        key = (b.get("api_key") or "").strip()
        if not key and mid:  # édition sans re-saisir la clé -> réutilise la stockée
            stored = {m["id"]: m for m in model_store.load(remote_store_path)}
            key = stored.get(mid, {}).get("api_key", "")
        if not (base_url and model):
            return {"ok": False, "message": "base_url et model requis"}, 400
        ok, msg = client.ping_remote(base_url, key, model)
        return {"ok": ok, "message": msg}

    @app.post("/models/remote")
    def models_remote_upsert():
        if not remote_store_path:
            return {"error": "store des modèles indisponible"}, 500
        b = request.get_json(silent=True) or {}
        mid = (b.get("id") or "").strip()
        base_url = (b.get("base_url") or "").strip().rstrip("/")
        model = (b.get("model") or "").strip()
        if not (mid and base_url and model):
            return {"error": "id, base_url et model sont requis"}, 400
        if mid in models and mid not in remote_model_ids:
            return {"error": f"'{mid}' est déjà un modèle local"}, 400
        stored = {m["id"]: m for m in model_store.load(remote_store_path)}
        # Clé : si vide, on garde l'existante — soit du store géré, soit de la route montée
        # (cas d'un modèle défini en config qu'on édite sans re-saisir la clé).
        key = (
            (b.get("api_key") or "").strip()
            or stored.get(mid, {}).get("api_key", "")
            or client.remote_api_key(mid)
        )
        rec = {
            "id": mid,
            "base_url": base_url,
            "model": model,
            "api_key": key,
            "context": int(b["context"]) if b.get("context") else None,
            "max_tokens": int(b["max_tokens"]) if b.get("max_tokens") else None,
            "vision": bool(b.get("vision")),
        }
        # Un modèle DÉFINI EN CONFIG (monté mais absent du store géré) reste dans local.toml :
        # on l'y édite en place (tomlkit, commentaires préservés). Sinon store JSON géré par l'UI.
        is_config = mid in remote_model_ids and mid not in stored
        with _toml_lock:
            if is_config and config_local_path:
                model_store.upsert_remote_in_toml(config_local_path, rec)
            else:
                model_store.upsert(remote_store_path, rec)
        _mount_remote(rec)
        return {"ok": True, "models": _models_payload(), "remotes": _remote_list()}

    @app.delete("/models/remote/<mid>")
    def models_remote_delete(mid):
        if not remote_store_path:
            return {"error": "store des modèles indisponible"}, 500
        managed = {m.get("id") for m in model_store.load(remote_store_path)}
        if mid not in managed:
            return {"error": "modèle non géré par l'UI (défini dans local.toml)"}, 400
        with _toml_lock:
            model_store.delete(remote_store_path, mid)
        client.remove_remote_route(mid)
        remote_model_ids.discard(mid)
        remote_model_names.pop(mid, None)
        model_contexts.pop(mid, None)
        model_max_tokens.pop(mid, None)
        model_prices.pop(mid, None)
        vision_models.discard(mid)
        if mid in models:
            models.remove(mid)
        return {"ok": True, "models": _models_payload(), "remotes": _remote_list()}

    # ---- Modèles LOCAUX : liste + édition du tuning MACHINE (offload GPU) dans model.toml.
    # La définition (repo/filename/n_layers) est commune au modèle -> lecture seule ici ; le
    # tuning (context/n_gpu_layers/cpu_moe/n_cpu_moe) est propre à cette machine -> éditable.
    _LOCAL_EDITABLE = {
        "context": "int",
        "n_gpu_layers": "int",
        "cpu_moe": "bool",
        "n_cpu_moe": "int",
    }

    @app.get("/models/local")
    def models_local():
        import tomllib

        out = []
        for m in local_model_specs:
            cur = {k: v for k, v in m.items() if k != "dir"}
            d = m.get("dir")
            if d:
                tp = Path(d) / "model.toml"
                if tp.exists():
                    try:
                        raw = tomllib.loads(tp.read_text(encoding="utf-8"))
                        for k in _LOCAL_EDITABLE:
                            if k in raw:
                                cur[k] = raw[k]
                    except (OSError, ValueError):
                        pass
            out.append(cur)
        return {"models": out}

    @app.post("/models/local/set")
    def models_local_set():
        import tomlkit

        b = request.get_json(silent=True) or {}
        mid = (b.get("id") or "").strip()
        key = (b.get("key") or "").strip()
        if key not in _LOCAL_EDITABLE:
            return {"error": "champ non éditable"}, 400
        spec = next((m for m in local_model_specs if m.get("id") == mid), None)
        if not spec or not spec.get("dir"):
            return {"error": "modèle local inconnu"}, 404
        tp = Path(spec["dir"]) / "model.toml"
        if not tp.exists():
            return {"error": "model.toml introuvable"}, 404
        raw = b.get("value")
        t = _LOCAL_EDITABLE[key]
        empty = raw is None or (
            isinstance(raw, str) and raw.strip() == "" and t == "int"
        )
        truthy = ("1", "true", "on", "yes")
        try:
            with _toml_lock:  # sérialise le read-modify-write (Flask threaded)
                doc = tomlkit.parse(tp.read_text(encoding="utf-8"))
                if empty:
                    if key in doc:
                        del doc[key]
                elif t == "int":
                    doc[key] = int(raw)
                else:  # bool
                    doc[key] = (
                        raw if isinstance(raw, bool) else str(raw).lower() in truthy
                    )
                tp.write_text(tomlkit.dumps(doc), encoding="utf-8")
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)[:120]}, 400
        # Applique À CHAUD côté serveur modèle : régénère le yaml (llama-swap -watch-config le
        # recharge) + décharge CE modèle -> il se relance avec le nouveau tuning au prochain
        # usage, sans toucher au TOML à la main ni tout redémarrer.
        applied = _regen_swap_yaml()
        if applied:
            threading.Thread(
                target=lambda: client.unload_local(mid),
                daemon=True,
                name="loom-reload-model",
            ).start()
        return {"ok": True, "applies": "model-reload" if applied else "restart"}

    # ---- Console de configuration : introspection + édition des vrais fichiers TOML (deux
    # couches commun/système), commentaires préservés via tomlkit (loom.runtime.config_schema).
    def _cfg_paths_ok():
        return bool(config_defaults_path and config_local_path)

    @app.get("/config")
    def config_describe():
        if not _cfg_paths_ok():
            return {"error": "chemins de config indisponibles"}, 500
        from loom.runtime import config_schema

        return config_schema.describe(config_defaults_path, config_local_path)

    @app.post("/config/set")
    def config_set():
        if not _cfg_paths_ok():
            return {"error": "chemins de config indisponibles"}, 500
        from loom.runtime import config_schema

        b = request.get_json(silent=True) or {}
        section = (b.get("section") or "").strip()
        key = (b.get("key") or "").strip()
        if not (section and key):
            return {"error": "section et key requis"}, 400
        try:
            with _toml_lock:
                res = config_schema.set_value(
                    config_defaults_path,
                    config_local_path,
                    section,
                    key,
                    b.get("value"),
                )
        except (ValueError, OSError) as e:
            return {"ok": False, "error": str(e)[:160]}, 400
        if res.get("ok"):
            _reload_app_config()  # applique À CHAUD les params app (permissions, tokens…)
            _apply_to_model_server(section)  # régénère le yaml si param serveur/modèle
        code = 200 if res.get("ok") else 400
        return res, code

    @app.post("/config/reset")
    def config_reset():
        if not _cfg_paths_ok():
            return {"error": "chemins de config indisponibles"}, 500
        from loom.runtime import config_schema

        b = request.get_json(silent=True) or {}
        section = (b.get("section") or "").strip()
        key = (b.get("key") or "").strip()
        if not (section and key):
            return {"error": "section et key requis"}, 400
        with _toml_lock:
            res = config_schema.reset_value(
                config_defaults_path, config_local_path, section, key
            )
        if res.get("ok"):
            _reload_app_config()
            _apply_to_model_server(section)
        return res, (200 if res.get("ok") else 400)

    @app.get("/config/effective")
    def config_effective():
        """Valeurs de config ACTUELLEMENT en vigueur dans l'app en cours (mémoire vive). Sert
        à vérifier qu'une édition s'applique à chaud, sans redémarrer loom.web."""
        return dict(_settings)

    @app.get("/machine_state")
    def machine_state():
        # État du modèle SUR LA MACHINE, pour l'indicateur UI. Vérité = llama-swap /running
        # (best-effort ; le modèle peut aussi s'être déchargé seul via son TTL). On teste par
        # sous-chaîne quel modèle est chargé, sans coupler au schéma JSON de llama-swap.
        conv, _ = _ctx()
        model = conv.model
        remote = model in remote_model_ids
        reachable, running_txt = client.running_local()
        # /running est parsé quand c'est possible : llama-swap distingue « starting »
        # (chargement en cours) de « ready » (servable). Sans ça, le chip disait
        # « chargé » dès le début du chargement, et un unload pendant « starting » est
        # ignoré par llama-swap -> le bouton « décharger » doit se cacher à ce moment.
        # Repli sous-chaîne si le JSON change (on reste découplé du schéma).
        states: dict[str, str] = {}
        try:
            for entry in json.loads(running_txt).get("running", []):
                states[str(entry.get("model", ""))] = str(entry.get("state", ""))
        except (ValueError, AttributeError):
            pass
        if states or reachable:
            model_loaded = bool(model and states.get(model) == "ready")
            model_loading = bool(model and states.get(model) == "starting")
            any_loaded = bool(
                any(states.get(mid) in ("ready", "starting") for mid in local_model_ids)
            )
        else:
            model_loaded = bool(reachable and model and model in running_txt)
            model_loading = False
            any_loaded = bool(
                reachable and any(mid in running_txt for mid in local_model_ids)
            )
        if reachable:
            server_manager.confirm_started()  # démarrage confirmé -> fin de l'état « démarrage »
        return {
            "mode": "remote" if remote else "home",
            "model": model,
            "reachable": reachable,
            "model_loaded": model_loaded,
            "loading": model_loading,
            "any_loaded": any_loaded,
            # Serveur GÉRÉ (lancé par loom.web) : conditionne le bouton « éteindre » —
            # on ne propose jamais de tuer une stack lancée à la main hors Loom.
            "managed": server_manager.owns_running(),
            "starting": server_manager.starting,
        }

    @app.post("/machine/unload")
    def machine_unload():
        # Déchargement À LA DEMANDE (bouton UI sous le chip machine) : libère la VRAM sans
        # changer de modèle sélectionné. Synchrone : la réponse reflète le résultat réel
        # (llama-swap tue le llama-server en ~1-2 s). Rechargé à la prochaine requête.
        return {"ok": client.unload_local()}

    @app.post("/machine/server/start")
    def machine_server_start():
        # Trigger MANUEL (bouton « démarrer le serveur ») : lance sans bloquer la requête ;
        # l'UI suit la progression via /machine_state (état « démarrage… »). Puis warmup du
        # modèle local sélectionné en fond : « démarrer » = « rendre prêt à répondre ».
        ok = server_manager.start()
        conv, _ = _ctx()
        if conv.model and conv.model not in remote_model_ids:

            def _warm(m=conv.model):
                if _ensure_local_server(wait=90.0):
                    client.warmup_local(m)

            threading.Thread(target=_warm, daemon=True, name="loom-warmup").start()
        return {"ok": ok}

    @app.post("/machine/server/stop")
    def machine_server_stop():
        # Éteint l'arbre complet (serve.py + llama-swap + llama-server) et libère RAM/VRAM.
        # Ne concerne QUE l'instance gérée par loom.web (cf. managed dans /machine_state).
        return {"ok": server_manager.stop()}

    @app.get("/sysmon")
    def sysmon_metrics():
        # Métriques système LIVE (CPU/RAM/GPU) pour le moniteur affiché avec un modèle LOCAL.
        # nvidia-smi + psutil ; champs à None si une source manque (le front s'adapte).
        from loom.runtime.sysmon import read_metrics

        return read_metrics()

    # --- Sessions : liste / nouvelle / bascule / suppression ---

    @app.get("/sessions")
    def sessions_list():

        active = _session().id

        return {
            "sessions": [
                {"id": m.id, "title": m.title, "workspace": m.workspace}
                for m in session_store.list()
            ],
            "active": active,
        }

    @app.get("/session_state")
    def session_state():
        # État CLIENT d'une session, pour OUVRIR un onglet sans recharger la page : messages,
        # modèle, thinking, workspace, outils actifs, compteur. Le multi-onglets s'appuie
        # dessus (chaque onglet hydrate sa session à l'ouverture).
        sid = (request.args.get("id") or "").strip()
        sess = _get_session(sid)
        if sess is None:
            return Response("session inconnue", status=404)
        conv = sess.conversation
        return {
            "id": sess.id,
            "title": sess.title,
            "workspace": sess.workspace,
            "messages": conv.messages,
            "thinking": conv.thinking,
            "model": conv.model,
            "active_tools": conv.active_tools,
            "usage_totals": _totals(conv),
            # A-t-on un journal d'affichage à rejouer ? (sinon l'UI retombe sur `messages`).
            "has_timeline": bool(session_store.read_timeline(sess.id)),
        }

    @app.get("/session/<sid>/timeline")
    def session_timeline(sid):
        """Journal d'affichage temps réel d'une session, pour REJOUER l'UI au rechargement
        (raisonnement, texte, cartes d'outils exactement comme en direct). Les chunks 'text'/
        'reasoning' consécutifs sont recollés pour un rejeu léger."""
        out: list[dict] = []
        for e in session_store.read_timeline(sid):
            ev = e.get("event")
            d = e.get("data") or {}
            if ev in ("text", "reasoning") and out and out[-1].get("event") == ev:
                out[-1]["data"]["text"] = (out[-1]["data"].get("text") or "") + (
                    d.get("text") or ""
                )
            else:
                out.append({"event": ev, "data": dict(d)})
        return {"events": out}

    @app.post("/session/new")
    def session_new():

        ws = (request.form.get("workspace") or "").strip() or workspace_dir

        title = (request.form.get("title") or "").strip()

        # Sessions FANTÔMES : créées puis jamais utilisées (« Nouvelle session » et
        # zéro message) — balayées quand on en crée une nouvelle : elles ne font que
        # polluer la sidebar. On épargne toute session dont le verrou de génération
        # est tenu (un tour vient peut-être de démarrer).
        for meta in session_store.list():
            if meta.title != "Nouvelle session":
                continue
            _lk = _sess_locks.get(meta.id)
            if _lk is not None and _lk.locked():
                continue
            ghost = _get_session(meta.id)
            if ghost is not None and not ghost.conversation.messages:
                session_store.delete(meta.id)
                with _gen_guard:
                    _sessions_cache.pop(meta.id, None)
                if _cur["session"] is not None and _cur["session"].id == meta.id:
                    _cur["session"] = None

        sess = session_store.create(workspace=ws, title=title)

        with _gen_guard:
            _sessions_cache[sess.id] = sess

        _cur["session"] = sess

        return {"id": sess.id, "title": sess.title, "workspace": sess.workspace}

    @app.post("/session/activate")
    def session_activate():

        sid = (request.form.get("id") or "").strip()

        loaded = _get_session(sid)

        if loaded is None:
            return Response("session inconnue", status=404)

        session_store.set_active(sid)

        _cur["session"] = loaded

        return {"id": loaded.id}

    @app.post("/session/workspace")
    def session_workspace():

        # Réaffecte le dossier de travail de la SESSION ACTIVE (appelé par le sélecteur

        # de dossier). Sans ça, choisir un dossier ne s'appliquerait qu'à la création

        # d'une nouvelle session -> les outils continueraient de cibler l'ancien.

        ws = (request.form.get("workspace") or "").strip()

        if not ws:
            return Response("workspace manquant", status=400)

        # Cible la session de l'ONGLET appelant (session_id), pas la session focus
        # globale : avec le multi-onglets, _session() peut désigner un autre fil ->
        # le dossier choisi s'écrivait ailleurs et le tour partait sur l'ancien
        # workspace (bug constaté le 2026-07-10).
        req_sid = (request.form.get("session_id") or "").strip()
        sess = (_get_session(req_sid) if req_sid else None) or _session()

        sess.workspace = ws

        session_store.save(sess)

        return {"workspace": sess.workspace}

    @app.post("/session/delete")
    def session_delete():

        sid = (request.form.get("id") or "").strip()

        session_store.delete(sid)

        # Si on supprime la session courante, on recharge l'active (ou on en crée une).

        if _cur["session"] is not None and _cur["session"].id == sid:
            _cur["session"] = None

            _session()

        return {"ok": True}

    def _prime_slot(sess) -> bool:
        """Ré-amorce le cache KV du slot local avec le fil de `sess` : re-prefill
        silencieux du MÊME préfixe que le prochain tour (system prompt + messages +
        schémas d'outils — mêmes ingrédients que /chat, sinon zéro réutilisation).
        Le message suivant ne préfille alors que son delta. False si rien à amorcer
        (modèle distant : cache provider + appel payant ; fil vide)."""
        try:
            conv = sess.conversation
            model = conv.model
            if not model or model in remote_model_ids:
                return False
            msgs = conv.to_messages()
            if not msgs:
                return False
            system_prompt, _strong = _build_system_prompt(conv)
            registry = tool_factory(conv.active_tools, sess.workspace, conv)
            return client.warm_context(
                msgs,
                system_prompt,
                model=model,
                registry=registry if (registry is not None and len(registry)) else None,
                thinking=conv.thinking,
            )
        except Exception as e:  # noqa: BLE001 - amorçage best-effort, jamais bloquant
            print(f"[prime] erreur ignorée : {e}", flush=True)
            return False

    def _post_turn_maintenance(
        sess, msgs, actions, answer, model, do_reflect, kv_saved=False
    ):
        """Fin de tour déportée hors du flux SSE : reflect (apprentissage) PUIS
        restauration du cache de la conversation (save fait en fin de génération ;
        repli = ré-amorçage par re-prefill si le save a échoué). Local : sérialisé
        derrière le verrou (attend la fermeture du flux ; si l'utilisateur a déjà
        relancé, on passe après son tour). Distant : reflect seul."""
        is_local = bool(model) and model not in remote_model_ids

        if is_local and not _local_gen_lock.acquire(timeout=600):
            return

        try:
            if do_reflect:
                try:
                    from loom.agent.reflect import reflect as _reflect

                    _res = _reflect(
                        msgs,
                        actions,
                        answer,
                        client=client,
                        model=model or reflect_model,
                        provider=reflect_stores.provider,
                        paths=reflect_stores.paths,
                        learned_dir=reflect_stores.learned_dir,
                    )

                    # Trace VISIBLE (console/serve.log) : sinon l'apprentissage est
                    # une boîte noire — on ne sait pas s'il a tourné ni retenu quoi.
                    if _res is None:
                        print(
                            "[reflect] rien retenu (tour peu généralisable)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[reflect] retenu : {len(_res.new_skills)} skill(s), "
                            f"{len(_res.improved_skills)} amélioré(s), "
                            f"{len(_res.episodes)} épisode(s), "
                            f"{len(_res.memory_updates) + len(_res.user_updates) + len(_res.soul_updates)} "
                            "note(s) identité",
                            flush=True,
                        )

                except Exception as _e:  # noqa: BLE001 - best-effort, jamais bloquant
                    print(f"[reflect] erreur ignorée : {_e}", flush=True)

            if is_local:
                if kv_saved and client.restore_slot(model, "turnend.kv"):
                    print(
                        "[slot] cache de la conversation RESTAURÉ après fin de tour "
                        "(~ms, save/restore du slot KV)",
                        flush=True,
                    )
                else:
                    _ok = _prime_slot(sess)
                    print(
                        f"[prime] repli ré-amorçage par re-prefill : "
                        f"{'ok' if _ok else 'échec/sans objet'}",
                        flush=True,
                    )
                _last_activity[0] = time.time()

        finally:
            if is_local:
                _local_gen_lock.release()

    # --- Keep-warm : empêche l'OS d'évincer le modèle inactif (cold start après pause). --

    # Thread daemon qui ping le modèle de la session ACTIVE (1 token) quand : keep-warm

    # activé, une vraie requête a déjà eu lieu (_last_activity > 0), et on est resté idle

    # depuis >= keepwarm_interval. `_local_gen_lock` non bloquant => on ne ping JAMAIS pendant

    # une génération LOCALE (--parallel 1). On ne ping QUE le modèle déjà chargé => pas de swap.

    def _keepwarm_loop():

        while True:
            interval = float(_settings["keepwarm_interval"])  # relu à chaud
            time.sleep(max(15.0, min(interval / 3.0, 60.0)))

            # Activable/désactivable à chaud : si coupé, on ne ping pas (thread reste en veille).
            if not _settings["keepwarm_enabled"]:
                continue

            last = _last_activity[0]

            if last <= 0 or (time.time() - last) < interval:
                continue

            if not _local_gen_lock.acquire(blocking=False):
                continue  # génération locale en cours => déjà chaud

            try:
                sess = _cur["session"]

                model = sess.conversation.model if sess else None

                if not model:
                    continue

                # Keep-warm = garder chaud le modèle LOCAL (éviter le cold start). Un modèle
                # DISTANT n'a pas de cold start côté machine ET est PAYANT à l'appel : le
                # pinger en boucle brûlerait des crédits pour rien -> on saute.
                if model in remote_model_ids:
                    continue

                # Keep-warm v2 : on ré-amorce le PRÉFIXE DE LA CONVERSATION au lieu
                # d'un « ping » — l'ancien ping gardait le modèle chaud mais ÉCRASAIT
                # le cache KV du fil (slot unique) : chaque reprise re-préfillait
                # TOUT (bug 2026-07-10). Ici : modèle chaud ET cache chaud ; si le
                # cache est déjà bon, le prefill est ~nul -> quasi gratuit. Repli
                # ping pour une session encore vide (rien à amorcer, juste chauffer).
                if not _prime_slot(sess):
                    for _kind, _chunk in client.stream_chat(
                        [{"role": "user", "content": "ping"}],
                        "",
                        1,
                        model=model,
                        thinking=False,
                    ):
                        pass

                _last_activity[0] = time.time()  # gardé chaud => relance un intervalle

            except Exception:  # noqa: BLE001 - keep-warm best-effort, jamais bloquant
                pass

            finally:
                _local_gen_lock.release()

    # Thread toujours lancé (il dort à l'idle) : ainsi activer/désactiver keep-warm dans la
    # console prend effet à chaud, sans redémarrer loom.web.
    threading.Thread(target=_keepwarm_loop, daemon=True, name="loom-keepwarm").start()

    return app
