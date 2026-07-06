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

from pathlib import Path


from flask import Flask, Response, render_template, request


from loom.agent import context

from loom.agent.client import set_debug_log_path

from loom.extend.skills import (
    collect_skills,
    effective_skills,
    read_skill_source,
    render_catalog,
    write_skill_source,
)

from loom.prompts import CHAT_SYSTEM_STRONG
from loom.runtime import model_store
from loom.runtime.platform_info import detect as platform_detect

from loom.runtime.models_profile import load_profile


MAX_IMAGE_BYTES = 10 * 1024 * 1024

MAX_IMAGES = 6  # nb max d'images jointes ÃƒÂ  un message (au-delÃƒÂ  : ignorÃƒÂ©es)


# DÃƒÂ©tection d'un chemin ABSOLU dans un message (Windows `C:\...` / `C:/...` ou POSIX

# `/...`). Sert ÃƒÂ  l'auto-adoption du dossier de travail : si l'utilisateur dÃƒÂ©signe un

# dossier existant, la session l'adopte -> run_shell tourne dedans, les chemins relatifs

# s'y rÃƒÂ©solvent, et il n'a PAS ÃƒÂ  pointer le dossier dans l'UI.

_PATH_RE = re.compile(r"""(?:[A-Za-z]:[\\/]|[\\/])[^\s"'`<>|*?]*""")


def _detect_workspace(message: str, root: str | None = None) -> str | None:
    """Renvoie le dossier EXISTANT le plus spÃƒÂ©cifique citÃƒÂ© dans `message` (rÃƒÂ©solu absolu),

    ou None. Un fichier existant -> son dossier parent. N'adopte QUE du rÃƒÂ©el (isdir/isfile),

    donc un chemin de rÃƒÂ©fÃƒÂ©rence faux n'a aucun effet.



    Si `root` est fourni, on accepte aussi un PROJET citÃƒÂ© par son seul NOM quand c'est un

    sous-dossier direct de `root` (ex. Ã‚Â« ... pour energy-data-platform Ã‚Â» sans le chemin

    complet). Match EXACT sur un sous-dossier rÃƒÂ©el -> pas de faux positif sur un mot courant.

    """

    found: list[str] = []

    for raw in _PATH_RE.findall(message):
        p = raw.rstrip(".,;:!?)]}Ã‚Â»\"'`").strip()

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

    best = max(found, key=len)  # le chemin le plus long = le plus spÃƒÂ©cifique

    try:
        return str(Path(best).resolve())

    except OSError:
        return None


def _sse(event_type: str, **fields) -> str:
    """SÃƒÂ©rialise un ÃƒÂ©vÃƒÂ©nement Server-Sent Events (UTF-8, accents prÃƒÂ©servÃƒÂ©s)."""

    return f"data: {json.dumps({'type': event_type, **fields}, ensure_ascii=False)}\n\n"


def _infer_title(client, model, message: str) -> str:
    """Titre court (3-5 mots) infÃƒÂ©rÃƒÂ© par le modÃƒÂ¨le depuis la 1re demande, pour ne pas laisser
    la session Ã‚Â« Nouvelle session Ã‚Â». La logique (thinking coupÃƒÂ© + tentatives) vit dans
    client.infer_title ; ici on ne garde QUE le repli sur le dÃƒÂ©but du message si le modÃƒÂ¨le ne
    renvoie rien d'exploitable."""
    try:
        title = client.infer_title(model, message)
    except Exception:  # noqa: BLE001 - un titre est cosmÃƒÂ©tique, jamais bloquant
        title = ""
    if not title:
        title = message.strip().splitlines()[0][:48].strip() or "Session"
    return title


def _build_user_content(message, images, *, is_vision, stash_dir) -> str | list:
    """Construit le contenu du message user ÃƒÂ  partir de N images jointes (max MAX_IMAGES).



    - ModÃƒÂ¨le VISION : images EMBARQUÃƒâ€°ES (data URI) dans un message multimodal Ã¢â‚¬â€ il les VOIT.

    - ModÃƒÂ¨le TEXTE-ONLY : images ENREGISTRÃƒâ€°ES sur disque (stash_dir) ; le message reste du

      texte, avec les chemins + consigne d'inspecter via read_image (qui route vers un VLM).



    LÃƒÂ¨ve ValueError (-> 400) si une image est trop grande ou n'est pas une image.

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
        # EMBARQUÃƒâ€°ES : le modÃƒÂ¨le multimodal les VOIT dÃƒÂ©jÃƒÂ . Sans ce rappel, il croit devoir

        # les rouvrir via read_image, devine un chemin, ÃƒÂ©choue.

        note = (
            f"[{len(read)} image(s) jointe(s) ÃƒÂ  ce message Ã¢â‚¬â€ tu les VOIS dÃƒÂ©jÃƒÂ  directement "
            "ci-dessous. N'utilise PAS read_image pour elles, ne devine aucun chemin.]\n"
        )

        parts: list = [{"type": "text", "text": note + message}]

        for _name, mime, blob in read:
            b64 = base64.b64encode(blob).decode("ascii")

            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )

        return parts

    # TEXTE-ONLY : on enregistre sur disque et on donne les chemins ; read_image (routÃƒÂ© vers

    # un VLM) dÃƒÂ©crira ÃƒÂ  la demande. Le modÃƒÂ¨le ne voit rien inline, inutile de l'embarquer.

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
        f"[{len(paths)} image(s) jointe(s) ÃƒÂ  ce message, enregistrÃƒÂ©e(s) sur disque. Ton "
        "modÃƒÂ¨le ne les voit pas directement : inspecte-les avec read_image(path[, question]) "
        "Ã¢â‚¬â€ un modÃƒÂ¨le vision te les dÃƒÂ©crira. Chemins :\n" + listing + "]\n"
    )

    return note + message


# Mots qui EFFACENT l'objectif de session via Ã‚Â« /goal <mot> Ã‚Â» (faÃƒÂ§on /goal de Claude Code).

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
    """Consigne dÃƒÂ©pliÃƒÂ©e par la commande /init : fait explorer le dossier `target_display`
    et ÃƒÂ©crire `target_display/loom.md` (fiche projet). Fonction pure (testable)."""
    return (
        f"GÃƒÂ©nÃƒÂ¨re la fiche projet du dossier de travail Ã‚Â« {target_display} Ã‚Â». Explore-le "
        "avec tes outils (list_dir, find_files, read_file) Ã¢â‚¬â€ n'invente RIEN, base-toi "
        "seulement sur ce que tu lis vraiment. RepÃƒÂ¨re : le but du projet, la "
        "stack/langages/frameworks, l'arborescence importante, comment l'installer / le "
        "lancer / le tester, et les conventions (lint, gestionnaire de paquets, CI/CD). "
        "Puis Ãƒâ€°CRIS le fichier avec write_file au chemin EXACT "
        f"Ã‚Â« {target_display}/loom.md Ã‚Â» (ÃƒÂ  la racine de CE dossier, nulle part ailleurs), "
        "en markdown structurÃƒÂ© : titre du projet, puis les sections `## But`, `## Stack`, "
        "`## Arborescence`, `## Lancer / Tester`, `## Conventions`, `## Points d'attention`. "
        "Concis et factuel ; si une info manque, dis-le plutÃƒÂ´t que de la deviner. Termine "
        "en confirmant le chemin ÃƒÂ©crit."
    )


# Verbe compact par outil pour la TRACE D'ACTIONS persistÃƒÂ©e (anti-amnÃƒÂ©sie). Les outils

# de navigation (find/search/list) en sont absents : on mÃƒÂ©morise les LECTURES et les

# CHANGEMENTS d'ÃƒÂ©tat, pas les allers-retours d'exploration.

_TRACE_VERB = {
    "read_file": "lu",
    "read_document": "lu",
    "read_image": "vu",
    "write_file": "crÃƒÂ©ÃƒÂ©",
    "append_file": "complÃƒÂ©tÃƒÂ©",
    "edit_file": "modifiÃƒÂ©",
    "run_shell": "exÃƒÂ©cutÃƒÂ©",
    "dispatch_agent": "dÃƒÂ©lÃƒÂ©guÃƒÂ©",
}

_WRITE_NAMES = {
    "write_file",
    "append_file",
    "edit_file",
}


def _action_trace_line(evt: dict) -> str | None:
    """Rend un `tool_result` en ligne compacte pour la trace, ou None s'il ne mÃƒÂ©rite pas

    d'ÃƒÂªtre mÃƒÂ©morisÃƒÂ© (navigation, ÃƒÂ©criture ÃƒÂ©chouÃƒÂ©e/diffÃƒÂ©rÃƒÂ©e)."""

    name = evt.get("name") or ""

    verb = _TRACE_VERB.get(name)

    if verb is None:
        return None

    ok = bool(evt.get("ok"))

    # Une ÃƒÂ©criture ÃƒÂ©chouÃƒÂ©e/diffÃƒÂ©rÃƒÂ©e n'est pas un changement d'ÃƒÂ©tat ÃƒÂ  retenir (si elle

    # rÃƒÂ©ussit ensuite dans le mÃƒÂªme tour, c'est cette rÃƒÂ©ussite-lÃƒÂ  qui sera tracÃƒÂ©e).

    if name in _WRITE_NAMES and not ok:
        return None

    mark = "" if ok else "Ã¢Å“â€” "

    if name == "run_shell":
        head = (evt.get("preview") or "").split("\n")[0][:60]

        return f"{mark}{verb} shell: {head}".strip()

    if name == "dispatch_agent":
        return f"{mark}{verb} une sous-tÃƒÂ¢che"

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
    remote_store_path=None,
    config_defaults_path=None,
    config_local_path=None,
    local_models=None,
) -> Flask:

    app = Flask(__name__)

    # Recharge le template ÃƒÂ  chaque requÃƒÂªte : ÃƒÂ©diter index.html ne nÃƒÂ©cessite pas de

    # redÃƒÂ©marrer le serveur (sinon Jinja sert la version compilÃƒÂ©e au dÃƒÂ©marrage).

    app.config["TEMPLATES_AUTO_RELOAD"] = True

    app.jinja_env.auto_reload = True

    # Pas de cache navigateur sur les statiques (app.js/css) : ÃƒÂ©diter le frontend prend effet

    # au simple rechargement, sans hard-refresh. Sinon un app.js mis ÃƒÂ  jour reste servi depuis

    # le cache et diverge du template rechargÃƒÂ© cÃƒÂ´tÃƒÂ© serveur (bugs fantÃƒÂ´mes).

    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Windows sert les .js/.css statiques avec un mimetype tirÃƒÂ© du registre, souvent SANS
    # `charset` -> le navigateur les dÃƒÂ©code en Windows-1252 et les glyphes UTF-8 deviennent
    # du mojibake (ÃƒÂ© -> ÃƒÆ’Ã‚Â©, fleche -> ÃƒÂ¢Ã¢â‚¬Â ', check -> ÃƒÂ¢Ã…â€œ"). On force `charset=utf-8` sur toute
    # rÃƒÂ©ponse textuelle qui n'en dÃƒÂ©clare pas, pour un dÃƒÂ©codage client correct quelle que
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

    # Seuil de microcompact INTERNE ÃƒÂ  la boucle d'outils : on vide les vieux rÃƒÂ©sultats

    # d'outils quand le contexte vivant approche la fenÃƒÂªtre du modÃƒÂ¨le (en rÃƒÂ©servant la place

    # de la rÃƒÂ©ponse). Distinct du rÃƒÂ©sumÃƒÂ© inter-tours (context_budget) qui ne porte que sur

    # l'historique persistÃƒÂ©. CalculÃƒÂ© PAR MODÃƒË†LE (_model_limits) : un modÃƒÂ¨le distant ÃƒÂ  grande

    # fenÃƒÂªtre exploite SA fenÃƒÂªtre + son max_tokens au lieu du global, sans toucher au rÃƒÂ©glage

    # local. Repli sur le global pour tout id non listÃƒÂ©.

    model_contexts = dict(model_contexts or {})

    model_max_tokens = dict(model_max_tokens or {})
    # Prix par modÃƒÂ¨le ($/M tokens) : id -> (input, output, cached). Local/absent -> (0,0,0).
    model_prices = dict(model_prices or {})

    # Config VIVANTE de loom.web : au lieu de figer ces valeurs au dÃƒÂ©marrage, on les tient dans
    # un holder mutable que le runtime consulte Ãƒâ‚¬ CHAQUE usage, et qu'on RECHARGE depuis le
    # disque aprÃƒÂ¨s chaque ÃƒÂ©dition (/config/set). RÃƒÂ©sultat : permissions, plafonds, budgets,
    # reflect et keep-warm prennent effet Ãƒâ‚¬ CHAUD, sans redÃƒÂ©marrer loom.web. (Les params du
    # SERVEUR MODÃƒË†LE restent hors de portÃƒÂ©e : autre process, cf. serve.py.)
    _settings = {
        "max_tokens": max_tokens,
        "context_budget": context_budget,
        "keep_recent": keep_recent,
        "identity_max_tokens": identity_max_tokens,
        "reflect_enabled": reflect_enabled,
        "reflect_min_actions": reflect_min_actions,
        "keepwarm_enabled": keepwarm_enabled,
        "keepwarm_interval": keepwarm_interval,
        "permission_mode": permission_mode,
    }
    # La fonction de permission capture cfg.permissions ; pour appliquer un changement de mode
    # ÃƒÂ  chaud on la remplace dans ce holder (le runtime lit _perm["fn"]).
    _perm = {"fn": permission}

    def _reload_app_config():
        """Relit defaults.toml + local.toml et met ÃƒÂ  jour le holder + la permission (ÃƒÂ  chaud).
        Best-effort : une config invalide ne casse pas l'app en cours (on garde l'ancienne)."""
        if not (config_defaults_path and config_local_path):
            return
        try:
            from loom.agent.context import effective_context_budget
            from loom.config import load_config
            from loom.permissions import evaluate

            c = load_config(config_defaults_path, config_local_path)
        except Exception:  # noqa: BLE001 - reload best-effort, jamais fatal
            return
        _settings.update(
            max_tokens=c.chat.max_tokens,
            context_budget=effective_context_budget(
                c.chat.context_token_budget, c.context, c.chat.max_tokens
            ),
            keep_recent=c.chat.keep_recent_messages,
            identity_max_tokens=c.chat.identity_max_tokens,
            reflect_enabled=c.chat.reflect_enabled,
            reflect_min_actions=c.chat.reflect_min_actions,
            keepwarm_enabled=c.chat.keepwarm_enabled,
            keepwarm_interval=c.chat.keepwarm_interval,
            permission_mode=c.permissions.mode,
        )
        _perm["fn"] = lambda name, args: evaluate(name, args, c.permissions)

    def _regen_swap_yaml():
        """RÃƒÂ©gÃƒÂ©nÃƒÂ¨re le llama-swap.yaml depuis la config (llama-swap -watch-config le recharge).
        Best-effort, silencieux si serve indispo. Renvoie True si ÃƒÂ©crit."""
        if not (config_defaults_path and config_local_path):
            return False
        try:
            from loom.runtime.serve import regenerate_swap_yaml

            return bool(regenerate_swap_yaml(config_defaults_path, config_local_path))
        except Exception:  # noqa: BLE001
            return False

    def _apply_to_model_server(section):
        """Param SERVEUR/OVERRIDE (affecte le lancement de llama-server) : rÃƒÂ©gÃƒÂ©nÃƒÂ¨re le yaml et
        dÃƒÂ©charge les modÃƒÂ¨les locaux -> ils se relancent avec les nouveaux args au prochain usage
        (llama-swap -watch-config). Ne touche pas les autres process. Best-effort, en tÃƒÂ¢che de fond."""
        if section not in ("server", "override"):
            return
        if _regen_swap_yaml():
            threading.Thread(
                target=client.unload_local, daemon=True, name="loom-reload-models"
            ).start()

    def _price_of(model_id):
        return model_prices.get(model_id, (0.0, 0.0, 0.0))

    def _ctx_info(model_id):
        """(fenÃƒÂªtre de contexte, source) du modÃƒÂ¨le -> dÃƒÂ©nominateur de la jauge + provenance.

        Distant : on demande D'ABORD au PROVIDER (`client.remote_context`, mis en cache) Ã¢â‚¬â€
        c'est le modÃƒÂ¨le lui-mÃƒÂªme qui fait autoritÃƒÂ©. S'il ne publie rien (Z.ai/OpenAI), repli
        sur la valeur dÃƒÂ©clarÃƒÂ©e en config. Local : la fenÃƒÂªtre est celle qu'on a ALLOUÃƒâ€°E au
        serveur (n_ctx) = notre limite volontaire, signalÃƒÂ©e comme telle. Sources possibles :
        `provider` (fait autoritÃƒÂ©), `config` (dÃƒÂ©clarÃƒÂ©, non vÃƒÂ©rifiable), `local` (notre limite)."""
        declared = model_contexts.get(model_id) or context_window
        if model_id in remote_model_ids:
            provided = client.remote_context(model_id)
            if provided:
                return provided, "provider"
            return declared, "config"
        return declared, "local"

    def _model_limits(model_id):
        """(plafond de sortie, seuil de microcompact) pour `model_id`.

        Le max_tokens global est une contrainte LOCALE (calibrÃƒÂ©e pour la VRAM de la machine).
        Un modÃƒÂ¨le DISTANT ne l'hÃƒÂ©rite PAS : sa machine est plus puissante. Non dÃƒÂ©fini -> None
        (plafond OMIS dans la requÃƒÂªte, le provider applique SA limite). La rÃƒÂ©serve de
        microcompact reste modeste cÃƒÂ´tÃƒÂ© distant (leur fenÃƒÂªtre est large, le seuil compte peu)."""
        win = model_contexts.get(model_id) or context_window
        explicit = model_max_tokens.get(model_id)
        if model_id in remote_model_ids:
            cap = explicit  # None possible -> pas de cap imposÃƒÂ©
            reserve = explicit or 8192
        else:
            cap = explicit or _settings["max_tokens"]  # local : plafond global
            reserve = cap
        return cap, max(1024, win - reserve - 1024)

    def _totals(conv):
        """Compteurs de session + fenÃƒÂªtre du modÃƒÂ¨le (jauge de remplissage du contexte).
        La fenÃƒÂªtre dÃƒÂ©pend du modÃƒÂ¨le (que l'app connaÃƒÂ®t), pas de la Conversation -> jointe ici,
        avec sa source (provider/config/local) pour que l'UI signale si le chiffre fait autoritÃƒÂ©."""
        win, src = _ctx_info(conv.model)
        return {**conv.usage_totals(), "context_window": win, "context_source": src}

    models = list(models or [])
    # SÃƒÂ©rialise TOUTES les ÃƒÂ©critures de fichiers de config/modÃƒÂ¨les (model.toml, local.toml,
    # defaults.toml, store JSON) : Flask est threaded -> deux ÃƒÂ©ditions concurrentes du mÃƒÂªme
    # fichier feraient une course read-modify-write (la derniÃƒÂ¨re ÃƒÂ©crase l'autre).
    _toml_lock = threading.Lock()

    vision_models = set(vision_models or [])  # ids des modÃƒÂ¨les avec mmproj (vision)

    remote_model_ids = set(remote_model_ids or [])  # ids servis par une API distante
    # ids des modÃƒÂ¨les LOCAUX (servis par llama-swap sur la machine) = tout sauf les distants.
    # Sert ÃƒÂ  /machine_state (quel modÃƒÂ¨le machine est chargÃƒÂ©).
    local_model_ids = [m for m in (models or []) if m not in remote_model_ids]
    # DÃƒÂ©tails des modÃƒÂ¨les LOCAUX (onglet ModÃƒÂ¨les locaux) : id/dir/offload/context. `dir` porte
    # le model.toml -> ÃƒÂ©dition du tuning machine via tomlkit.
    local_model_specs = list(local_models or [])

    remote_model_names = dict(
        remote_model_names or {}
    )  # id Loom -> vrai modÃƒÂ¨le provider

    available_tools = list(available_tools or [])

    # Concurrence PAR SESSION : chaque session a son propre verrou de gÃƒÂ©nÃƒÂ©ration et son
    # signal d'annulation -> plusieurs sessions (onglets) gÃƒÂ©nÃƒÂ¨rent EN PARALLÃƒË†LE. Une nouvelle
    # soumission n'interrompt QUE la gÃƒÂ©nÃƒÂ©ration de SA session (pas les autres). Verrou GLOBAL
    # `_local_gen_lock` en plus pour les modÃƒÂ¨les LOCAUX : llama-swap n'en sert qu'un ÃƒÂ  la fois
    # -> deux gÃƒÂ©nÃƒÂ©rations locales se sÃƒÂ©rialisent (limitation machine connue, signalÃƒÂ©e ÃƒÂ  l'UI).
    _gen_guard = threading.Lock()
    _sess_locks: dict[str, threading.Lock] = {}
    _sess_cancel: dict[str, threading.Event] = {}
    _local_gen_lock = threading.Lock()
    # Ãƒâ€°vÃƒÂ©nement d'annulation de la gÃƒÂ©nÃƒÂ©ration EN COURS sur CE thread (pour _confirm, qui tourne
    # dans le thread de gÃƒÂ©nÃƒÂ©ration) Ã¢â‚¬â€ posÃƒÂ© au dÃƒÂ©but de /chat.
    _confirm_local = threading.local()

    def _lock_for(sid: str) -> threading.Lock:
        with _gen_guard:
            return _sess_locks.setdefault(sid, threading.Lock())

    def _cancel_for(sid: str) -> threading.Event:
        with _gen_guard:
            return _sess_cancel.setdefault(sid, threading.Event())

    # DÃƒÂ©cisions de confirmation en attente : tool_call_id -> {event, approved}.

    # RenseignÃƒÂ©es par la route /tool_decision (autre thread), consommÃƒÂ©es par _confirm.

    pending: dict = {}

    # Horodatage de la derniÃƒÂ¨re fin de gÃƒÂ©nÃƒÂ©ration (0 = jamais). Le keep-warm ne pinge

    # qu'aprÃƒÂ¨s une vraie activitÃƒÂ© et seulement ÃƒÂ  l'idle (cf. thread plus bas).

    _last_activity = [0.0]

    # Session active : un fil persistant par projet. Tout passe par la session courante

    # (conversation + persistance) ; un seul mode, plus de legacy.

    _cur: dict = {"session": None}

    # Cache d'OBJETS session en mÃƒÂ©moire : UNE instance par session, partagÃƒÂ©e entre requÃƒÂªtes
    # (onglets) -> pas de sauvegardes qui se clobberent quand plusieurs sessions tournent en
    # parallÃƒÂ¨le. `_cur` pointe la session FOCUS (dÃƒÂ©faut de l'index).
    _sessions_cache: dict = {}

    def _ensure_model(sess):
        # Une session neuve peut naÃƒÂ®tre sans modÃƒÂ¨le -> requÃƒÂªte model="" -> llama-swap renvoie
        # 404. On garantit un modÃƒÂ¨le valide (le 1er = dÃƒÂ©faut) ; corrige aussi les vides.
        if sess is not None and not sess.conversation.model and models:
            sess.conversation.set_model(models[0])
            session_store.save(sess)
        return sess

    def _get_session(sid: str):
        """Session par id, depuis le cache (une instance) ou chargÃƒÂ©e du disque. None si absente."""
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

        persistance. Point de vÃƒÂ©ritÃƒÂ© unique pour tous les endpoints."""

        sess = _session()

        return sess.conversation, (lambda: session_store.save(sess))

    def _confirm(tool_id: str, name: str, args: dict) -> bool:
        """Bloque jusqu'ÃƒÂ  la dÃƒÂ©cision UI (OK/Refuser). Interruptible et bornÃƒÂ©.



        Renvoie False si refus, timeout, ou si une nouvelle soumission annule

        (cancel_event) Ã¢â‚¬â€ ÃƒÂ©vite tout deadlock sur le verrou de chat.

        """

        ev = threading.Event()

        pending[tool_id] = {"event": ev, "approved": False}

        deadline = time.monotonic() + confirm_timeout

        # Annulation de LA session dont on exÃƒÂ©cute la gÃƒÂ©nÃƒÂ©ration (thread-local, posÃƒÂ© par /chat).
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
        """Contexte du panneau Skills : la liste COMPLÃƒË†TE (pour les cases), l'ensemble des
        skills ACTIFS (non dÃƒÂ©sactivÃƒÂ©s) et ceux qui ont un override de session (badge UI)."""
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
            "current_model": conv.model,
            "thinking": conv.thinking,
            "available_tools": available_tools,
            "active_tools": conv.active_tools,
            "workspace_dir": ws,
            "sessions": sessions,
            "active_session": active_id,
            "permission_mode": _settings["permission_mode"],
            # Ãƒâ€°tat initial pour l'hydratation cÃƒÂ´tÃƒÂ© client (Preact). On ÃƒÂ©chappe '<'
            # pour ne pas pouvoir fermer la balise <script> depuis le contenu.
            "init_json": json.dumps(
                {
                    "messages": conv.messages,
                    "thinking": conv.thinking,
                    "usage_totals": _totals(conv),
                    # Onglet initial : la session active (id/titre/modÃƒÂ¨le/workspace) + toutes
                    # les sessions (pour la sidebar). Le multi-onglets s'hydrate lÃƒÂ -dessus.
                    "active_session": active_id,
                    "title": sess.title,
                    "model": conv.model,
                    "workspace": ws,
                    "sessions": sessions,
                },
                ensure_ascii=False,
            ).replace("<", "\\u003c"),
        }

    # Garde CSRF : le serveur ÃƒÂ©coute sur 127.0.0.1 SANS auth, et tourne souvent en

    # mode=allow (outils exÃƒÂ©cutÃƒÂ©s sans confirmation). Une page web tierce OUVERTE dans le

    # navigateur de l'utilisateur peut POSTer en cross-origin vers 127.0.0.1 (requÃƒÂªtes

    # Ã‚Â« simples Ã‚Â», sans preflight) et piloter l'agent local -> exÃƒÂ©cution d'outils. Le

    # binding localhost NE protÃƒÂ¨ge PAS de ÃƒÂ§a. On refuse les POST dont l'en-tÃƒÂªte

    # Sec-Fetch-Site (envoyÃƒÂ© par tous les navigateurs modernes) trahit une origine tierce.

    # `same-origin`/`none` (notre propre page, barre d'adresse) passent ; un client non-

    # navigateur (curl, tests) n'envoie pas l'en-tÃƒÂªte -> autorisÃƒÂ©, on ne casse rien.

    @app.before_request
    def _csrf_guard():

        if request.method != "POST":
            return None

        if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            return Response("requÃƒÂªte cross-origin refusÃƒÂ©e (CSRF)", status=403)

        return None

    @app.get("/")
    def index() -> str:

        return render_template("index.html", **_index_context())

    @app.post("/reset")
    def reset() -> str:

        conv, save = _ctx()

        conv.reset()

        save()

        # Le fil repart ÃƒÂ  neuf -> on efface aussi le journal d'affichage temps rÃƒÂ©el.
        session_store.clear_timeline(_session().id)

        return render_template("index.html", **_index_context())

    @app.post("/chat")
    def chat():

        message = (request.form.get("message") or "").strip()

        if not message or len(message) > 5000:
            return Response("message invalide", status=400)

        # Session CIBLE : par `session_id` (onglet) sinon la session focus. Chaque session a
        # son verrou : une nouvelle soumission n'interrompt QUE la gÃƒÂ©nÃƒÂ©ration de SA session,
        # les autres onglets continuent en parallÃƒÂ¨le.
        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(req_sid) or _session()
        _cur["session"] = sess  # focus (dÃƒÂ©faut de l'index)
        sid = sess.id
        chat_lock = _lock_for(sid)
        cancel_event = _cancel_for(sid)

        if not chat_lock.acquire(blocking=False):
            # Une gÃƒÂ©nÃƒÂ©ration de CETTE session tourne dÃƒÂ©jÃƒÂ  : on l'annule et on attend le verrou.

            cancel_event.set()

            if not chat_lock.acquire(timeout=interrupt_wait):
                return Response("occupÃƒÂ© : cette session gÃƒÂ©nÃƒÂ¨re dÃƒÂ©jÃƒÂ ", status=429)

        # On tient le verrou : repartir d'un signal d'annulation propre.

        cancel_event.clear()

        conv = sess.conversation
        save = lambda: session_store.save(sess)  # noqa: E731

        # Commande /goal : pilote l'OBJECTIF de complÃƒÂ©tion de la session.

        # Ã‚Â« /goal <condition> Ã‚Â» pose l'objectif ET DÃƒâ€°MARRE aussitÃƒÂ´t une itÃƒÂ©ration (comme /goal

        # de Claude Code : Ã‚Â« exÃƒÂ©cute une premiÃƒÂ¨re itÃƒÂ©ration immÃƒÂ©diatement Ã‚Â») Ã¢â‚¬â€ l'objectif maintient

        # ensuite l'agent au travail (garde cÃƒÂ´tÃƒÂ© client.py) jusqu'ÃƒÂ  ce qu'un ÃƒÂ©valuateur le juge

        # PROUVÃƒâ€° atteint. Ã‚Â« /goal Ã‚Â» seul = statut ; Ã‚Â« /goal clear|stop|Ã¢â‚¬Â¦ Ã‚Â» = efface. Ces deux-lÃƒÂ 

        # ne lancent pas de tour modÃƒÂ¨le (ack immÃƒÂ©diat) ; poser un objectif, si.

        if message == "/goal" or message.startswith("/goal "):
            arg = message[len("/goal") :].strip()

            if arg and arg.lower() not in _GOAL_CLEAR_WORDS:
                # Pose l'objectif et AMORCE le travail : on remplace le message par une consigne

                # de dÃƒÂ©marrage et on laisse le flux normal tourner, objectif dÃƒÂ©sormais actif.

                conv.set_goal(arg)

                save()

                message = (
                    f"Objectif ÃƒÂ  atteindre : {arg}\n"
                    "Commence MAINTENANT ÃƒÂ  agir pour l'atteindre, et PROUVE-le (exÃƒÂ©cute, montre "
                    "la sortie rÃƒÂ©elle). Ne t'arrÃƒÂªte pas tant qu'il n'est pas dÃƒÂ©montrÃƒÂ© atteint."
                )

                # (pas de return : on tombe dans la gÃƒÂ©nÃƒÂ©ration normale ci-dessous)

            else:
                if not arg:
                    ack = (
                        f"Objectif courant : Ã‚Â« {conv.goal} Ã‚Â» (actif jusqu'ÃƒÂ  preuve d'atteinte, "
                        "/goal clear pour l'effacer)."
                        if conv.goal
                        else "Aucun objectif actif. Pose-en un : /goal <condition vÃƒÂ©rifiable>."
                    )

                else:
                    conv.set_goal("")

                    save()

                    ack = "Objectif effacÃƒÂ© Ã¢â‚¬â€ retour au mode normal (arrÃƒÂªt au stop naturel)."

                chat_lock.release()

                def _goal_ack():

                    yield _sse("text", text=ack)

                    yield _sse("done")

                return Response(_goal_ack(), mimetype="text/event-stream")

        # Commande /init : gÃƒÂ©nÃƒÂ¨re une fiche projet `loom.md` Ãƒâ‚¬ LA RACINE DU DOSSIER DE TRAVAIL
        # de la session (celui dÃƒÂ©fini dans l'UI, ou ciblÃƒÂ© par Ã‚Â« /init <chemin> Ã‚Â»). Macro de
        # prompt (pas de scanner figÃƒÂ©) : on dÃƒÂ©plie une consigne et la boucle tool-use explore
        # puis ÃƒÂ©crit le fichier. `/init <dossier existant>` adopte d'abord ce dossier comme
        # workspace (exploration + ÃƒÂ©criture cohÃƒÂ©rentes).
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
            # (pas de return : le flux normal ci-dessous exÃƒÂ©cute la consigne)

        # Plus de garde bloquant : un modÃƒÂ¨le texte-only ne reÃƒÂ§oit PAS l'image inline (qui

        # ferait planter un llama-server sans mmproj) Ã¢â‚¬â€ on la stocke sur disque et il l'inspecte

        # via read_image, routÃƒÂ© vers un modÃƒÂ¨le vision (cf. _build_user_content plus bas).

        # Logs PAR SESSION (au mÃƒÂªme titre que session.json) : (1) trace des ÃƒÂ©changes modÃƒÂ¨le

        # routÃƒÂ©e vers sessions/<id>/debug.log ; (2) copie du log serveur modÃƒÂ¨le global

        # (var/logs/serve.log) dans la session Ã¢â‚¬â€ doublon assumÃƒÂ©, pour tout avoir sous la main.

        _sdir = session_store.session_dir(_session().id)

        set_debug_log_path(_sdir / "debug.log")

        _serve_log = session_store.root.parent / "logs" / "serve.log"

        if _serve_log.exists():
            try:
                _sdir.mkdir(parents=True, exist_ok=True)

                shutil.copyfile(_serve_log, _sdir / "serve.log")

            except OSError:
                pass

        # Auto-adoption du dossier de travail : si le message dÃƒÂ©signe un dossier EXISTANT,

        # la session l'adopte avant le tour -> run_shell tourne dedans et les chemins

        # relatifs s'y rÃƒÂ©solvent, sans que l'utilisateur ait ÃƒÂ  pointer le dossier dans l'UI.

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

            # Journal d'affichage temps rÃƒÂ©el : on y consigne le message user (le journal est la
            # source de RÃƒâ€°-AFFICHAGE au rechargement -> il doit ÃƒÂªtre complet, user inclus).
            session_store.append_event(sess.id, "user", {"content": message})

            # Gestion du contexte : rÃƒÂ©sumÃƒÂ© auto si trop long

            if context.summarize(
                conv, client, _settings["context_budget"], _settings["keep_recent"]
            ):
                save()

            skills = effective_skills(
                collect_skills(skills_dir, plugins_dir, learned_dir=learned_skills_dir),
                overrides=conv.skill_overrides,
                disabled=conv.disabled_skills,
            )

            catalog = render_catalog(skills)

            # IdentitÃƒÂ© always-on (SOUL/USER/MEMORY) EN TÃƒÅ TE : c'est la dÃƒÂ©finition qui FAIT FOI

            # de qui est Loom (rÃƒÂ´le, persona, style). Le mode d'emploi opÃƒÂ©rationnel (outils,

            # rÃƒÂ¨gles) de chat.system.md vient APRÃƒË†S et s'y conforme Ã¢â‚¬â€ on ne plante plus un

            # cadrage gÃƒÂ©nÃƒÂ©rique d'abord pour le corriger 12k caractÃƒÂ¨res plus loin. Always-on =>

            # survit toujours ÃƒÂ  la microcompaction/summarization (qui ne touchent que

            # l'historique). BornÃƒÂ©e par identity_max_tokens. Cf. design Ã‚Â§5.6.

            _idblk = ""

            if identity_paths:
                from loom.memory.identity import identity_block

                _idblk = identity_block(
                    identity_paths["soul_path"],
                    identity_paths["user_path"],
                    identity_paths["memory_md_path"],
                    max_tokens=_settings["identity_max_tokens"],
                )

            # TIER du harnais : un modÃƒÂ¨le DISTANT (API, non quantifiÃƒÂ©) se pilote seul -> prompt

            # ALLÃƒâ€°GÃƒâ€° (identitÃƒÂ© + outils + mÃƒÂ©moire + sÃƒÂ©curitÃƒÂ©), sans le scaffolding de comportement

            # de chat.system.md qui ne sert qu'ÃƒÂ  un petit modÃƒÂ¨le local. Le flag `strong` sert

            # aussi (plus bas) ÃƒÂ  couper les gardes de comportement dans la boucle d'outils.

            strong = bool(conv.model and conv.model in remote_model_ids)

            base_prompt = CHAT_SYSTEM_STRONG if strong else conv.system_prompt

            system_prompt = f"{_idblk}\n\n{base_prompt}" if _idblk else base_prompt

            # Distant (strong) : la machine du provider encaisse le parallÃƒÂ©lisme -> on incite ÃƒÂ 
            # GROUPER les sous-agents indÃƒÂ©pendants dans un mÃƒÂªme tour (ils tournent en parallÃƒÂ¨le).
            if strong:
                system_prompt += (
                    "\n\nParallÃƒÂ©lisme : quand plusieurs sous-tÃƒÂ¢ches sont INDÃƒâ€°PENDANTES (auditer/"
                    "explorer des pans distincts), ÃƒÂ©mets PLUSIEURS dispatch_agent dans le MÃƒÅ ME "
                    "tour Ã¢â‚¬â€ ils s'exÃƒÂ©cutent EN PARALLÃƒË†LE, bien plus vite qu'un par tour. Un pan = "
                    "un agent, lance-les ensemble."
                )

            if catalog:
                system_prompt += f"\n\n{catalog}"

            # Le modÃƒÂ¨le ignore par dÃƒÂ©faut sous quel backend il tourne (le prompt dit

            # "Tu es Loom") -> il baratine quand on lui demande "quel modÃƒÂ¨le ?". On lui

            # injecte son modÃƒÂ¨le courant pour qu'il rÃƒÂ©ponde honnÃƒÂªtement. DISTANT vs LOCAL :

            # sans ÃƒÂ§a un modÃƒÂ¨le servi par une API rÃƒÂ©pÃƒÂ©tait Ã‚Â« je tourne en local/offline sur

            # llama.cpp Ã‚Â» (la persona de Loom est Ã‚Â« agent local Ã‚Â») -> confabulation d'infra.

            if conv.model:
                if conv.model in remote_model_ids:
                    _pm = remote_model_names.get(conv.model)

                    _label = (
                        f"Ã‚Â« {_pm} Ã‚Â» (route Ã‚Â« {conv.model} Ã‚Â»)"
                        if _pm
                        else f"Ã‚Â« {conv.model} Ã‚Â»"
                    )

                    system_prompt += (
                        f"\n\n# Ton moteur\nTon raisonnement est servi par le modÃƒÂ¨le DISTANT "
                        f"{_label}, via une API externe Ã¢â‚¬â€ PAS en local. Tes OUTILS, eux, "
                        "s'exÃƒÂ©cutent bien sur la machine de l'utilisateur, mais toi (le cerveau) "
                        "non. Ne prÃƒÂ©tends donc JAMAIS ÃƒÂªtre offline, ni tourner sur llama.cpp / "
                        "llama-swap / une carte graphique locale : ce serait faux. Si on te "
                        "demande quel modÃƒÂ¨le/moteur tu utilises, donne ce nom honnÃƒÂªtement, sans "
                        "inventer de dÃƒÂ©tails d'infrastructure."
                    )

                else:
                    system_prompt += (
                        f"\n\n# Ton moteur\nTu tournes sur le modÃƒÂ¨le local Ã‚Â« {conv.model} Ã‚Â». "
                        "Si on te demande quel modÃƒÂ¨le/moteur tu utilises, rÃƒÂ©ponds-le "
                        "honnÃƒÂªtement et directement (ce nom), sans esquiver."
                    )

            # SystÃƒÂ¨me : Loom dÃƒÂ©tecte SEUL l'OS et injecte ses conventions (shell, commandes,
            # chemins) -> le modÃƒÂ¨le produit du PowerShell sous Windows, du bash/unix sous
            # macOS/Linux, sans qu'on code l'OS en dur dans le prompt. Source unique partagÃƒÂ©e
            # avec run_shell (loom.runtime.platform_info) : jamais de divergence.

            system_prompt += "\n\n" + platform_detect().prompt_block()

            # Dossier de travail courant : le modÃƒÂ¨le l'IGNORE sinon et le devine en sondant

            # (git rev-parse ÃƒÂ  l'aveugle, list_dirÃ¢â‚¬Â¦) -> tours gaspillÃƒÂ©s. On le lui dit, avec

            # le rÃƒÂ©flexe anti-tÃƒÂ¢tonnement quand ce dossier n'est pas un repo git. Reste EN BAS

            # (contexte volatil, prÃƒÂ¨s de l'action).

            _ws = _session().workspace

            system_prompt += (
                f"\n\n# Dossier de travail courant\nTes commandes (run_shell) tournent dans "
                f"`{_ws}` et les chemins relatifs s'y rÃƒÂ©solvent Ã¢â‚¬â€ n'y rÃƒÂ©pÃƒÂ¨te pas le nom de ce "
                "dossier dans tes chemins. Si une commande git ÃƒÂ©choue par Ã‚Â« not a git "
                "repository Ã‚Â», c'est que CE dossier n'est pas un repo : fais UN list_dir pour "
                "repÃƒÂ©rer le bon sous-dossier (puis `git -C <sous-dossier>`), ne relance pas la "
                "mÃƒÂªme commande ÃƒÂ  l'identique."
            )

            # Objectif de session (/goal), en DIRECTIVE DOUCE : pas de juge externe qui te

            # contredit (retirÃƒÂ© Ã¢â‚¬â€ il recalait des preuves correctes). Tu restes seul maÃƒÂ®tre de

            # ta propre vÃƒÂ©rification : ne te dÃƒÂ©clare pas fini tant que l'objectif n'est pas

            # ATTEINT ET PROUVÃƒâ€° par tes exÃƒÂ©cutions (montre la sortie rÃƒÂ©elle) ; une fois prouvÃƒÂ©,

            # dis-le et arrÃƒÂªte-toi. L'utilisateur l'efface avec Ã‚Â« /goal clear Ã‚Â».

            if conv.goal:
                system_prompt += (
                    f"\n\n# Objectif de session\nTant qu'il est actif, oriente ton travail vers "
                    f"cet objectif et ne le dÃƒÂ©clare atteint qu'une fois PROUVÃƒâ€° par tes propres "
                    f"exÃƒÂ©cutions (sortie rÃƒÂ©elle affichÃƒÂ©e) :\n{conv.goal}"
                )

        except ValueError as exc:
            chat_lock.release()

            return Response(str(exc), status=400)

        except Exception as exc:  # noqa: BLE001
            chat_lock.release()

            return Response(f"erreur: {exc}", status=500)

        def generate():

            # Annulation de CETTE session, lue par _confirm (mÃƒÂªme thread de gÃƒÂ©nÃƒÂ©ration).
            _confirm_local.ev = cancel_event
            # Verrou modÃƒÂ¨le LOCAL : pris dans le try ci-dessous (avant le 1er appel modÃƒÂ¨le),
            # libÃƒÂ©rÃƒÂ© au finally. Distant -> jamais pris (vrai parallÃƒÂ¨le entre onglets).
            _local_held = False

            # Profil du modÃƒÂ¨le : correctifs dÃƒÂ©terministes (cadratins, guillemets

            # typographiques) appliquÃƒÂ©s au texte streamÃƒÂ© du chat. Le profil existe

            # dÃƒÂ©jÃƒÂ  pour les outils d'ÃƒÂ©criture (via tool_factory) ; on le recharge ici

            # pour l'appliquer AUSSI aux rÃƒÂ©ponses du modÃƒÂ¨le, pas seulement aux fichiers.

            _profile = load_profile(conv.model) if conv.model else None

            if adopted_ws:  # informe l'UI que le dossier de travail a ÃƒÂ©tÃƒÂ© adoptÃƒÂ©
                yield _sse("workspace", path=adopted_ws)

            answer = ""

            actions: list[str] = []  # trace compacte des outils (anti-amnÃƒÂ©sie)

            saved = False
            # Persistance AU FIL DE L'EAU : au lieu de tout sauver une seule fois EN FIN de tour
            # (un long audit interrompu/rechargÃƒÂ©/relancÃƒÂ© perdait TOUT), on met ÃƒÂ  jour EN PLACE
            # l'unique message assistant du tour (rÃƒÂ©ponse en cours + trace compacte des actions)
            # et on sauve ÃƒÂ  CHAQUE ÃƒÂ©tape marquante (outil terminÃƒÂ©, flux de texte) + ÃƒÂ  la fin.
            _turn = {"idx": None, "last": 0.0}

            # Journal d'affichage TEMPS RÃƒâ€°EL : chaque ÃƒÂ©vÃƒÂ©nement visible est ÃƒÂ©crit ÃƒÂ  l'instant
            # dans timeline.jsonl (append, zÃƒÂ©ro batch) -> rejouable au rechargement. On y met
            # les ÃƒÂ©vÃƒÂ©nements qui reconstruisent la vue (raisonnement, texte, cartes d'outils) ;
            # pas les compteurs (metrics/totals) ni les dÃƒÂ©corations live (tool_stream/args).
            _TL = {"reasoning", "text", "tool_call", "tool_result", "phase", "notice"}

            def _tl(event, **data):
                if event in _TL:
                    session_store.append_event(sess.id, event, data)
                return _sse(event, **data)

            def _persist(final=False):
                # On NE persiste pas les messages `tool` bruts (gonflerait le contexte + casserait
                # le rÃƒÂ©sumeur) : seulement le texte + la trace des actions. Un mÃƒÂªme tour = UN seul
                # message assistant, mis ÃƒÂ  jour en place (pas de doublons).
                nonlocal saved

                body = answer

                if actions:
                    trace = "[Actions de ce tour : " + " Ã‚Â· ".join(actions[:20]) + "]"

                    body = f"{body}\n\n{trace}" if body else trace

                if not body:  # rien ÃƒÂ  dire ET rien fait -> pas de bulle vide
                    return

                # PilotÃƒÂ© par Ãƒâ€°VÃƒâ€°NEMENT (chaque outil terminÃƒÂ© + fin), plus par un timer : le
                # temps rÃƒÂ©el de l'affichage vient du journal `timeline.jsonl`, pas d'ici. Ce
                # session.json ne porte que le contexte lean du modÃƒÂ¨le, inutile ÃƒÂ  chaque token.
                if _turn["idx"] is None:
                    conv.add("assistant", body)
                    _turn["idx"] = len(conv.messages) - 1
                else:
                    conv.messages[_turn["idx"]]["content"] = body

                save()
                saved = True

            # Registre construit selon les outils activÃƒÂ©s pour CETTE conversation

            # (toggles UI) ET le workspace de la session active : sans ÃƒÂ§a les outils

            # (write/edit/run_shell + sous-agent) retombent sur cfg.chat.workspace_dir

            # et ÃƒÂ©crivent ÃƒÂ  cÃƒÂ´tÃƒÂ© du dossier ciblÃƒÂ©.

            ws = _session().workspace

            registry = tool_factory(conv.active_tools, ws, conv)

            use_tools = registry is not None and len(registry)

            # Limites du modÃƒÂ¨le courant (distant = sa grande fenÃƒÂªtre ; local = global).

            eff_max_tokens, eff_compact = _model_limits(conv.model)

            # `strong` (tier distant=fort) est calculÃƒÂ© plus haut, ÃƒÂ  la construction du prompt :

            # il coupe ici les gardes de comportement (act_nudge, claim_audit, coupe non-progrÃƒÂ¨s).

            # On ne garde que outils + mÃƒÂ©moire + sÃƒÂ©curitÃƒÂ©. Un modÃƒÂ¨le local garde le harnais complet.

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

            recv_confirmed = 0  # reÃƒÂ§us confirmÃƒÂ©s par l'usage (tool-calls inclus)

            cur_turn = 0  # reÃƒÂ§us live du tour en cours (reset ÃƒÂ  chaque usage)

            sent_tokens = 0  # envoyÃƒÂ©s (prompt) cumulÃƒÂ©s via l'usage

            last_rate = 0.0  # dernier dÃƒÂ©bit mesurÃƒÂ©

            burst_start = None  # dÃƒÂ©but de rafale (dÃƒÂ©bit hors pauses outils)

            burst_tokens = 0

            last_tok = None

            # Auto-titre DÃƒË†S L'ENVOI (le titre dÃƒÂ©rive du MESSAGE, pas de la rÃƒÂ©ponse). Pour un
            # modÃƒÂ¨le DISTANT : on l'infÃƒÂ¨re en tÃƒÂ¢che de fond tout de suite et on le pousse au
            # client dÃƒÂ¨s qu'il est prÃƒÂªt (interleavÃƒÂ©), sans attendre la fin du tour -> l'onglet
            # prend son vrai nom en ~1-2s mÃƒÂªme sur une gÃƒÂ©nÃƒÂ©ration longue. Pour un modÃƒÂ¨le LOCAL,
            # on NE le fait PAS ici (llama-swap = 1 slot ; un appel concurrent contendrait avec
            # la gÃƒÂ©nÃƒÂ©ration) : on garde le titrage en fin de tour, quand le slot est libre.
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
                # ModÃƒÂ¨le LOCAL : llama-swap n'en sert qu'UN ÃƒÂ  la fois -> on sÃƒÂ©rialise via le
                # verrou global (limitation machine connue, signalÃƒÂ©e ÃƒÂ  l'UI). ModÃƒÂ¨le DISTANT :
                # pas de verrou -> cette session gÃƒÂ©nÃƒÂ¨re EN PARALLÃƒË†LE des autres onglets.
                if conv.model and conv.model not in remote_model_ids:
                    if not _local_gen_lock.acquire(blocking=False):
                        yield _sse(
                            "notice",
                            text=(
                                "modÃƒÂ¨le local occupÃƒÂ© : une autre session gÃƒÂ©nÃƒÂ¨re dÃƒÂ©jÃƒÂ  sur la "
                                "machine Ã¢â‚¬â€ mise en file (le parallÃƒÂ¨le rÃƒÂ©el n'existe qu'avec un "
                                "modÃƒÂ¨le distant)."
                            ),
                        )
                        _local_gen_lock.acquire()
                    _local_held = True

                for kind, payload in source:
                    # Titre distant prÃƒÂªt (thread de fond) -> on le pousse dÃƒÂ¨s la 1re occasion.
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
                        # Une nouvelle soumission demande l'arrÃƒÂªt : on stoppe net

                        # et on persiste ce qui a dÃƒÂ©jÃƒÂ  ÃƒÂ©tÃƒÂ© gÃƒÂ©nÃƒÂ©rÃƒÂ©.

                        interrupted = True

                        break

                    if kind == "reasoning":
                        yield _tl("reasoning", text=payload)

                    elif kind == "content":
                        if _profile is not None:
                            payload = _profile.apply_to_text(payload)
                        answer += payload

                        # Temps rÃƒÂ©el : le texte est journalisÃƒÂ© ÃƒÂ  l'instant (rejouable). Le
                        # session.json (contexte) se met ÃƒÂ  jour aux frontiÃƒÂ¨res d'outils + fin.
                        yield _tl("text", text=payload)

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
                        # Fin d'un tour : llama-server donne le prompt rÃƒÂ©el et le completion

                        # EXACT (tool-calls inclus) -> on cumule envoyÃƒÂ©s/reÃƒÂ§us ÃƒÂ  travers les

                        # tours ET les outils, et on rÃƒÂ©concilie le tour courant.

                        _p = payload.get("prompt_tokens", 0) or 0
                        _c = payload.get("completion_tokens", 0) or 0
                        _cached = payload.get("cached_tokens", 0) or 0

                        sent_tokens += _p

                        recv_confirmed += _c

                        cur_turn = 0

                        # Cumul RÃƒâ€°EL de la session : chaque appel refacture tout le contexte en
                        # INPUT -> on somme input/output/cache/coÃƒÂ»t sur TOUS les appels
                        # (persistÃƒÂ©), pas seulement le tour. C'est LA vraie somme facturÃƒÂ©e, et
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
                        # Conso d'un SOUS-AGENT (dispatch_agent) : ses tokens sont RÃƒâ€°ELS et
                        # facturÃƒÂ©s -> on les ajoute aux totaux de session (coÃƒÂ»t, NÃƒâ€”, in/out/
                        # cache). `set_context=False` : son prompt n'est PAS le contexte du fil
                        # principal, on ne touche donc pas la jauge de remplissage ni les
                        # mÃƒÂ©triques per-tour (sent/recv) qui dÃƒÂ©crivent le tour principal.
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

                    # 1 vrai token streamÃƒÂ© par llama-server. On compte aussi tool_args

                    # pour que le compteur avance pendant la gÃƒÂ©nÃƒÂ©ration d'un appel (gros

                    # write_file inclus) au lieu de se figer. On affiche le cumul + un dÃƒÂ©bit

                    # mesurÃƒÂ© sur la rafale courante ; le timer se rÃƒÂ©initialise aprÃƒÂ¨s >1s sans

                    # token (pause d'exÃƒÂ©cution) pour que les tok/s reflÃƒÂ¨tent la gÃƒÂ©nÃƒÂ©ration.

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
                    answer = "(le modÃƒÂ¨le a seulement rÃƒÂ©flÃƒÂ©chi Ã¢â‚¬â€ augmente max_tokens)"

                    yield _sse("text", text=answer)

                _persist(final=True)  # fin de tour : ÃƒÂ©criture finale garantie

                # Apprentissage post-tour (HORS de la loop d'action) : ne s'exÃƒÂ©cute que si le

                # tour a fait du vrai travail (>= reflect_min_actions). Toute dÃƒÂ©faillance est

                # avalÃƒÂ©e Ã¢â‚¬â€ la rÃƒÂ©ponse utilisateur est dÃƒÂ©jÃƒÂ  rendue (design Ã‚Â§6, Ã‚Â§11).

                if (
                    _settings["reflect_enabled"]
                    and reflect_stores is not None
                    and saved
                    and len(actions) >= _settings["reflect_min_actions"]
                ):
                    try:
                        from loom.agent.reflect import reflect as _reflect

                        _res = _reflect(
                            conv.to_messages(),
                            actions,
                            answer,
                            client=client,
                            model=conv.model or reflect_model,
                            provider=reflect_stores.provider,
                            paths=reflect_stores.paths,
                            learned_dir=reflect_stores.learned_dir,
                        )

                        # Trace VISIBLE (console/serve.log) : sinon l'apprentissage est une

                        # boÃƒÂ®te noire Ã¢â‚¬â€ on ne sait pas s'il a tournÃƒÂ© ni ce qu'il a retenu.

                        if _res is None:
                            print(
                                "[reflect] rien retenu (tour peu gÃƒÂ©nÃƒÂ©ralisable)",
                                flush=True,
                            )

                        else:
                            print(
                                f"[reflect] retenu : {len(_res.new_skills)} skill(s), "
                                f"{len(_res.improved_skills)} amÃƒÂ©liorÃƒÂ©(s), "
                                f"{len(_res.episodes)} ÃƒÂ©pisode(s), "
                                f"{len(_res.memory_updates) + len(_res.user_updates) + len(_res.soul_updates)} "
                                "note(s) identitÃƒÂ©",
                                flush=True,
                            )

                    except Exception as _e:  # noqa: BLE001 - best-effort, jamais bloquant
                        print(f"[reflect] erreur ignorÃƒÂ©e : {_e}", flush=True)

                # Auto-titre : ÃƒÂ  la 1re vraie rÃƒÂ©ponse, nommer la session (le modÃƒÂ¨le infÃƒÂ¨re le
                # sujet). On titre LA session de CETTE gÃƒÂ©nÃƒÂ©ration (`sess`), pas la session
                # focus (_cur) Ã¢â‚¬â€ sinon, en multi-onglets concurrent, on titrerait la mauvaise.

                if _immediate_title:
                    # Distant : filet de secours si le thread de titre n'a pas fini avant la
                    # fin de la boucle (ou tour sans ÃƒÂ©vÃƒÂ©nement) -> on l'attend briÃƒÂ¨vement.
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
                # L'utilisateur a soumis un nouveau message : le client a fermÃƒÂ© le

                # flux. On persiste la rÃƒÂ©ponse PARTIELLE dÃƒÂ©jÃƒÂ  reÃƒÂ§ue, puis on relaie

                # l'interruption (re-raise obligatoire pour le protocole gÃƒÂ©nÃƒÂ©rateur).

                _persist(final=True)  # client parti : ÃƒÂ©criture finale garantie

                raise

            except Exception as exc:  # noqa: BLE001 - on remonte l'erreur au client SSE
                yield _sse("error", message=str(exc))

            finally:
                _last_activity[0] = time.time()  # marque l'activitÃƒÂ© pour le keep-warm

                if _local_held:
                    _local_gen_lock.release()

                chat_lock.release()

        return Response(generate(), mimetype="text/event-stream")

    @app.post("/fork")
    def fork():
        """Repart d'un message utilisateur : tronque l'historique APRES ce message (exclus),

        renvoie son texte pour pre-remplir l'input. user_index = N-ieme message user (0-based)."""

        user_index = int(request.form.get("user_index", "-1"))

        conv, save = _ctx()

        msgs = conv.messages

        # Trouve le N-ieme message user dans l'historique persiste

        user_msgs = [i for i, m in enumerate(msgs) if m.get("role") == "user"]

        if user_index < 0 or user_index >= len(user_msgs):
            return Response("index invalide", status=400)

        target_idx = user_msgs[user_index]

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

    @app.post("/cancel")
    def cancel():

        # Bouton Stop : pose le signal d'annulation de LA session ciblÃƒÂ©e (par session_id, sinon
        # la session focus) -> SA boucle /chat s'arrÃƒÂªte net et libÃƒÂ¨re son verrou. Les AUTRES
        # sessions (onglets) ne sont PAS touchÃƒÂ©es. Sans effet si rien ne tourne pour elle.

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
        # Toggle des skills (faÃƒÂ§on /tools) : le formulaire porte les skills COCHÃƒâ€°S. Les
        # dÃƒÂ©cochÃƒÂ©s (tous les autres) deviennent `disabled_skills` de la session -> retirÃƒÂ©s
        # du catalogue et de use_skill. Re-render le panneau (case maÃƒÂ®tre incluse).
        conv, save = _ctx()
        enabled = set(request.form.getlist("skill"))
        all_names = [s.name for s in _all_skills()]
        conv.set_disabled_skills([n for n in all_names if n not in enabled])
        save()
        return render_template("_skills.html", **_skills_ctx(conv))

    @app.get("/skill")
    def skill_get():
        # Source d'un skill pour l'ÃƒÂ©diteur : texte brut du SKILL.md, ou l'override de session
        # s'il existe (ce que le modÃƒÂ¨le voit rÃƒÂ©ellement pour cette session).
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
        # Enregistre l'ÃƒÂ©dition d'un skill. scope=session -> override de session (n'ÃƒÂ©crit
        # PAS le disque) ; scope=global -> ÃƒÂ©crit le SKILL.md pour TOUTES les sessions et
        # lÃƒÂ¨ve l'override de session (le fichier fait dÃƒÂ©sormais foi).
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
                return {"error": f"ÃƒÂ©criture impossible : {exc}"}, 400
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

        # Sous-processus : ÃƒÂ©vite les soucis tkinter hors du thread principal de Flask.

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
                "error": (proc.stderr or "sÃƒÂ©lecteur indisponible")[:200],
            }

        return {"path": path}

    @app.post("/model")
    def model_update():

        conv, save = _ctx()

        model = request.form.get("model", "")

        conv.set_model(model)

        save()

        # MÃƒÂ©morise ce choix : il devient le dÃƒÂ©faut des prochaines sessions / lancements.

        session_store.set_default_model(model)

        # Cycle de vie du modÃƒÂ¨le SUR LA MACHINE. SÃƒÂ©lectionner un modÃƒÂ¨le LOCAL le CHARGE
        # (warmup : llama-swap charge ÃƒÂ  la 1re requÃƒÂªte, et swap l'ancien si besoin) ;
        # passer ÃƒÂ  un modÃƒÂ¨le DISTANT (API) DÃƒâ€°CHARGE le local pour LIBÃƒâ€°RER LA VRAM. Les deux
        # en tÃƒÂ¢che de fond (best-effort) : la rÃƒÂ©ponse UI reste instantanÃƒÂ©e, l'indicateur
        # d'ÃƒÂ©tat (/machine_state) reflÃƒÂ¨te ensuite le rÃƒÂ©sultat rÃƒÂ©el via llama-swap.
        if model in remote_model_ids:
            threading.Thread(
                target=client.unload_local, daemon=True, name="loom-unload"
            ).start()
        elif model:
            threading.Thread(
                target=lambda m=model: client.warmup_local(m),
                daemon=True,
                name="loom-warmup",
            ).start()

        return render_template(
            "_models.html",
            models=models,
            current_model=conv.model,
            remote_model_ids=remote_model_ids,
        )

    # ---- Gestionnaire de modÃƒÂ¨les (UI) : ajouter/tester/supprimer un modÃƒÂ¨le DISTANT ÃƒÂ  chaud,
    # sans redÃƒÂ©marrer. Un distant = URL + clÃƒÂ© (rien en VRAM) -> l'ajout monte une route et met
    # ÃƒÂ  jour les registres partagÃƒÂ©s en place. PersistÃƒÂ© dans le store JSON (remote_store_path).
    def _models_payload():
        """Liste ordonnÃƒÂ©e pour reconstruire le <select> cÃƒÂ´tÃƒÂ© client (id + local/distant)."""
        return [{"id": m, "remote": m in remote_model_ids} for m in models]

    def _remote_list():
        """ModÃƒÂ¨les distants montÃƒÂ©s, pour le panneau de config. Jamais la clÃƒÂ© en clair :
        seulement sa prÃƒÂ©sence. `managed` = ajoutÃƒÂ© via l'UI (ÃƒÂ©ditable/supprimable) vs dÃƒÂ©clarÃƒÂ©
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
                    # Indice masquÃƒÂ© (4 derniers car.) : l'utilisateur voit sa propre clÃƒÂ© de
                    # faÃƒÂ§on partielle, jamais la clÃƒÂ© entiÃƒÂ¨re renvoyÃƒÂ©e au client.
                    "key_hint": ("Ã¢â‚¬Â¦" + key[-4:]) if key else "",
                    "managed": mid in managed_ids,
                }
            )
        return sorted(out, key=lambda x: x["id"])

    def _mount_remote(rec):
        """Monte ÃƒÂ  chaud un modÃƒÂ¨le distant `rec` (dict) dans TOUS les registres partagÃƒÂ©s."""
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
        if not key and mid:  # ÃƒÂ©dition sans re-saisir la clÃƒÂ© -> rÃƒÂ©utilise la stockÃƒÂ©e
            stored = {m["id"]: m for m in model_store.load(remote_store_path)}
            key = stored.get(mid, {}).get("api_key", "")
        if not (base_url and model):
            return {"ok": False, "message": "base_url et model requis"}, 400
        ok, msg = client.ping_remote(base_url, key, model)
        return {"ok": ok, "message": msg}

    @app.post("/models/remote")
    def models_remote_upsert():
        if not remote_store_path:
            return {"error": "store des modÃƒÂ¨les indisponible"}, 500
        b = request.get_json(silent=True) or {}
        mid = (b.get("id") or "").strip()
        base_url = (b.get("base_url") or "").strip().rstrip("/")
        model = (b.get("model") or "").strip()
        if not (mid and base_url and model):
            return {"error": "id, base_url et model sont requis"}, 400
        if mid in models and mid not in remote_model_ids:
            return {"error": f"'{mid}' est dÃƒÂ©jÃƒÂ  un modÃƒÂ¨le local"}, 400
        stored = {m["id"]: m for m in model_store.load(remote_store_path)}
        # ClÃƒÂ© : si vide, on garde l'existante Ã¢â‚¬â€ soit du store gÃƒÂ©rÃƒÂ©, soit de la route montÃƒÂ©e
        # (cas d'un modÃƒÂ¨le dÃƒÂ©fini en config qu'on ÃƒÂ©dite sans re-saisir la clÃƒÂ©).
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
        # Un modÃƒÂ¨le DÃƒâ€°FINI EN CONFIG (montÃƒÂ© mais absent du store gÃƒÂ©rÃƒÂ©) reste dans local.toml :
        # on l'y ÃƒÂ©dite en place (tomlkit, commentaires prÃƒÂ©servÃƒÂ©s). Sinon store JSON gÃƒÂ©rÃƒÂ© par l'UI.
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
            return {"error": "store des modÃƒÂ¨les indisponible"}, 500
        managed = {m.get("id") for m in model_store.load(remote_store_path)}
        if mid not in managed:
            return {"error": "modÃƒÂ¨le non gÃƒÂ©rÃƒÂ© par l'UI (dÃƒÂ©fini dans local.toml)"}, 400
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

    # ---- ModÃƒÂ¨les LOCAUX : liste + ÃƒÂ©dition du tuning MACHINE (offload GPU) dans model.toml.
    # La dÃƒÂ©finition (repo/filename/n_layers) est commune au modÃƒÂ¨le -> lecture seule ici ; le
    # tuning (context/n_gpu_layers/cpu_moe/n_cpu_moe) est propre ÃƒÂ  cette machine -> ÃƒÂ©ditable.
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
            return {"error": "champ non ÃƒÂ©ditable"}, 400
        spec = next((m for m in local_model_specs if m.get("id") == mid), None)
        if not spec or not spec.get("dir"):
            return {"error": "modÃƒÂ¨le local inconnu"}, 404
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
            with _toml_lock:  # sÃƒÂ©rialise le read-modify-write (Flask threaded)
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
        # Applique Ãƒâ‚¬ CHAUD cÃƒÂ´tÃƒÂ© serveur modÃƒÂ¨le : rÃƒÂ©gÃƒÂ©nÃƒÂ¨re le yaml (llama-swap -watch-config le
        # recharge) + dÃƒÂ©charge CE modÃƒÂ¨le -> il se relance avec le nouveau tuning au prochain
        # usage, sans toucher au TOML ÃƒÂ  la main ni tout redÃƒÂ©marrer.
        applied = _regen_swap_yaml()
        if applied:
            threading.Thread(
                target=lambda: client.unload_local(mid),
                daemon=True,
                name="loom-reload-model",
            ).start()
        return {"ok": True, "applies": "model-reload" if applied else "restart"}

    # ---- Console de configuration : introspection + ÃƒÂ©dition des vrais fichiers TOML (deux
    # couches commun/systÃƒÂ¨me), commentaires prÃƒÂ©servÃƒÂ©s via tomlkit (loom.runtime.config_schema).
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
            _reload_app_config()  # applique Ãƒâ‚¬ CHAUD les params app (permissions, tokensÃ¢â‚¬Â¦)
            _apply_to_model_server(section)  # rÃƒÂ©gÃƒÂ©nÃƒÂ¨re le yaml si param serveur/modÃƒÂ¨le
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
        """Valeurs de config ACTUELLEMENT en vigueur dans l'app en cours (mÃƒÂ©moire vive). Sert
        ÃƒÂ  vÃƒÂ©rifier qu'une ÃƒÂ©dition s'applique ÃƒÂ  chaud, sans redÃƒÂ©marrer loom.web."""
        return dict(_settings)

    @app.get("/machine_state")
    def machine_state():
        # Ãƒâ€°tat du modÃƒÂ¨le SUR LA MACHINE, pour l'indicateur UI. VÃƒÂ©ritÃƒÂ© = llama-swap /running
        # (best-effort ; le modÃƒÂ¨le peut aussi s'ÃƒÂªtre dÃƒÂ©chargÃƒÂ© seul via son TTL). On teste par
        # sous-chaÃƒÂ®ne quel modÃƒÂ¨le est chargÃƒÂ©, sans coupler au schÃƒÂ©ma JSON de llama-swap.
        conv, _ = _ctx()
        model = conv.model
        remote = model in remote_model_ids
        reachable, running_txt = client.running_local()
        model_loaded = bool(reachable and model and model in running_txt)
        any_loaded = bool(
            reachable and any(mid in running_txt for mid in local_model_ids)
        )
        return {
            "mode": "remote" if remote else "home",
            "model": model,
            "reachable": reachable,
            "model_loaded": model_loaded,
            "any_loaded": any_loaded,
        }

    @app.get("/sysmon")
    def sysmon_metrics():
        # MÃƒÂ©triques systÃƒÂ¨me LIVE (CPU/RAM/GPU) pour le moniteur affichÃƒÂ© avec un modÃƒÂ¨le LOCAL.
        # nvidia-smi + psutil ; champs ÃƒÂ  None si une source manque (le front s'adapte).
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
        # Ãƒâ€°tat CLIENT d'une session, pour OUVRIR un onglet sans recharger la page : messages,
        # modÃƒÂ¨le, thinking, workspace, outils actifs, compteur. Le multi-onglets s'appuie
        # dessus (chaque onglet hydrate sa session ÃƒÂ  l'ouverture).
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
            # A-t-on un journal d'affichage ÃƒÂ  rejouer ? (sinon l'UI retombe sur `messages`).
            "has_timeline": bool(session_store.read_timeline(sess.id)),
        }

    @app.get("/session/<sid>/timeline")
    def session_timeline(sid):
        """Journal d'affichage temps rÃƒÂ©el d'une session, pour REJOUER l'UI au rechargement
        (raisonnement, texte, cartes d'outils exactement comme en direct). Les chunks 'text'/
        'reasoning' consÃƒÂ©cutifs sont recollÃƒÂ©s pour un rejeu lÃƒÂ©ger."""
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

        # RÃƒÂ©affecte le dossier de travail de la SESSION ACTIVE (appelÃƒÂ© par le sÃƒÂ©lecteur

        # de dossier). Sans ÃƒÂ§a, choisir un dossier ne s'appliquerait qu'ÃƒÂ  la crÃƒÂ©ation

        # d'une nouvelle session -> les outils continueraient de cibler l'ancien.

        ws = (request.form.get("workspace") or "").strip()

        if not ws:
            return Response("workspace manquant", status=400)

        sess = _session()

        sess.workspace = ws

        session_store.save(sess)

        return {"workspace": sess.workspace}

    @app.post("/session/delete")
    def session_delete():

        sid = (request.form.get("id") or "").strip()

        session_store.delete(sid)

        # Si on supprime la session courante, on recharge l'active (ou on en crÃƒÂ©e une).

        if _cur["session"] is not None and _cur["session"].id == sid:
            _cur["session"] = None

            _session()

        return {"ok": True}

    # --- Keep-warm : empÃƒÂªche l'OS d'ÃƒÂ©vincer le modÃƒÂ¨le inactif (cold start aprÃƒÂ¨s pause). --

    # Thread daemon qui ping le modÃƒÂ¨le de la session ACTIVE (1 token) quand : keep-warm

    # activÃƒÂ©, une vraie requÃƒÂªte a dÃƒÂ©jÃƒÂ  eu lieu (_last_activity > 0), et on est restÃƒÂ© idle

    # depuis >= keepwarm_interval. `_local_gen_lock` non bloquant => on ne ping JAMAIS pendant

    # une gÃƒÂ©nÃƒÂ©ration LOCALE (--parallel 1). On ne ping QUE le modÃƒÂ¨le dÃƒÂ©jÃƒÂ  chargÃƒÂ© => pas de swap.

    def _keepwarm_loop():

        while True:
            interval = float(_settings["keepwarm_interval"])  # relu ÃƒÂ  chaud
            time.sleep(max(15.0, min(interval / 3.0, 60.0)))

            # Activable/dÃƒÂ©sactivable ÃƒÂ  chaud : si coupÃƒÂ©, on ne ping pas (thread reste en veille).
            if not _settings["keepwarm_enabled"]:
                continue

            last = _last_activity[0]

            if last <= 0 or (time.time() - last) < interval:
                continue

            if not _local_gen_lock.acquire(blocking=False):
                continue  # gÃƒÂ©nÃƒÂ©ration locale en cours => dÃƒÂ©jÃƒÂ  chaud

            try:
                sess = _cur["session"]

                model = sess.conversation.model if sess else None

                if not model:
                    continue

                # Keep-warm = garder chaud le modÃƒÂ¨le LOCAL (ÃƒÂ©viter le cold start). Un modÃƒÂ¨le
                # DISTANT n'a pas de cold start cÃƒÂ´tÃƒÂ© machine ET est PAYANT ÃƒÂ  l'appel : le
                # pinger en boucle brÃƒÂ»lerait des crÃƒÂ©dits pour rien -> on saute.
                if model in remote_model_ids:
                    continue

                for _kind, _chunk in client.stream_chat(
                    [{"role": "user", "content": "ping"}],
                    "",
                    1,
                    model=model,
                    thinking=False,
                ):
                    pass

                _last_activity[0] = time.time()  # gardÃƒÂ© chaud => relance un intervalle

            except Exception:  # noqa: BLE001 - keep-warm best-effort, jamais bloquant
                pass

            finally:
                _local_gen_lock.release()

    # Thread toujours lancÃƒÂ© (il dort ÃƒÂ  l'idle) : ainsi activer/dÃƒÂ©sactiver keep-warm dans la
    # console prend effet ÃƒÂ  chaud, sans redÃƒÂ©marrer loom.web.
    threading.Thread(target=_keepwarm_loop, daemon=True, name="loom-keepwarm").start()

    return app
