# loom/web/app.py

"""Application Flask de Loom Chat : page + endpoints (chat SSE, reset, skills,

thinking, model).

`create_app` construit l'état partagé (SimpleNamespace `S`) et délègue les routes aux
fonctions d'enregistrement de loom/web/routes.py — même surface HTTP, même comportement."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, Response, request

from loom.runtime.manager import ModelServerManager
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


# Notes en vol (« btw » natif) : remarques utilisateur arrivées PENDANT une
# génération, par session. Poussées par /note, drainées par la boucle tool-use
# juste avant chaque appel modèle (cf. notes_provider de stream_chat_tools) —
# elles infléchissent le tour SANS l'interrompre.
# Bornes anti-abus (la file part TELLE QUELLE dans le contexte du modèle) :
# même plafond de longueur que /chat, et une file courte — elle est drainée
# avant CHAQUE appel modèle, 10 notes en attente = déjà anormal.
_NOTE_MAX_CHARS = 5000
_NOTES_CAP = 10


class NotesQueue:
    """File de notes en vol, par session, bornée à _NOTES_CAP entrées."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._queues: dict[str, list[str]] = {}

    def push(self, sid: str, text: str) -> int:
        """Empile une note ; renvoie la taille de la file, ou -1 si elle est pleine."""
        with self._guard:
            q = self._queues.setdefault(sid, [])
            if len(q) >= _NOTES_CAP:
                return -1
            q.append(text)
            return len(q)

    def drain(self, sid: str) -> list[str]:
        with self._guard:
            q = self._queues.get(sid) or []
            self._queues[sid] = []
            return q


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
    user_skills_dir="var/skills_user",
    reflect_stores=None,
    reflect_enabled=False,
    reflect_min_actions=1,
    reflect_model=None,
    model_contexts=None,
    model_max_tokens=None,
    remote_model_ids=None,
    remote_weak_ids=None,
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

    models = list(models or [])
    # Sérialise TOUTES les écritures de fichiers de config/modèles (model.toml, local.toml,
    # defaults.toml, store JSON) : Flask est threaded -> deux éditions concurrentes du même
    # fichier feraient une course read-modify-write (la dernière écrase l'autre).
    _toml_lock = threading.Lock()

    vision_models = set(vision_models or [])  # ids des modèles avec mmproj (vision)

    remote_model_ids = set(remote_model_ids or [])  # ids servis par une API distante
    # Distants FAIBLES (strong=false en config) : gardent le harnais complet — le tier
    # « fort » n'est plus déduit de la seule distance (leçon GLM-4.7-Flash 2026-07-15).
    remote_weak_ids = set(remote_weak_ids or [])
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

    remote_model_names = dict(
        remote_model_names or {}
    )  # id Loom -> vrai modèle provider

    available_tools = list(available_tools or [])

    # Concurrence PAR SESSION : chaque session a son propre verrou de génération et son
    # signal d'annulation -> plusieurs sessions (onglets) génèrent EN PARALLÈLE. Une nouvelle
    # soumission n'interrompt QUE la génération de SA session (pas les autres). Verrou GLOBAL
    # `local_gen_lock` en plus pour les modèles LOCAUX : llama-swap n'en sert qu'un à la fois
    # -> deux générations locales se sérialisent (limitation machine connue, signalée à l'UI).
    # `stay_awake` : tant qu'une génération tourne, on empêche la VEILLE du système (l'écran
    # peut s'éteindre) -> le travail continue en arrière-plan au lieu de geler à la mise en
    # veille (« network error »). No-op hors Windows.
    # `confirm_local` : événement d'annulation de la génération EN COURS sur CE thread (pour
    # _confirm, qui tourne dans le thread de génération) — posé au début de /chat.

    # Objet d'état PARTAGÉ des routes (remplace les fermetures géantes de create_app) :
    # tout ce que les endpoints se partagent vit ici, et les fonctions d'enregistrement
    # de loom/web/routes.py opèrent dessus.
    S = SimpleNamespace(
        client=client,
        session_store=session_store,
        skills_dir=skills_dir,
        plugins_dir=plugins_dir,
        workspace_dir=workspace_dir,
        learned_skills_dir=learned_skills_dir,
        user_skills_dir=user_skills_dir,
        identity_paths=identity_paths,
        reflect_stores=reflect_stores,
        reflect_model=reflect_model,
        confirm_timeout=confirm_timeout,
        tool_factory=tool_factory,
        available_tools=available_tools,
        context_window=context_window,
        models=models,
        vision_models=vision_models,
        remote_model_ids=remote_model_ids,
        remote_weak_ids=remote_weak_ids,
        remote_model_names=remote_model_names,
        local_model_ids=local_model_ids,
        local_model_specs=local_model_specs,
        model_contexts=model_contexts,
        model_max_tokens=model_max_tokens,
        model_prices=model_prices,
        model_descriptions=model_descriptions,
        image_model_ids=image_model_ids,
        video_model_ids=video_model_ids,
        image_by_id=_image_by_id,
        # Moteurs image (ComfyUI) : un par (dir, port) — cf. _engine_for (routes.py).
        engines={},
        engines_lock=threading.Lock(),
        generated_dir=_generated_dir,
        server_manager=server_manager,
        settings=_settings,
        perm=_perm,
        config_defaults_path=config_defaults_path,
        config_local_path=config_local_path,
        remote_store_path=remote_store_path,
        toml_lock=_toml_lock,
        gen_guard=threading.Lock(),
        sess_locks={},
        sess_cancel={},
        local_gen_lock=threading.Lock(),
        stay_awake=StayAwake(),
        confirm_local=threading.local(),
        notes=NotesQueue(),
        # Boucle de feedback de la note de recentrage (retour user 2026-07-10 : « une
        # consigne corrective = un ÉPISODE, pas une rengaine ») : True = un épisode de
        # troncature a déjà été GÉRÉ proprement par le modèle (stop naturel) -> le harnais
        # arrête de ré-injecter la note à chaque tour d'une session saturée. Ré-armée si
        # le modèle dérape (non-progrès/boucle) ou quand la pression retombe (prochain
        # épisode = nouvelle note).
        refocus_handled={},
        # Décisions de confirmation en attente : tool_call_id -> {event, approved}.
        # Renseignées par la route /tool_decision (autre thread), consommées par _confirm.
        pending={},
        # Horodatage de la dernière fin de génération (0 = jamais). Le keep-warm ne pinge
        # qu'après une vraie activité et seulement à l'idle (cf. thread plus bas).
        last_activity=[0.0],
        # Session active : un fil persistant par projet. Tout passe par la session courante
        # (conversation + persistance) ; un seul mode, plus de legacy.
        cur={"session": None},
        # Cache d'OBJETS session en mémoire : UNE instance par session, partagée entre requêtes
        # (onglets) -> pas de sauvegardes qui se clobberent quand plusieurs sessions tournent en
        # parallèle. `cur` pointe la session FOCUS (défaut de l'index).
        sessions_cache={},
    )

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

    # Routes : enregistrées par groupes depuis loom/web/routes.py (import ici et pas en
    # tête de module : routes.py importe les helpers purs de CE module).
    from loom.web.routes import (
        _boot_prime,
        _keepwarm_loop,
        _register_chat_routes,
        _register_config_routes,
        _register_misc_routes,
        _register_model_routes,
        _register_session_routes,
        _register_skill_routes,
    )

    _register_misc_routes(app, S)
    _register_chat_routes(app, S)
    _register_session_routes(app, S)
    _register_skill_routes(app, S)
    _register_model_routes(app, S)
    _register_config_routes(app, S)

    # Thread keep-warm toujours lancé (il dort à l'idle) : ainsi activer/désactiver keep-warm
    # dans la console prend effet à chaud, sans redémarrer loom.web.
    threading.Thread(
        target=_keepwarm_loop, args=(S,), daemon=True, name="loom-keepwarm"
    ).start()

    # Amorce au boot : si le serveur modèle tourne déjà (restart de loom.web), le
    # préfixe de la session active est pré-préfillé sans attendre le premier message.
    _boot_prime(S)

    return app
