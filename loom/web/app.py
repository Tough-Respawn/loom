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


# Détecter un dossier absolu explicite pour y résoudre les commandes de la session.

_PATH_RE = re.compile(r"""(?:[A-Za-z]:[\\/]|[\\/])[^\s"'`<>|*?]*""")


def _detect_workspace(message: str, root: str | None = None) -> str | None:
    """Renvoie le dossier EXISTANT le plus spécifique cité dans `message` (résolu absolu),

    ou None. Un fichier existant -> son dossier parent. N'adopte QUE du réel (isdir/isfile),

    donc un chemin de référence faux n'a aucun effet.



    Si `root` est fourni, on accepte aussi un PROJET cité par son seul NOM quand c'est un

    sous-dossier direct de `root` (ex. « ... pour energy-data-platform » sans le chemin

    complet) — mais SEULEMENT si ce nom est un slug (contient -, _ ou .) : un dossier au

    nom de mot courant (« cas », « games »…) est indistinguable de la prose, et le mot

    « cas » d'un message collé a réellement basculé le workspace sur un dossier personnel

    (2026-07-19). Nom simple -> chemin absolu ou sélecteur UI, jamais l'adoption par nom.

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
                if e.is_dir() and any(c in e.name for c in "-_.")
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


def _should_adopt(current_ws: str | None, candidate: str) -> bool:
    """Faut-il adopter `candidate` comme nouveau dossier de travail ?

    Un chemin cité À L'INTÉRIEUR du projet courant n'est PAS un changement de
    contexte : les outils l'atteignent déjà, et l'adoption modifie le system prompt
    en tête de préfixe -> cache KV invalidé -> re-prefill INTÉGRAL (~9,2k tokens,
    52 s vécus le 2026-07-19 en citant var/sessions/<id> au modèle). On n'adopte un
    sous-chemin du workspace que si celui-ci n'est pas une racine de projet (.git /
    loom.md) — ex. ~/Documents -> focus d'un projet cité par nom, toujours voulu."""
    if not current_ws:
        return True
    try:
        cur = Path(current_ws).resolve()
        cand = Path(candidate).resolve()
    except OSError:
        return True
    if cand == cur:
        return False  # déjà dessus : rien à changer, cache préservé
    if cur not in cand.parents:
        return True  # hors du workspace = vrai changement de contexte
    return not any((cur / m).exists() for m in (".git", "loom.md"))


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

      texte, avec les chemins + le CONSTAT franc que ce modèle ne voit pas (proposer un

      modèle VISION du sélecteur — read_image n'est pas exposé hors vision, et il n'y a

      pas de repli automatique vers un autre modèle, décisions 2026-07-09/15).



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
        # Rappeler que les images inline sont déjà visibles évite un faux appel à read_image.

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

    # Stocker les images permet de les relire après bascule explicite vers un modèle vision.

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
        f"[{len(paths)} image(s) jointe(s) à ce message, enregistrée(s) sur disque. Tu es "
        "un modèle SANS vision : tu ne peux ni les voir ni les lire — ne devine pas leur "
        "contenu. Dis-le franchement et propose à l'utilisateur de basculer sur un modèle "
        "marqué VISION dans le sélecteur : dans cette même session, read_image saura alors "
        "lire ces chemins :\n" + listing + "]\n"
    )

    return note + message


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


# Tracer les lectures et mutations, pas les allers-retours de navigation.

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

    # Ne tracer une mutation que lorsqu'elle a réellement réussi.

    if name in _WRITE_NAMES and not ok:
        return None

    mark = "" if ok else "✗ "

    if name == "run_shell":
        head = (evt.get("preview") or "").split("\n")[0][:60]

        return f"{mark}{verb} shell: {head}".strip()

    if name == "dispatch_agent":
        return f"{mark}{verb} une sous-tâche"

    return f"{mark}{verb} {evt.get('path') or '?'}"


# Les notes infléchissent le tour sans l'interrompre et restent bornées avant injection.
_NOTE_MAX_CHARS = 5000
_NOTES_CAP = 10


class NotesQueue:
    """File de notes en vol, par session, bornée à _NOTES_CAP entrées."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        # Les handoffs ajoutent des métadonnées sans invalider les anciennes notes texte.
        self._queues: dict[str, list[str | dict]] = {}

    def push(self, sid: str, text: str | dict) -> int:
        """Empile une note ; renvoie la taille de la file, ou -1 si elle est pleine."""
        with self._guard:
            q = self._queues.setdefault(sid, [])
            if len(q) >= _NOTES_CAP:
                return -1
            q.append(text)
            return len(q)

    def drain(self, sid: str) -> list[str | dict]:
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
    monitor_hub=None,
    available_tools=None,
    permission=None,
    permission_mode="ask",
    confirm_timeout=300.0,
    workspace_dir=".",
    plugins_dir="loom/plugins",
    keepwarm_enabled=True,
    keepwarm_interval=150.0,
    identity_paths=None,
    memory_db_path=None,
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
    models_dir=None,
    models_roots=None,
) -> Flask:
    app = Flask(__name__)

    # Recharger le template permet d'éditer l'interface sans redémarrer Flask.

    app.config["TEMPLATES_AUTO_RELOAD"] = True

    app.jinja_env.auto_reload = True

    # Désactiver le cache statique évite qu'il diverge du template rechargé à chaud.

    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    # Forcer UTF-8 contourne les MIME Windows qui omettent le charset des statiques.
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

    # La microcompaction suit la fenêtre de chaque modèle et réserve sa future réponse.

    model_contexts = dict(model_contexts or {})

    model_max_tokens = dict(model_max_tokens or {})
    model_prices = dict(model_prices or {})
    model_descriptions = dict(model_descriptions or {})

    # Ce holder mutable applique à chaud les réglages propres au processus web.
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
    # Remplacer la closure permet d'appliquer aussi le mode de permission à chaud.
    _perm = {"fn": permission}

    models = list(models or [])
    # Sérialiser les écritures évite les courses read-modify-write de Flask threaded.
    _toml_lock = threading.Lock()

    vision_models = set(vision_models or [])  # ids des modèles avec mmproj (vision)

    remote_model_ids = set(remote_model_ids or [])  # ids servis par une API distante
    # La distance ne suffit pas à déterminer si un modèle peut alléger le harnais.
    remote_weak_ids = set(remote_weak_ids or [])
    local_model_ids = [m for m in (models or []) if m not in remote_model_ids]

    # Les moteurs ComfyUI sont externes et partagés par couple dossier/port.
    image_models = list(image_models or [])
    image_model_ids = {m.id for m in image_models}
    # Image et vidéo partagent le runtime; seul leur libellé diffère.
    video_model_ids = {m.id for m in image_models if m.kind == "video"}
    _image_by_id = {m.id: m for m in image_models}
    models = list(models or []) + [m.id for m in image_models]

    _generated_dir = (
        Path(remote_store_path).resolve().parent / "generated"
        if remote_store_path
        else Path("var") / "generated"
    )
    local_model_specs = list(local_models or [])

    # Le serveur enfant meurt avec l'app et peut être démarré à la demande.
    server_manager = ModelServerManager()

    remote_model_names = dict(
        remote_model_names or {}
    )  # id Loom -> vrai modèle provider

    available_tools = list(available_tools or [])

    # Les sessions sont parallèles, mais les modèles locaux partagent un verrou machine.
    # Centraliser l'état évite des closures de routes divergentes.
    S = SimpleNamespace(
        client=client,
        session_store=session_store,
        skills_dir=skills_dir,
        plugins_dir=plugins_dir,
        workspace_dir=workspace_dir,
        learned_skills_dir=learned_skills_dir,
        user_skills_dir=user_skills_dir,
        identity_paths=identity_paths,
        memory_db_path=memory_db_path,
        reflect_stores=reflect_stores,
        reflect_model=reflect_model,
        confirm_timeout=confirm_timeout,
        tool_factory=tool_factory,
        monitor_hub=monitor_hub,
        available_tools=available_tools,
        context_window=context_window,
        models=models,
        vision_models=vision_models,
        remote_model_ids=remote_model_ids,
        remote_weak_ids=remote_weak_ids,
        remote_model_names=remote_model_names,
        local_model_ids=local_model_ids,
        local_model_specs=local_model_specs,
        models_dir=models_dir,
        models_roots=models_roots,
        model_contexts=model_contexts,
        model_max_tokens=model_max_tokens,
        model_prices=model_prices,
        model_descriptions=model_descriptions,
        image_model_ids=image_model_ids,
        video_model_ids=video_model_ids,
        image_by_id=_image_by_id,
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
        # Fermer ce stream permet à `/cancel` de débloquer une API distante figée.
        active_streams={},
        # Après STOP, borner l'attente du teardown avant de reprendre la session.
        interrupt_wait=interrupt_wait,
        local_gen_lock=threading.Lock(),
        # La raison du verrou alimente un message de mise en file exact.
        local_busy={"reason": ""},
        stay_awake=StayAwake(),
        confirm_local=threading.local(),
        notes=NotesQueue(),
        # Une note de recentrage ne doit être injectée qu'une fois par épisode.
        refocus_handled={},
        pending={},
        last_activity=[0.0],
        cur={"session": None},
        # Partager une instance par session évite des sauvegardes concurrentes divergentes.
        sessions_cache={},
    )

    # Localhost n'empêche pas un site tiers de POSTer; Sec-Fetch-Site borne ce risque CSRF.

    @app.before_request
    def _csrf_guard():
        if request.method != "POST":
            return None

        if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            return Response("requête cross-origin refusée (CSRF)", status=403)

        return None

    # Import tardif requis car les routes réutilisent les helpers purs de ce module.
    from loom.web.routes import (
        _boot_prime,
        _keepwarm_loop,
        _register_chat_routes,
        _register_config_routes,
        _register_misc_routes,
        _register_model_routes,
        _register_session_routes,
        _register_skill_routes,
        _register_soul_routes,
    )

    _register_misc_routes(app, S)
    _register_chat_routes(app, S)
    _register_session_routes(app, S)
    _register_skill_routes(app, S)
    _register_model_routes(app, S)
    _register_soul_routes(app, S)
    _register_config_routes(app, S)

    # Le thread dormant permet d'activer keep-warm à chaud.
    threading.Thread(
        target=_keepwarm_loop, args=(S,), daemon=True, name="loom-keepwarm"
    ).start()

    # Réamorcer au boot seulement si le serveur modèle tourne déjà.
    _boot_prime(S)

    # Exposer l'état partagé permet aux tests d'inspecter directement ses invariants.
    app.S = S

    return app
