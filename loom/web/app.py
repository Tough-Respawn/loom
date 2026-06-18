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
from loom.extend.skills import collect_skills, render_catalog

MAX_IMAGE_BYTES = 10 * 1024 * 1024

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
    """Titre court (3-5 mots) inféré par le modèle depuis la 1re demande, pour ne pas
    laisser la session « Nouvelle session ». Thinking OFF + peu de tokens (cosmétique, pas
    cher). Repli sur le début du message si le modèle ne renvoie rien d'exploitable."""
    prompt = (
        "Donne un titre TRÈS court (3 à 5 mots) résumant cette demande, en français, "
        "sans guillemets ni ponctuation finale. Réponds UNIQUEMENT par le titre.\n\n"
        "Demande : " + message[:500]
    )
    title = ""
    try:
        parts = [
            chunk
            for kind, chunk in client.stream_chat(
                [{"role": "user", "content": prompt}],
                "Tu génères des titres de conversation courts et clairs.",
                32,
                model=model,
                thinking=False,
            )
            if kind == "content"
        ]
        merged = "".join(parts).strip().strip('"').strip("'").strip()
        title = merged.splitlines()[0][:60].strip() if merged else ""
    except Exception:  # noqa: BLE001 - un titre est cosmétique, jamais bloquant
        title = ""
    if not title:
        title = message.strip().splitlines()[0][:48].strip() or "Session"
    return title


def _build_user_content(message: str, image) -> str | list:
    """Construit le contenu du message user : texte seul, ou multimodal si image.

    Lève ValueError (-> 400) si l'image est trop grande ou n'est pas une image.
    """
    if not (image and image.filename):
        return message
    blob = image.read()
    if len(blob) > MAX_IMAGE_BYTES:
        raise ValueError("image trop grande")
    mime = image.mimetype or "image/png"
    if not mime.startswith("image/"):
        raise ValueError("fichier non-image")
    b64 = base64.b64encode(blob).decode("ascii")
    # L'image est EMBARQUÉE dans le message (data URI) : le modèle multimodal la VOIT
    # déjà. Sans ce rappel, il croit devoir l'ouvrir via read_image, DEVINE un chemin,
    # échoue, puis cherche les images du disque. On coupe court.
    note = (
        "[Une image est jointe à ce message — tu la VOIS déjà directement ci-dessous. "
        "N'utilise PAS read_image pour elle et ne devine AUCUN chemin de fichier : "
        "analyse l'image telle qu'elle est.]\n"
    )
    return [
        {"type": "text", "text": note + message},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]


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
    "replace_lines": "modifié",
    "insert_lines": "modifié",
    "run_shell": "exécuté",
    "dispatch_agent": "délégué",
}
_WRITE_NAMES = {
    "write_file",
    "append_file",
    "edit_file",
    "replace_lines",
    "insert_lines",
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
    learned_skills_dir=None,
    reflect_stores=None,
    reflect_enabled=False,
    reflect_min_actions=1,
    reflect_model=None,
) -> Flask:
    app = Flask(__name__)
    # Recharge le template à chaque requête : éditer index.html ne nécessite pas de
    # redémarrer le serveur (sinon Jinja sert la version compilée au démarrage).
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    skills_dir = str(skills_dir)
    workspace_dir = str(workspace_dir)
    plugins_dir = str(plugins_dir)
    # Seuil de microcompact INTERNE à la boucle d'outils : on vide les vieux résultats
    # d'outils quand le contexte vivant approche la fenêtre du modèle (en réservant la
    # place de la réponse). Distinct du résumé inter-tours (context_budget) qui, lui,
    # ne porte que sur l'historique persisté.
    compact_after_tokens = max(1024, context_window - max_tokens - 1024)
    models = list(models or [])
    vision_models = set(vision_models or [])  # ids des modèles avec mmproj (vision)
    available_tools = list(available_tools or [])
    chat_lock = threading.Lock()
    # Signal d'annulation : une nouvelle soumission le pose pour stopper net la
    # génération en cours (la boucle le vérifie à chaque token). Plus fiable que
    # d'attendre la détection de déconnexion par le serveur WSGI.
    cancel_event = threading.Event()
    # Décisions de confirmation en attente : tool_call_id -> {event, approved}.
    # Renseignées par la route /tool_decision (autre thread), consommées par _confirm.
    pending: dict = {}
    app.config["_chat_lock"] = chat_lock
    app.config["_cancel_event"] = cancel_event
    app.config["_pending"] = pending
    # Horodatage de la dernière fin de génération (0 = jamais). Le keep-warm ne pinge
    # qu'après une vraie activité et seulement à l'idle (cf. thread plus bas).
    _last_activity = [0.0]

    # Session active : un fil persistant par projet. Tout passe par la session courante
    # (conversation + persistance) ; un seul mode, plus de legacy.
    _cur: dict = {"session": None}
    app.config["_session_holder"] = _cur

    def _session():
        if _cur["session"] is None:
            _cur["session"] = session_store.active() or session_store.create(
                workspace=workspace_dir
            )
        sess = _cur["session"]
        # Une session neuve peut naître sans modèle -> requête model="" -> llama-swap
        # renvoie 404 'no router for requested model'. On garantit un modèle valide
        # (le 1er = défaut) ; corrige aussi les sessions déjà créées vides.
        if not sess.conversation.model and models:
            sess.conversation.set_model(models[0])
            session_store.save(sess)
        return sess

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
        try:
            while not ev.wait(0.2):
                if cancel_event.is_set() or time.monotonic() > deadline:
                    return False
            return bool(pending[tool_id]["approved"])
        finally:
            pending.pop(tool_id, None)

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
            "skills": collect_skills(
                skills_dir, plugins_dir, learned_dir=learned_skills_dir
            ),
            "models": models,
            "current_model": conv.model,
            "thinking": conv.thinking,
            "available_tools": available_tools,
            "active_tools": conv.active_tools,
            "workspace_dir": ws,
            "sessions": sessions,
            "active_session": active_id,
            "permission_mode": permission_mode,
            # État initial pour l'hydratation côté client (Preact). On échappe '<'
            # pour ne pas pouvoir fermer la balise <script> depuis le contenu.
            "init_json": json.dumps(
                {"messages": conv.messages, "thinking": conv.thinking},
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

    @app.post("/reset")
    def reset() -> str:
        conv, save = _ctx()
        conv.reset()
        save()
        return render_template("index.html", **_index_context())

    @app.post("/chat")
    def chat():
        message = (request.form.get("message") or "").strip()
        if not message or len(message) > 5000:
            return Response("message invalide", status=400)
        if not chat_lock.acquire(blocking=False):
            # Un échange est en cours : on demande son annulation (interruption
            # par nouvelle soumission) et on attend qu'il libère le verrou.
            cancel_event.set()
            if not chat_lock.acquire(timeout=interrupt_wait):
                return Response("occupé : un échange est déjà en cours", status=429)
        # On tient le verrou : repartir d'un signal d'annulation propre.
        cancel_event.clear()
        conv, save = _ctx()

        # Gate vision : une image envoyée à un modèle SANS mmproj fait planter llama-server
        # (500 "image input not supported") en plein run. On refuse EN AMONT, message clair,
        # plutôt qu'un crash après avoir déjà agi. (read_image suit le même garde côté tool.)
        _img = request.files.get("image")
        if _img and _img.filename and conv.model and conv.model not in vision_models:
            chat_lock.release()
            others = ", ".join(sorted(vision_models)) or "aucun (configure un mmproj)"
            msg = (
                f"Le modèle actif « {conv.model} » ne voit pas les images (pas de mmproj). "
                f"Bascule sur un modèle avec vision : {others}."
            )

            def _gate():
                yield _sse("error", message=msg)

            return Response(_gate(), mimetype="text/event-stream")

        # Logs PAR SESSION (au même titre que session.json) : (1) trace des échanges modèle
        # routée vers sessions/<id>/debug.log ; (2) copie du log serveur modèle global
        # (loom/data/serve.log) dans la session — doublon assumé, pour tout avoir sous la main.
        _sdir = session_store.session_dir(_session().id)
        set_debug_log_path(_sdir / "debug.log")
        _serve_log = session_store.root.parent / "serve.log"
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
            content = _build_user_content(message, request.files.get("image"))
            conv.add("user", content)
            save()

            # Gestion du contexte : résumé auto si trop long
            if context.summarize(conv, client, context_budget, keep_recent):
                save()

            skills = collect_skills(
                skills_dir, plugins_dir, learned_dir=learned_skills_dir
            )
            catalog = render_catalog(skills)
            # Identité always-on (SOUL/USER/MEMORY) EN TÊTE : c'est la définition qui FAIT FOI
            # de qui est Loom (rôle, persona, style). Le mode d'emploi opérationnel (outils,
            # règles) de chat.system.md vient APRÈS et s'y conforme — on ne plante plus un
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
                    max_tokens=identity_max_tokens,
                )
            system_prompt = (
                f"{_idblk}\n\n{conv.system_prompt}" if _idblk else conv.system_prompt
            )
            if catalog:
                system_prompt += f"\n\n{catalog}"
            # Le modèle ignore par défaut sous quel backend il tourne (le prompt dit
            # "Tu es Loom") -> il baratine quand on lui demande "quel modèle ?". On lui
            # injecte son modèle courant pour qu'il réponde honnêtement.
            if conv.model:
                system_prompt += (
                    f"\n\n# Ton moteur\nTu tournes sur le modèle local « {conv.model} ». "
                    "Si on te demande quel modèle/moteur tu utilises, réponds-le "
                    "honnêtement et directement (ce nom), sans esquiver."
                )
            # Dossier de travail courant : le modèle l'IGNORE sinon et le devine en sondant
            # (git rev-parse à l'aveugle, list_dir…) -> tours gaspillés. On le lui dit, avec
            # le réflexe anti-tâtonnement quand ce dossier n'est pas un repo git. Reste EN BAS
            # (contexte volatil, près de l'action).
            _ws = _session().workspace
            system_prompt += (
                f"\n\n# Dossier de travail courant\nTes commandes (run_shell) tournent dans "
                f"`{_ws}` et les chemins relatifs s'y résolvent — n'y répète pas le nom de ce "
                "dossier dans tes chemins. Si une commande git échoue par « not a git "
                "repository », c'est que CE dossier n'est pas un repo : fais UN list_dir pour "
                "repérer le bon sous-dossier (puis `git -C <sous-dossier>`), ne relance pas la "
                "même commande à l'identique."
            )
        except ValueError as exc:
            chat_lock.release()
            return Response(str(exc), status=400)
        except Exception as exc:  # noqa: BLE001
            chat_lock.release()
            return Response(f"erreur: {exc}", status=500)

        def generate():
            if adopted_ws:  # informe l'UI que le dossier de travail a été adopté
                yield _sse("workspace", path=adopted_ws)
            answer = ""
            actions: list[str] = []  # trace compacte des outils (anti-amnésie)
            saved = False

            def _persist():
                # Idempotent. Persiste le texte final + une TRACE COMPACTE des actions
                # (chemins lus/écrits, commandes) : sans elle, l'historique persisté est
                # amnésique de ce que l'agent a fait (seul son texte survivait) et le tour
                # suivant repartait à l'aveugle. On NE persiste pas les messages `tool`
                # bruts (gonflerait le contexte + casserait le résumeur).
                nonlocal saved
                if saved:
                    return
                body = answer
                if actions:
                    trace = "[Actions de ce tour : " + " · ".join(actions[:20]) + "]"
                    body = f"{body}\n\n{trace}" if body else trace
                if not body:  # rien à dire ET rien fait -> pas de bulle vide
                    return
                saved = True
                conv.add("assistant", body)
                save()

            # Registre construit selon les outils activés pour CETTE conversation
            # (toggles UI) ET le workspace de la session active : sans ça les outils
            # (write/edit/run_shell + sous-agent) retombent sur cfg.chat.workspace_dir
            # et écrivent à côté du dossier ciblé.
            ws = _session().workspace
            registry = tool_factory(conv.active_tools, ws, conv)
            use_tools = registry is not None and len(registry)
            if use_tools:
                source = client.stream_chat_tools(
                    conv.to_messages(),
                    system_prompt,
                    max_tokens,
                    model=conv.model or None,
                    registry=registry,
                    thinking=conv.thinking,
                    permission=permission,
                    confirm=_confirm,
                    compact_after_tokens=compact_after_tokens,
                )
            else:
                source = client.stream_chat(
                    conv.to_messages(),
                    system_prompt,
                    max_tokens,
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
            try:
                for kind, payload in source:
                    if cancel_event.is_set():
                        # Une nouvelle soumission demande l'arrêt : on stoppe net
                        # et on persiste ce qui a déjà été généré.
                        interrupted = True
                        break
                    if kind == "reasoning":
                        yield _sse("reasoning", text=payload)
                    elif kind == "content":
                        answer += payload
                        yield _sse("text", text=payload)
                    elif kind == "tool_call":
                        yield _sse("tool_call", **payload)
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
                        yield _sse("tool_result", **payload)
                    elif kind == "usage":
                        # Fin d'un tour : llama-server donne le prompt réel et le completion
                        # EXACT (tool-calls inclus) -> on cumule envoyés/reçus à travers les
                        # tours ET les outils, et on réconcilie le tour courant.
                        sent_tokens += payload.get("prompt_tokens", 0) or 0
                        recv_confirmed += payload.get("completion_tokens", 0) or 0
                        cur_turn = 0
                        yield _sse("usage", **payload)
                        yield _sse(
                            "metrics",
                            sent=sent_tokens,
                            recv=recv_confirmed,
                            tok_s=last_rate,
                        )
                    elif kind == "phase":
                        yield _sse("phase", **payload)
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
                    _persist()  # partiel sauvé, pas de 'done' (le client est parti)
                    return
                if not answer.strip():
                    answer = "(le modèle a seulement réfléchi — augmente max_tokens)"
                    yield _sse("text", text=answer)
                _persist()
                # Apprentissage post-tour (HORS de la loop d'action) : ne s'exécute que si le
                # tour a fait du vrai travail (>= reflect_min_actions). Toute défaillance est
                # avalée — la réponse utilisateur est déjà rendue (design §6, §11).
                if (
                    reflect_enabled
                    and reflect_stores is not None
                    and saved
                    and len(actions) >= reflect_min_actions
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
                        # boîte noire — on ne sait pas s'il a tourné ni ce qu'il a retenu.
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
                # Auto-titre : à la 1re vraie réponse, nommer la session (le modèle infère
                # le sujet) au lieu de la laisser « Nouvelle session ».
                _sess = _session()
                if saved and _sess.title == "Nouvelle session":
                    _title = _infer_title(client, conv.model or None, message)
                    if _title:
                        _sess.title = _title
                        session_store.save(_sess)
                        yield _sse("session_title", id=_sess.id, title=_title)
                yield _sse("done")
            except GeneratorExit:
                # L'utilisateur a soumis un nouveau message : le client a fermé le
                # flux. On persiste la réponse PARTIELLE déjà reçue, puis on relaie
                # l'interruption (re-raise obligatoire pour le protocole générateur).
                _persist()
                raise
            except Exception as exc:  # noqa: BLE001 - on remonte l'erreur au client SSE
                yield _sse("error", message=str(exc))
            finally:
                _last_activity[0] = time.time()  # marque l'activité pour le keep-warm
                chat_lock.release()

        return Response(generate(), mimetype="text/event-stream")

    @app.post("/cancel")
    def cancel():
        # Bouton Stop : pose le signal d'annulation que la boucle de /chat vérifie à
        # chaque token -> la génération s'arrête net et libère le verrou de chat. Sans
        # effet si rien ne tourne (le prochain /chat le remet à zéro avant de générer).
        cancel_event.set()
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
        return render_template("_models.html", models=models, current_model=conv.model)

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

    @app.post("/session/new")
    def session_new():
        ws = (request.form.get("workspace") or "").strip() or workspace_dir
        title = (request.form.get("title") or "").strip()
        sess = session_store.create(workspace=ws, title=title)
        _cur["session"] = sess
        return {"id": sess.id, "title": sess.title, "workspace": sess.workspace}

    @app.post("/session/activate")
    def session_activate():
        sid = (request.form.get("id") or "").strip()
        loaded = session_store.load(sid)
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
        sess = _session()
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

    # --- Keep-warm : empêche l'OS d'évincer le modèle inactif (cold start après pause). --
    # Thread daemon qui ping le modèle de la session ACTIVE (1 token) quand : keep-warm
    # activé, une vraie requête a déjà eu lieu (_last_activity > 0), et on est resté idle
    # depuis >= keepwarm_interval. `chat_lock` non bloquant => on ne ping JAMAIS pendant une
    # génération (--parallel 1). On ne ping QUE le modèle déjà chargé => pas de swap parasite.
    def _keepwarm_loop():
        tick = max(15.0, min(float(keepwarm_interval) / 3.0, 60.0))
        while True:
            time.sleep(tick)
            last = _last_activity[0]
            if last <= 0 or (time.time() - last) < float(keepwarm_interval):
                continue
            if not chat_lock.acquire(blocking=False):
                continue  # génération en cours => déjà chaud
            try:
                sess = _cur["session"]
                model = sess.conversation.model if sess else None
                if not model:
                    continue
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
                chat_lock.release()

    if keepwarm_enabled:
        threading.Thread(
            target=_keepwarm_loop, daemon=True, name="loom-keepwarm"
        ).start()

    return app
