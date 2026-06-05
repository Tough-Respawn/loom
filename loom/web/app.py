# loom/web/app.py
"""Application Flask de Loom Chat : page + endpoints (chat SSE, reset, skills,
thinking, model)."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time

from flask import Flask, Response, render_template, request

from loom import context
from loom.skills import compose_system_prompt, list_skills, load_skill

MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _sse(event_type: str, **fields) -> str:
    """Sérialise un événement Server-Sent Events (UTF-8, accents préservés)."""
    return f"data: {json.dumps({'type': event_type, **fields}, ensure_ascii=False)}\n\n"


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
    return [
        {"type": "text", "text": message},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]


def create_app(
    conversation,
    client,
    history_path,
    skills_dir,
    *,
    max_tokens=2048,
    context_budget=3000,
    keep_recent=6,
    models=None,
    interrupt_wait=15.0,
    tool_registry=None,
    tool_factory=None,
    available_tools=None,
    permission=None,
    confirm_timeout=300.0,
    workspace_dir=".",
    session_store=None,
) -> Flask:
    app = Flask(__name__)
    # Recharge le template à chaque requête : éditer index.html ne nécessite pas de
    # redémarrer le serveur (sinon Jinja sert la version compilée au démarrage).
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    history_path = str(history_path)
    skills_dir = str(skills_dir)
    workspace_dir = str(workspace_dir)
    models = list(models or [])
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

    # Session active : un fil persistant par projet. Mode session (session_store fourni)
    # -> conversation/persistance pointent sur la session active. Mode legacy (absent)
    # -> une seule conversation persistée dans history_path (tests).
    _cur: dict = {"session": None}
    app.config["_session_holder"] = _cur

    def _session():
        if _cur["session"] is None:
            _cur["session"] = session_store.active() or session_store.create(
                workspace=workspace_dir
            )
        return _cur["session"]

    def _ctx():
        """Renvoie (conversation, save) : la conversation active et sa persistance.

        Un seul point de vérité : les endpoints n'ont pas à savoir s'ils sont en mode
        session ou legacy, ils manipulent `conv` et appellent `save()`."""
        if session_store is not None:
            sess = _session()
            return sess.conversation, (lambda: session_store.save(sess))
        return conversation, (lambda: conversation.save(history_path))

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
        conv, _ = _ctx()
        # En mode session, le workspace et la liste des sessions viennent de la session
        # active ; en legacy, du paramètre et liste vide (l'UI sessions reste inerte).
        if session_store is not None:
            sess = _session()
            ws = sess.workspace
            sessions = [
                {"id": m.id, "title": m.title, "workspace": m.workspace}
                for m in session_store.list()
            ]
            active_id = sess.id
        else:
            ws = workspace_dir
            sessions = []
            active_id = ""
        return {
            "messages": conv.messages,
            "skills": list_skills(skills_dir),
            "active_skills": conv.active_skills,
            "models": models,
            "current_model": conv.model,
            "thinking": conv.thinking,
            "available_tools": available_tools,
            "active_tools": conv.active_tools,
            "workspace_dir": ws,
            "sessions": sessions,
            "active_session": active_id,
            # État initial pour l'hydratation côté client (Preact). On échappe '<'
            # pour ne pas pouvoir fermer la balise <script> depuis le contenu.
            "init_json": json.dumps(
                {"messages": conv.messages, "thinking": conv.thinking},
                ensure_ascii=False,
            ).replace("<", "\\u003c"),
        }

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

        try:
            content = _build_user_content(message, request.files.get("image"))
            conv.add("user", content)
            save()

            # Gestion du contexte : résumé auto si trop long
            if context.summarize(conv, client, context_budget, keep_recent):
                save()

            active = [
                s for s in (load_skill(skills_dir, n) for n in conv.active_skills) if s
            ]
            system_prompt = compose_system_prompt(conv.system_prompt, active)
            # Le modèle ignore par défaut sous quel backend il tourne (le prompt dit
            # "Tu es Loom") -> il baratine quand on lui demande "quel modèle ?". On lui
            # injecte son modèle courant pour qu'il réponde honnêtement.
            if conv.model:
                system_prompt += (
                    f"\n\n# Ton moteur\nTu tournes sur le modèle local « {conv.model} ». "
                    "Si on te demande quel modèle/moteur tu utilises, réponds-le "
                    "honnêtement et directement (ce nom), sans esquiver."
                )
        except ValueError as exc:
            chat_lock.release()
            return Response(str(exc), status=400)
        except Exception as exc:  # noqa: BLE001
            chat_lock.release()
            return Response(f"erreur: {exc}", status=500)

        def generate():
            answer = ""
            saved = False

            def _persist():
                # Idempotent : ne persiste que s'il y a du contenu (le placeholder
                # "réfléchi" est déjà injecté dans `answer` sur le chemin normal).
                # Sur interruption sans aucun token reçu, answer == "" -> on ne
                # pollue pas l'historique avec une bulle assistant vide.
                nonlocal saved
                if saved or not answer:
                    return
                saved = True
                conv.add("assistant", answer)
                save()

            # Registre construit selon les outils activés pour CETTE conversation
            # (toggles UI) ET le workspace de la session active : sans ça les outils
            # (write/edit/run_shell + sous-agent) retombent sur cfg.chat.workspace_dir
            # et écrivent à côté du dossier ciblé. À défaut de factory, registre
            # statique (compat tests).
            ws = _session().workspace if session_store is not None else workspace_dir
            registry = (
                tool_factory(conv.active_tools, ws) if tool_factory else tool_registry
            )
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
                    elif kind == "tool_stream":
                        yield _sse("tool_stream", **payload)
                    elif kind == "tool_result":
                        yield _sse("tool_result", **payload)
                    elif kind == "usage":
                        yield _sse("usage", **payload)
                if interrupted:
                    _persist()  # partiel sauvé, pas de 'done' (le client est parti)
                    return
                if not answer.strip():
                    answer = "(le modèle a seulement réfléchi — augmente max_tokens)"
                    yield _sse("text", text=answer)
                _persist()
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
                chat_lock.release()

        return Response(generate(), mimetype="text/event-stream")

    @app.post("/skills")
    def skills_update():
        conv, save = _ctx()
        conv.set_skills(request.form.getlist("skill"))
        save()
        skills = list_skills(skills_dir)
        return render_template(
            "_skills.html", skills=skills, active_skills=conv.active_skills
        )

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
        conv.set_model(request.form.get("model", ""))
        save()
        return render_template("_models.html", models=models, current_model=conv.model)

    # --- Sessions : liste / nouvelle / bascule / suppression (mode session seulement) ---

    @app.get("/sessions")
    def sessions_list():
        if session_store is None:
            return {"sessions": [], "active": ""}
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
        if session_store is None:
            return Response("sessions désactivées", status=404)
        ws = (request.form.get("workspace") or "").strip() or workspace_dir
        title = (request.form.get("title") or "").strip()
        sess = session_store.create(workspace=ws, title=title)
        _cur["session"] = sess
        return {"id": sess.id, "title": sess.title, "workspace": sess.workspace}

    @app.post("/session/activate")
    def session_activate():
        if session_store is None:
            return Response("sessions désactivées", status=404)
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
        if session_store is None:
            return Response("sessions désactivées", status=404)
        ws = (request.form.get("workspace") or "").strip()
        if not ws:
            return Response("workspace manquant", status=400)
        sess = _session()
        sess.workspace = ws
        session_store.save(sess)
        return {"workspace": sess.workspace}

    @app.post("/session/delete")
    def session_delete():
        if session_store is None:
            return Response("sessions désactivées", status=404)
        sid = (request.form.get("id") or "").strip()
        session_store.delete(sid)
        # Si on supprime la session courante, on recharge l'active (ou on en crée une).
        if _cur["session"] is not None and _cur["session"].id == sid:
            _cur["session"] = None
            _session()
        return {"ok": True}

    return app
