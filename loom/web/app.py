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
from pathlib import Path

from flask import Flask, Response, render_template, request

from loom import context
from loom.agents import resolve_agents
from loom.orchestrator import run_build, run_pipeline
from loom.session import RunRecord
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
    agents=None,
    pipeline=None,
    max_revisions=1,
    verifier=None,
    workspace_dir=".",
    server_context=8192,
    n_parallel=1,
    session_store=None,
    lesson_store=None,
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
    agents = list(agents or [])
    pipeline = list(pipeline or [])
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

    # Session active : abstraction unique pour chat ET build. Mode session
    # (session_store fourni) -> conversation/persistance pointent sur la session active
    # et le build y journalise ses runs (le modèle reprend le fil au tour suivant).
    # Mode legacy (absent) -> une seule conversation persistée dans history_path (tests).
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

    # Run exécuté EN TÂCHE DE FOND (thread) qui pousse ses events dans un buffer. Le
    # client se (re)branche dessus via /run/replay : un reload ne perd plus le run et le
    # voit continuer en direct. Un seul run à la fois (chat_lock sérialise déjà).
    run_buffer = {"id": 0, "events": [], "done": True}
    run_cv = threading.Condition()

    def _tail(rid: int, from_idx: int = 0):
        """Streame les events du run `rid` depuis `from_idx`, puis suit les nouveaux en
        direct jusqu'à la fin. Réutilisé par /run (live) ET /run/replay (reload)."""
        i = from_idx
        while True:
            with run_cv:
                while (
                    i >= len(run_buffer["events"])
                    and not run_buffer["done"]
                    and run_buffer["id"] == rid
                ):
                    run_cv.wait(timeout=1.0)
                if run_buffer["id"] != rid:
                    return  # un run plus récent a pris la place
                evs = run_buffer["events"][i:]
                i = len(run_buffer["events"])
                done = run_buffer["done"]
            for ev in evs:
                if ev["type"] != "run_done":
                    yield _sse(
                        ev["type"], **{k: v for k, v in ev.items() if k != "type"}
                    )
            if done:
                yield _sse("done")
                return

    def _run_record(task: str, events: list) -> RunRecord:
        """Résume un run (fichiers écrits + verdict du vérificateur) pour la session."""
        files: list[str] = []
        ok = False
        for ev in events:
            if (
                ev.get("type") == "tool_result"
                and ev.get("ok")
                and ev.get("name") in ("write_file", "edit_file")
                and ev.get("path")
                and ev["path"] not in files
            ):
                files.append(ev["path"])
            if ev.get("type") == "verify":
                ok = bool(ev.get("ok"))
        summary = f"{len(files)} fichier(s) généré(s)."
        return RunRecord(task=task, summary=summary, files=files, ok=ok)

    def _run_worker(rid: int, gen, sess=None, task: str = ""):
        """Consomme le générateur de run EN ARRIÈRE-PLAN (indépendant du client) et pousse
        chaque event dans le buffer. Libère chat_lock à la fin (run_done/erreur). En mode
        session, journalise un RunRecord (le modèle reprend le fil au tour suivant)."""
        try:
            for ev in gen:
                if cancel_event.is_set():
                    break
                with run_cv:
                    if run_buffer["id"] != rid:
                        return
                    run_buffer["events"].append(ev)
                    run_cv.notify_all()
        except Exception as exc:  # noqa: BLE001 - relaie au client via le buffer
            with run_cv:
                if run_buffer["id"] == rid:
                    run_buffer["events"].append({"type": "error", "message": str(exc)})
                    run_cv.notify_all()
        finally:
            if sess is not None and session_store is not None:
                try:
                    with run_cv:
                        events = list(run_buffer["events"])
                    sess.add_run(_run_record(task, events))
                    session_store.save(sess)
                except Exception:  # noqa: BLE001 - journaliser ne doit pas casser le run
                    pass
            with run_cv:
                if run_buffer["id"] == rid:
                    run_buffer["done"] = True
                    run_cv.notify_all()
            chat_lock.release()

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
            "pipeline": resolve_agents(agents, pipeline),
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
            # (toggles UI). À défaut de factory, registre statique (compat tests).
            registry = (
                tool_factory(conv.active_tools) if tool_factory else tool_registry
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

    @app.post("/run")
    def run():
        task = (request.form.get("task") or "").strip()
        if not task or len(task) > 5000:
            return Response("tâche invalide", status=400)
        # Dossier cible (workspace) pour CE run : champ UI, sinon le workspace de la
        # session active (mode session), sinon le défaut config. Les agents écrivent
        # des chemins RELATIFS, résolus sous ce dossier.
        sess = _session() if session_store is not None else None
        default_ws = sess.workspace if sess is not None else workspace_dir
        target = (request.form.get("workspace") or "").strip() or default_ws
        try:
            target = str(Path(target).expanduser().resolve())
            Path(target).mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            return Response(f"dossier cible invalide : {exc}", status=400)
        if not chat_lock.acquire(blocking=False):
            cancel_event.set()
            if not chat_lock.acquire(timeout=interrupt_wait):
                return Response("occupé : un échange est déjà en cours", status=429)
        cancel_event.clear()

        selected = resolve_agents(agents, pipeline)
        # mode="build" (défaut) = fan-out (plan détaillé + génération parallèle isolée :
        # pas d'overflow de tool-call, GPU batché). mode="pipeline" = ancien tool-loop.
        mode = (request.form.get("mode") or "build").strip()
        semantic_review = (request.form.get("semantic") or "").strip() in (
            "1",
            "true",
            "on",
            "yes",
        )
        build_model = (selected[0].model if selected else None) or (
            models[0] if models else None
        )
        # Closures par-run : lient les outils ET le vérificateur au dossier cible
        # choisi (le tool_factory/verifier de base acceptent un workspace optionnel).
        run_factory = (
            (lambda active: tool_factory(active, workspace=target))
            if tool_factory
            else None
        )
        run_verifier = (
            (lambda paths: verifier(paths, workspace=target)) if verifier else None
        )

        def _write_to_target(path, content):
            # Écriture atomique bornée au dossier cible, renvoie le chemin absolu (le
            # vérificateur déterministe tourne ensuite sur l'ensemble des fichiers écrits).
            from loom.tools.base import _resolve_in_root
            from loom.tools.fs import _atomic_write

            p = _resolve_in_root(Path(target), path)
            _atomic_write(p, content)
            return str(p)

        if mode == "pipeline":
            if not selected:
                chat_lock.release()
                return Response("aucun agent configuré", status=400)
            gen = run_pipeline(
                selected,
                task,
                client,
                skills_dir,
                max_tokens=max_tokens,
                tool_factory=run_factory,
                permission=permission,
                confirm=_confirm,
                max_revisions=max_revisions,
                verifier=run_verifier,
            )
        else:
            gen = run_build(
                task,
                client,
                model=build_model,
                write=_write_to_target,
                workspace=target,
                verifier=run_verifier,
                max_tokens=max_tokens,
                context=server_context,
                n_parallel=n_parallel,
                semantic_review=semantic_review,
                lesson_store=lesson_store,
            )
        # En mode session, mémorise le dossier choisi sur la session (les prochains runs
        # le reprennent par défaut) et journalise le run sous cette session.
        if sess is not None:
            sess.workspace = target
            session_store.save(sess)
        # Démarre le run en arrière-plan (survit au reload) et renvoie son flux live.
        with run_cv:
            run_buffer["id"] += 1
            rid = run_buffer["id"]
            run_buffer["events"] = [{"type": "run_info", "workspace": target}]
            run_buffer["done"] = False
        threading.Thread(
            target=_run_worker, args=(rid, gen, sess, task), daemon=True
        ).start()
        return Response(_tail(rid, 0), mimetype="text/event-stream")

    @app.get("/run/active")
    def run_active():
        with run_cv:
            return {
                "id": run_buffer["id"],
                "running": not run_buffer["done"],
                "has": len(run_buffer["events"]) > 0,
            }

    @app.get("/run/replay")
    def run_replay():
        with run_cv:
            rid = run_buffer["id"]
            has = len(run_buffer["events"]) > 0
        if not has:
            return Response(_sse("done"), mimetype="text/event-stream")
        return Response(_tail(rid, 0), mimetype="text/event-stream")

    @app.post("/skills")
    def skills_update():
        conv, save = _ctx()
        conv.set_skills(request.form.getlist("skill"))
        save()
        skills = list_skills(skills_dir)
        return render_template(
            "_skills.html", skills=skills, active_skills=conv.active_skills
        )

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

    @app.route("/classify", methods=["POST"])
    def classify():
        message = (request.form.get("message") or "").strip()
        if not message:
            return {"mode": "chat"}
        if not chat_lock.acquire(blocking=False):
            return {"mode": "chat"}  # occupé -> défaut sûr
        try:
            from loom.classify import classify_intent

            mode = classify_intent(
                client, message, model=(models[0] if models else None)
            )
        finally:
            chat_lock.release()
        return {"mode": mode}

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
