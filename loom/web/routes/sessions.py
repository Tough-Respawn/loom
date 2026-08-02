from __future__ import annotations

import re

from flask import Response, render_template, request

from loom.agent.streaming import _close
from loom.web.app import (
    _NOTE_MAX_CHARS,
)
from loom.web.routes.helpers import (
    _cancel_for,
    _ctx,
    _ensure_model,
    _get_session,
    _lock_for,
    _session,
    _totals,
)
from loom.web.routes.priming import _prime_async
from loom.web.routes.skills import _index_context



def _register_session_routes(app, S):
    @app.post("/reset")
    def reset() -> str:
        conv, save = _ctx(S)

        if S.monitor_hub is not None:
            S.monitor_hub.stop_session(_session(S).id)

        conv.reset()

        save()

        # Un nouveau fil ne doit pas rejouer l'ancienne timeline.
        S.session_store.clear_timeline(_session(S).id)

        return render_template("index.html", **_index_context(S))

    @app.post("/fork")
    def fork():
        """Repart d'un message utilisateur : tronque l'historique APRES ce message (exclus),

        renvoie son texte pour pre-remplir l'input. user_index = N-ieme message user (0-based)
        COMPTÉ SUR L'AFFICHAGE — or la compaction fusionne les vieux tours côté serveur, donc
        l'index peut avoir glissé : on retrouve alors le message par son CONTENU (`text`,
        envoyé par l'UI), et sinon on répond une erreur EXPLICITE (plus d'échec muet)."""

        user_index = int(request.form.get("user_index", "-1"))
        ui_text = (request.form.get("text") or "").strip()

        # Cibler l'onglet appelant évite de tronquer une autre session en split view.
        req_sid = (request.form.get("session_id") or "").strip()
        if req_sid:
            sess = _get_session(S, req_sid)
            if sess is None:
                return Response("session inconnue", status=404)
        else:
            sess = _session(S)
        conv, save = sess.conversation, (lambda: S.session_store.save(sess))

        msgs = conv.messages

        def _as_text(content) -> str:
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                ).strip()
            return str(content).strip()


        user_msgs = [i for i, m in enumerate(msgs) if m.get("role") == "user"]

        target_idx = None
        if 0 <= user_index < len(user_msgs):
            cand = user_msgs[user_index]
            # Un index n'est fiable que si le contenu correspond encore.
            if not ui_text or _as_text(msgs[cand].get("content", "")) == ui_text:
                target_idx = cand
        if target_idx is None and ui_text:
            # Après compaction, rechercher le dernier message au même contenu.
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
        sess = _get_session(S, req_sid) if req_sid else _session(S)
        if sess is None:
            return Response("session introuvable", status=404)
        lock = _lock_for(S, sess.id)
        if not lock.acquire(blocking=False):
            return Response("occupé : cette session génère déjà", status=429)
        try:
            conv = sess.conversation
            # La compaction manuelle reste déterministe et n'appelle aucun modèle.
            target_chars = len(conv.system_prompt) + 4000
            new_msgs, freed = S.client.compact_conversation(
                conv.messages,
                system_prompt=conv.system_prompt,
                target_chars=target_chars,
            )
            if freed:
                conv.messages = new_msgs
                # Retrancher le delta préserve les coûts de schéma déjà inclus dans la jauge.
                conv.context_tokens = max(0, conv.context_tokens - freed)
                S.session_store.save(sess)
        finally:
            lock.release()
        return {**_totals(S, conv), "collapsed": freed}

    @app.post("/cancel")
    def cancel():
        # Le STOP vise une session sans interrompre les autres onglets.

        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(S, req_sid) if req_sid else S.cur["session"]
        if sess is not None:
            _cancel_for(S, sess.id).set()
            # Fermer un stream distant figé rend son teardown et son verrou bornés.
            holder = S.active_streams.get(sess.id)
            if holder is not None and holder.get("stream") is not None:
                _close(holder["stream"])

        return Response("", status=204)

    @app.post("/note")
    def note():
        # Une note reste en file jusqu'au prochain point d'injection, même au tour suivant.
        text = (request.form.get("text") or "").strip()
        if not text:
            return {"error": "note vide"}, 400
        if len(text) > _NOTE_MAX_CHARS:
            return {
                "error": f"note trop longue (max {_NOTE_MAX_CHARS} caractères)"
            }, 413
        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(S, req_sid) if req_sid else S.cur["session"]
        if sess is None:
            return {"error": "session inconnue"}, 404
        queued = S.notes.push(sess.id, text)
        if queued < 0:
            return {
                "error": "file de notes pleine — attendre le prochain point d'arrêt"
            }, 429
        return {"ok": True, "queued": queued}


    @app.get("/sessions")
    def sessions_list():
        active = _session(S).id

        return {
            "sessions": [
                {"id": m.id, "title": m.title, "workspace": m.workspace}
                for m in S.session_store.list()
            ],
            "active": active,
        }

    @app.get("/session_state")
    def session_state():
        # Renvoyer assez d'état pour hydrater un onglet sans recharger la page.
        sid = (request.args.get("id") or "").strip()
        sess = _get_session(S, sid)
        if sess is None:
            return Response("session inconnue", status=404)
        conv = sess.conversation
        return {
            "id": sess.id,
            "title": sess.title,
            "workspace": sess.workspace,
            "messages": conv.messages,
            "thinking": conv.thinking,
            "local_only": conv.local_only,
            "model": conv.model,
            "active_tools": conv.active_tools,
            "usage_totals": _totals(S, conv),
            # Sans timeline, l'UI retombe sur les messages persistés.
            "has_timeline": bool(S.session_store.read_timeline(sess.id)),
        }

    @app.get("/session/<sid>/timeline")
    def session_timeline(sid):
        """Journal d'affichage temps réel d'une session, pour REJOUER l'UI au rechargement
        (raisonnement, texte, cartes d'outils exactement comme en direct). Les chunks 'text'/
        'reasoning' consécutifs sont recollés pour un rejeu léger."""
        out: list[dict] = []
        for e in S.session_store.read_timeline(sid):
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
        ws = (request.form.get("workspace") or "").strip() or S.workspace_dir

        title = (request.form.get("title") or "").strip()

        # Supprimer les sessions vierges inactives sans toucher à celles qui génèrent.
        for meta in S.session_store.list():
            if meta.title != "Nouvelle session":
                continue
            _lk = S.sess_locks.get(meta.id)
            if _lk is not None and _lk.locked():
                continue
            ghost = _get_session(S, meta.id)
            if ghost is not None and not ghost.conversation.messages:
                if S.monitor_hub is not None:
                    S.monitor_hub.stop_session(meta.id)
                S.session_store.delete(meta.id)
                with S.gen_guard:
                    S.sessions_cache.pop(meta.id, None)
                if S.cur["session"] is not None and S.cur["session"].id == meta.id:
                    S.cur["session"] = None

        sess = S.session_store.create(workspace=ws, title=title)

        with S.gen_guard:
            S.sessions_cache[sess.id] = sess

        S.cur["session"] = sess

        # Préremplir le préfixe en fond pendant que l'utilisateur tape.
        _prime_async(S, _ensure_model(S, sess), require_running=True)

        return {"id": sess.id, "title": sess.title, "workspace": sess.workspace}


    @app.get("/session/<sid>/export")
    def session_export(sid):
        data = S.session_store.export_zip(sid)
        if data is None:
            return {"error": "session inconnue"}, 404
        meta = S.session_store.load(sid)
        slug = (
            re.sub(r"[^A-Za-z0-9._-]+", "-", meta.title if meta else "session").strip(
                "-."
            )[:40]
            or "session"
        )
        return Response(
            data,
            mimetype="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="loom-session-{slug}-{sid}.zip"'
                )
            },
        )

    @app.post("/session/import")
    def session_import():
        f = request.files.get("file")
        if f is None:
            return {"error": "fichier manquant (champ multipart « file »)"}, 400
        data = f.read(S.session_store.MAX_IMPORT_BYTES + 1)
        if len(data) > S.session_store.MAX_IMPORT_BYTES:
            return {"error": "archive trop grosse — import refusé"}, 413
        try:
            sess = S.session_store.import_zip(data)
        except ValueError as exc:
            return {"error": str(exc)}, 400
        with S.gen_guard:
            S.sessions_cache[sess.id] = sess
        S.cur["session"] = sess
        _prime_async(S, _ensure_model(S, sess), require_running=True)
        return {
            "id": sess.id,
            "title": sess.title,
            "workspace": sess.workspace,
            "model": sess.conversation.model,
        }

    @app.post("/session/activate")
    def session_activate():
        sid = (request.form.get("id") or "").strip()

        loaded = _get_session(S, sid)

        if loaded is None:
            return Response("session inconnue", status=404)

        S.session_store.set_active(sid)

        S.cur["session"] = loaded

        # Réamorcer en fond sans démarrer le serveur uniquement pour cette bascule.
        _prime_async(S, loaded, require_running=True)

        # Renvoyer le workspace évite de conserver celui de l'onglet précédent.
        return {
            "id": loaded.id,
            "title": loaded.title,
            "workspace": loaded.workspace,
            "model": loaded.conversation.model,
        }

    @app.post("/session/workspace")
    def session_workspace():
        # Le sélecteur doit réaffecter la session existante, pas seulement les futures.

        ws = (request.form.get("workspace") or "").strip()

        if not ws:
            return Response("workspace manquant", status=400)

        # Utiliser l'identifiant de l'onglet pour éviter une course avec le focus global.
        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(S, req_sid) if req_sid else _session(S)
        if sess is None:
            return Response("session introuvable", status=404)

        sess.workspace = ws

        S.session_store.save(sess)

        return {"workspace": sess.workspace}

    @app.post("/session/delete")
    def session_delete():
        sid = (request.form.get("id") or "").strip()

        # Valider l'id avant `setdefault` évite des verrous orphelins.
        if _get_session(S, sid) is None:
            return Response("session introuvable", status=404)

        # Une sauvegarde en vol pourrait recréer une session supprimée.
        lock = _lock_for(S, sid)
        if not lock.acquire(blocking=False):
            return Response("occupé : cette session génère — Stop d'abord", status=409)
        try:
            if S.monitor_hub is not None:
                S.monitor_hub.stop_session(sid)
            S.session_store.delete(sid)

            # Purger l'objet en mémoire empêche sa réapparition lors d'une sauvegarde.
            with S.gen_guard:
                S.sessions_cache.pop(sid, None)
                S.sess_cancel.pop(sid, None)

            if S.cur["session"] is not None and S.cur["session"].id == sid:
                S.cur["session"] = None
                _session(S)
        finally:
            lock.release()
            # Conserver le verrou évite que deux instances coexistent pendant une course.

        return {"ok": True}
