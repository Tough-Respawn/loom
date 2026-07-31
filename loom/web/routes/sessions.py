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

# ---- Routes : sessions (fil, notes, fork, compaction) ---------------------------------


def _register_session_routes(app, S):
    @app.post("/reset")
    def reset() -> str:
        conv, save = _ctx(S)

        conv.reset()

        save()

        # Le fil repart à neuf -> on efface aussi le journal d'affichage temps réel.
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

        # Cible la session de l'ONGLET appelant (session_id), pas la session focus
        # globale : avec la split view, le bouton « repartir » existe dans des panneaux
        # non-focus, et la resynchro du focus (/session/activate) COURT contre ce POST —
        # sans session_id, /fork pouvait tronquer une AUTRE session que celle affichée.
        # Un session_id fourni mais INCONNU est une erreur franche : surtout pas de
        # repli silencieux sur la session focus (ce serait re-tronquer la mauvaise).
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
        sess = _get_session(S, req_sid) if req_sid else _session(S)
        if sess is None:
            return Response("session introuvable", status=404)
        lock = _lock_for(S, sess.id)
        if not lock.acquire(blocking=False):
            return Response("occupé : cette session génère déjà", status=429)
        try:
            conv = sess.conversation
            # DÉTERMINISTE et INSTANTANÉ (aucun appel modèle -> pas de blocage de plusieurs
            # minutes). Cible = prompt système (INCOMPRESSIBLE) + ~4000 car. (~1,3k tokens)
            # de conversation : on clippe le reste de l'historique.
            target_chars = len(conv.system_prompt) + 4000
            new_msgs, freed = S.client.compact_conversation(
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
                S.session_store.save(sess)
        finally:
            lock.release()
        return {**_totals(S, conv), "collapsed": freed}

    @app.post("/cancel")
    def cancel():
        # Bouton Stop : pose le signal d'annulation de LA session ciblée (par session_id, sinon
        # la session focus) -> SA boucle /chat s'arrête net et libère son verrou. Les AUTRES
        # sessions (onglets) ne sont PAS touchées. Sans effet si rien ne tourne pour elle.

        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(S, req_sid) if req_sid else S.cur["session"]
        if sess is not None:
            _cancel_for(S, sess.id).set()
            # Modèle distant lent/bloqué : cancel_event n'est lu qu'ENTRE deux chunks.
            # Si l'itération du stream ne rend jamais la main, le finally qui relâche le
            # verrou de session n'est jamais atteint. On FERME donc le stream en cours :
            # close() lève httpx.ReadError (attrapée par la boucle) -> teardown borné.
            holder = S.active_streams.get(sess.id)
            if holder is not None and holder.get("stream") is not None:
                _close(holder["stream"])

        return Response("", status=204)

    @app.post("/note")
    def note():
        # Note en vol (« btw » natif) : remarque envoyée PENDANT une génération.
        # Mise en file par session ; la boucle tool-use l'injecte au prochain point
        # d'arrêt (avant l'appel modèle suivant) SANS interrompre le tour. Si le
        # tour se termine avant l'injection, la note reste en file et part au début
        # du tour suivant — jamais perdue.
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

    # --- Sessions : liste / nouvelle / bascule / suppression ---

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
        # État CLIENT d'une session, pour OUVRIR un onglet sans recharger la page : messages,
        # modèle, thinking, workspace, outils actifs, compteur. Le multi-onglets s'appuie
        # dessus (chaque onglet hydrate sa session à l'ouverture).
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
            # A-t-on un journal d'affichage à rejouer ? (sinon l'UI retombe sur `messages`).
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

        # Sessions FANTÔMES : créées puis jamais utilisées (« Nouvelle session » et
        # zéro message) — balayées quand on en crée une nouvelle : elles ne font que
        # polluer la sidebar. On épargne toute session dont le verrou de génération
        # est tenu (un tour vient peut-être de démarrer).
        for meta in S.session_store.list():
            if meta.title != "Nouvelle session":
                continue
            _lk = S.sess_locks.get(meta.id)
            if _lk is not None and _lk.locked():
                continue
            ghost = _get_session(S, meta.id)
            if ghost is not None and not ghost.conversation.messages:
                S.session_store.delete(meta.id)
                with S.gen_guard:
                    S.sessions_cache.pop(meta.id, None)
                if S.cur["session"] is not None and S.cur["session"].id == meta.id:
                    S.cur["session"] = None

        sess = S.session_store.create(workspace=ws, title=title)

        with S.gen_guard:
            S.sessions_cache[sess.id] = sess

        S.cur["session"] = sess

        # Amorce du cache KV en fond (si le serveur modèle tourne déjà) : le préfixe
        # statique de la session neuve se préfille pendant que l'utilisateur tape.
        _prime_async(S, _ensure_model(S, sess), require_running=True)

        return {"id": sess.id, "title": sess.title, "workspace": sess.workspace}

    # ---- Export / import de session (.zip clair, cf. SessionStore.export_zip) ----

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
        # Même prise en charge qu'une session neuve : cache, focus, prime en fond.
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

        # Bascule = intention de reprendre CE fil : on ré-amorce son préfixe (fil
        # complet) en fond si le serveur tourne, sans jamais le démarrer pour ça.
        _prime_async(S, loaded, require_running=True)

        # Renvoie l'état complet de la session activée : le front s'en sert pour
        # rafraîchir l'affichage du path/workspace à CHAQUE bascule d'onglet.
        # Sans ça, un onglet dont le workspace a changé (ou est vide) garde
        # l'affichage du path de l'onglet précédent.
        return {
            "id": loaded.id,
            "title": loaded.title,
            "workspace": loaded.workspace,
            "model": loaded.conversation.model,
        }

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
        sess = (_get_session(S, req_sid) if req_sid else None) or _session(S)

        sess.workspace = ws

        S.session_store.save(sess)

        return {"workspace": sess.workspace}

    @app.post("/session/delete")
    def session_delete():
        sid = (request.form.get("id") or "").strip()

        S.session_store.delete(sid)

        # Si on supprime la session courante, on recharge l'active (ou on en crée une).

        if S.cur["session"] is not None and S.cur["session"].id == sid:
            S.cur["session"] = None

            _session(S)

        return {"ok": True}
