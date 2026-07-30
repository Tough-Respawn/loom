from __future__ import annotations

from __future__ import annotations
from pathlib import Path
from flask import request
from loom.agent.client import log_event




def _register_soul_routes(app, S):
    """L'Âme : export/import chiffré de l'état portable (sessions, mémoire,
    identité, skills). AUCUN secret ne part (remote_models.json exclu par
    construction dans soul.build_archive). Spec : specs/2026-07-21-ame-*."""
    from loom.web import soul as soul_mod

    def _soul_paths():
        idp = S.identity_paths or {}
        identity_dir = (
            Path(idp["soul_path"]).parent
            if idp.get("soul_path")
            else Path("var/identity")
        )
        return soul_mod.SoulPaths(
            sessions_root=Path(S.session_store.root),
            memory_db=Path(S.memory_db_path or "var/memory/memory.db"),
            identity_dir=identity_dir,
            learned_skills_dir=Path(S.learned_skills_dir or "var/skills_learned"),
            user_skills_dir=Path(S.user_skills_dir),
        )

    @app.get("/soul/sessions")
    def soul_sessions():
        return {
            "sessions": [
                {"id": m.id, "title": m.title, "updated_at": m.updated_at}
                for m in S.session_store.list()
            ]
        }

    @app.post("/soul/passphrase/check")
    def soul_passphrase_check():
        return soul_mod.check_passphrase(request.form.get("passphrase", ""))

    @app.post("/soul/passphrase/generate")
    def soul_passphrase_generate():
        return {"passphrase": soul_mod.generate_passphrase()}

    @app.post("/soul/export")
    def soul_export():
        passphrase = request.form.get("passphrase", "")
        if not soul_mod.check_passphrase(passphrase)["ok"]:
            return {"error": "passphrase trop faible (score zxcvbn < 3)"}, 400
        ids = [s for s in request.form.get("session_ids", "").split(",") if s.strip()]
        try:
            recap = soul_mod.export_soul(
                _soul_paths(), ids, request.form.get("dest_dir", ""), passphrase
            )
        except soul_mod.SoulError as e:
            return {"error": str(e)}, 400
        except OSError as e:
            # USB pleine/retirée, dossier en lecture seule : vraie cause au user,
            # pas un 500 que l'UI traduirait en faux « échec réseau ».
            return {"error": f"écriture impossible : {e}"}, 400
        log_event("soul.export", sessions=recap["sessions"], path=recap["path"])
        return recap

    @app.post("/soul/import")
    def soul_import():
        # Tout l'import sous gen_guard : (1) refus net si une génération est en vol
        # (l'objet session vivant sauverait son état par-dessus les fichiers importés),
        # (2) aucune génération ne peut DÉMARRER pendant la fusion (elle chargerait
        # l'objet périmé du cache), (3) purge du cache + rechargement de la session
        # focus AVANT de relâcher — sinon le prochain save de l'objet mémoire écrase
        # silencieusement les données importées (« rien n'est jamais détruit »).
        with S.gen_guard:
            if any(lk.locked() for lk in S.sess_locks.values()):
                return {
                    "error": "génération en cours — réessaie quand les sessions "
                    "sont au repos"
                }, 409
            try:
                report = soul_mod.import_soul(
                    _soul_paths(),
                    request.form.get("file", ""),
                    request.form.get("passphrase", ""),
                )
            except soul_mod.SoulError as e:
                return {"error": str(e)}, 400
            S.sessions_cache.clear()
            cur = S.cur.get("session")
            if cur is not None:
                fresh = S.session_store.load(cur.id)
                if fresh is not None:
                    S.sessions_cache[cur.id] = fresh
                    S.cur["session"] = fresh
        log_event(
            "soul.import",
            ajoutees=report["sessions"]["ajoutees"],
            remplacees=report["sessions"]["remplacees"],
        )
        return {"report": report}
