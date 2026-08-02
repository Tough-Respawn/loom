from __future__ import annotations

from __future__ import annotations
import json
import re
import shutil
from pathlib import Path
from flask import render_template, request
from loom.extend.skills import (
    collect_skills,
    read_skill_source,
    write_skill_source,
)

from loom.web.routes.helpers import _ctx, _ensure_local_server, _session, _totals






def _all_skills(S) -> list:
    return collect_skills(
        S.skills_dir,
        S.plugins_dir,
        learned_dir=S.learned_skills_dir,
        user_dir=S.user_skills_dir,
    )


def _skills_ctx(S, conv) -> dict:
    """Contexte du panneau Skills : la liste COMPLÈTE (pour les cases), l'ensemble des
    skills ACTIFS (non désactivés) et ceux qui ont un override de session (badge UI)."""
    skills = _all_skills(S)
    disabled = set(conv.disabled_skills)
    return {
        "skills": skills,
        "active_skills": [s.name for s in skills if s.name not in disabled],
        "overridden_skills": [s.name for s in skills if s.name in conv.skill_overrides],
    }


def _skill_deletable(S, skill) -> bool:
    """Un skill n'est supprimable depuis l'UI que s'il vient d'un dossier GÉRÉ par
    l'utilisateur (appris ou ajouté) ET que son dossier est bien confiné dedans —
    jamais un skill du package (versionné) ni d'un plugin (géré par le store)."""
    if getattr(skill, "origin", "") not in ("learned", "user"):
        return False
    if not skill.base_dir:
        return False
    roots = [
        d for d in (S.learned_skills_dir, S.user_skills_dir) if d
    ]  # dossiers gérés
    base = Path(skill.base_dir).resolve()
    for root in roots:
        r = Path(root).resolve()
        if base != r and r in base.parents:
            return True
    return False


def _index_context(S) -> dict:
    sess = _session(S)

    conv = sess.conversation

    ws = sess.workspace

    sessions = [
        {"id": m.id, "title": m.title, "workspace": m.workspace}
        for m in S.session_store.list()
    ]

    active_id = sess.id

    return {
        "messages": conv.messages,
        **_skills_ctx(S, conv),
        "usage_totals": _totals(S, conv),
        "models": S.models,
        "remote_model_ids": S.remote_model_ids,
        "image_model_ids": S.image_model_ids,
        "video_model_ids": S.video_model_ids,
        "model_descriptions": S.model_descriptions,
        "current_model": conv.model,
        "thinking": conv.thinking,
        "local_only": conv.local_only,
        "available_tools": S.available_tools,
        "active_tools": conv.active_tools,
        "workspace_dir": ws,
        "sessions": sessions,
        "active_session": active_id,
        "permission_mode": S.settings["permission_mode"],
        # Échapper `<` empêche le contenu hydraté de fermer sa balise script.
        "init_json": json.dumps(
            {
                "messages": conv.messages,
                "thinking": conv.thinking,
                "local_only": conv.local_only,
                "usage_totals": _totals(S, conv),
                # Hydrater l'onglet actif et la liste utilisée par la sidebar.
                "active_session": active_id,
                "title": sess.title,
                "model": conv.model,
                "workspace": ws,
                "sessions": sessions,
                # Le front dérive de cette racine les chemins copiables des sessions.
                "sessions_root": str(S.session_store.root.resolve()),
            },
            ensure_ascii=False,
        ).replace("<", "\\u003c"),
    }




def _register_skill_routes(app, S):
    @app.post("/skills")
    def skills_update():
        # Les skills absents du formulaire deviennent désactivés pour cette session.
        conv, save = _ctx(S)
        enabled = set(request.form.getlist("skill"))
        all_names = [s.name for s in _all_skills(S)]
        conv.set_disabled_skills([n for n in all_names if n not in enabled])
        save()
        return render_template("_skills.html", **_skills_ctx(S, conv))

    @app.get("/skill")
    def skill_get():
        # Montrer l'override de session lorsqu'il masque le fichier global.
        conv, _ = _ctx(S)
        name = request.args.get("name", "")
        skill = next((s for s in _all_skills(S) if s.name == name), None)
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
            "origin": getattr(skill, "origin", "loom"),
            "deletable": _skill_deletable(S, skill),
        }

    @app.post("/skill/save")
    def skill_save():
        # Une édition globale remplace le fichier et supprime l'override de session.
        conv, save = _ctx(S)
        name = request.form.get("name", "")
        body = request.form.get("body", "")
        scope = request.form.get("scope", "session")
        skill = next((s for s in _all_skills(S) if s.name == name), None)
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

    @app.post("/skill/create")
    def skill_create():
        # Créer les skills utilisateur hors du package versionné, avec un slug sûr.
        raw = (request.form.get("name") or "").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
        if not slug or len(slug) < 3:
            return {
                "error": "nom invalide (3+ caractères, lettres/chiffres/tirets)"
            }, 400
        if any(s.name == slug for s in _all_skills(S)):
            return {"error": f"un skill « {slug} » existe déjà"}, 409
        desc = (request.form.get("description") or "").strip()
        body = (request.form.get("body") or "").strip()
        # Forcer le nom au slug du dossier évite les collisions d'identité.
        fm_end = body.find("\n---", 3) if body.startswith("---") else -1
        if fm_end != -1:
            front_lines = [
                ln
                for ln in body[3:fm_end].splitlines()
                if ln.strip() and not ln.strip().lower().startswith("name:")
            ]
            content = (
                "---\nname: " + slug + "\n" + "\n".join(front_lines) + body[fm_end:]
            )
            if not content.endswith("\n"):
                content += "\n"
        elif body.startswith("---"):
            # Un frontmatter incomplet retombera sur le nom du dossier.
            content = body if body.endswith("\n") else body + "\n"
        else:
            content = (
                f"---\nname: {slug}\ndescription: {desc}\n---\n\n"
                + (
                    body
                    or "Décris ici la méthode : quand ce skill s'applique, les "
                    "étapes à suivre, les pièges à éviter."
                )
                + "\n"
            )
        target = Path(S.user_skills_dir) / slug
        try:
            target.mkdir(parents=True, exist_ok=False)
            (target / "SKILL.md").write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"error": f"création impossible : {exc}"}, 400
        return {"ok": True, "name": slug}

    @app.post("/skill/generate")
    def skill_generate():
        # La génération locale cède la priorité au chat et restaure son slot KV ensuite.
        conv, _ = _ctx(S)
        desc = (request.form.get("description") or "").strip()
        if not desc:
            return {"error": "décris d'abord le skill"}, 400
        name = (request.form.get("name") or "nouveau-skill").strip()
        model = conv.model
        is_local = bool(model) and model not in S.remote_model_ids
        if is_local and not S.local_gen_lock.acquire(blocking=False):
            return {
                "error": "modèle local occupé — réessaie après la génération en cours"
            }, 409
        if is_local:
            S.local_busy["reason"] = "skill"
        try:
            # Démarrer le serveur comme le chat s'il est encore éteint.
            if is_local and not _ensure_local_server(S, wait=45.0):
                return {
                    "error": "serveur modèle indisponible (démarrage trop long) — "
                    "réessaie dans quelques secondes"
                }, 503
            saved = S.client.save_slot(model, "uigen.kv") if is_local else False
            prompt = (
                "Write a Loom SKILL.md file for the skill idea below. Output ONLY the "
                "file content, nothing else: a YAML frontmatter (---\\nname: "
                f"{name}\\ndescription: <one factual trigger line>\\n---) followed by "
                "a concise, actionable markdown body (when it applies, the steps, the "
                "pitfalls). Write the body in the SAME language as the idea.\n\n"
                f"Skill idea: {desc}"
            )
            chunks: list[str] = []
            try:
                for kind, chunk in S.client.stream_chat(
                    [{"role": "user", "content": prompt}],
                    "",
                    900,
                    model=model or None,
                    thinking=False,
                ):
                    if kind == "content":
                        chunks.append(chunk)
            except Exception as exc:  # noqa: BLE001 - erreur PROPRE côté UI, pas un 500
                return {"error": f"modèle injoignable : {str(exc)[:140]}"}, 502
            finally:
                if saved:
                    S.client.restore_slot(model, "uigen.kv")
            text = "".join(chunks).strip()
            # Retirer un bloc Markdown éventuel autour du fichier généré.
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.rstrip().endswith("```"):
                    text = text.rstrip()[:-3].rstrip() + "\n"
            if not text.strip():
                return {"error": "le modèle n'a rien produit — réessaie"}, 502
            return {"ok": True, "source": text}
        finally:
            if is_local:
                S.local_busy["reason"] = ""
                S.local_gen_lock.release()

    @app.post("/skill/delete")
    def skill_delete():
        # Seuls les skills utilisateur/appris sont supprimables; les autres se désactivent.
        conv, save = _ctx(S)
        name = request.form.get("name", "")
        skill = next((s for s in _all_skills(S) if s.name == name), None)
        if skill is None:
            return {"error": f"skill inconnu : {name}"}, 404
        if not _skill_deletable(S, skill):
            return {
                "error": "seuls les skills appris ou ajoutés par l'utilisateur sont "
                "supprimables (pour un skill du package ou d'un plugin : décoche-le)"
            }, 403
        try:
            shutil.rmtree(skill.base_dir)
        except OSError as exc:
            return {"error": f"suppression impossible : {exc}"}, 400
        conv.set_skill_override(name, None)
        conv.set_disabled_skills([n for n in conv.disabled_skills if n != name])
        save()
        return {"ok": True}
