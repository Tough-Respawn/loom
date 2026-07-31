# Caractérisation des routes de create_app (P2-1) : statuts, formes de réponse,
# effets de bord observables — le filet avant découpage en services/blueprints.
# Périmètre : tout ce qui est testable sans modèle ni processus externe.
from __future__ import annotations

import json


def _sse_types(body: bytes) -> list[str]:
    out = []
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:])["type"])
    return out


# ---------- socle / sessions ----------


def test_commands_catalogue_de_la_palette(web):
    # GET /commands : source de vérité de la palette « / » du composer — chaque
    # commande porte name/usage/description, et les handlers connus y figurent.
    data = web.get("/commands").get_json()
    cmds = data["commands"]
    names = {c["name"] for c in cmds}
    assert {"/add-model", "/remove-model", "/goal", "/init", "/cancel"} <= names
    assert all(c["usage"] and c["description"] for c in cmds)


def test_index_cree_une_session(web):
    r = web.get("/")
    assert r.status_code == 200
    sessions = web.get("/sessions").get_json()
    assert len(sessions["sessions"]) == 1
    assert sessions["active"]


def test_session_new_fait_le_menage_des_fantomes(web):
    # /session/new supprime les sessions VIDES non verrouillées avant de créer
    # (app.py ~3040) : créer b fait disparaître a (vide).
    a = web.post("/session/new", data={}).get_json()
    b = web.post("/session/new", data={}).get_json()
    assert a["id"] != b["id"]
    ids = [s["id"] for s in web.get("/sessions").get_json()["sessions"]]
    assert a["id"] not in ids
    assert b["id"] in ids


def test_session_activate_delete(web):
    b = web.post("/session/new", data={}).get_json()
    assert web.post("/session/activate", data={"id": b["id"]}).status_code == 200
    assert web.get("/sessions").get_json()["active"] == b["id"]
    assert web.post("/session/activate", data={"id": "inconnu"}).status_code == 404

    assert web.post("/session/delete", data={"id": b["id"]}).get_json()["ok"] is True
    ids = [s["id"] for s in web.get("/sessions").get_json()["sessions"]]
    assert b["id"] not in ids


def test_delete_purge_le_cache_pas_de_resurrection(web):
    # Après suppression, l'objet ne doit plus être servi depuis S.sessions_cache
    # (sinon /session_state le retrouve et une save ultérieure recrée le dossier).
    b = web.post("/session/new", data={}).get_json()
    assert web.get("/session_state", query_string={"id": b["id"]}).status_code == 200
    assert web.post("/session/delete", data={"id": b["id"]}).status_code == 200
    assert web.get("/session_state", query_string={"id": b["id"]}).status_code == 404
    assert b["id"] not in web.application.S.sessions_cache


def test_delete_refuse_si_generation_en_cours(web):
    # Le verrou de génération tenu => la route refuse (409) et ne supprime pas :
    # supprimer sous une boucle active laissait une session zombie.
    from loom.web.routes.helpers import _lock_for

    b = web.post("/session/new", data={}).get_json()
    S = web.application.S
    lock = _lock_for(S, b["id"])
    lock.acquire()
    try:
        assert web.post("/session/delete", data={"id": b["id"]}).status_code == 409
        assert (
            web.get("/session_state", query_string={"id": b["id"]}).status_code == 200
        )
    finally:
        lock.release()


def test_delete_id_inconnu_404_sans_fuite_de_verrou(web):
    # /session/delete ne doit pas créer d'entrée dans sess_locks pour un id inconnu
    # (le verrou n'étant plus retiré, ce serait une fuite mémoire sur POST répétés) :
    # on vérifie l'existence AVANT _lock_for -> 404 net, aucun verrou créé.
    S = web.application.S
    before = set(S.sess_locks)
    r = web.post("/session/delete", data={"id": "deadbeefdead"})
    assert r.status_code == 404
    assert set(S.sess_locks) == before, (
        "un id inconnu a créé une entrée dans sess_locks"
    )


def test_workspace_id_explicite_inconnu_repond_404(web_sess, tmp_env):
    r = web_sess.post(
        "/session/workspace",
        data={"workspace": str(tmp_env), "session_id": "deadbeefdead"},
    )
    assert r.status_code == 404


def test_session_state(web_sess):
    sid = web_sess.get("/sessions").get_json()["active"]
    r = web_sess.get("/session_state", query_string={"id": sid})
    assert r.status_code == 200
    assert web_sess.get("/session_state", query_string={"id": "xxx"}).status_code == 404


def test_session_workspace(web_sess, tmp_env):
    ws = tmp_env / "autre-ws"
    ws.mkdir()
    r = web_sess.post("/session/workspace", data={"workspace": str(ws)})
    assert r.status_code == 200
    assert r.get_json()["workspace"] == str(ws)


def test_timeline_vide(web_sess):
    sid = web_sess.get("/sessions").get_json()["active"]
    r = web_sess.get(f"/session/{sid}/timeline")
    assert r.status_code == 200
    assert r.get_json()["events"] == []


def test_reset(web_sess):
    assert web_sess.post("/reset").status_code == 200


# ---------- garde CSRF (before_request) ----------


def test_csrf_cross_site_403(web_sess):
    r = web_sess.post(
        "/note", data={"text": "x"}, headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert r.status_code == 403


def test_csrf_same_origin_passe(web_sess):
    r = web_sess.post(
        "/note", data={"text": "x"}, headers={"Sec-Fetch-Site": "same-origin"}
    )
    assert r.status_code == 200


# ---------- /chat : branches synchrones sans modèle ----------


def test_chat_message_vide_400(web_sess):
    assert web_sess.post("/chat", data={"message": "  "}).status_code == 400


def test_chat_message_trop_long_400(web_sess):
    assert web_sess.post("/chat", data={"message": "x" * 5001}).status_code == 400


def test_chat_goal_statut_ack_sse(web_sess):
    r = web_sess.post("/chat", data={"message": "/goal"})
    assert r.status_code == 200
    assert r.content_type.startswith("text/event-stream")
    types = _sse_types(r.data)
    assert types[-1] == "done"
    assert "text" in types


def test_chat_goal_clear_ack_sse(web_sess):
    r = web_sess.post("/chat", data={"message": "/goal clear"})
    assert r.status_code == 200
    assert _sse_types(r.data)[-1] == "done"


# ---------- petites routes d'état ----------


def test_cancel_204(web_sess):
    assert web_sess.post("/cancel", data={}).status_code == 204


def test_tool_decision_204(web_sess):
    assert (
        web_sess.post("/tool_decision", data={"id": "x", "approve": "1"}).status_code
        == 204
    )


def test_thinking_toggle(web_sess):
    assert web_sess.post("/thinking", data={"thinking": "1"}).data == b"1"
    assert web_sess.post("/thinking", data={"thinking": "0"}).data == b"0"


def test_tools_et_skills_html(web_sess):
    assert web_sess.post("/tools", data={"tool": ["read_file"]}).status_code == 200
    assert web_sess.post("/skills", data={"skill": []}).status_code == 200


# ---------- skills (source/save/delete) ----------


def test_skill_source_404(web):
    assert web.get("/skill", query_string={"name": "inconnu"}).status_code == 404


def test_skill_save_scope_session(web_sess):
    web_sess.post(
        "/skill/create",
        data={"name": "skill-a-editer", "description": "d", "body": "corps"},
    )
    r = web_sess.post(
        "/skill/save",
        data={"name": "skill-a-editer", "body": "nouveau corps", "scope": "session"},
    )
    assert r.status_code == 200
    assert r.get_json()["scope"] == "session"


def test_skill_delete(web, tmp_env):
    web.post("/skill/create", data={"name": "skill-a-suppr", "description": "d"})
    assert (tmp_env / "skills_user" / "skill-a-suppr").exists()
    r = web.post("/skill/delete", data={"name": "skill-a-suppr"})
    assert r.status_code == 200
    assert not (tmp_env / "skills_user" / "skill-a-suppr").exists()
    assert web.post("/skill/delete", data={"name": "skill-a-suppr"}).status_code == 404


def test_skill_generate_sans_description_400(web_sess):
    assert web_sess.post("/skill/generate", data={"description": ""}).status_code == 400


# ---------- modèles / config ----------


def test_models_local(web):
    r = web.get("/models/local")
    assert r.status_code == 200
    assert "models" in r.get_json()


def test_models_config(web):
    r = web.get("/models/config")
    assert r.status_code == 200
    data = r.get_json()
    assert data["remotes"] == []


def test_models_remote_test_champs_manquants_400(web):
    r = web.post(
        "/models/remote/test", data=json.dumps({}), content_type="application/json"
    )
    assert r.status_code == 400


def test_config_sans_chemins_500(web):
    assert web.get("/config").status_code == 500


def test_config_effective(web):
    r = web.get("/config/effective")
    assert r.status_code == 200
    assert r.get_json()["keepwarm_enabled"] is False


def test_compact_session_inconnue_404(web_sess):
    r = web_sess.post("/compact", data={"session_id": "inconnue"})
    assert r.status_code == 404


# ---------- helpers module-level (logique pure, P2-2) ----------


def test_sse_format():
    from loom.web.app import _sse

    s = _sse("text", content="héllo")
    assert s.startswith("data: ") and s.endswith("\n\n")
    assert json.loads(s[6:]) == {"type": "text", "content": "héllo"}


def test_detect_workspace(tmp_env):
    from loom.web.app import _detect_workspace

    assert _detect_workspace("aucun chemin ici", str(tmp_env)) is None
    ws = tmp_env / "projet-x"
    ws.mkdir()
    detected = _detect_workspace(str(ws), str(tmp_env))
    assert detected and "projet-x" in detected


def test_detect_workspace_nom_nu(tmp_env):
    from loom.web.app import _detect_workspace

    (tmp_env / "cas").mkdir()
    (tmp_env / "energy-data-platform").mkdir()

    # Un dossier au nom de mot courant ne doit JAMAIS être adopté par son seul nom :
    # « cas » dans la prose d'un message collé basculait le workspace sur Documents/cas
    # (hijack constaté le 2026-07-19, fuite de contenu personnel vers le modèle).
    assert _detect_workspace("analyse ce cas limite du parseur", str(tmp_env)) is None

    # Un nom de projet « slug » (séparateur -/_/.) reste adoptable par son seul nom.
    d = _detect_workspace("travaille sur energy-data-platform stp", str(tmp_env))
    assert d and d.endswith("energy-data-platform")

    # Le chemin ABSOLU d'un dossier au nom courant marche toujours.
    d2 = _detect_workspace(str(tmp_env / "cas"), str(tmp_env))
    assert d2 and d2.endswith("cas")


def test_should_adopt_pas_de_sous_chemin_dun_projet(tmp_env):
    # Citer un chemin INTERNE au projet courant n'est pas un changement de contexte :
    # l'adoption cassait le cache KV (system prompt modifié -> re-prefill intégral de
    # ~9,2k tokens, 52 s — vécu 2026-07-19 en donnant var/sessions/<id> au modèle).
    from loom.web.app import _should_adopt

    proj = tmp_env / "mon-projet"
    (proj / ".git").mkdir(parents=True)
    sub = proj / "var" / "sessions" / "abc"
    sub.mkdir(parents=True)
    # Sous-chemin d'une racine de projet (.git) -> PAS d'adoption.
    assert _should_adopt(str(proj), str(sub)) is False
    # Même dossier -> pas d'adoption (rien ne change).
    assert _should_adopt(str(proj), str(proj)) is False
    # Chemin HORS du workspace -> vrai changement de contexte, adoption.
    autre = tmp_env / "autre-projet"
    autre.mkdir()
    assert _should_adopt(str(proj), str(autre)) is True
    # Workspace générique (pas une racine de projet) -> focus d'un sous-projet OK.
    assert _should_adopt(str(tmp_env), str(proj)) is True
    # loom.md marque aussi une racine de projet.
    proj2 = tmp_env / "proj-fiche"
    proj2.mkdir()
    (proj2 / "loom.md").write_text("fiche", encoding="utf-8")
    assert _should_adopt(str(proj2), str(proj2 / "sub")) is False


def test_init_message_pure():
    from loom.web.app import _init_message

    msg = _init_message("mon-projet")
    assert "loom.md" in msg and "mon-projet" in msg


def test_build_user_content_texte_sans_images(tmp_env):
    from loom.web.app import _build_user_content

    out = _build_user_content(
        "bonjour", [], is_vision=False, stash_dir=str(tmp_env / "stash")
    )
    assert out == "bonjour"


class _FakeUpload:
    def __init__(self, filename, blob, mimetype="image/png"):
        self.filename = filename
        self.mimetype = mimetype
        self._blob = blob

    def read(self):
        return self._blob


def test_build_user_content_image_modele_texte_note_honnete(tmp_env):
    # Modèle SANS vision + image jointe : la note ne promet plus read_image/un VLM
    # (read_image est gaté hors vision, pas de repli) — elle dit franchement que le
    # modèle ne voit pas et cape sur le sélecteur VISION. Les chemins restent listés
    # (une bascule vision dans la même session pourra les lire).
    from loom.web.app import _build_user_content

    out = _build_user_content(
        "ton avis ?",
        [_FakeUpload("photo.png", b"\x89PNG fake")],
        is_vision=False,
        stash_dir=str(tmp_env / "stash"),
    )
    assert isinstance(out, str)
    assert "read_image(path" not in out and "te les décrira" not in out
    assert "SANS vision" in out and "VISION" in out
    assert "photo.png" in out  # chemin stashé listé
    assert (tmp_env / "stash").exists()
