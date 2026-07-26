# Export/import de session : archive .zip claire (session.json + timeline.jsonl +
# manifeste), import sous un id NEUF, remap du modèle inconnu, rejets actionnables.
from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app


def _store(tmp_path, known=("m1", "m2"), default="m1"):
    return SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=default,
        known_models=list(known),
    )


def _fill(store, title="Chasse au bug", model="m2"):
    sess = store.create(workspace="C:/projets/x", title=title)
    sess.conversation.set_model(model)
    sess.conversation.add("user", "bonjour")
    sess.conversation.add("assistant", "salut !")
    store.save(sess)
    store.append_event(sess.id, "text", {"text": "salut !"})
    return sess


# ---------- SessionStore.export_zip / import_zip ----------


def test_export_zip_contient_session_timeline_et_manifeste(tmp_path):
    store = _store(tmp_path)
    sess = _fill(store)
    data = store.export_zip(sess.id)
    z = zipfile.ZipFile(io.BytesIO(data))
    assert set(z.namelist()) == {"loom-session.json", "session.json", "timeline.jsonl"}
    manifest = json.loads(z.read("loom-session.json"))
    assert manifest["format"] == "loom-session" and manifest["version"] == 1
    payload = json.loads(z.read("session.json"))
    assert payload["title"] == "Chasse au bug"
    # debug.log n'est JAMAIS embarqué (log runtime machine)
    (store.session_dir(sess.id) / "debug.log").write_text("x", encoding="utf-8")
    z2 = zipfile.ZipFile(io.BytesIO(store.export_zip(sess.id)))
    assert "debug.log" not in z2.namelist()


def test_export_zip_session_inconnue(tmp_path):
    assert _store(tmp_path).export_zip("nexiste-pas") is None


def test_import_zip_roundtrip_id_neuf_et_contenu_preserve(tmp_path):
    store = _store(tmp_path)
    src = _fill(store)
    data = store.export_zip(src.id)
    imported = store.import_zip(data)
    assert imported.id != src.id  # jamais d'écrasement
    assert imported.title == src.title
    assert imported.workspace == src.workspace
    assert imported.conversation.model == "m2"  # modèle connu : préservé
    texts = [m for m in imported.conversation.messages]
    assert len(texts) == len(src.conversation.messages)
    # timeline rejouable + session active = l'importée
    assert store.read_timeline(imported.id) == store.read_timeline(src.id)
    assert store.active().id == imported.id
    # le fichier session.json porte bien l'id NEUF (pas celui de l'archive)
    on_disk = json.loads(
        (store.session_dir(imported.id) / "session.json").read_text(encoding="utf-8")
    )
    assert on_disk["id"] == imported.id


def test_import_zip_remap_modele_inconnu(tmp_path):
    store = _store(tmp_path, known=("m1", "m2"), default="m1")
    src = _fill(store, model="m2")
    data = store.export_zip(src.id)
    # nouvelle « machine » : m2 n'existe pas ici
    other = _store(tmp_path / "autre", known=("m1",), default="m1")
    imported = other.import_zip(data)
    assert imported.conversation.model == "m1"


def test_import_zip_rejets_actionnables(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="zip"):
        store.import_zip(b"pas un zip du tout")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("autre.txt", "x")
    with pytest.raises(ValueError, match="session.json"):
        store.import_zip(buf.getvalue())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("session.json", "{pas du json")
    with pytest.raises(ValueError, match="illisible"):
        store.import_zip(buf.getvalue())


# ---------- routes Flask ----------


@pytest.fixture()
def env(tmp_path):
    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    store = _store(tmp_path, known=("fake-model",), default="fake-model")
    app = create_app(
        client=None,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=["fake-model"],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
    )
    web = app.test_client()
    r = web.post("/session/new", data={"title": "À exporter"})
    return SimpleNamespace(web=web, store=store, sid=r.get_json()["id"])


def test_route_export_telecharge_un_zip(env):
    r = env.web.get(f"/session/{env.sid}/export")
    assert r.status_code == 200
    assert r.mimetype == "application/zip"
    assert "attachment" in r.headers["Content-Disposition"]
    assert env.sid in r.headers["Content-Disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.data))
    assert "session.json" in z.namelist()


def test_route_export_404_session_inconnue(env):
    assert env.web.get("/session/zzz/export").status_code == 404


def test_route_import_cree_et_active_une_session_neuve(env):
    data = env.web.get(f"/session/{env.sid}/export").data
    r = env.web.post(
        "/session/import",
        data={"file": (io.BytesIO(data), "session.zip")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    d = r.get_json()
    assert d["id"] and d["id"] != env.sid
    assert d["title"] == "À exporter"
    # ouverte comme une session neuve : présente dans la liste, active
    assert env.store.active().id == d["id"]


def test_route_import_rejette_fichier_invalide(env):
    r = env.web.post(
        "/session/import",
        data={"file": (io.BytesIO(b"garbage"), "x.zip")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "zip" in r.get_json()["error"]
    assert env.web.post("/session/import", data={}).status_code == 400
