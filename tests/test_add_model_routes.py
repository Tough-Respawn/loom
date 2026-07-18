# Intégration Flask du wizard /add-model : deps HF fakées (monkeypatch de
# loom.web.routes._wizard_deps), download fake instantané. Aucun réseau.
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

FAKE_MODEL = "fake-model"


def _sse_texts(body: bytes) -> str:
    out = []
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if evt["type"] == "text":
                out.append(evt.get("text", ""))
    return "\n".join(out)


@pytest.fixture()
def env(tmp_path):
    for d in ("skills", "skills_user", "workspace", "models"):
        (tmp_path / d).mkdir()
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL],
    )
    app = create_app(
        client=None,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_MODEL],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        models_dir=str(tmp_path / "models"),
    )
    web = app.test_client()
    assert web.post("/session/new", data={}).status_code == 200
    return SimpleNamespace(web=web, tmp=tmp_path)


def _fake_deps(monkeypatch, hits=(), files=()):
    from loom.web import routes

    deps = SimpleNamespace(
        existing_ids={FAKE_MODEL},
        search_models=lambda q: list(hits),
        list_gguf_files=lambda repo: list(files),
        recommend=lambda fs: [dict(f, fits=True, recommended=True) for f in fs],
        derive_id=lambda repo: "nouveau-modele",
        list_remote_models=lambda base_url, key: None,  # provider muet -> saisie libre
        removable_models=lambda: [
            {"id": "glm-test", "kind": "remote", "label": "glm-test — distant"}
        ],
    )
    monkeypatch.setattr(routes, "_wizard_deps", lambda S: deps)


def test_add_model_ack_et_etat_persiste(env, monkeypatch):
    _fake_deps(monkeypatch)
    r = env.web.post("/chat", data={"message": "/add-model"})
    assert r.status_code == 200
    assert r.content_type.startswith("text/event-stream")
    assert "1. local" in _sse_texts(r.data)
    # l'étape suivante est bien routée vers le wizard (état persisté sur la session)
    r2 = env.web.post("/chat", data={"message": "2"})
    assert "distant 1/5" in _sse_texts(r2.data)


def test_add_model_cancel(env, monkeypatch):
    _fake_deps(monkeypatch)
    env.web.post("/chat", data={"message": "/add-model"})
    r = env.web.post("/chat", data={"message": "/cancel"})
    assert "annulé" in _sse_texts(r.data).lower()


def test_add_model_distant_persiste_le_store(env, monkeypatch):
    _fake_deps(monkeypatch)
    # nouvel ordre : la CLÉ vient avant le modèle (elle sert à lister GET /models)
    for msg in [
        "/add-model",
        "2",
        "glm-test",
        "https://api.exemple/v4",
        "aucune",
        "glm-test-model",
    ]:
        env.web.post("/chat", data={"message": msg})
    r = env.web.post("/chat", data={"message": "non"})
    assert "ajouté" in _sse_texts(r.data)
    stored = json.loads((env.tmp / "remote_models.json").read_text(encoding="utf-8"))
    assert stored[0]["id"] == "glm-test"
    assert stored[0]["base_url"] == "https://api.exemple/v4"


def test_remove_model_distant_vide_le_store(env, monkeypatch):
    _fake_deps(monkeypatch)
    # ajoute d'abord un distant (flux complet), puis le supprime via /remove-model
    for msg in [
        "/add-model",
        "2",
        "glm-test",
        "https://api.exemple/v4",
        "aucune",
        "glm-test-model",
        "non",
    ]:
        env.web.post("/chat", data={"message": msg})
    env.web.post("/chat", data={"message": "/remove-model"})
    env.web.post("/chat", data={"message": "1"})
    r = env.web.post("/chat", data={"message": "oui"})
    assert "retiré" in _sse_texts(r.data)
    stored = json.loads((env.tmp / "remote_models.json").read_text(encoding="utf-8"))
    assert stored == []


def test_add_model_local_installe(env, monkeypatch):
    hits = [{"repo_id": "org/mon-GGUF", "downloads": 10, "likes": 1}]
    files = [
        {
            "filename": "m.Q4.gguf",
            "part_files": ["m.Q4.gguf"],
            "size_mb": 5,
            "is_mmproj": False,
        }
    ]
    _fake_deps(monkeypatch, hits=hits, files=files)

    # download fake : écrit le fichier puis appelle on_done tout de suite (synchrone)
    from loom.runtime import model_install

    def fake_start(repo, filenames, dest_dir, total_mb, on_done=None):
        job = model_install.DownloadJob(repo, filenames, dest_dir, total_mb)
        (job.dest_dir / filenames[0]).write_bytes(b"GGUFxxxx")
        job.done = True
        if on_done:
            on_done(job)
        return job

    monkeypatch.setattr(model_install, "start_download", fake_start)

    for msg in ["/add-model qwen", "1", "1"]:
        env.web.post("/chat", data={"message": msg})
    r = env.web.post("/chat", data={"message": "ok"})
    body = _sse_texts(r.data)
    assert "téléchargement" in body.lower()

    mdir = env.tmp / "models" / "nouveau-modele"
    assert (mdir / "model.toml").exists()
    assert 'repo = "org/mon-GGUF"' in (mdir / "model.toml").read_text(encoding="utf-8")
    # monté à chaud : visible dans la liste des modèles sélectionnables
    payload = env.web.get("/models/config").get_json()
    assert any(m["id"] == "nouveau-modele" for m in payload["models"])
