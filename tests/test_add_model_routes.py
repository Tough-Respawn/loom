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


def test_remove_model_confirmation_porte_des_boutons(env, monkeypatch):
    _fake_deps(monkeypatch)
    env.web.post("/chat", data={"message": "/remove-model"})
    r = env.web.post("/chat", data={"message": "1"})
    events = [
        json.loads(line[6:])
        for line in r.data.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]
    assert {"type": "choices", "options": ["oui", "annuler"]} in events


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


# ---------- tous types : liste complète, image/vidéo, distant config ----------


def test_removable_models_liste_les_4_familles(tmp_path):
    from loom.runtime import model_store as ms
    from loom.web import routes

    store_path = tmp_path / "remote_models.json"
    ms.save(store_path, [{"id": "ui", "base_url": "https://x", "model": "m-ui"}])
    cfg = tmp_path / "local.toml"
    cfg.write_text('[chat]\ndefault_model = "cfg"\n', encoding="utf-8")
    S = SimpleNamespace(
        local_model_specs=[{"id": "loc", "size_mb": 1024}],
        remote_store_path=str(store_path),
        remote_model_ids={"ui", "cfg"},
        remote_model_names={"cfg": "m-cfg"},
        config_local_path=str(cfg),
        image_by_id={
            "img": SimpleNamespace(id="img", dir=str(tmp_path / "img")),
            "vid": SimpleNamespace(id="vid", dir=str(tmp_path / "vid")),
        },
        video_model_ids={"vid"},
    )
    items = routes._removable_models(S)
    kinds = {i["id"]: i["kind"] for i in items}
    assert kinds == {
        "loc": "local",
        "ui": "remote",
        "cfg": "remote_config",
        "img": "image",
        "vid": "video",
    }
    assert next(i for i in items if i["id"] == "cfg")["is_default"] is True


@pytest.fixture()
def env_img(tmp_path):
    # Arbo RÉALISTE <root>/local/{text,image,video} : _image_base_dir en dépend.
    root = tmp_path / "root"
    (root / "local" / "text").mkdir(parents=True)
    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    cfg = tmp_path / "local.toml"
    cfg.write_text(
        '[chat]\ndefault_model = "cfg-distant"\n\n'
        "[[remote_models]]\n"
        'id = "cfg-distant"\n'
        'base_url = "https://api.exemple/v4"\n'
        'model = "m-cfg"\n',
        encoding="utf-8",
    )
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL, "cfg-distant"],
    )
    # Client factice : juste ce que /models/config et _forget_remote consomment.
    fake_client = SimpleNamespace(
        remote_route_info=lambda mid: {
            "base_url": "https://api.exemple/v4",
            "model": "m-cfg",
            "has_key": False,
        },
        remote_api_key=lambda mid: "",
        remove_remote_route=lambda mid: None,
    )
    app = create_app(
        client=fake_client,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_MODEL, "cfg-distant"],
        remote_model_ids=["cfg-distant"],
        remote_model_names={"cfg-distant": "m-cfg"},
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        config_local_path=str(cfg),
        models_dir=str(root / "local" / "text"),
    )
    web = app.test_client()
    assert web.post("/session/new", data={}).status_code == 200
    return SimpleNamespace(web=web, tmp=tmp_path, root=root, cfg=cfg)


def _real_img_deps(monkeypatch):
    """Deps wizard : helpers image/remove RÉELS (S de l'app), HF stubbé (aucun réseau)."""
    from loom.web import routes

    def fake(S):
        return SimpleNamespace(
            existing_ids=set(S.models),
            search_models=lambda q: [],
            list_gguf_files=lambda repo: [],
            recommend=lambda fs: fs,
            derive_id=lambda repo: "x",
            list_remote_models=lambda base_url, key: None,
            removable_models=lambda: routes._removable_models(S),
            image_dir_state=lambda k, m: routes._image_dir_state(S, k, m),
            check_workflow=routes._check_workflow,
        )

    monkeypatch.setattr(routes, "_wizard_deps", fake)


def _install_image(env_img, mid="zz-img"):
    wf = env_img.tmp / "wf_api.json"
    wf.write_text('{"1": {"inputs": {"text": "{PROMPT}"}}}', encoding="utf-8")
    for msg in ["/add-model image", mid, "ok", "mon générateur"]:
        env_img.web.post("/chat", data={"message": msg})
    return env_img.web.post("/chat", data={"message": str(wf)})


def test_install_image_scaffold_et_montage(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    r = _install_image(env_img)
    body = _sse_texts(r.data)
    assert "✅" in body and "sélecteur" in body
    mdir = env_img.root / "local" / "image" / "zz-img"
    assert (mdir / "model.toml").exists() and (mdir / "workflow.json").exists()
    toml = (mdir / "model.toml").read_text(encoding="utf-8")
    assert "width = 1024" in toml and 'description = "mon générateur"' in toml
    payload = env_img.web.get("/models/config").get_json()
    m = next(m for m in payload["models"] if m["id"] == "zz-img")
    assert m["image"] is True


def test_install_image_plus_tard_scaffold_sans_montage(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    for msg in ["/add-model image", "zz-later", "ok", "non"]:
        env_img.web.post("/chat", data={"message": msg})
    r = env_img.web.post("/chat", data={"message": "plus tard"})
    body = _sse_texts(r.data)
    assert "workflow.json" in body  # le chemin où déposer l'export est annoncé
    mdir = env_img.root / "local" / "image" / "zz-later"
    assert (mdir / "model.toml").exists()
    assert not (mdir / "workflow.json").exists()
    payload = env_img.web.get("/models/config").get_json()
    assert not any(m["id"] == "zz-later" for m in payload["models"])


def test_reprise_dossier_complete_monte_directement(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    # scaffold « plus tard », dépôt manuel de la recette, puis re-run /add-model image
    for msg in ["/add-model image", "zz-resume", "ok", "non", "plus tard"]:
        env_img.web.post("/chat", data={"message": msg})
    mdir = env_img.root / "local" / "image" / "zz-resume"
    (mdir / "workflow.json").write_text('{"1": "{PROMPT}"}', encoding="utf-8")
    env_img.web.post("/chat", data={"message": "/add-model image"})
    r = env_img.web.post("/chat", data={"message": "zz-resume"})
    assert "montage direct" in _sse_texts(r.data)
    payload = env_img.web.get("/models/config").get_json()
    assert any(m["id"] == "zz-resume" for m in payload["models"])


def test_remove_image_rmtree_dossier_et_demonte(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    _install_image(env_img)
    mdir = env_img.root / "local" / "image" / "zz-img"
    env_img.web.post("/chat", data={"message": "/remove-model"})
    # liste : 1. cfg-distant (remote_config), 2. zz-img (image)
    env_img.web.post("/chat", data={"message": "2"})
    r = env_img.web.post("/chat", data={"message": "oui"})
    assert "PAS touchés" in _sse_texts(r.data)
    assert not mdir.exists()
    payload = env_img.web.get("/models/config").get_json()
    assert not any(m["id"] == "zz-img" for m in payload["models"])


def test_remove_remote_config_edite_local_toml_et_demonte(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    env_img.web.post("/chat", data={"message": "/remove-model"})
    r = env_img.web.post("/chat", data={"message": "1"})  # cfg-distant
    body = _sse_texts(r.data)
    assert (
        "config/local.toml" in body and "défaut" in body
    )  # avertissement default_model
    r = env_img.web.post("/chat", data={"message": "oui"})
    assert "retiré" in _sse_texts(r.data)
    assert "cfg-distant" not in env_img.cfg.read_text(encoding="utf-8").replace(
        'default_model = "cfg-distant"', ""
    )
    payload = env_img.web.get("/models/config").get_json()
    assert not any(m["id"] == "cfg-distant" for m in payload["models"])


def test_extra_reply_persistee_dans_le_journal(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    _install_image(env_img)
    env_img.web.post("/chat", data={"message": "/remove-model"})
    env_img.web.post("/chat", data={"message": "2"})
    env_img.web.post("/chat", data={"message": "oui"})
    timelines = list((env_img.tmp / "sessions").rglob("timeline.jsonl"))
    assert timelines
    events = [
        json.loads(line)
        for line in timelines[0].read_text(encoding="utf-8").splitlines()
    ]
    texts = [e["data"].get("text", "") for e in events if e["event"] == "text"]
    assert any("✅" in t for t in texts)  # le résultat survit au rechargement du fil


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
