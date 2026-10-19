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
    if body.lstrip().startswith(b"{"):
        data = json.loads(body)
        return data.get("reply", data.get("message", ""))
    out = []
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if evt["type"] == "text":
                out.append(evt.get("text", ""))
    return "\n".join(out)


@pytest.fixture()
def env(tmp_path):
    # Arbo réaliste : la racine des modèles = tmp/root (local/text + remote).
    root = tmp_path / "root"
    (root / "local" / "text").mkdir(parents=True)
    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL],
    )
    app = create_app(
        client=SimpleNamespace(
            add_remote_route=lambda *a: None, remove_remote_route=lambda *a: None
        ),
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_MODEL],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        config_local_path=str(tmp_path / "local.toml"),
        models_dir=str(root / "local" / "text"),
        models_roots=[str(root)],
    )
    web = app.test_client()
    assert web.post("/session/new", data={}).status_code == 200
    return SimpleNamespace(web=web, tmp=tmp_path, root=root)


def _fake_deps(monkeypatch, hits=(), files=()):
    from loom.web import routes

    deps = SimpleNamespace(
        existing_ids={FAKE_MODEL},
        search_models=lambda q: list(hits),
        list_gguf_files=lambda repo: list(files),
        recommend=lambda fs, repo=None: [
            dict(f, fits=True, recommended=True) for f in fs
        ],
        derive_id=lambda repo: "nouveau-modele",
        list_remote_models=lambda base_url, key: None,  # provider muet -> saisie libre
        removable_models=lambda: [
            {"id": "glm-test", "kind": "remote", "label": "glm-test — distant"}
        ],
        install_roots=lambda: [{"path": "/modeles-prioritaires", "free_gb": 100}],
    )
    monkeypatch.setattr(routes.models, "_wizard_deps", lambda S: deps)


def test_add_model_ack_et_etat_persiste(env, monkeypatch):
    _fake_deps(monkeypatch)
    r = env.web.post("/models/add", data={"message": "/add-model"})
    assert r.status_code == 200
    assert r.is_json
    assert "1. local" in _sse_texts(r.data)
    # l'étape suivante est bien routée vers le wizard (état persisté sur la session)
    r2 = env.web.post("/models/add", data={"message": "2"})
    assert "distant 1/5" in _sse_texts(r2.data)


def test_add_model_cancel(env, monkeypatch):
    _fake_deps(monkeypatch)
    env.web.post("/models/add", data={"message": "/add-model"})
    r = env.web.post("/models/add", data={"message": "/cancel"})
    assert "annulé" in _sse_texts(r.data).lower()


@pytest.mark.parametrize(
    "command", ["/add-model", "/add-model distant", "/rebench", "/rebench x", "/cancel"]
)
@pytest.mark.parametrize("busy", [False, True])
def test_commandes_interface_ne_generent_ni_note_ni_journal(
    env, command, busy, monkeypatch
):
    from loom.web.routes import models
    from loom.web.routes.helpers import _cancel_for, _lock_for

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Une commande d’interface ne doit pas construire les dépendances du wizard"
        )

    monkeypatch.setattr(models, "_wizard_deps", forbidden)
    S = env.web.application.S
    sess = S.cur["session"]
    lock = _lock_for(S, sess.id)
    if busy:
        lock.acquire()
    before = {p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")}
    try:
        for endpoint, key in (("/chat", "message"), ("/note", "text")):
            r = env.web.post(endpoint, data={key: command, "session_id": sess.id})
            assert r.status_code == 200
            expected = {"/cancel": "cancel_interface", "/rebench": "rebench"}.get(
                command.split()[0], "add_model"
            )
            assert r.get_json()["ui_action"] == expected
        assert S.notes.drain(sess.id) == []
        assert not _cancel_for(S, sess.id).is_set()
        assert lock.locked() == busy
        assert before == {
            p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")
        }
    finally:
        if busy:
            lock.release()


def test_panneau_ajout_independant_du_chat_et_annulation_sans_deps(env, monkeypatch):
    from loom.web.routes import models
    from loom.web.routes.helpers import _lock_for

    _fake_deps(monkeypatch)
    S = env.web.application.S
    sess = S.cur["session"]
    before = {p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")}
    lock = _lock_for(S, sess.id)
    lock.acquire()
    try:
        assert env.web.post("/models/add", data={"message": "/add-model"}).get_json()[
            "active"
        ]
        assert env.web.post("/models/add", data={"message": "2"}).get_json()["active"]
        assert "distant" in env.web.get("/models/add").get_json()["reply"]
        assert sess.conversation.wizard is None
        monkeypatch.setattr(
            models,
            "_wizard_deps",
            lambda S: (_ for _ in ()).throw(ImportError("GPU_KV_BYTES")),
        )
        assert env.web.post("/chat", data={"message": "/cancel"}).status_code == 200
        assert not env.web.get("/models/add").get_json()["active"]
        assert lock.locked()
        assert before == {
            p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")
        }
    finally:
        lock.release()


def test_import_echoue_puis_cancel_ne_lance_pas_le_modele(env, monkeypatch):
    from loom.web.routes import models

    def failed(S):
        raise ImportError("cannot import name 'GPU_KV_BYTES'")

    monkeypatch.setattr(models, "_wizard_deps", failed)
    r = env.web.post("/models/add", data={"message": "/add-model"})
    assert "Redémarre Loom" in r.get_json()["reply"]
    assert not r.get_json()["active"]
    assert (
        env.web.post("/chat", data={"message": "/cancel"}).get_json()["ui_action"]
        == "cancel_interface"
    )


def test_cancel_efface_ancien_wizard_sans_dependances(env, monkeypatch):
    from loom.web.routes import models

    S = env.web.application.S
    conv = S.cur["session"].conversation
    conv.set_wizard({"step": "r_key", "id": "old"})
    monkeypatch.setattr(
        models,
        "_wizard_deps",
        lambda S: pytest.fail("cancel doit être indépendant des imports"),
    )
    messages = list(conv.messages)
    assert env.web.post("/chat", data={"message": "/cancel"}).status_code == 200
    assert conv.wizard is None
    assert conv.messages == messages


def test_panneaux_ajout_isoles_entre_sessions(env, monkeypatch):
    _fake_deps(monkeypatch)
    S = env.web.application.S
    first = S.cur["session"].id
    env.web.post("/models/add", data={"message": "/add-model", "session_id": first})
    env.web.post("/session/new", data={})
    second = S.cur["session"].id
    assert not env.web.get(
        "/models/add", query_string={"session_id": second}
    ).get_json()["active"]
    env.web.post("/models/add", data={"message": "/cancel", "session_id": second})
    assert env.web.get("/models/add", query_string={"session_id": first}).get_json()[
        "active"
    ]


def test_navigateur_commandes_interface_preservent_generation(env, monkeypatch):
    """Exercise actual keyboard routing and panel JS, with a fake provider."""
    import threading
    from pathlib import Path

    from werkzeug.serving import make_server

    playwright = pytest.importorskip("playwright.sync_api")
    _fake_deps(monkeypatch)
    env.web.application.S.client = None  # no model or provider is ever invoked
    with playwright.sync_playwright() as p:
        if not Path(p.chromium.executable_path).exists():
            pytest.skip("Chromium non installé")
        server = make_server("127.0.0.1", 0, env.web.application, threaded=True)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        browser = None
        try:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            chat_requests, errors = [], []
            page.on(
                "request",
                lambda r: (
                    chat_requests.append(r.url)
                    if r.url.endswith(("/chat", "/note"))
                    else None
                ),
            )
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            assert not errors
            composer = page.locator(".chat-form textarea").first
            composer.wait_for()
            page.evaluate("""async () => {
              const {state, tab} = await import('/static/state.js');
              const current = tab(state.active);
              current.streaming = true;
              window.originalAbort = current.abort = new AbortController();
            }""")
            composer.fill("/cancel")
            composer.press("Enter")
            playwright.expect(composer).to_have_value("")
            composer.fill("/add-model distant")
            composer.press("Enter")
            playwright.expect(page.locator("#model-add-reply")).to_contain_text(
                "distant 1/5"
            )
            page.locator("#model-add-input").fill("test-remote")
            page.locator("#model-add-form button[type=submit]").click()
            playwright.expect(page.locator("#model-add-reply")).to_contain_text(
                "base_url"
            )
            page.locator("#model-add-input").fill("https://example.org/v1")
            page.locator("#model-add-form button[type=submit]").click()
            playwright.expect(page.locator("#model-add-input")).to_have_attribute(
                "type", "password"
            )
            page.locator("#model-add-cancel").click()
            playwright.expect(page.locator("#model-add-reply")).to_contain_text(
                "annulé"
            )
            # Installation completion refreshes the actual selector, without switching
            # the model of the running conversation or losing the remote panel.
            selected = page.locator("#model-select").input_value()
            env.web.application.S.models.append("installed-model")
            page.evaluate(
                "document.dispatchEvent(new CustomEvent('loom:models-changed'))"
            )
            playwright.expect(
                page.locator('#model-select option[value="installed-model"]')
            ).to_have_count(1)
            playwright.expect(page.locator("#model-select")).to_have_value(selected)
            assert not page.evaluate("window.originalAbort.signal.aborted")
            assert not chat_requests
            assert not errors
        finally:
            if browser:
                browser.close()
            server.shutdown()
            worker.join(timeout=5)


def test_add_model_distant_persiste_le_dossier_remote(env, monkeypatch):
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
        env.web.post("/models/add", data={"message": msg})
    r = env.web.post("/models/add", data={"message": "non"})
    assert "ajouté" in _sse_texts(r.data)
    # Un distant = un dossier remote/<id>/model.toml sur la racine prioritaire —
    # ni store JSON, ni [[remote_models]] dans local.toml.
    import tomllib

    p = env.root / "remote" / "glm-test" / "model.toml"
    assert p.exists()
    d = tomllib.loads(p.read_text(encoding="utf-8"))
    assert d["base_url"] == "https://api.exemple/v4"
    assert d["model"] == "glm-test-model"
    assert not (env.tmp / "remote_models.json").exists()
    assert not (env.tmp / "local.toml").exists()


def test_remove_model_ouvre_interface_sans_ecrire_dans_session(env, monkeypatch):
    _fake_deps(monkeypatch)
    before = {p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")}
    for endpoint, key in (("/chat", "message"), ("/note", "text")):
        r = env.web.post(endpoint, data={key: "/remove-model"})
        assert r.get_json() == {"ui_action": "manage_models"}
    assert before == {
        p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")
    }


def test_remove_model_pendant_generation_ne_note_ni_annule(env):
    from loom.web.routes.helpers import _cancel_for, _lock_for

    S = env.web.application.S
    sess = S.cur["session"]
    lock = _lock_for(S, sess.id)
    lock.acquire()
    try:
        for endpoint, key in (("/chat", "message"), ("/note", "text")):
            r = env.web.post(
                endpoint, data={key: "/remove-model", "session_id": sess.id}
            )
            assert r.get_json() == {"ui_action": "manage_models"}
        assert not _cancel_for(S, sess.id).is_set()
        assert S.notes.drain(sess.id) == []
        assert lock.locked()
    finally:
        lock.release()


def test_model_occupe_non_supprimable_et_autre_modele_independant(env):
    from loom.web.routes.helpers import _lock_for

    S = env.web.application.S
    for mid in (FAKE_MODEL, "unused"):
        folder = env.root / "local" / "text" / mid
        folder.mkdir()
        (folder / "weights.gguf").write_bytes(b"test")
        S.local_model_specs.append({"id": mid, "dir": str(folder)})
    sess = S.cur["session"]
    lock = _lock_for(S, sess.id)
    lock.acquire()
    try:
        assert env.web.delete("/models/installed/" + FAKE_MODEL).status_code == 409
        assert (env.root / "local" / "text" / FAKE_MODEL / "weights.gguf").exists()
        assert env.web.delete("/models/installed/unused").status_code == 200
        assert lock.locked()
    finally:
        lock.release()


def test_suppression_refuse_dossier_parent(env):
    S = env.web.application.S
    S.local_model_specs.append({"id": "bad", "dir": str(env.root / "local" / "text")})
    assert env.web.delete("/models/installed/bad").status_code == 409
    assert (env.root / "local" / "text").is_dir()


def test_remove_model_distant_supprime_le_dossier(env, monkeypatch):
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
        env.web.post("/models/add", data={"message": msg})
    assert (env.root / "remote" / "glm-test").is_dir()
    r = env.web.delete("/models/installed/glm-test")
    assert r.status_code == 200
    assert "retiré" in r.get_json()["message"]
    assert not (env.root / "remote" / "glm-test").exists()


# ---------- tous types : liste complète, image/vidéo, distant config ----------


def test_removable_models_liste_les_4_familles(tmp_path):
    from loom.web import routes

    cfg = tmp_path / "local.toml"
    cfg.write_text('[chat]\ndefault_model = "cfg"\n', encoding="utf-8")
    S = SimpleNamespace(
        local_model_specs=[{"id": "loc", "size_mb": 1024}],
        remote_store_path=str(tmp_path / "remote_models.json"),
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
    # Tous les distants sortent en kind "remote" : config/local.toml, source unique.
    assert kinds == {
        "loc": "local",
        "ui": "remote",
        "cfg": "remote",
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
            recommend=lambda fs, repo=None: fs,
            derive_id=lambda repo: "x",
            list_remote_models=lambda base_url, key: None,
            removable_models=lambda: routes._removable_models(S),
            image_dir_state=lambda k, m: routes._image_dir_state(S, k, m),
            check_workflow=routes._check_workflow,
        )

    monkeypatch.setattr(routes.models, "_wizard_deps", fake)


def _install_image(env_img, mid="zz-img"):
    wf = env_img.tmp / "wf_api.json"
    wf.write_text('{"1": {"inputs": {"text": "{PROMPT}"}}}', encoding="utf-8")
    for msg in ["/add-model image", mid, "ok", "mon générateur"]:
        env_img.web.post("/models/add", data={"message": msg})
    return env_img.web.post("/models/add", data={"message": str(wf)})


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
        env_img.web.post("/models/add", data={"message": msg})
    r = env_img.web.post("/models/add", data={"message": "plus tard"})
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
        env_img.web.post("/models/add", data={"message": msg})
    mdir = env_img.root / "local" / "image" / "zz-resume"
    (mdir / "workflow.json").write_text('{"1": "{PROMPT}"}', encoding="utf-8")
    env_img.web.post("/models/add", data={"message": "/add-model image"})
    r = env_img.web.post("/models/add", data={"message": "zz-resume"})
    assert "montage direct" in _sse_texts(r.data)
    payload = env_img.web.get("/models/config").get_json()
    assert any(m["id"] == "zz-resume" for m in payload["models"])


def test_remove_image_rmtree_dossier_et_demonte(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    _install_image(env_img)
    mdir = env_img.root / "local" / "image" / "zz-img"
    r = env_img.web.delete("/models/installed/zz-img")
    assert r.status_code == 200
    assert "PAS touchés" in r.get_json()["message"]
    assert not mdir.exists()
    payload = env_img.web.get("/models/config").get_json()
    assert not any(m["id"] == "zz-img" for m in payload["models"])


def test_remove_remote_config_edite_local_toml_et_demonte(env_img, monkeypatch):
    # Distant hérité encore déclaré en [[remote_models]] (pas migré : pas de boot
    # load_config dans ce fixture) : /remove-model doit le purger du fichier AUSSI.
    _real_img_deps(monkeypatch)
    items = env_img.web.get("/models/removable").get_json()["models"]
    assert next(m for m in items if m["id"] == "cfg-distant")["is_default"]
    r = env_img.web.delete("/models/installed/cfg-distant")
    assert r.status_code == 200
    assert "retiré" in r.get_json()["message"]
    assert "cfg-distant" not in env_img.cfg.read_text(encoding="utf-8").replace(
        'default_model = "cfg-distant"', ""
    )
    payload = env_img.web.get("/models/config").get_json()
    assert not any(m["id"] == "cfg-distant" for m in payload["models"])


def test_remote_delete_preserves_remaining_editor_list(env_img):
    S = env_img.web.application.S
    S.remote_model_ids.add("remaining")
    S.models.append("remaining")
    response = env_img.web.delete("/models/remote/cfg-distant")
    assert response.status_code == 200
    assert [m["id"] for m in response.get_json()["remotes"]] == ["remaining"]


def test_suppression_ne_pollue_pas_le_journal_conversation(env_img, monkeypatch):
    _real_img_deps(monkeypatch)
    _install_image(env_img)
    files = list((env_img.tmp / "sessions").rglob("timeline.jsonl"))
    before = {p: p.read_bytes() for p in files}
    assert env_img.web.delete("/models/installed/zz-img").status_code == 200
    assert before == {p: p.read_bytes() for p in files}


def _install_local(env, monkeypatch, probe):
    """/add-model local de bout en bout, téléchargement simulé ; `probe` = résultat
    du test de chargement (jamais un vrai llama-server en test unitaire)."""
    from loom.web.routes import model_admin

    monkeypatch.setattr(model_admin, "_probe_install", lambda gguf: probe)
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
        env.web.post("/models/add", data={"message": msg})
    r = env.web.post("/models/add", data={"message": "ok"})
    body = _sse_texts(r.data)
    assert "téléchargement" in body.lower()

    mdir = env.root / "local" / "text" / "nouveau-modele"
    assert (mdir / "model.toml").exists()
    assert 'repo = "org/mon-GGUF"' in (mdir / "model.toml").read_text(encoding="utf-8")
    payload = env.web.get("/models/config").get_json()
    return any(m["id"] == "nouveau-modele" for m in payload["models"])


def test_add_model_local_installe(env, monkeypatch):
    # monté à chaud : visible dans la liste des modèles sélectionnables
    assert _install_local(env, monkeypatch, probe=(True, ""))


def test_installation_panneau_pendant_generation_sans_journal(env, monkeypatch):
    from loom.web.routes.helpers import _lock_for

    S = env.web.application.S
    sess = S.cur["session"]
    before = {p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")}
    lock = _lock_for(S, sess.id)
    lock.acquire()
    try:
        assert _install_local(env, monkeypatch, probe=(True, ""))
        assert lock.locked()
        assert before == {
            p: p.read_bytes() for p in (env.tmp / "sessions").rglob("*.json*")
        }
    finally:
        lock.release()


def test_add_model_local_illisible_non_monte(env, monkeypatch):
    # llama-server refuse le GGUF (ex. quant d'un fork) : téléchargé mais PAS monté
    refus = (False, "tensor 'output.weight' has invalid ggml type 142")
    assert not _install_local(env, monkeypatch, probe=refus)


def test_add_model_local_test_non_concluant_monte(env, monkeypatch):
    # test impossible (binaire introuvable) : on ne bloque pas l'installation
    assert _install_local(env, monkeypatch, probe=(None, "binaire introuvable"))


def test_modele_refuse_marque_liste_et_supprimable(env, monkeypatch):
    # Vécu 2026-10-17 : Bonsai PQ2_0 refusé, 8 Go orphelins, absent de /remove-model.
    from loom.config import _discover_models
    from loom.web import routes
    from loom.web.routes import model_admin

    refus = (False, "tensor 'output.weight' has invalid ggml type 142")
    assert not _install_local(env, monkeypatch, probe=refus)
    mdir = env.root / "local" / "text" / "nouveau-modele"
    assert "unloadable" in (mdir / "model.toml").read_text(encoding="utf-8")
    # Pas remonté au prochain démarrage.
    assert "nouveau-modele" not in [m.id for m in _discover_models(mdir.parent)]

    real = {}

    def deps_reels(S):
        real["S"] = S
        return SimpleNamespace(
            removable_models=lambda: model_admin._removable_models(S)
        )

    monkeypatch.setattr(routes.models, "_wizard_deps", deps_reels)
    items = env.web.get("/models/removable").get_json()["models"]
    assert any(
        m["id"] == "nouveau-modele" and "non chargeable" in m["label"] for m in items
    )
    r = env.web.delete("/models/installed/nouveau-modele")
    assert r.status_code == 200
    assert "supprimé" in r.get_json()["message"]
    assert not mdir.exists()
