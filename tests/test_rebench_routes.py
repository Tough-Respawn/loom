# /rebench côté routes : job de calibration STUBBÉ (routes._run_calibration),
# verdict persisté, état b_apply, application au model.toml, verrou anti-double.
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

FAKE_MODEL = "fake-model"
CALIB = {
    "context": 8192,
    "mode": "capacite",
    "mecanisme": "pente 9.0 Ko/token mesurée, vitesse validée",
    "slope_kb_tok": 9.0,
    "valide_jusqua": 7000,
}


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
    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    mdir = tmp_path / "models" / "loc-test"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'filename = "fake.gguf"\ncontext = 4096\n', encoding="utf-8"
    )
    (mdir / "fake.gguf").write_bytes(b"GGUF")
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL, "loc-test"],
    )
    app = create_app(
        client=None,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_MODEL, "loc-test"],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        models_dir=str(tmp_path / "models"),
        local_models=[
            {"id": "loc-test", "dir": str(mdir), "size_mb": 1024, "context": 4096}
        ],
    )
    web = app.test_client()
    assert web.post("/session/new", data={}).status_code == 200
    return SimpleNamespace(web=web, tmp=tmp_path, mdir=mdir)


def _wait_verdict(env, timeout=10.0):
    """Attend que le worker ait posté le verdict dans le journal (thread réel)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for p in (env.tmp / "sessions").rglob("timeline.jsonl"):
            txt = p.read_text(encoding="utf-8")
            if "Verdict" in txt or "déjà au top" in txt or "échouée" in txt:
                return txt
        time.sleep(0.1)
    raise AssertionError("verdict jamais posté")


def _launch(env, monkeypatch, calib=CALIB, error=None):
    from loom.web import routes

    monkeypatch.setitem(routes._REBENCH, "job", None)

    def fake_run(S, spec, progress):
        progress("sonde 4096")
        if error is not None:
            raise error
        return calib, env.mdir / "fake.gguf"

    monkeypatch.setattr(routes, "_run_calibration", fake_run)
    env.web.post("/chat", data={"message": "/rebench loc-test"})
    return env.web.post("/chat", data={"message": "oui"})


def test_rebench_job_poste_verdict_et_etat_apply(env, monkeypatch):
    r = _launch(env, monkeypatch)
    assert "lancée" in _sse_texts(r.data)
    txt = _wait_verdict(env)
    assert "4096 → 8192" in txt and "oui" in txt
    # l'état b_apply est actif : « oui » applique
    r = env.web.post("/chat", data={"message": "oui"})
    assert "Application" in _sse_texts(r.data)
    toml = (env.mdir / "model.toml").read_text(encoding="utf-8")
    assert "context = 8192" in toml
    # visible aussi par l'endpoint disque (onglet Modèles locaux)
    payload = env.web.get("/models/local").get_json()
    m = next(x for x in payload["models"] if x["id"] == "loc-test")
    assert m["context"] == 8192


def test_rebench_deja_au_top(env, monkeypatch):
    r = _launch(env, monkeypatch, calib=dict(CALIB, context=4096))
    assert "lancée" in _sse_texts(r.data)
    txt = _wait_verdict(env)
    assert "déjà au top" in txt
    # PAS d'état wizard b_apply persisté : rien à « appliquer »
    sessions = "".join(
        p.read_text(encoding="utf-8")
        for p in (env.tmp / "sessions").rglob("session.json")
    )
    assert "b_apply" not in sessions
    assert "context = 4096" in (env.mdir / "model.toml").read_text(encoding="utf-8")


def test_rebench_echec_calibration_message_persiste(env, monkeypatch):
    _launch(env, monkeypatch, error=RuntimeError("binaire llama-server introuvable"))
    txt = _wait_verdict(env)
    assert "échouée" in txt and "introuvable" in txt
    assert "context = 4096" in (env.mdir / "model.toml").read_text(encoding="utf-8")


def test_rebench_refus_types_et_inconnu(env, monkeypatch):
    r = env.web.post("/chat", data={"message": "/rebench nexiste-pas"})
    assert "inconnu" in _sse_texts(r.data)


def test_rebench_verrou_un_seul_job(env, monkeypatch):
    from loom.web import routes

    monkeypatch.setitem(
        routes._REBENCH, "job", SimpleNamespace(done=False, label="", final=None)
    )
    env.web.post("/chat", data={"message": "/rebench loc-test"})
    r = env.web.post("/chat", data={"message": "oui"})
    assert "déjà en cours" in _sse_texts(r.data)
