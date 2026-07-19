# Cycle de vie à la SÉLECTION d'un modèle (POST /model) — invariant multi-onglets
# (2026-07-19) : au plus UN modèle local chargé (garanti par llama-swap), et les
# distants n'imposent AUCUNE limite. Sélectionner un distant ne doit donc JAMAIS
# décharger le local (une autre session l'utilise peut-être, voire génère dessus).
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

FAKE_LOCAL = "loc-model"
FAKE_REMOTE = "rem-model"


@pytest.fixture()
def env(tmp_path):
    for d in ("skills", "skills_user", "workspace", "models"):
        (tmp_path / d).mkdir()
    calls = []
    fake_client = SimpleNamespace(
        unload_local=lambda: calls.append("unload") or True,
        remote_route_info=lambda mid: {
            "base_url": "https://x",
            "model": "m",
            "has_key": False,
        },
        remote_api_key=lambda mid: "",
        remove_remote_route=lambda mid: None,
    )
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_LOCAL,
        known_models=[FAKE_LOCAL, FAKE_REMOTE],
    )
    app = create_app(
        client=fake_client,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_LOCAL, FAKE_REMOTE],
        remote_model_ids=[FAKE_REMOTE],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        models_dir=str(tmp_path / "models"),
    )
    web = app.test_client()
    assert web.post("/session/new", data={}).status_code == 200
    return SimpleNamespace(web=web, calls=calls)


def test_selection_distant_ne_decharge_jamais_le_local(env):
    r = env.web.post("/model", data={"model": FAKE_REMOTE})
    assert r.status_code == 200
    time.sleep(0.3)  # l'ancien déchargement partait dans un thread de fond
    assert env.calls == []  # le local d'une autre session reste chargé
