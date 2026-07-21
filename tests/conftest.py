# Fixtures des tests de NON-RÉGRESSION (caractérisation) — filet posé AVANT les
# refactors P2-1 (create_app), P2-3/P2-4 (stream_chat_tools). Ces tests figent le
# comportement ACTUEL observé ; après refactor, on garde ceux qui protègent un
# invariant durable et on jette le reste (décision 2026-07-13).
from __future__ import annotations

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

FAKE_MODEL = "fake-model"


@pytest.fixture()
def tmp_env(tmp_path):
    """Arborescence isolée : tout ce que l'app écrit reste sous tmp_path."""
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills_user").mkdir()
    (tmp_path / "workspace").mkdir()
    return tmp_path


@pytest.fixture()
def app(tmp_env):
    store = SessionStore(
        tmp_env / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL],
    )
    return create_app(
        client=None,
        skills_dir=str(tmp_env / "skills"),
        session_store=store,
        models=[FAKE_MODEL],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_env / "workspace"),
        user_skills_dir=str(tmp_env / "skills_user"),
        plugins_dir=str(tmp_env / "plugins"),
        remote_store_path=str(tmp_env / "remote_models.json"),
        learned_skills_dir=str(tmp_env / "skills_learned"),
        memory_db_path=str(tmp_env / "memory" / "memory.db"),
        identity_paths={
            "soul_path": str(tmp_env / "identity" / "SOUL.md"),
            "user_path": str(tmp_env / "identity" / "USER.md"),
            "memory_md_path": str(tmp_env / "identity" / "MEMORY.md"),
        },
    )


@pytest.fixture()
def web(app):
    """Client de test Flask ; une session focus existe après /session/new."""
    return app.test_client()


@pytest.fixture()
def web_sess(web):
    """Client + session active créée (la plupart des routes exigent une session)."""
    r = web.post("/session/new", data={})
    assert r.status_code == 200, r.data
    return web
