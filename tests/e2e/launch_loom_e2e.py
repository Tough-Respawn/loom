# Lance une instance loom.web ISOLÉE pour le banc E2E Playwright : vraie UI,
# vrai LoomClient, modèle distant = stub local (tests/e2e/stub_openai.py, port
# 18081). Données sous le temp système, aucun serveur modèle local géré.
# Lancement : uv run python tests/e2e/launch_loom_e2e.py  (port 18090)
# À TUER après le test (processus au premier plan, Ctrl+C).
from __future__ import annotations

import tempfile
from pathlib import Path

from loom.agent.client import LoomClient
from loom.agent.session import SessionStore
from loom.tools import build_registry
from loom.web.app import create_app

DATA = Path(tempfile.gettempdir()) / "loom-e2e-data"
for d in ("skills", "skills_user", "workspace", "plugins"):
    (DATA / d).mkdir(parents=True, exist_ok=True)

MODEL = "stub-remote"
client = LoomClient("http://127.0.0.1:9/v1")  # local inexistant, jamais appelé
client.add_remote_route(
    MODEL,
    {"base_url": "http://127.0.0.1:18081/v1", "api_key": "stub", "model": "stub"},
)

store = SessionStore(
    DATA / "sessions",
    default_system_prompt="Tu es l'agent de test E2E. Réponds brièvement.",
    default_model=MODEL,
    known_models=[MODEL],
    default_tools=["read_file", "list_dir"],
)


def make_registry(active, workspace=None, conversation=None):
    return build_registry(
        workspace_dir=workspace or str(DATA / "workspace"),
        max_bytes=200_000,
        enabled=active or [],
    )


app = create_app(
    client=client,
    skills_dir=str(DATA / "skills"),
    session_store=store,
    models=[MODEL],
    remote_model_ids=[MODEL],
    keepwarm_enabled=False,
    workspace_dir=str(DATA / "workspace"),
    user_skills_dir=str(DATA / "skills_user"),
    plugins_dir=str(DATA / "plugins"),
    remote_store_path=str(DATA / "remote_models.json"),
    tool_factory=make_registry,
)

if __name__ == "__main__":
    print(f"[e2e] données sous {DATA}")
    app.run(host="127.0.0.1", port=18090, threaded=True)
