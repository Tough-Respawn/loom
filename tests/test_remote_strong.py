# Tier « fort » par MODÈLE distant, plus par simple distance (2026-07-15).
# Constat live : GLM-4.7-Flash (free, MoE 30B-A3B — gabarit d'un local) recevait le
# prompt allégé strong avec les gardes coupées -> il ANNONCE « je lance les 3
# chercheurs » et s'arrête sans un seul tool_call. Un distant faible (strong=false)
# doit garder le harnais complet (prompt local + act-nudge).
from __future__ import annotations

from .fakes import FakeOAI, FakeRegistry, turn_text


def test_config_remote_strong_par_defaut_et_explicite():
    from loom.config import _parse_remote_model

    base = {"id": "m", "base_url": "https://x/v1", "model": "prov/m"}
    assert _parse_remote_model(base).strong is True  # défaut : distant = fort
    assert _parse_remote_model({**base, "strong": False}).strong is False


def test_distant_faible_recoit_le_harnais_complet(tmp_path):
    # Modèle distant marqué faible -> system prompt COMPLET de la session (pas
    # CHAT_SYSTEM_STRONG) et gardes actives : l'intention sans tool_call est
    # relancée (act-nudge) au lieu de passer.
    from loom.agent.client import LoomClient
    from loom.agent.session import SessionStore
    from loom.web.app import create_app

    MODEL = "remote-faible"
    client = LoomClient("http://127.0.0.1:9/v1")
    client.add_remote_route(
        MODEL,
        {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "fake/x"},
    )
    fake = FakeOAI(
        [
            turn_text("Je vais maintenant lire le fichier de configuration."),
            turn_text("Terminé."),
        ]
    )
    client._routes[MODEL]["client"] = fake
    registry = FakeRegistry({"list_dir": lambda a: "a.txt"})
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt harnais complet de test",
        default_model=MODEL,
        known_models=[MODEL],
    )
    app = create_app(
        client=client,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[MODEL],
        remote_model_ids=[MODEL],
        remote_weak_ids=[MODEL],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        tool_factory=lambda tools, ws, conv: registry,
    )
    web = app.test_client()
    r = web.post("/session/new", data={"title": "session testée"})
    assert r.status_code == 200
    sid = r.get_json()["id"]
    r = web.post("/chat", data={"message": "fais le travail", "session_id": sid})
    assert r.status_code == 200
    r.get_data()  # draine le SSE
    # Harnais complet : le system prompt est celui de la session, pas le strong FR.
    system = fake.calls[0]["messages"][0]["content"]
    assert "prompt harnais complet de test" in system
    assert "Tu es Loom" not in system
    # Gardes actives : l'intention nue a été relancée -> 2 appels modèle.
    assert len(fake.calls) == 2
