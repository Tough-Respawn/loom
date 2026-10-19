# Journal du serveur modèle suivi en fond (llama-swap /logs/stream/upstream) et exposé
# au fil par GET /server/log : préfixe retiré, numérotation, rejeu de tampon dédupliqué.
from __future__ import annotations

from loom.runtime.upstream_log import UpstreamLogTail, strip_prefix


def test_strip_prefix_retire_horodatage_et_niveau():
    assert strip_prefix("0.01.139.576 I srv    load_model: loading model 'x'\n") == (
        "srv    load_model: loading model 'x'"
    )
    assert strip_prefix("\tdry_multiplier = 0.000") == "\tdry_multiplier = 0.000"


def test_since_rend_les_lignes_posterieures_et_le_seq_courant():
    tail = UpstreamLogTail("http://127.0.0.1:1")
    tail.feed(["0.00.1.1 I a\n", "\n", "0.00.1.2 I b\n"])
    seq, lines = tail.since(0)
    assert seq == 2 and lines == ["a", "b"]
    tail.feed(["0.00.1.3 W c\n"])
    assert tail.since(seq) == (3, ["c"])
    assert tail.since(99) == (3, [])


def test_rejeu_de_tampon_a_la_reconnexion_ne_duplique_pas():
    tail = UpstreamLogTail("http://127.0.0.1:1")
    tail.feed(["0.0.0.1 I un", "0.0.0.2 I deux"])
    # Reconnexion : llama-swap rejoue « un, deux » puis envoie du neuf.
    added = tail.feed(
        ["0.0.0.1 I un", "0.0.0.2 I deux", "0.0.0.3 I trois"], resync=True
    )
    assert added == 1
    assert tail.since(0)[1] == ["un", "deux", "trois"]


def test_route_server_log_sans_suivi_puis_avec(tmp_path):
    from loom.agent.session import SessionStore
    from loom.web.app import create_app

    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="p",
        default_model="m",
        known_models=["m"],
    )
    app = create_app(
        client=None,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=["m"],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
    )
    web = app.test_client()
    assert web.get("/server/log").get_json() == {"seq": 0, "lines": []}
    tail = UpstreamLogTail("http://127.0.0.1:1")
    tail.feed(["0.0.0.1 I a", "0.0.0.2 I b"])
    app.S.upstream_log = tail
    assert web.get("/server/log?since=1").get_json() == {"seq": 2, "lines": ["b"]}
    assert web.get("/server/log?since=zz").get_json()["lines"] == ["a", "b"]
