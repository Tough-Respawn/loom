"""Sauvegarde du slot de la conversation : hors du chemin critique, en tête de la
maintenance (sous le verrou, avant que reflect ou le titre touchent un slot)."""

import threading
from types import SimpleNamespace as NS

from loom.web.routes import maintenance


def _S(tmp_path, calls, abort=False):
    client = NS(
        save_slot=lambda model, name, session_id=None: (
            calls.append(("save", name)) or True
        ),
        restore_slot=lambda model, name: calls.append(("restore", name)) or True,
    )
    return NS(
        client=client,
        remote_model_ids=set(),
        warm_holder={"abort": abort},
        local_gen_lock=threading.Lock(),
        local_busy={"reason": ""},
        last_activity=[0.0],
        session_store=NS(session_dir=lambda sid: tmp_path),
    )


def test_sauvegarde_avant_reflect(tmp_path, monkeypatch):
    calls = []
    S = _S(tmp_path, calls)

    def fake_reflect(*a, **k):
        calls.append(("reflect", ""))

    monkeypatch.setattr("loom.agent.reflect.reflect", fake_reflect)
    S.reflect_model = "m"
    S.reflect_stores = NS(provider=None, paths=None, learned_dir=None)
    maintenance._post_turn_maintenance(S, NS(id="s1"), [], [], "ok", "m", True)
    assert calls == [("save", "turnend.kv"), ("reflect", ""), ("restore", "turnend.kv")]
    assert not S.local_gen_lock.locked()


def test_message_en_attente_saute_la_sauvegarde(tmp_path):
    calls = []
    S = _S(tmp_path, calls, abort=True)
    maintenance._post_turn_maintenance(S, NS(id="s1"), [], [], "ok", "m", False)
    assert calls == []
