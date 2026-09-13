"""Étape 2 de l'audit timings (2026-09-13) : affectation EXPLICITE des slots.

Constat : llama-server choisit le slot par LRU/similarité, la conversation
s'installait sur le slot 1 et Loom sauvait `/slots/0` (n_saved = 114 = le
prompt de titre). Contrat : fil principal et warm sur le slot 0, appels annexes
(titre, résumé, reflect, sous-agent) sur le slot 1 quand le modèle en a deux,
save refusé quand il ne contient aucun token.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace as NS

from loom.agent.client import LoomClient
from loom.agent.streaming import build_create_kwargs
from loom.tools.agent import SubAgentRunner, make_dispatch_agent
from loom.tools.base import ToolRegistry, ToolSpec

from .fakes import FakeRegistry, collect, make_client, turn_text, turn_tools

# ---- payload ----------------------------------------------------------------


def test_build_create_kwargs_epingle_le_slot_local():
    kw = build_create_kwargs("m", [], "sys", 8, id_slot=0)
    assert kw["extra_body"]["id_slot"] == 0
    kw = build_create_kwargs("m", [], "sys", 8, id_slot=1)
    assert kw["extra_body"]["id_slot"] == 1


def test_build_create_kwargs_sans_slot_ni_pour_le_distant():
    assert "id_slot" not in build_create_kwargs("m", [], "sys", 8)["extra_body"]
    # Une API distante n'a pas de slots : jamais d'extension llama.cpp.
    assert "extra_body" not in build_create_kwargs(
        "m", [], "sys", 8, native_extras=False, id_slot=0
    )


# ---- choix du slot annexe ---------------------------------------------------


def test_annex_slot_1_seulement_si_le_modele_a_deux_slots():
    c = LoomClient(base_url="http://127.0.0.1:8080/v1")
    c.slot_counts = {"orn": 2, "qwen": 1}
    assert c.annex_slot("orn") == 1
    assert c.annex_slot("qwen") == 0
    assert c.annex_slot("inconnu") == 0  # défaut : un seul slot


# ---- fil principal et warm sur le slot 0 -----------------------------------


def test_stream_chat_tools_epingle_le_fil_principal_sur_le_slot_0():
    client, fake = make_client([turn_text("ok")])
    collect(
        client.stream_chat_tools(
            [{"role": "user", "content": "hi"}], "sys", 8, registry=FakeRegistry()
        )
    )
    assert fake.calls[0]["extra_body"]["id_slot"] == 0


def test_stream_chat_tools_accepte_un_slot_explicite():
    client, fake = make_client([turn_text("ok")])
    collect(
        client.stream_chat_tools(
            [{"role": "user", "content": "hi"}],
            "sys",
            8,
            registry=FakeRegistry(),
            id_slot=1,
        )
    )
    assert fake.calls[0]["extra_body"]["id_slot"] == 1


def test_warm_context_amorce_le_slot_0():
    client, fake = make_client([turn_text("x")])
    assert client.warm_context([{"role": "user", "content": "."}], "sys") is True
    assert fake.calls[0]["extra_body"]["id_slot"] == 0


# ---- appels annexes sur le slot annexe -------------------------------------


def test_stream_chat_sans_outils_va_sur_le_slot_annexe():
    client, fake = make_client([turn_text("ok")])
    client.slot_counts = {"local": 2}
    list(client.stream_chat([{"role": "user", "content": "hi"}], "sys", 8))
    assert fake.calls[0]["extra_body"]["id_slot"] == 1


def test_summarize_slice_va_sur_le_slot_annexe():
    client, fake = make_client([])
    client.slot_counts = {"local": 2}
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        return NS(choices=[NS(message=NS(content="résumé"))])

    fake.chat.completions.create = create
    assert client.summarize_slice([{"role": "user", "content": "a" * 50}]) == "résumé"
    assert seen[0]["extra_body"]["id_slot"] == 1


def test_infer_title_va_sur_le_slot_annexe():
    client, fake = make_client([])
    client.slot_counts = {"local": 2}
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        return NS(choices=[NS(message=NS(content="Titre court"))])

    fake.with_options = lambda **_: NS(chat=NS(completions=NS(create=create)))
    assert client.infer_title(None, "bonjour, aide-moi") == "Titre court"
    assert seen[0]["extra_body"]["id_slot"] == 1


def test_sous_agent_local_va_sur_le_slot_annexe():
    client, fake = make_client(
        [
            turn_tools([("p1", "dispatch_agent", '{"task": "liste"}')]),
            turn_text("sous-résultat"),
            turn_text("fini"),
        ]
    )
    client.slot_counts = {"local": 2}
    client.save_slot = lambda *a, **k: False
    client.restore_slot = lambda *a, **k: None

    def sub_registry():
        return ToolRegistry(
            [
                ToolSpec(
                    name="noop",
                    description="rien",
                    parameters={"type": "object", "properties": {}},
                    run=lambda a: "",
                )
            ]
        )

    runner = SubAgentRunner(client, sub_registry, system_prompt="sub", model=None)
    parent = ToolRegistry(
        [make_dispatch_agent(client, sub_registry, system_prompt="sub", runner=runner)]
    )
    collect(
        client.stream_chat_tools(
            [{"role": "user", "content": "go"}], "sys", 8, registry=parent
        )
    )
    assert fake.calls[0]["extra_body"]["id_slot"] == 0  # parent
    assert fake.calls[1]["extra_body"]["id_slot"] == 1  # sous-agent
    assert fake.calls[2]["extra_body"]["id_slot"] == 0  # parent, tour final


# ---- save vérifié -----------------------------------------------------------


def _fake_urlopen(body: dict):
    def urlopen(req, timeout=0):
        return io.BytesIO(json.dumps(body).encode())

    return urlopen


def test_save_slot_refuse_une_sauvegarde_vide(monkeypatch, tmp_path):
    import urllib.request

    c = LoomClient(base_url="http://127.0.0.1:8080/v1")
    c.hot_resume_enabled = True
    c.slots_dir_override = str(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen({"n_saved": 0}))
    assert c.save_slot("orn", "turnend.kv", session_id="s1") is False
    assert not (tmp_path / "turnend.kv.meta.json").exists()
    assert "orn" not in c._warm_slots()


def test_save_slot_accepte_une_sauvegarde_pleine(monkeypatch, tmp_path):
    import urllib.request

    c = LoomClient(base_url="http://127.0.0.1:8080/v1")
    c.hot_resume_enabled = True
    c.slots_dir_override = str(tmp_path)
    monkeypatch.setattr(
        urllib.request, "urlopen", _fake_urlopen({"n_saved": 9697, "n_written": 5})
    )
    assert c.save_slot("orn", "turnend.kv", session_id="s1") is True
    assert (tmp_path / "turnend.kv.meta.json").exists()


# ---- revue 2026-09-13 : distant, save vide après save valide, restore vide -----


def test_jamais_d_id_slot_vers_une_route_distante_meme_avec_extras_natifs():
    # `_resolve` renvoie enable_thinking_param comme drapeau « extras natifs » :
    # une route distante qui l'active recevait id_slot. Un slot n'a de sens qu'en local.
    client, fake = make_client([turn_text("ok")], remote=True)
    client._routes["remote-x"]["enable_thinking_param"] = True
    collect(
        client.stream_chat_tools(
            [{"role": "user", "content": "hi"}],
            "sys",
            8,
            model="remote-x",
            registry=FakeRegistry(),
        )
    )
    body = fake.calls[0]["extra_body"]
    assert "repeat_penalty" in body  # les extras natifs partent toujours
    assert "id_slot" not in body


def test_stream_chat_distant_sans_id_slot():
    client, fake = make_client([turn_text("ok")], remote=True)
    client._routes["remote-x"]["enable_thinking_param"] = True
    list(
        client.stream_chat(
            [{"role": "user", "content": "hi"}], "sys", 8, model="remote-x"
        )
    )
    assert "id_slot" not in fake.calls[0]["extra_body"]


def test_save_vide_invalide_la_meta_du_save_precedent(monkeypatch, tmp_path):
    import urllib.request

    c = LoomClient(base_url="http://127.0.0.1:8080/v1")
    c.hot_resume_enabled = True
    c.slots_dir_override = str(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen({"n_saved": 9697}))
    assert c.save_slot("orn", "turnend.kv", session_id="s1") is True
    assert (tmp_path / "turnend.kv.meta.json").exists()
    # Le serveur a DÉJÀ écrasé le fichier par un save vide : la meta ment désormais.
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen({"n_saved": 0}))
    assert c.save_slot("orn", "turnend.kv", session_id="s1") is False
    assert not (tmp_path / "turnend.kv.meta.json").exists()
    assert "orn" not in c._warm_slots()


def test_restore_vide_est_un_echec(monkeypatch, tmp_path):
    import urllib.request

    c = LoomClient(base_url="http://127.0.0.1:8080/v1")
    c.hot_resume_enabled = True
    c.slots_dir_override = str(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen({"n_restored": 0}))
    assert c.restore_slot("orn", "turnend.kv", force=True) is False
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen({"n_restored": 9697}))
    assert c.restore_slot("orn", "turnend.kv", force=True) is True


def test_compute_slot_counts_suit_la_config():
    from loom.runtime.server_args import compute_slot_counts

    models = [NS(id="orn", cache_isolation=True), NS(id="qwen", cache_isolation=False)]
    assert compute_slot_counts(models, n_parallel=1) == {"orn": 2, "qwen": 1}
    assert compute_slot_counts(models, n_parallel=3) == {"orn": 3, "qwen": 3}
