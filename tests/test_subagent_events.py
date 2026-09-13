# tests/test_subagent_events.py
"""ST-02 : contrat d'événements des sous-agents (identité + chronologie).

Le runner émet subagent_start / subagent_tool_call / subagent_tool_result /
subagent_usage / subagent_end, corrélés par un agent_id stable, pour les DEUX
chemins (dispatch_agent et run_workflow). Télémétrie pure : le contexte du
parent ne reçoit toujours que la synthèse.
"""

from __future__ import annotations

import loom.tools.agent as agent_mod
from loom.tools.agent import SubAgentRunner, make_dispatch_agent
from loom.tools.base import ToolRegistry, ToolSpec
from loom.tools.workflow import make_run_workflow

from .fakes import make_client, turn_text, turn_tools
from .test_dispatch_routing import ChainClient, FakeRegistry

_SYNTHESE = "Inventaire : alpha_load, alpha_save, beta_run, gamma_parse, delta_merge."


def _sub_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="list_dir",
                description="liste",
                parameters={"type": "object", "properties": {}},
                run=lambda a: f"{a.get('path') or 'src'}/alpha.py lib/gamma.py",
            )
        ]
    )


def _parent_stream(scripts):
    """Chemin réel parent -> dispatch_agent -> sous-boucle (FakeOAI partagé)."""
    client, fake = make_client(scripts)
    client.save_slot = lambda *a, **k: False
    client.restore_slot = lambda *a, **k: None
    runner = SubAgentRunner(client, _sub_registry, system_prompt="sub-sys", model=None)
    parent_reg = ToolRegistry(
        [
            make_dispatch_agent(
                client, _sub_registry, system_prompt="sub-sys", runner=runner
            )
        ]
    )
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "Inventorie le projet."}],
            "parent-sys",
            1024,
            model=None,
            registry=parent_reg,
            thinking=False,
            max_iters=6,
        )
    )
    return events, fake


def _healthy_scripts():
    return [
        turn_tools([("p1", "dispatch_agent", '{"task": "inventorie le projet"}')]),
        turn_tools([("s1", "list_dir", "{}")]),
        turn_text(_SYNTHESE),
        turn_text("Terminé."),
    ]


def _subevents(events):
    return [(k, p) for k, p in events if k.startswith("subagent_")]


# --- 1. ouvrier réussi : séquence complète, corrélée, bornée -------------------


def test_ouvrier_reussi_sequence_complete():
    events, _ = _parent_stream(_healthy_scripts())
    subs = _subevents(events)
    kinds = [k for k, _ in subs]
    assert kinds[0] == "subagent_start" and kinds[-1] == "subagent_end"
    assert "subagent_tool_call" in kinds and "subagent_tool_result" in kinds
    assert "subagent_usage" in kinds

    # UN agent_id stable sur toute la délégation.
    ids = {p["agent_id"] for _, p in subs}
    assert len(ids) == 1

    start = subs[0][1]
    assert start["parent"] == "dispatch"
    assert start["label"] == "inventorie le projet"
    assert start["model_requested"] == "local" and start["model_resolved"] == "local"

    end = subs[-1][1]
    assert end["status"] == "completed"
    assert end["stop_reason"] == "natural"
    assert end["duration_s"] >= 0
    assert end["events_dropped"] == 0

    # Usage CUMULÉ : le dernier événement porte le total des 2 tours (100 + 100).
    usages = [p for k, p in subs if k == "subagent_usage"]
    assert usages[-1]["prompt_tokens"] == 200
    assert usages[-1]["completion_tokens"] == 10

    # Aperçu de résultat borné, outil nommé.
    res = next(p for k, p in subs if k == "subagent_tool_result")
    assert res["name"] == "list_dir" and res["ok"] is True
    assert len(res["preview"]) <= 160


def test_zero_injection_dans_le_contexte_du_parent():
    """La télémétrie ne touche PAS le prompt : le tour final du parent voit la
    synthèse comme message tool, et aucun message ne mentionne l'agent_id."""
    events, fake = _parent_stream(_healthy_scripts())
    aid = _subevents(events)[0][1]["agent_id"]
    final_msgs = fake.calls[3]["messages"]  # dernier appel API du parent
    tool_msgs = [m for m in final_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["content"] == _SYNTHESE
    assert not any(aid in str(m.get("content", "")) for m in final_msgs)
    assert not any("subagent" in str(m.get("content", "")) for m in final_msgs)


def test_chronologie_bornee_compte_les_surplus(monkeypatch):
    monkeypatch.setattr(agent_mod, "SUBAGENT_EVENT_CAP", 2)
    # Arguments DISTINCTS à chaque appel : trois list_dir({}) identiques
    # déclencheraient le garde anti-répétition de la sous-boucle (repeat_stop,
    # constaté en écrivant ce test) avant même d'éprouver le plafond.
    scripts = [
        turn_tools([("p1", "dispatch_agent", '{"task": "t"}')]),
        turn_tools([("s1", "list_dir", '{"path": "a"}')]),
        turn_tools([("s2", "list_dir", '{"path": "b"}')]),
        turn_tools([("s3", "list_dir", '{"path": "c"}')]),
        turn_text(_SYNTHESE),
        turn_text("Fini."),
    ]
    events, _ = _parent_stream(scripts)
    subs = _subevents(events)
    mirrors = [
        k for k, _ in subs if k in ("subagent_tool_call", "subagent_tool_result")
    ]
    assert len(mirrors) == 2  # plafond respecté
    end = subs[-1][1]
    assert end["events_dropped"] == 4  # 3 outils x2 mirrors - 2 émis
    assert end["status"] == "completed"  # usage/fin passent toujours


# --- 2. échec API : relève visible, identité stable ; échec total = failed -----


def test_releve_api_error_garde_l_identite_et_note_le_modele_final():
    client = ChainClient(
        {"flash": "api_error", "zai": "ok"}, remote_ids={"flash", "zai"}
    )
    runner = SubAgentRunner(
        client,
        lambda: FakeRegistry(),
        system_prompt="s",
        model="local-x",
        model_chain=["flash", "zai"],
    )
    events = list(runner.stream("t"))
    subs = _subevents(events)
    assert len({p["agent_id"] for _, p in subs}) == 1
    start, end = subs[0][1], subs[-1][1]
    assert start["model_resolved"] == "flash"  # premier tier appelé
    assert end["status"] == "completed" and end["model"] == "zai"  # tier final
    assert any("relève" in str(p) for k, p in events if k == "content")


def test_tous_tiers_morts_status_failed():
    client = ChainClient(
        {"flash": "api_error", "local-x": "api_error"}, remote_ids={"flash"}
    )
    runner = SubAgentRunner(
        client,
        lambda: FakeRegistry(),
        system_prompt="s",
        model="local-x",
        model_chain=["flash"],
    )
    subs = _subevents(list(runner.stream("t")))
    end = subs[-1][1]
    assert end["status"] == "failed" and end["stop_reason"] == "api_error"


# --- 2 bis. modèle demandé BRUT vs résolu --------------------------------------


def test_role_brut_preserve_avant_resolution():
    """model='cheap' : subagent_start montre le RÔLE demandé tel quel, et le tier
    concret résolu — c'est la paire qui rend un mauvais routage diagnosticable."""
    client = ChainClient({"flash": "ok"}, remote_ids={"flash", "zai"})
    runner = SubAgentRunner(
        client,
        lambda: FakeRegistry(),
        system_prompt="s",
        model="local-x",
        model_chain=["flash", "zai"],
        model_roles={"cheap": "flash"},
    )
    subs = _subevents(list(runner.stream("t", model="cheap")))
    start = subs[0][1]
    assert start["model_requested"] == "cheap"  # le rôle BRUT, pas sa résolution
    assert start["model_resolved"] == "flash"  # l'id concret réellement appelé


def test_routage_automatique_sans_modele_explicite():
    # Aucune demande explicite : le routage automatique = la tête de chaîne,
    # et requested == resolved (rien n'a été « demandé » puis traduit).
    client = ChainClient({"flash": "ok"}, remote_ids={"flash"})
    runner = SubAgentRunner(
        client,
        lambda: FakeRegistry(),
        system_prompt="s",
        model="local-x",
        model_chain=["flash"],
    )
    start = _subevents(list(runner.stream("t")))[0][1]
    assert start["model_requested"] == "flash"
    assert start["model_resolved"] == "flash"


# --- 2 ter. table des états terminaux ------------------------------------------


class _StopClient:
    """Sous-boucle minimale qui s'arrête avec la raison donnée."""

    def __init__(self, reason):
        self.reason = reason

    def annex_slot(self, model):
        return 0  # un seul slot : rien à isoler

    def is_remote(self, model):
        return False

    def save_slot(self, *a, **k):
        return False

    def restore_slot(self, *a, **k):
        pass

    def stream_chat_tools(self, *a, **kw):
        yield ("content", "x")
        yield ("done", {"reason": self.reason})


def test_table_des_etats_terminaux():
    """natural -> completed ; chaque arrêt de garde-fou/API -> failed.
    (cancelled : réservé ST-03, jamais émis par cette table.)"""
    attendu = {
        "natural": "completed",
        "": "completed",  # stop vide hérité : pas un signal d'échec
        "api_error": "failed",
        "empty_response": "failed",
        "repeat_stop": "failed",
        "loop_degenerate": "failed",
        "max_iters": "failed",
        "context_irreducible": "failed",
        "output_overflow": "failed",
        "worker_timeout": "failed",
        "api_call_budget": "failed",
        "prompt_budget": "failed",
        "context_budget": "failed",
        "result_cycle": "failed",
        "tool_error_budget": "failed",
    }
    for reason, status in attendu.items():
        runner = SubAgentRunner(
            _StopClient(reason), lambda: FakeRegistry(), system_prompt="s", model="m"
        )
        end = _subevents(list(runner.stream("t")))[-1][1]
        assert end["status"] == status, f"stop={reason!r} -> {end['status']}"
        assert end["stop_reason"] == reason


class _EventClient(_StopClient):
    def __init__(self, events):
        super().__init__("")
        self.events = events

    def stream_chat_tools(self, *a, **kw):
        yield from self.events


def _budget_end(events, **limits):
    options = {
        "max_duration_s": None,
        "max_api_calls": None,
        "max_prompt_tokens": None,
    }
    options.update(limits)
    runner = SubAgentRunner(
        _EventClient(events),
        lambda: FakeRegistry(),
        system_prompt="s",
        model="m",
        **options,
    )
    emitted = list(runner.stream("t"))
    return emitted, _subevents(emitted)[-1][1]


def test_budget_tokens_cumules_coupe_l_ouvrier():
    events, end = _budget_end(
        [
            ("usage", {"prompt_tokens": 60, "completion_tokens": 1}),
            ("usage", {"prompt_tokens": 60, "completion_tokens": 1}),
            ("content", "ne doit pas être atteint"),
        ],
        max_prompt_tokens=100,
    )
    assert end["status"] == "failed" and end["stop_reason"] == "prompt_budget"
    assert end["api_calls"] == 2 and end["prompt_tokens"] == 120
    assert any(k == "done" and p["reason"] == "prompt_budget" for k, p in events)


def test_budgets_appels_api_et_contexte_coupent_l_ouvrier():
    _, api_end = _budget_end(
        [
            ("usage", {"prompt_tokens": 10, "completion_tokens": 1}),
            ("usage", {"prompt_tokens": 10, "completion_tokens": 1}),
        ],
        max_api_calls=2,
    )
    assert api_end["stop_reason"] == "api_call_budget"

    _, context_end = _budget_end(
        [("usage", {"prompt_tokens": 101, "completion_tokens": 1})],
        max_context_tokens=100,
    )
    assert context_end["stop_reason"] == "context_budget"


def test_budget_de_duree_coupe_avant_un_nouveau_tour():
    _, end = _budget_end([("content", "ne doit pas être atteint")], max_duration_s=0)
    assert end["status"] == "failed" and end["stop_reason"] == "worker_timeout"


def test_cycle_de_resultats_identiques_ignore_les_arguments():
    result = {
        "name": "search_text",
        "ok": True,
        "preview": "mêmes lignes",
        "out_full": "src/a.ts:10: valeur identique",
    }
    events, end = _budget_end(
        [("tool_result", {**result, "in_full": f"pattern-{i}"}) for i in range(3)]
        + [("content", "ne doit pas être atteint")],
        result_repeat_limit=3,
    )
    assert end["status"] == "failed" and end["stop_reason"] == "result_cycle"
    assert any(k == "done" and p["reason"] == "result_cycle" for k, p in events)


def test_erreurs_consecutives_du_meme_outil_coupent_l_ouvrier():
    error = {
        "name": "run_shell",
        "ok": False,
        "preview": "commande en échec",
        "out_full": "commande en échec",
    }
    _, end = _budget_end(
        [("tool_result", dict(error)) for _ in range(3)],
        max_consecutive_tool_errors=3,
    )
    assert end["status"] == "failed"
    assert end["stop_reason"] == "tool_error_budget"


# --- 3. deux ouvriers concurrents : aucune fuite entre agent_id ----------------


def test_deux_ouvriers_concurrents_ne_se_melangent_pas(tmp_path):
    (tmp_path / "wf.py").write_text(
        "meta = {'name': 'x'}\n"
        "return parallel([lambda: agent('tache A'), lambda: agent('tache B')])\n",
        encoding="utf-8",
    )
    client = ChainClient({"m1": "ok"}, remote_ids={"m1"})
    runner = SubAgentRunner(
        client,
        lambda: FakeRegistry(),
        system_prompt="s",
        model="local-x",
        model_chain=["m1"],
    )
    reg = ToolRegistry([make_run_workflow(runner, str(tmp_path), is_remote=True)])
    events = list(reg.run_stream("run_workflow", {"path": "wf.py"}))
    subs = _subevents(events)
    by_id: dict = {}
    for k, p in subs:
        by_id.setdefault(p["agent_id"], []).append(k)
    assert len(by_id) == 2  # un id par ouvrier
    for kinds in by_id.values():
        # Chaque ouvrier a une chronologie COMPLÈTE et ordonnée sous SON id.
        assert kinds[0] == "subagent_start" and kinds[-1] == "subagent_end"
    parents = {p["parent"] for _, p in subs}
    assert parents == {"workflow"}
    labels = {p["label"] for k, p in subs if k == "subagent_start"}
    assert labels == {"tache A", "tache B"}


# --- 4. persistance : corrélables après rechargement de session ----------------


def test_rechargement_de_session_conserve_la_correlation(tmp_path):
    from loom.agent.session import SessionStore

    store = SessionStore(tmp_path / "sessions", default_system_prompt="p")
    sess = store.create(workspace=".")
    events, _ = _parent_stream(_healthy_scripts())
    for kind, payload in _subevents(events):
        store.append_event(sess.id, kind, payload)

    replayed = store.read_timeline(sess.id)
    subs = [e for e in replayed if e["event"].startswith("subagent_")]
    assert [e["event"] for e in subs] == [k for k, _ in _subevents(events)]
    ids = {e["data"]["agent_id"] for e in subs}
    assert len(ids) == 1  # corrélation intacte après relecture disque
    end = subs[-1]["data"]
    assert end["status"] == "completed" and "duration_s" in end
