# tests/test_dispatch_agent_path.py
"""ST-01 : le VRAI chemin parent -> dispatch_agent -> sous-boucle -> résultat.

Aucun mock de SubAgentRunner : le parent tourne dans client.stream_chat_tools
(FakeOAI scripté), l'outil dispatch_agent est le vrai make_dispatch_agent, la
sous-boucle est le vrai runner sur le MÊME client, et le registre du sous-agent
est un vrai ToolRegistry. Le script FakeOAI est séquentiel : tour parent,
tours du sous-agent, tour final du parent.

Porte go/no-go ST-01 : les tests `_check_dispatch` prouvent que le grader du cas
d'éval distingue une délégation SAINE d'une délégation CASSÉE par injection
contrôlée (sous-agent muet, sous-agent qui hallucine).
"""

from __future__ import annotations

from evals.cases import _check_dispatch
from evals.run_eval import case_passed
from loom.tools.agent import SubAgentRunner, make_dispatch_agent
from loom.tools.base import ToolRegistry, ToolSpec

from .fakes import make_client, turn_text, turn_tools

_SYNTHESE = (
    "Inventaire : src/ contient alpha_load, alpha_save (alpha.py) et beta_run "
    "(beta.py) ; lib/ contient gamma_parse et delta_merge."
)


def _sub_registry() -> ToolRegistry:
    """Registre RÉEL du sous-agent : un outil de lecture minimal."""
    return ToolRegistry(
        [
            ToolSpec(
                name="list_dir",
                description="liste les fonctions du projet de test",
                parameters={"type": "object", "properties": {}},
                run=lambda a: (
                    "src/alpha.py: alpha_load, alpha_save | src/beta.py: beta_run | "
                    "lib/gamma.py: gamma_parse | lib/delta.py: delta_merge"
                ),
            )
        ]
    )


def _parent_stream(scripts, record=False):
    """Monte le chemin complet et déroule le tour parent. Renvoie
    (events, fake) — plus la liste `sub_agents` capturée si record=True
    (instrumentation du harnais d'éval, _record_subagents)."""
    client, fake = make_client(scripts)
    # Le slot KV local ferait un vrai aller-retour HTTP vers un port mort.
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
    sub_agents: list = []
    if record:
        from evals.run_eval import _record_subagents

        stream_fn, restore = _record_subagents(client, "modele-eval", sub_agents)
    else:
        stream_fn, restore = client.stream_chat_tools, lambda: None
    try:
        events = list(
            stream_fn(
                [{"role": "user", "content": "Fais l'inventaire des fonctions."}],
                "parent-sys",
                1024,
                model=None,
                registry=parent_reg,
                thinking=False,
                max_iters=6,
            )
        )
    finally:
        restore()
    if record:
        return events, fake, sub_agents
    return events, fake


def _healthy_scripts():
    return [
        turn_tools([("p1", "dispatch_agent", '{"task": "inventorie les fonctions"}')]),
        turn_tools([("s1", "list_dir", "{}")]),  # le sous-agent explore
        turn_text(_SYNTHESE),  # sa synthèse
        turn_text(f"Voici l'inventaire. {_SYNTHESE}"),  # tour final du parent
    ]


def _traj(events):
    """Reconstruit une Trajectory d'éval depuis les events du parent, comme
    run_eval.run_one : MÊME mapping, donc mêmes graders applicables."""
    from evals.run_eval import Trajectory

    traj = Trajectory()
    for kind, payload in events:
        if kind == "content":
            traj.final_text += payload
        elif kind == "tool_result":
            traj.tool_calls.append((payload.get("name"), {}))
            traj.tool_results.append(
                {
                    "name": payload.get("name"),
                    "ok": payload.get("ok"),
                    "preview": str(payload.get("preview", ""))[:300],
                }
            )
        elif kind == "done":
            traj.stop_reason = payload.get("reason") or ""
    return traj


# --- ST-01a : chemin complet, délégation saine ---------------------------------


def test_chemin_complet_parent_dispatch_sousboucle_resultat(tmp_path):
    events, fake = _parent_stream(_healthy_scripts())

    # 4 appels API : parent, sous-agent x2, parent final — dans cet ordre.
    assert len(fake.calls) == 4
    parent_tools = [t["function"]["name"] for t in fake.calls[0]["tools"]]
    sub_tools = [t["function"]["name"] for t in fake.calls[1]["tools"]]
    assert parent_tools == ["dispatch_agent"]
    assert "list_dir" in sub_tools and "dispatch_agent" not in sub_tools

    # Le sous-agent reçoit SON prompt système, pas celui du parent.
    sub_msgs = fake.calls[1]["messages"]
    assert any(m.get("content") == "sub-sys" for m in sub_msgs if m["role"] == "system")

    # L'activité du sous-agent est relayée live (tool_stream) et sa synthèse
    # devient le résultat de l'outil dispatch_agent côté parent.
    streams = [p["text"] for k, p in events if k == "tool_stream"]
    assert any("list_dir" in s for s in streams)
    results = [p for k, p in events if k == "tool_result"]
    assert len(results) == 1 and results[0]["ok"]
    assert "alpha_load" in results[0]["out_full"]

    # Fin de tour naturelle, réponse finale du parent présente.
    done = [p for k, p in events if k == "done"][-1]
    assert done.get("reason") == "natural"
    final = "".join(p for k, p in events if k == "content")
    assert "gamma_parse" in final


def test_usage_du_sous_agent_remonte_au_parent(tmp_path):
    events, _ = _parent_stream(_healthy_scripts())
    # La conso du sous-agent est relayée (sub_usage) : les totaux de session
    # restent exacts — c'est la capture « tokens » exigée par ST-01.
    assert any(k == "sub_usage" for k, _ in events)


# --- ST-01b : porte go/no-go — le grader distingue sain / cassé ----------------


def test_grader_accepte_une_delegation_saine(tmp_path):
    events, _ = _parent_stream(_healthy_scripts())
    checks = _check_dispatch(_traj(events), tmp_path)
    assert case_passed(checks), checks
    assert checks["a délégué (dispatch_agent)"] is True


def test_grader_refuse_une_exploration_directe_sans_delegation(tmp_path):
    """Porte durcie : un run qui rend le BON inventaire sans déléguer n'exerce
    pas le chemin mesuré (prompt sous-agent, routage) -> échec BLOQUANT."""
    scripts = [turn_text(f"Inventaire fait moi-même. {_SYNTHESE}")]
    events, _ = _parent_stream(scripts)
    checks = _check_dispatch(_traj(events), tmp_path)
    assert checks["synthèse correcte (>=3 fonctions nommées)"] is True
    assert checks["a délégué (dispatch_agent)"] is False
    assert not case_passed(checks), checks


def test_grader_refuse_un_sous_agent_muet(tmp_path):
    """Injection contrôlée n°1 : le sous-agent ne rend RIEN (prompt sous-agent
    cassé, routage mort…). Le parent ne peut pas nommer les fonctions -> le cas
    doit ÉCHOUER, pas rester vert."""
    # Constaté en écrivant ce test : la sous-boucle RELANCE un EOS vide
    # (garde empty_response, max_empty_retries=2) — le harnais se défend déjà
    # contre un sous-agent muet. L'injection doit rester muette sur les TROIS
    # tours (initial + 2 relances) pour éprouver le grader derrière le garde.
    scripts = [
        turn_tools([("p1", "dispatch_agent", '{"task": "inventorie"}')]),
        turn_text(""),  # sous-agent muet…
        turn_text(""),  # …au premier retry du garde-fou…
        turn_text(""),  # …et au second : la sous-boucle rend 'empty_response'
        turn_text("L'inventaire a échoué, le sous-agent n'a rien rendu."),
    ]
    events, _ = _parent_stream(scripts)
    results = [p for k, p in events if k == "tool_result"]
    # Le résultat rendu au parent est le constat du garde-fou, pas une synthèse.
    assert "réponse vide" in results[0]["preview"]
    checks = _check_dispatch(_traj(events), tmp_path)
    assert not case_passed(checks), checks


# --- ST-01 (porte durcie) : capture de l'activité RÉELLE du sous-agent --------

_SYNTHESE_LONGUE = (
    _SYNTHESE
    + " Détail complet : chaque fichier a été ouvert et analysé ligne par ligne, "
    "aucune classe n'est définie, et la convention de nommage <dossier>_<verbe> "
    "est respectée partout dans le projet."
)


def _long_scripts():
    return [
        turn_tools([("p1", "dispatch_agent", '{"task": "inventorie"}')]),
        turn_tools([("s1", "list_dir", "{}")]),
        turn_text(_SYNTHESE_LONGUE),
        turn_text(f"Inventaire terminé, via la délégation. {_SYNTHESE}"),
    ]


def test_recorder_capture_l_activite_reelle_du_sous_agent():
    _, _, subs = _parent_stream(_long_scripts(), record=True)
    assert len(subs) == 1  # une entrée par tier appelé — jamais le parent
    s = subs[0]
    assert s["model"] == "modele-eval"  # tier local None -> modèle de campagne
    assert s["model_turns"] == 2
    assert s["tools"] == ["list_dir"]
    assert s["prompt_tokens"] > 0 and s["completion_tokens"] > 0
    assert s["stop_reason"] == "natural"
    assert s["synthesis"] == _SYNTHESE_LONGUE  # ENTIÈRE, pas un aperçu
    assert len(s["synthesis"]) > 160


def test_report_record_porte_les_sous_agents():
    import json as _json

    from evals.run_eval import _run_record

    events, _, subs = _parent_stream(_long_scripts(), record=True)
    traj = _traj(events)
    traj.sub_agents = subs
    rec = _run_record(traj, {"ok": True}, None, "modele-eval")
    assert rec["sub_agents"][0]["synthesis"] == _SYNTHESE_LONGUE
    assert rec["sub_agents"][0]["stop_reason"] == "natural"
    _json.dumps(rec, ensure_ascii=False)  # sérialisable tel quel dans report.json


def test_transcript_porte_la_section_sous_agents(tmp_path):
    from evals.run_eval import _save_transcript

    events, _, subs = _parent_stream(_long_scripts(), record=True)
    traj = _traj(events)
    traj.sub_agents = subs
    _save_transcript(
        tmp_path, "new", "dispatch_probe", 0, traj, {"x": True}, None, model="m-eval"
    )
    txt = (tmp_path / "new" / "dispatch_probe_run1.md").read_text(encoding="utf-8")
    assert "## Sous-agents" in txt
    assert "modèle=modele-eval" in txt
    assert _SYNTHESE_LONGUE in txt  # la synthèse ENTIÈRE, pas 160 caractères
    assert "stop=natural" in txt


def test_grader_refuse_une_synthese_hallucinee(tmp_path):
    """Injection contrôlée n°2 : le sous-agent répond à côté (fonctions
    inventées). Le grader exige les VRAIS noms -> échec attendu."""
    scripts = [
        turn_tools([("p1", "dispatch_agent", '{"task": "inventorie"}')]),
        turn_text("Fonctions trouvées : foo_init, bar_main, baz_run."),
        turn_text("Inventaire : foo_init, bar_main et baz_run."),
    ]
    events, _ = _parent_stream(scripts)
    checks = _check_dispatch(_traj(events), tmp_path)
    assert not case_passed(checks), checks
