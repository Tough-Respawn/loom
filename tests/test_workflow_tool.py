# tests/test_workflow_tool.py
"""Sortie structurée (submit_result) et outil run_workflow, sur un vrai LoomClient
branché sur un FakeOAI : le chemin complet script -> runner -> sous-boucle -> outil.
"""

from __future__ import annotations

import json

from loom.tools.agent import SubAgentRunner
from loom.tools.base import ToolRegistry
from loom.tools.workflow import make_run_workflow

from .fakes import make_client, turn_text, turn_tools


def _runner(scripts):
    client, fake = make_client(scripts)
    # Le cache KV du slot local est hors sujet ici et ferait un VRAI aller-retour HTTP
    # vers un port mort (~2 s par test). Le save/restore a ses propres tests.
    client.save_slot = lambda *a, **k: False
    client.restore_slot = lambda *a, **k: None
    runner = SubAgentRunner(
        client,
        lambda: ToolRegistry([]),  # registre réel : submit_result s'y ajoute
        system_prompt="s",
        model=None,
    )
    return runner, fake


def _submit(payload: str):
    """Script d'un sous-agent qui appelle submit_result puis conclut."""
    return [
        turn_tools([("c1", "submit_result", payload)]),
        turn_text("fini"),
    ]


# --- sortie structurée --------------------------------------------------------


def test_agent_avec_schema_rend_un_dict():
    runner, _ = _runner(_submit('{"bugs": 2, "file": "a.py"}'))
    sink: list = []
    schema = {
        "type": "object",
        "properties": {"bugs": {"type": "integer"}, "file": {"type": "string"}},
        "required": ["bugs"],
    }
    list(runner.stream("audite a.py", schema=schema, sink=sink))
    assert sink[-1] == {"bugs": 2, "file": "a.py"}


def test_submit_result_herite_de_la_coercition():
    """L'argument de vente du choix « outil de sortie » plutôt que response_format :
    validate_and_coerce s'applique GRATUITEMENT, donc les fautes de type d'un petit
    modèle ('2' au lieu de 2) sont rattrapées comme pour n'importe quel outil."""
    runner, _ = _runner(_submit('{"bugs": "2"}'))
    sink: list = []
    schema = {
        "type": "object",
        "properties": {"bugs": {"type": "integer"}},
        "required": ["bugs"],
    }
    list(runner.stream("t", schema=schema, sink=sink))
    assert sink[-1] == {"bugs": 2}


def test_submit_result_autorise_doffice_par_la_politique():
    """RÉGRESSION E2E (2026-07-16, constatée deux fois) : submit_result n'était dans
    aucune catégorie de permissions -> branche « outil inconnu -> ask », même en mode
    'allow'. Or un sous-agent tourne SANS UI, donc tout 'ask' y est refusé par défaut :
    le résultat n'était jamais enregistré, agent(schema=…) rendait None pour TOUS les
    agents, et le modèle en concluait que la sortie structurée ne marchait pas.
    Rendre son résultat n'est pas une action à autoriser."""
    from loom.permissions import PermissionConfig, evaluate

    for mode in ("allow", "ask", "allowlist"):
        d = evaluate("submit_result", {"bug": "x"}, PermissionConfig(mode=mode))
        assert d.action == "allow", f"mode {mode} -> {d.action}"


def test_agent_avec_schema_survit_a_une_politique_ask():
    """Le vrai chemin du harnais : une politique est branchée sur la sous-boucle."""
    from loom.permissions import PermissionConfig, evaluate

    runner, _ = _runner(_submit('{"bugs": 3}'))
    runner.permission = lambda name, args: evaluate(
        name, args, PermissionConfig(mode="ask")
    )
    sink: list = []
    schema = {
        "type": "object",
        "properties": {"bugs": {"type": "integer"}},
        "required": ["bugs"],
    }
    list(runner.stream("t", schema=schema, sink=sink))
    assert sink == [{"bugs": 3}]  # pas de refus silencieux


def test_submit_result_absent_du_registre_sans_schema():
    """Sans schéma demandé, l'outil de sortie n'existe pas : pas de schéma parasite
    dans le prompt d'un sous-agent qui n'en a pas besoin."""
    runner, fake = _runner([turn_text("synthèse libre")])
    out = "".join(p for k, p in runner.stream("t") if k == "content")
    assert out == "synthèse libre"
    names = [t["function"]["name"] for t in (fake.calls[0].get("tools") or [])]
    assert "submit_result" not in names


def test_schema_expose_tel_quel_au_modele():
    runner, fake = _runner(_submit('{"bugs": 1}'))
    schema = {
        "type": "object",
        "properties": {"bugs": {"type": "integer"}},
        "required": ["bugs"],
    }
    list(runner.stream("t", schema=schema, sink=[]))
    tools = {t["function"]["name"]: t["function"] for t in fake.calls[0]["tools"]}
    assert tools["submit_result"]["parameters"] == schema


# --- outil run_workflow -------------------------------------------------------


def _tool(tmp_path, scripts):
    """Renvoie (registre, fake). On passe par un VRAI ToolRegistry : c'est lui qui
    convertit ToolError en 'erreur: …' — tester spec.run() en direct court-circuiterait
    la frontière d'entrée et ne dirait rien du comportement réel."""
    runner, fake = _runner(scripts)
    reg = ToolRegistry([make_run_workflow(runner, str(tmp_path))])
    return reg, fake


def _run_tool(reg, args):
    return reg.run("run_workflow", args)


def test_run_workflow_execute_le_script_et_rend_son_retour(tmp_path):
    (tmp_path / "wf.py").write_text(
        "meta = {'name': 'audit', 'description': 'd'}\n"
        "r = agent('audite a.py')\n"
        "return {'verdict': r}\n",
        encoding="utf-8",
    )
    reg, _ = _tool(tmp_path, [turn_text("aucun bug")])
    out = _run_tool(reg, {"path": "wf.py"})
    assert json.loads(out) == {"verdict": "aucun bug"}


def test_run_workflow_ne_fuit_pas_les_syntheses_dans_le_resultat(tmp_path):
    """LA raison d'être : les synthèses des ouvriers restent dans les variables du
    script. Ici l'agent bavarde, mais le script ne renvoie qu'un compte — c'est tout
    ce que le modèle appelant doit voir."""
    (tmp_path / "wf.py").write_text(
        "meta = {'name': 'x'}\n"
        "rapports = [agent('a'), agent('b')]\n"
        "return f'{len(rapports)} fichiers audités'\n",
        encoding="utf-8",
    )
    reg, _ = _tool(
        tmp_path,
        [turn_text("PAVÉ INTERMINABLE " * 50), turn_text("AUTRE PAVÉ " * 50)],
    )
    out = _run_tool(reg, {"path": "wf.py"})
    assert out == "2 fichiers audités"
    assert "PAVÉ" not in out


def test_run_workflow_relaie_la_progression(tmp_path):
    (tmp_path / "wf.py").write_text(
        "meta = {'name': 'x'}\nphase('Scan')\nlog('go')\nagent('t')\nreturn 'ok'\n",
        encoding="utf-8",
    )
    reg, _ = _tool(tmp_path, [turn_text("r")])
    events = list(reg.run_stream("run_workflow", {"path": "wf.py"}))
    calls = [p.get("name", "") for k, p in events if k == "tool_call"]
    assert any("workflow: x" in c for c in calls)
    assert any("phase — Scan" in c for c in calls)
    assert any("agent 1" in c for c in calls)
    assert events[-1] == ("content", "ok")


def test_run_workflow_args_transmis(tmp_path):
    (tmp_path / "wf.py").write_text(
        "meta = {'name': 'x'}\nreturn args['n'] + 1\n", encoding="utf-8"
    )
    reg, _ = _tool(tmp_path, [])
    assert _run_tool(reg, {"path": "wf.py", "args": {"n": 41}}) == "42"


def test_script_introuvable_erreur_actionnable(tmp_path):
    reg, _ = _tool(tmp_path, [])
    out = _run_tool(reg, {"path": "absent.py"})
    assert out.startswith("erreur")
    assert "write_file" in out


def test_meta_manquant_refuse_avant_execution(tmp_path):
    (tmp_path / "wf.py").write_text(
        "import pathlib\npathlib.Path('effet_de_bord').write_text('x')\n",
        encoding="utf-8",
    )
    reg, _ = _tool(tmp_path, [])
    out = _run_tool(reg, {"path": "wf.py"})
    assert out.startswith("erreur")
    assert not (tmp_path / "effet_de_bord").exists()  # rien n'a tourné


def test_erreur_du_script_remonte_comme_erreur_d_outil(tmp_path):
    (tmp_path / "wf.py").write_text(
        "meta = {'name': 'x'}\nreturn 1 / 0\n", encoding="utf-8"
    )
    reg, _ = _tool(tmp_path, [])
    out = _run_tool(reg, {"path": "wf.py"})
    assert out.startswith("erreur")
    assert "ZeroDivisionError" in out
