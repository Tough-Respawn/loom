# tests/test_workflow_runtime.py
"""Runtime des workflows : parsing, primitives, concurrence, tolérance aux pannes.

`agent_fn` est bouchonné partout : on teste l'ORCHESTRATION, pas les modèles.
"""

from __future__ import annotations

import threading
import time

import pytest

from loom.workflow import MAX_ITEMS, WorkflowError, parse_meta, run_workflow


def _echo(prompt, *, schema=None, label=None, model=None):
    return f"[{prompt}]"


def _run(source, **kw):
    kw.setdefault("agent_fn", _echo)
    return run_workflow(source, **kw)


# --- meta ---------------------------------------------------------------------


def test_meta_extrait_sans_executer():
    meta = parse_meta("meta = {'name': 'x', 'description': 'd'}\nraise SystemExit(1)")
    assert meta == {"name": "x", "description": "d"}


def test_meta_manquant_est_une_erreur_lisible():
    with pytest.raises(WorkflowError, match="sans bloc `meta`"):
        parse_meta("x = 1")


def test_meta_calcule_refuse():
    # Un meta non littéral ne peut pas être lu sans exécuter le script — c'est le
    # point de l'exigence, pas un caprice de parseur.
    with pytest.raises(WorkflowError, match="littéral pur"):
        parse_meta("name = 'x'\nmeta = {'name': name}")


def test_syntaxe_invalide_nomme_la_ligne():
    with pytest.raises(WorkflowError, match="ligne 2"):
        parse_meta("meta = {'name': 'x'}\ndef (:\n")


# --- enveloppe AST ------------------------------------------------------------


def test_return_au_niveau_du_script():
    assert _run("meta = {'name': 'x'}\nreturn 42") == 42


def test_chaine_multiligne_intacte():
    # Régression : envelopper par RÉ-INDENTATION du texte corromprait le contenu des
    # chaînes multi-lignes — or un script de workflow est fait de prompts multi-lignes.
    src = 'meta = {"name": "x"}\np = """ligne1\nligne2\n"""\nreturn p'
    assert _run(src) == "ligne1\nligne2\n"


def test_script_sans_return_rend_none():
    assert _run("meta = {'name': 'x'}\nx = 1") is None


def test_erreur_du_script_remonte():
    with pytest.raises(ZeroDivisionError):
        _run("meta = {'name': 'x'}\nreturn 1 / 0")


# --- agent() ------------------------------------------------------------------


def test_agent_rend_la_reponse():
    assert _run("meta = {'name': 'x'}\nreturn agent('salut')") == "[salut]"


def test_agent_avec_schema_passe_le_schema():
    seen = {}

    def fn(prompt, *, schema=None, label=None, model=None):
        seen["schema"] = schema
        return {"ok": True}

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    src = "meta = {'name': 'x'}\nreturn agent('t', schema=args)"
    assert _run(src, agent_fn=fn, args=schema) == {"ok": True}
    assert seen["schema"] == schema


def test_agent_qui_leve_rend_none_sans_tuer_le_run():
    def boom(prompt, *, schema=None, label=None, model=None):
        raise RuntimeError("ouvrier mort")

    assert _run("meta = {'name': 'x'}\nreturn agent('t')", agent_fn=boom) is None


def test_schema_malforme_arrete_le_run_au_lieu_de_rendre_none():
    """Régression E2E 2026-07-16 : un schéma invalide partait à l'API, qui rejetait
    l'appel -> les N agents rendaient None ensemble. Le modèle en a conclu « la sortie
    structurée ne marche pas » et l'a abandonnée. Une faute du SCRIPT doit se voir
    comme telle, pas se déguiser en ouvriers défaillants."""
    src = "meta = {'name': 'x'}\nreturn agent('t', schema=args)"
    for bad, motif in [
        ("pas un dict", "dict JSON Schema"),
        ({"type": "array", "items": {}}, "doit être"),
        ({"type": "object"}, "properties"),
        ({"type": "object", "properties": {"a": {}}, "required": "a"}, "liste"),
        ({"type": "object", "properties": {"a": {}}, "required": ["b"]}, "absents"),
    ]:
        with pytest.raises(WorkflowError, match=motif):
            _run(src, args=bad)


def test_schema_valide_passe():
    schema = {
        "type": "object",
        "properties": {"bug": {"type": "string"}},
        "required": ["bug"],
    }
    assert _run("meta = {'name': 'x'}\nreturn agent('t', schema=args)", args=schema)


def test_agent_transmet_le_modele_epingle():
    """agent(model=...) : le script route un agent sur un modèle précis (ex.
    vérificateurs sur le fort) — transmis à l'exécuteur tel quel."""
    seen = {}

    def fn(prompt, *, schema=None, label=None, model=None):
        seen["model"] = model
        return "ok"

    src = "meta = {'name': 'x'}\nreturn agent('t', model='glm-zai')"
    assert _run(src, agent_fn=fn) == "ok"
    assert seen["model"] == "glm-zai"


def test_plafond_agents():
    from loom.workflow import runtime as rt

    original = rt.MAX_AGENTS
    rt.MAX_AGENTS = 3
    try:
        src = "meta = {'name': 'x'}\nwhile True:\n    agent('t')"
        with pytest.raises(WorkflowError, match="plafond de 3 agents"):
            _run(src)
    finally:
        rt.MAX_AGENTS = original


# --- parallel() ---------------------------------------------------------------


def test_parallel_rend_les_resultats_dans_l_ordre():
    src = (
        "meta = {'name': 'x'}\n"
        "return parallel([lambda: agent('a'), lambda: agent('b')])"
    )
    assert _run(src, is_remote=True) == ["[a]", "[b]"]


def test_parallel_thunk_qui_leve_donne_none():
    def _boom():
        raise ValueError

    src = "meta = {'name': 'x'}\nreturn parallel([lambda: 1, args])"
    assert _run(src, args=_boom, is_remote=True) == [1, None]


def test_parallel_est_concurrent_en_distant():
    seen = []

    def slow(prompt, *, schema=None, label=None, model=None):
        seen.append(threading.current_thread().name)
        time.sleep(0.15)
        return prompt

    src = (
        "meta = {'name': 'x'}\n"
        # `i=i` : capture par VALEUR. Sans ça les 4 lambdas voient le dernier i —
        # piège de fermeture tardive propre à Python, absent du JS de Claude Code
        # (`let` y est lié par itération). Documenté dans le schéma de run_workflow.
        "return parallel([lambda i=i: agent(str(i)) for i in range(4)])"
    )
    t0 = time.monotonic()
    out = _run(src, agent_fn=slow, is_remote=True)
    elapsed = time.monotonic() - t0
    assert out == ["0", "1", "2", "3"]
    # 4 agents à 0.15 s : concurrent => ~0.15 s, sérialisé => ~0.6 s.
    assert elapsed < 0.45
    assert len(set(seen)) > 1


def test_parallel_est_serialise_en_local():
    """RÈGLE CARDINALE : un slot llama-swap -> pas de concurrence en local. Même
    sémantique, perf différente — c'est la dégradation assumée, pas un bug."""
    running = []
    peak = []
    lock = threading.Lock()

    def track(prompt, *, schema=None, label=None, model=None):
        with lock:
            running.append(1)
            peak.append(len(running))
        time.sleep(0.02)
        with lock:
            running.pop()
        return prompt

    src = (
        "meta = {'name': 'x'}\n"
        # `i=i` : capture par VALEUR. Sans ça les 4 lambdas voient le dernier i —
        # piège de fermeture tardive propre à Python, absent du JS de Claude Code
        # (`let` y est lié par itération). Documenté dans le schéma de run_workflow.
        "return parallel([lambda i=i: agent(str(i)) for i in range(4)])"
    )
    _run(src, agent_fn=track, is_remote=False)
    assert max(peak) == 1


def test_parallel_borne_le_nombre_d_items():
    src = "meta = {'name': 'x'}\nreturn parallel([lambda: 1] * (args + 1))"
    with pytest.raises(WorkflowError, match="plafond"):
        _run(src, args=MAX_ITEMS, is_remote=True)


# --- pipeline() ---------------------------------------------------------------


def test_pipeline_enchaine_les_etapes():
    src = (
        "meta = {'name': 'x'}\n"
        "return pipeline([1, 2], lambda v: v * 10, lambda v: v + 1)"
    )
    assert _run(src, is_remote=True) == [11, 21]


def test_pipeline_passe_item_et_index():
    src = (
        "meta = {'name': 'x'}\n"
        "return pipeline(['a', 'b'], lambda v: v.upper(), "
        "lambda r, item, i: f'{i}{item}{r}')"
    )
    assert _run(src, is_remote=True) == ["0aA", "1bB"]


def test_pipeline_etape_qui_leve_isole_l_item():
    src = (
        "meta = {'name': 'x'}\n"
        "def s(v):\n"
        "    if v == 2:\n"
        "        raise ValueError('nope')\n"
        "    return v\n"
        "return pipeline([1, 2, 3], s, lambda v: v * 10)"
    )
    assert _run(src, is_remote=True) == [10, None, 30]


def test_pipeline_sans_etape_rend_les_items():
    assert _run("meta = {'name': 'x'}\nreturn pipeline([1, 2])") == [1, 2]


def test_pipeline_pas_de_barriere_entre_etapes():
    """Un item lent en étape 1 ne doit pas retenir un item rapide en étape 2 : c'est
    tout l'intérêt du pipeline sur un parallel() par étape."""
    order = []

    src = (
        "meta = {'name': 'x'}\n"
        "def s1(v):\n"
        "    args['sleep'](v)\n"
        "    return v\n"
        "return pipeline([0.3, 0.0], s1, args['mark'])"
    )
    _run(
        src,
        args={"sleep": time.sleep, "mark": lambda v: order.append(v) or v},
        is_remote=True,
    )
    # L'item rapide (0.0) atteint l'étape 2 AVANT que le lent (0.3) ait fini l'étape 1.
    assert order == [0.0, 0.3]


# --- events -------------------------------------------------------------------


def test_phase_et_log_emettent_des_events():
    events = []
    src = (
        "meta = {'name': 'x'}\n"
        "phase('Scan')\n"
        "log('2 fichiers')\n"
        "agent('t')\n"
        "return 'fini'"
    )
    _run(src, on_event=lambda k, p: events.append((k, p)))
    kinds = [k for k, _ in events]
    assert kinds == ["phase", "log", "agent_start", "agent_end"]
    assert events[0][1]["title"] == "Scan"
    assert events[1][1]["message"] == "2 fichiers"
    assert events[2][1]["phase"] == "Scan"  # l'agent est rattaché à la phase courante
    assert events[3][1]["ok"] is True


def test_args_expose_au_script():
    assert _run("meta = {'name': 'x'}\nreturn args['n'] * 2", args={"n": 21}) == 42
