"""Juge des évals par lecture des logits (evals/run_eval.py::judge_logits) : mapping
Decision -> verdict {pass, score, reason}, sans serveur (décideur factice)."""

from __future__ import annotations

from types import SimpleNamespace

from evals.run_eval import judge_logits
from loom.agent.decide import Decision


class _FakeDecider:
    def __init__(self, values, probs, coverage):
        self.d = Decision(
            values=values, probs=probs, coverage=coverage, requests=3, ms=42.0
        )
        self.calls = []

    def decide(self, instructions, schema, context):
        self.calls.append((instructions, [f.name for f in schema], context))
        return self.d


def _traj(final="Fait : fichier écrit.", tools=(("write_file", {}),)):
    return SimpleNamespace(final_text=final, tool_calls=list(tools))


def _case():
    return SimpleNamespace(
        prompt="Écris hello.txt", rubric="Le fichier hello.txt existe."
    )


def test_judge_logits_maps_decision_to_verdict():
    dec = _FakeDecider(
        values={"pass": True, "score": 4},
        probs={"pass": {"true": 0.93, "false": 0.07}, "score": {"4": 0.6}},
        coverage={"pass": 0.99, "score": 0.97},
    )
    v = judge_logits(None, "m", _case(), _traj(), decider=dec)
    assert v["pass"] is True and v["score"] == 4
    assert "0.93" in v["reason"]
    # Le juge voit la tâche, le critère, les outils et la réponse finale.
    instr, names, ctx = dec.calls[0]
    assert names == ["pass", "score"]
    assert "hello.txt" in ctx and "write_file" in ctx


def test_judge_logits_fail_verdict_and_low_coverage_flagged():
    dec = _FakeDecider(
        values={"pass": False, "score": 1},
        probs={"pass": {"true": 0.2, "false": 0.8}, "score": {"1": 0.5}},
        coverage={"pass": 0.3, "score": 0.9},
    )
    v = judge_logits(None, "m", _case(), _traj(final=""), decider=dec)
    assert v["pass"] is False and v["score"] == 1
    assert "couverture" in v["reason"]  # signal faible signalé, jamais masqué


def test_judge_logits_never_raises_when_server_is_down():
    class _Boom:
        def decide(self, *a):
            raise ConnectionError("down")

    v = judge_logits(None, "m", _case(), _traj(), decider=_Boom())
    assert v["pass"] is None and v["score"] is None and "indisponible" in v["reason"]
