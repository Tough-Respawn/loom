"""Trace des appels modèle : sections du system prompt, verdict de préfixe, issue."""

from types import SimpleNamespace as NS

import pytest

from loom.agent import calltrace
from loom.agent.calltrace import (
    call_purpose,
    prefix_diff,
    request_elements,
    system_sections,
    traced_create,
)


@pytest.fixture(autouse=True)
def _events(monkeypatch):
    """Capture les log_event du module au lieu de les écrire."""
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        calltrace, "log_event", lambda ev, level="DEBUG", **f: seen.append((ev, f))
    )
    calltrace._last_elements.clear()
    return seen


def _kwargs(system: str, *messages, slot=0, tools=None):
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": r, "content": c} for r, c in messages]
    return {
        "model": "m",
        "messages": msgs,
        "stream": True,
        "tools": tools,
        "extra_body": {"id_slot": slot},
    }


def test_sections_du_system_prompt():
    sections = system_sections("intro\n# Mémoire durable\nnote\n# Dossier\nC:/x\n")
    assert [t for t, _, _ in sections] == ["(début)", "Mémoire durable", "Dossier"]


def test_verdicts_de_prefixe():
    a = [("outils", "1"), ("system « A »", "2")]
    assert prefix_diff(None, a) == ("premier", None)
    assert prefix_diff(a, list(a)) == ("identique", None)
    assert prefix_diff(a, a + [("message 1 (user)", "3")]) == ("ajout", None)
    assert prefix_diff(a, [("outils", "1"), ("system « A »", "X")]) == ("diverge", 1)
    assert prefix_diff(a, a[:1]) == ("diverge", 1)


def test_divergence_localisee_dans_la_section_modifiee(_events):
    base = "# Mémoire durable\nnote 1\n# Dossier de travail courant\nC:/a\n"
    k1 = _kwargs(base, ("user", "salut"))
    k2 = _kwargs(base.replace("C:/a", "C:/b"), ("user", "salut"), ("user", "suite"))
    calltrace.CallTrace("turn", k1)
    calltrace.CallTrace("turn", k2)
    diffs = [f for ev, f in _events if ev == "prefix.diff"]
    assert len(diffs) == 1
    assert diffs[0]["element"] == "system « Dossier de travail courant »"


def test_prefixe_suivi_par_slot(_events):
    # le cache est propre à un slot : un appel annexe sur le slot 1 ne compte pas
    calltrace.CallTrace("turn", _kwargs("S", ("user", "a"), slot=0))
    calltrace.CallTrace("reflect", _kwargs("autre", ("user", "b"), slot=1))
    calltrace.CallTrace("turn", _kwargs("S", ("user", "a"), ("user", "c"), slot=0))
    requests = [f for ev, f in _events if ev == "call.request"]
    assert [r["prefixe"] for r in requests] == ["premier", "premier", "ajout"]


def test_distant_un_titre_ne_compte_pas_contre_la_conversation(_events):
    # vécu 2026-09-30 (GLM-5.3) : un appel de titre sans outils entre deux tours
    # faisait signaler une divergence « outils » qui n'existait pas
    def remote(system, *msgs, tools=None):
        k = _kwargs(system, *msgs, tools=tools)
        k["extra_body"] = {}  # distant : pas de slot
        return k

    tools = [{"type": "function", "function": {"name": "read"}}]
    calltrace.CallTrace("turn", remote("S", ("user", "a"), tools=tools))
    calltrace.CallTrace("title", remote("T", ("user", "a")))
    calltrace.CallTrace("turn", remote("S", ("user", "a"), ("user", "b"), tools=tools))
    requests = [f for ev, f in _events if ev == "call.request"]
    assert [r["prefixe"] for r in requests] == ["premier", "premier", "ajout"]
    assert not [f for ev, f in _events if ev == "prefix.diff"]


def test_ordre_de_rendu_outils_puis_system():
    labels = [e for e, _ in request_elements(_kwargs("# A\nx\n", ("user", "u")))]
    assert labels[0] == "outils" and labels[1].startswith("system")
    assert labels[-1] == "message 1 (user)"


def _client(chunks):
    return NS(chat=NS(completions=NS(create=lambda **kw: iter(chunks))))


def test_flux_termine_journalise_les_timings(_events):
    tim = {"cache_n": 8863, "prompt_n": 18, "prompt_ms": 1700.0, "predicted_n": 5}
    chunks = [NS(timings=None, usage=None), NS(timings=tim, usage=None)]
    with call_purpose("reflect"):
        list(traced_create(_client(chunks), "annex", **_kwargs("S", ("user", "a"))))
    end = [f for ev, f in _events if ev == "call.end"][0]
    assert end["purpose"] == "reflect"  # le contexte prime sur le défaut
    assert end["issue"] == "ok" and end["cache_tok"] == 8863
    assert end["prefill_tok"] == 18


def test_flux_ferme_avant_la_fin_est_annule(_events):
    chunks = [NS(timings=None, usage=None)] * 3
    stream = traced_create(_client(chunks), "prime", stream=True)
    next(iter(stream))
    stream.close()
    ends = [f for ev, f in _events if ev == "call.end"]
    assert len(ends) == 1 and ends[0]["issue"] == "annule"


def test_erreur_a_l_ouverture(_events):
    def boom(**kw):
        raise RuntimeError("serveur mort")

    with pytest.raises(RuntimeError):
        traced_create(NS(chat=NS(completions=NS(create=boom))), "turn", stream=True)
    ends = [f for ev, f in _events if ev == "call.end"]
    assert ends[0]["issue"] == "erreur" and "serveur mort" in ends[0]["erreur"]
