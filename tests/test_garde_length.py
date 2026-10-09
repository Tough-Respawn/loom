# Garde `length` (session c81fcc4bd207, 2026-10-09) : dix appels consécutifs finissaient
# exactement à 32 768 tokens, le serveur renvoyait `length`, Loom relançait « CONTINUE »
# sans rien libérer, et le modèle re-réfléchissait dans la place restante jusqu'au 400.
# Trois cas désormais distincts : plafond de sortie atteint avec de la place (continuer),
# fenêtre pleine (compacter AVANT de relancer), relances sans progrès (arrêter).
from __future__ import annotations

from loom.agent.compaction import _CLEARED_TOOL
from loom.agent.guards import _length_cause

from .fakes import (
    FakeRegistry,
    chunk,
    collect,
    kinds,
    make_client,
    only,
    turn_text,
    turn_tools,
    usage_chunk,
)

USER = [{"role": "user", "content": "fais le travail"}]
SYSTEM = "tu es un agent de test"


def _compactions(events):
    return [
        p
        for p in only(events, "tool_result")
        if str(p.get("name", "")).startswith("(compaction")
    ]


def _coupe(text=None, reasoning=None, prompt=1_000, completion=8_192):
    """Un appel coupé par `length` avec l'usage réel correspondant."""
    return [
        chunk(content=text, reasoning=reasoning, finish="length"),
        usage_chunk(prompt=prompt, completion=completion),
    ]


# ---- relances sans progrès --------------------------------------------------------------


def test_deux_reflexions_coupees_de_suite_arretent_le_tour():
    # Réflexion coupée, aucun texte ni outil, deux fois : relancer encore ne produirait
    # qu'une troisième réflexion coupée (vécu : sept de suite). Arrêt explicite.
    client, fake = make_client(
        [_coupe(reasoning="r" * 50), _coupe(reasoning="r" * 50), turn_text("jamais")]
    )
    events, done = collect(
        client.stream_chat_tools(list(USER), SYSTEM, max_tokens=8_192)
    )
    assert done["reason"] == "length_no_progress"
    assert len(fake.calls) == 2
    relances = [p for p in only(events, "harness") if p["kind"] == "continuation"]
    assert len(relances) == 1
    assert "coupée" in "".join(only(events, "content"))


def test_un_fragment_de_texte_remet_le_compteur_a_zero():
    client, fake = make_client(
        [
            _coupe(reasoning="r" * 50),
            _coupe(text="début"),
            _coupe(reasoning="r" * 50),
            turn_text(" et fin."),
        ]
    )
    events, done = collect(
        client.stream_chat_tools(list(USER), SYSTEM, max_tokens=8_192)
    )
    assert done["reason"] == "natural"
    assert len(fake.calls) == 4


# ---- fenêtre pleine vs plafond de sortie -----------------------------------------------


def test_fenetre_pleine_compacte_avant_de_relancer():
    # Génération arrêtée à 1 200 tokens alors que le plafond est 8 192 : le slot était
    # plein (23 100 + 1 200). Libérer de la place AVANT la relance, sinon le modèle
    # re-réfléchit dans le reliquat et se fait recouper.
    gros = "x" * 9_000
    reg = FakeRegistry({"read_file": lambda a: gros})
    premier = turn_tools([("c1", "read_file", '{"path": "big"}')], with_usage=False)
    premier.append(usage_chunk(prompt=20_000))
    client, fake = make_client(
        [
            premier,
            _coupe(text="début", prompt=23_100, completion=1_200),
            turn_text(" et fin."),
        ]
    )
    events, done = collect(
        client.stream_chat_tools(
            list(USER), SYSTEM, registry=reg, max_tokens=8_192, keep_recent_tools=0
        )
    )
    assert done["reason"] == "natural"
    assert len(_compactions(events)) == 1
    ks = kinds(events)
    i_compaction = next(
        i
        for i, (k, p) in enumerate(events)
        if k == "tool_result" and str(p.get("name", "")).startswith("(compaction")
    )
    i_relance = next(
        i
        for i, (k, p) in enumerate(events)
        if k == "harness" and p["kind"] == "continuation"
    )
    assert i_compaction < i_relance, ks
    tool_msg = next(m for m in fake.calls[2]["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == _CLEARED_TOOL
    assert "".join(only(events, "content")) == "début et fin."


def test_plafond_de_sortie_atteint_relance_sans_compacter():
    client, fake = make_client([_coupe(text="début"), turn_text(" et fin.")])
    events, done = collect(
        client.stream_chat_tools(list(USER), SYSTEM, max_tokens=8_192)
    )
    assert done["reason"] == "natural"
    assert not _compactions(events)
    assert "".join(only(events, "content")) == "début et fin."


def test_cause_plafond_de_sortie_quand_la_generation_atteint_max_tokens():
    usage = {"prompt_tokens": 1_000, "completion_tokens": 8_192}
    assert _length_cause("length", usage, 8_192) == "output_cap"
    # Un token d'écart reste le plafond (arrondi du serveur), pas la fenêtre.
    assert _length_cause("length", {**usage, "completion_tokens": 8_191}, 8_192) == (
        "output_cap"
    )


def test_cause_fenetre_pleine_quand_la_generation_s_arrete_avant_le_plafond():
    usage = {"prompt_tokens": 31_400, "completion_tokens": 1_368}
    assert _length_cause("length", usage, 8_192) == "context_full"


def test_cause_inconnue_sans_plafond_ou_sans_usage_reel():
    usage = {"prompt_tokens": 1_000, "completion_tokens": 500}
    assert _length_cause("length", usage, None) == "unknown"
    assert _length_cause("length", {**usage, "estimated": True}, 8_192) == "unknown"
    assert _length_cause("length", None, 8_192) == "unknown"


def test_pas_de_cause_hors_length():
    assert _length_cause("stop", {"completion_tokens": 5}, 8_192) is None
    assert _length_cause(None, None, 8_192) is None
