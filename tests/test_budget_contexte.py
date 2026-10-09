# Budget de contexte RÉEL (session c81fcc4bd207, 2026-10-09) : la compaction préventive
# estimait caractères/3 du prompt et des messages, sans les schémas d'outils ni le
# raisonnement conservé après un appel d'outil -> jamais déclenchée alors que le
# serveur comptait 31k tokens sur 32k. Référence désormais : le `usage.prompt_tokens`
# de l'appel précédent, plus l'estimation des SEULS messages modifiés depuis.
from __future__ import annotations

import httpx
from openai import BadRequestError

from loom.agent.compaction import (
    _CLEARED_TOOL,
    _anchor_from_usage,
    _convo_chars,
    _ctx_estimate,
    _message_chars,
)

from .fakes import (
    NS,
    FakeRegistry,
    _FakeStream,
    chunk,
    collect,
    make_client,
    only,
    turn_text,
    turn_tools,
    usage_chunk,
)

SYSTEM = "s" * 300


def _assistant_with_reasoning(n_reasoning: int) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "reasoning_content": "r" * n_reasoning,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"x"}'},
            }
        ],
    }


def test_message_chars_compte_le_raisonnement_conserve():
    sans = {"role": "assistant", "content": "ok"}
    avec = {**sans, "reasoning_content": "x" * 900}
    assert _message_chars(avec) == _message_chars(sans) + 900


def test_estimation_sans_ancre_reste_caracteres_sur_3():
    convo = [{"role": "user", "content": "u" * 600}]
    assert _ctx_estimate(SYSTEM, convo) == (300 + 600) // 3


def test_estimation_ancree_part_du_compteur_serveur_plus_le_delta():
    convo = [{"role": "user", "content": "u" * 600}]
    # Requête précédente : 900 caractères envoyés, 20 000 tokens comptés par le serveur
    # (les schémas d'outils et le gabarit pèsent, invisibles côté caractères).
    ancre = {"prompt_tokens": 20_000, "chars": _convo_chars(SYSTEM, convo)}
    assert _ctx_estimate(SYSTEM, convo, ancre) == 20_000
    convo.append(_assistant_with_reasoning(3_000))
    convo.append({"role": "tool", "tool_call_id": "c1", "content": "t" * 6_000})
    delta = _convo_chars(SYSTEM, convo) - ancre["chars"]
    assert delta > 9_000  # raisonnement + résultat + appel
    assert _ctx_estimate(SYSTEM, convo, ancre) == 20_000 + delta // 3


def test_estimation_ancree_redescend_apres_compaction():
    convo = [{"role": "user", "content": "u" * 600}]
    ancre = {"prompt_tokens": 20_000, "chars": _convo_chars(SYSTEM, convo)}
    convo.append(_assistant_with_reasoning(0))
    convo.append({"role": "tool", "tool_call_id": "c1", "content": "t" * 6_000})
    avant = _ctx_estimate(SYSTEM, convo, ancre)
    convo[-1] = {**convo[-1], "content": _CLEARED_TOOL}
    apres = _ctx_estimate(SYSTEM, convo, ancre)
    assert apres < avant
    assert apres == 20_000 + (_convo_chars(SYSTEM, convo) - ancre["chars"]) // 3


def test_force_fit_reduit_le_raisonnement_plutot_que_de_supprimer_des_messages():
    from loom.agent.compaction import _force_fit

    convo = [
        {"role": "user", "content": "tâche"},
        _assistant_with_reasoning(20_000),
        {"role": "tool", "tool_call_id": "c1", "content": "petit résultat"},
        {"role": "user", "content": "[LOOM] continue"},
    ]
    assert _force_fit(convo, SYSTEM, budget_chars=3_000) is True
    assert len(convo) == 4  # aucun message supprimé : le volume était le raisonnement
    assert len(convo[1]["reasoning_content"]) < 20_000
    assert "tronqué" in convo[1]["reasoning_content"]


def test_ancre_ignore_un_usage_estime_ou_sans_compteur():
    assert _anchor_from_usage({"prompt_tokens": 500, "estimated": True}, 900) is None
    assert _anchor_from_usage({"completion_tokens": 5}, 900) is None
    assert _anchor_from_usage({"prompt_tokens": 0}, 900) is None
    assert _anchor_from_usage({"prompt_tokens": 500}, 900) == {
        "prompt_tokens": 500,
        "chars": 900,
    }


# ---- Boucle d'outils : l'ancre vient de l'usage réel de l'appel précédent -------------

USER = [{"role": "user", "content": "fais le travail"}]
PETIT_SYSTEM = "tu es un agent de test"


def _compactions(events):
    return [
        p
        for p in only(events, "tool_result")
        if str(p.get("name", "")).startswith("(compaction")
    ]


def test_compaction_preventive_se_declenche_sur_le_compteur_serveur():
    # Caractères/3 de tout le fil ≈ 2k tokens ; le serveur en a compté 20 000 pour la
    # requête précédente (outils, gabarit). Le résultat de 6 000 car. doit faire passer
    # le seuil de 20 500 et vider ce résultat avant l'appel suivant.
    gros = "x" * 6_000
    reg = FakeRegistry({"read_file": lambda a: gros})
    premier = turn_tools([("c1", "read_file", '{"path": "big"}')], with_usage=False)
    premier.append(usage_chunk(prompt=20_000))
    client, fake = make_client([premier, turn_text("fini.")])
    events, done = collect(
        client.stream_chat_tools(
            list(USER),
            PETIT_SYSTEM,
            registry=reg,
            compact_after_tokens=20_500,
            keep_recent_tools=0,
        )
    )
    assert done["reason"] == "natural"
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"] == _CLEARED_TOOL


def test_raisonnement_conserve_apres_outil_est_compte_et_reduit():
    # Le raisonnement reste dans le message assistant porteur de l'appel d'outil
    # (rejoué par le gabarit) : 9 000 car. ≈ 3 000 tokens de plus que le compteur.
    reg = FakeRegistry({"read_file": lambda a: "ok"})
    premier = [
        chunk(reasoning="r" * 9_000),
        chunk(tool_calls=[(0, "c1", "read_file", '{"path":"x"}')]),
        chunk(finish="tool_calls"),
        usage_chunk(prompt=20_000),
    ]
    client, fake = make_client([premier, turn_text("fini.")])
    events, done = collect(
        client.stream_chat_tools(
            list(USER), PETIT_SYSTEM, registry=reg, compact_after_tokens=20_500
        )
    )
    assert done["reason"] == "natural"
    assert _compactions(events), (
        "le dépassement vient du raisonnement : il doit compacter"
    )
    assistant = next(m for m in fake.calls[1]["messages"] if m.get("tool_calls"))
    assert len(assistant["reasoning_content"]) < 9_000
    assert "tronqué" in assistant["reasoning_content"]


def test_raisonnement_abandonne_apres_length_ne_compte_pas():
    # Même volume de raisonnement, mais coupé par `length` : la continuation ne garde
    # que le texte visible, donc rien de ce volume n'entre dans la prochaine requête.
    # Plafond de sortie atteint (completion = max_tokens) : il reste de la place.
    premier = [
        chunk(reasoning="r" * 9_000),
        chunk(content="début", finish="length"),
        usage_chunk(prompt=20_000, completion=2_048),
    ]
    client, fake = make_client([premier, turn_text(" et fin.")])
    events, done = collect(
        client.stream_chat_tools(
            list(USER), PETIT_SYSTEM, max_tokens=2_048, compact_after_tokens=20_500
        )
    )
    assert done["reason"] == "natural"
    assert not _compactions(events)
    coupe = next(
        m
        for m in fake.calls[1]["messages"]
        if m["role"] == "assistant" and m.get("content") == "début"
    )
    assert not coupe.get("reasoning_content")  # le raisonnement coupé n'est pas renvoyé


class _ServeurFenetre:
    """Faux serveur qui compte comme llama-server : `overhead` tokens sans caractères
    (schémas d'outils, gabarit) + caractères des messages / 3 ; 400 « exceeds the
    available context size » au-dessus de `window`, sinon le script suivant avec
    l'usage réel. Compte les 400 renvoyés."""

    def __init__(self, scripts, overhead: int, window: int = 32_768):
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.erreurs = 0
        self.overhead = overhead
        self.window = window
        self.chat = NS(completions=NS(create=self._create))

    def _tokens(self, messages) -> int:
        chars = 0
        for m in messages:
            chars += len(str(m.get("content") or ""))
            chars += len(m.get("reasoning_content") or "")
            for tc in m.get("tool_calls") or []:
                chars += len(tc["function"]["arguments"])
        return self.overhead + chars // 3

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        n = self._tokens(kwargs["messages"])
        if n > self.window:
            self.erreurs += 1
            request = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
            raise BadRequestError(
                f"Error code: 400 - request ({n} tokens) exceeds the available "
                f"context size ({self.window} tokens)",
                response=httpx.Response(400, request=request),
                body={"error": {"code": 400}},
            )
        return _FakeStream(list(self.scripts.pop(0)) + [usage_chunk(prompt=n)])


def test_recuperation_apres_400_repetes_sans_seuil_configure():
    # Requête précédente déjà à 32 010 tokens (outils lourds, peu de caractères) ; un
    # raisonnement conservé de 2 700 car. fait déborder la suivante. Le microcompact ne
    # peut rien (résultat d'outil minuscule), le résumé non plus (fil trop court) : le
    # force-fit doit réduire DÈS SA PREMIÈRE passe. Converti en tokens × 3 direct, son
    # budget (~69k car.) dépassait le fil (~2,8k car.) : huit passes sans réduction,
    # puis un faux « contexte irréductible ».
    from loom.agent.client import LoomClient

    reg = FakeRegistry({"read_file": lambda a: "ok"})
    premier = [
        chunk(reasoning="r" * 2_700),
        chunk(tool_calls=[(0, "c1", "read_file", '{"path":"x"}')]),
        chunk(finish="tool_calls"),
    ]
    serveur = _ServeurFenetre([premier, turn_text("fini.", with_usage=False)], 32_000)
    client = LoomClient("http://127.0.0.1:9/v1")
    client._client = serveur
    events, done = collect(
        client.stream_chat_tools(list(USER), PETIT_SYSTEM, registry=reg)
    )
    assert done["reason"] == "natural"
    forcees = [
        p for p in only(events, "tool_result") if p["name"] == "(compaction forcée)"
    ]
    assert len(forcees) == 1, "chaque passe de force-fit doit réduire réellement"
    assert serveur.erreurs == 3  # 2 microcompacts à vide, 1 résumé à vide, puis fit
    assistant = next(m for m in serveur.calls[-1]["messages"] if m.get("tool_calls"))
    assert "tronqué" in assistant["reasoning_content"]  # c'est lui qui a été réduit


def test_estimation_apres_un_400_part_du_compteur_serveur():
    # Sur un 400 « context exceeds », la jauge publiée doit refléter le compteur
    # serveur de l'appel précédent (30 000), pas caractères/3 du fil (~1 000).
    reg = FakeRegistry({"read_file": lambda a: "x" * 3_000})
    premier = turn_tools([("c1", "read_file", '{"path": "big"}')], with_usage=False)
    premier.append(usage_chunk(prompt=30_000))
    request = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
    response = httpx.Response(400, request=request)
    erreur = BadRequestError(
        "Error code: 400 - request (32786 tokens) exceeds the available context "
        "size (32768 tokens)",
        response=response,
        body={"error": {"code": 400}},
    )
    client, _fake = make_client([premier, erreur, turn_text("fini.")])
    events, done = collect(
        client.stream_chat_tools(list(USER), PETIT_SYSTEM, registry=reg)
    )
    assert done["reason"] == "natural"
    jauges = only(events, "context_estimate")
    assert jauges and jauges[0]["tokens"] >= 30_000
