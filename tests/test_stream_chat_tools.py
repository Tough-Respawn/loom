# Caractérisation de LoomClient.stream_chat_tools (P2-3) et des deux chemins
# d'exécution d'outils, parallèle vs séquentiel (P2-4). Ces tests figent le
# comportement OBSERVABLE actuel (kinds d'events, ordre, payloads, messages
# ajoutés à la conversation) avant refactor.
from __future__ import annotations

import json

from .fakes import (
    FakeRegistry,
    collect,
    kinds,
    make_client,
    only,
    turn_text,
    turn_tools,
)

USER = [{"role": "user", "content": "fais le travail"}]
SYSTEM = "tu es un agent de test"


def run(client, registry=None, model=None, **kw):
    return collect(
        client.stream_chat_tools(USER, SYSTEM, registry=registry, model=model, **kw)
    )


# ---------- fin de tour ----------


def test_stop_naturel():
    client, fake = make_client([turn_text("Bonjour, travail fini.")])
    events, done = run(client)
    assert done["reason"] == "natural"
    assert "".join(only(events, "content")) == "Bonjour, travail fini."
    assert len(fake.calls) == 1
    # usage réel relayé (pas estimé)
    usages = only(events, "usage")
    assert usages and "estimated" not in usages[0]


def test_kwargs_api_local():
    client, fake = make_client([turn_text("ok")])
    run(client)
    kw = fake.calls[0]
    assert kw["stream"] is True
    assert kw["messages"][0] == {"role": "system", "content": SYSTEM}
    assert kw["messages"][1:] == USER
    # chemin local : extra_body natif présent
    assert "extra_body" in kw


def test_reponse_vide_relances_puis_done():
    # max_empty_retries=2 (défaut) : 2 relances nudgées puis arrêt empty_response.
    client, fake = make_client([turn_text(""), turn_text(""), turn_text("")])
    events, done = run(client)
    assert done["reason"] == "empty_response"
    assert len(fake.calls) == 3
    # Les relances sont des interventions du HARNAIS (3e voix) -> events 'harness',
    # plus des tool_result déguisés (refonte 2026-07-23).
    vides = [p for p in only(events, "harness") if p["kind"] == "réponse vide"]
    assert len(vides) == 2 and all("vide" in p["text"].lower() for p in vides)


def test_continuation_length():
    client, fake = make_client(
        [turn_text("début de réponse", finish="length"), turn_text(" et la fin.")]
    )
    events, done = run(client)
    assert done["reason"] == "natural"
    assert len(fake.calls) == 2
    # le texte des deux tours a bien été streamé
    assert "".join(only(events, "content")) == "début de réponse et la fin."
    # au 2e appel : l'assistant text est conservé + nudge user de continuation
    msgs = fake.calls[1]["messages"]
    assert {"role": "assistant", "content": "début de réponse"} in msgs
    assert msgs[-1]["role"] == "user"


# ---------- outils, chemin SÉQUENTIEL ----------


def test_tool_call_sequentiel_nominal():
    reg = FakeRegistry({"read_file": lambda a: f"contenu de {a['path']}"})
    client, fake = make_client(
        [
            turn_tools([("call_1", "read_file", '{"path": "x.txt"}')]),
            turn_text("j'ai lu le fichier."),
        ]
    )
    events, done = run(client, registry=reg)
    assert done["reason"] == "natural"
    assert reg.calls == [("read_file", {"path": "x.txt"})]

    # ordre des events outil
    ks = kinds(events)
    assert ks.index("tool_begin") < ks.index("tool_call") < ks.index("tool_result")
    assert "parallel" not in ks  # local => jamais parallèle

    # payload tool_result séquentiel : champs figés (P2-4)
    tr = only(events, "tool_result")[0]
    assert tr["id"] == "call_1" and tr["name"] == "read_file" and tr["ok"] is True
    assert tr["preview"].startswith("contenu de x.txt")
    assert tr["path"] == "x.txt"
    assert "cmd" in tr  # présent (None) sur le chemin séquentiel
    assert tr["out_full"].startswith("contenu de x.txt")

    # conversation au 2e appel : assistant.tool_calls + message tool
    msgs = fake.calls[1]["messages"]
    asst = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    assert asst["tool_calls"][0]["function"]["name"] == "read_file"
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"].startswith("contenu de x.txt")


def test_tool_erreur_ok_false():
    reg = FakeRegistry({"read_file": lambda a: "erreur: fichier introuvable"})
    client, _ = make_client(
        [
            turn_tools([("call_1", "read_file", '{"path": "x"}')]),
            turn_text("tant pis."),
        ]
    )
    events, done = run(client, registry=reg)
    assert done["reason"] == "natural"
    tr = only(events, "tool_result")[0]
    assert tr["ok"] is False


def test_args_json_invalide_sans_executer():
    reg = FakeRegistry({"read_file": lambda a: "jamais appelé"})
    client, _ = make_client(
        [
            turn_tools([("call_1", "read_file", '{"path": "tronq')]),
            turn_text("bon."),
        ]
    )
    events, done = run(client, registry=reg)
    assert done["reason"] == "natural"
    assert reg.calls == []  # l'outil n'est PAS exécuté
    tr = only(events, "tool_result")[0]
    assert tr["ok"] is False


def test_permission_deny():
    reg = FakeRegistry({"run_shell": lambda a: "jamais appelé"})

    class Decision:
        action = "deny"
        reason = "interdit en test"

    client, fake = make_client(
        [
            turn_tools([("call_1", "run_shell", '{"command": "rm -rf /"}')]),
            turn_text("ok je n'y touche pas."),
        ]
    )
    events, done = run(client, registry=reg, permission=lambda n, a: Decision())
    assert done["reason"] == "natural"
    assert reg.calls == []
    tr = only(events, "tool_result")[0]
    assert tr["ok"] is False
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"].startswith("refusé")


def test_payload_write_file_specialise():
    # detail/in_full spécialisés pour write_file (perdus si le refactor unifie
    # naïvement la construction du payload — P2-4).
    reg = FakeRegistry({"write_file": lambda a: f"écrit {a['path']}"})
    args = {"path": "out.txt", "content": "hello monde"}
    client, _ = make_client(
        [
            turn_tools([("call_1", "write_file", json.dumps(args))]),
            turn_text("fichier écrit."),
        ]
    )
    events, done = run(client, registry=reg)
    assert done["reason"] == "natural"
    tr = only(events, "tool_result")[0]
    assert tr["ok"] is True
    assert tr["detail"] == "hello monde"
    assert tr["in_full"] == "out.txt\nhello monde"


# ---------- outils, chemin PARALLÈLE (distant + ≥2 appels parallel-safe) ----------


def _two_reads():
    return turn_tools(
        [
            ("call_1", "read_file", '{"path": "a.txt"}'),
            ("call_2", "list_dir", '{"path": "."}'),
        ]
    )


def test_parallele_distant():
    reg = FakeRegistry(
        {
            "read_file": lambda a: f"contenu {a['path']}",
            "list_dir": lambda a: "a.txt\nb.txt",
        }
    )
    client, fake = make_client(
        [_two_reads(), turn_text("fini.")],
        remote=True,
    )
    events, done = run(client, registry=reg, model="remote-x")
    assert done["reason"] == "natural"

    par = only(events, "parallel")
    assert par == [{"ids": ["call_1", "call_2"], "names": ["read_file", "list_dir"]}]
    # les deux pastilles tool_call sont émises AVANT tout tool_result
    ks = kinds(events)
    assert ks.index("tool_result") > len(ks) - 1 - ks[::-1].index("tool_call")

    # résultats dans l'ordre des tool_calls ; payload UNIFIÉ avec le séquentiel
    # (refactor P2-4 2026-07-13) : cmd présent mais None (aucun outil parallel-safe
    # n'a de `command` ; la divergence historique était accidentelle)
    trs = only(events, "tool_result")
    assert [t["id"] for t in trs] == ["call_1", "call_2"]
    assert all(t["cmd"] is None for t in trs)
    assert all(t["detail"] for t in trs)  # detail générique = résultat de l'outil
    assert trs[0]["ok"] is True and trs[0]["preview"] == "contenu a.txt"

    # conversation au 2e appel : deux messages tool, dans l'ordre
    tool_msgs = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2"]


def test_local_jamais_parallele():
    reg = FakeRegistry({"read_file": lambda a: "x", "list_dir": lambda a: "y"})
    client, _ = make_client([_two_reads(), turn_text("fini.")])
    events, done = run(client, registry=reg)
    assert done["reason"] == "natural"
    assert "parallel" not in kinds(events)
    assert [t["id"] for t in only(events, "tool_result")] == ["call_1", "call_2"]


def test_distant_outil_non_safe_reste_sequentiel():
    reg = FakeRegistry({"read_file": lambda a: "x", "write_file": lambda a: "écrit"})
    client, _ = make_client(
        [
            turn_tools(
                [
                    ("call_1", "read_file", '{"path": "a"}'),
                    ("call_2", "write_file", '{"path": "b", "content": "c"}'),
                ]
            ),
            turn_text("fini."),
        ],
        remote=True,
    )
    events, done = run(client, registry=reg, model="remote-x")
    assert done["reason"] == "natural"
    assert "parallel" not in kinds(events)


# ---------- garde-fous ----------


def test_notes_en_vol_injectees():
    pile = [["pense au README"]]
    reg = FakeRegistry({"read_file": lambda a: "contenu"})
    client, fake = make_client(
        [
            turn_tools([("call_1", "read_file", '{"path": "x"}')]),
            turn_text("fait."),
        ]
    )
    events, done = run(
        client, registry=reg, notes_provider=lambda: pile.pop(0) if pile else []
    )
    assert done["reason"] == "natural"
    notes = only(events, "note")
    assert notes == [
        "[User note received mid-turn — take it into account "
        "and continue the task] pense au README"
    ]
    # la note est bien dans la conversation envoyée au modèle (1er appel déjà)
    assert any(
        m["role"] == "user" and "pense au README" in m["content"]
        for m in fake.calls[0]["messages"]
    )


def test_harness_est_une_3e_voix_marquee_loom():
    # Le garde-fou (ici : réponse vide) parle au modèle en role:user MARQUÉ [LOOM]
    # (pas l'utilisateur) et émet un event 'harness' distinct pour l'UI (3e voix).
    client, fake = make_client([turn_text(""), turn_text("ok.")])
    events, done = run(client)
    # 1) event harness visible en UI, étiqueté
    harness = only(events, "harness")
    assert harness and harness[0]["kind"] == "réponse vide"
    # 2) dans la conversation vue par le modèle : role user, préfixé du tag MINIMAL
    #    [LOOM] (le sens est expliqué UNE fois dans le system prompt, pas répété ici).
    injected = [
        m
        for m in fake.calls[-1]["messages"]
        if m["role"] == "user" and m.get("content", "").startswith("[LOOM]")
    ]
    assert injected, "le nudge doit être marqué [LOOM] pour le modèle"


def test_system_prompt_explique_le_tag_loom():
    # Le SENS du tag [LOOM] est défini UNE fois dans les system prompts (stables,
    # en tête -> absorbés par le prefix-cache) : les modèles savent que ce n'est pas
    # l'utilisateur, sans re-consommer l'explication à chaque relance.
    from loom.prompts import CHAT_SYSTEM, CHAT_SYSTEM_STRONG

    for prompt in (CHAT_SYSTEM, CHAT_SYSTEM_STRONG):
        assert "[LOOM]" in prompt
        low = prompt.lower()
        assert "harness" in low or "harnais" in low  # le concept, en EN ou FR


def test_note_au_stop_naturel_relance():
    # Bug 2026-07-23 : une note postée PENDANT la génération finale (après le
    # drain d'avant-appel) était laissée en file jusqu'au prochain message
    # manuel (« ? » de relance). Elle doit maintenant être drainée au stop
    # naturel et déclencher une relance immédiate.
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        # note absente au 1er drain (avant 1er appel), présente au 2e (au stop)
        return ["regarde aussi le CHANGELOG"] if calls["n"] == 2 else []

    reg = FakeRegistry({})
    client, fake = make_client(
        [turn_text("premiere reponse."), turn_text("ok, vu le CHANGELOG.")]
    )
    events, done = run(client, registry=reg, notes_provider=provider)
    assert done["reason"] == "natural"
    notes = only(events, "note")
    assert notes == [
        "[User note received mid-turn — take it into account "
        "and continue the task] regarde aussi le CHANGELOG"
    ]
    # la note a provoqué un 2e appel modèle (relance), pas un « ? » de l'user
    assert len(fake.calls) == 2
    assert any(
        m["role"] == "user" and "CHANGELOG" in m["content"]
        for m in fake.calls[1]["messages"]
    )


def test_repeat_stop_meme_appel_repete():
    # repeat_limit=3 (défaut) : le même tool call non-vérif répété tour après tour
    # déclenche l'arrêt repeat_stop.
    reg = FakeRegistry({"read_file": lambda a: "toujours pareil"})
    same = lambda: turn_tools([("call_1", "read_file", '{"path": "x"}')])  # noqa: E731
    client, _ = make_client([same(), same(), same(), same(), same()])
    events, done = run(client, registry=reg)
    assert done["reason"] == "repeat_stop"


def test_boucle_degeneree_texte_repete():
    from .fakes import chunk, usage_chunk

    ligne = "cette ligne fait clairement plus de vingt-quatre caractères\n"
    tour_boucle = [chunk(content=ligne) for _ in range(12)]
    tour_boucle.append(chunk(finish="stop"))
    tour_boucle.append(usage_chunk())
    # max_loop_breaks=2 : deux nudges puis arrêt loop_degenerate au 3e tour bouclé.
    client, _ = make_client([list(tour_boucle), list(tour_boucle), list(tour_boucle)])
    events, done = run(client)
    assert done["reason"] == "loop_degenerate"
    boucles = [p for p in only(events, "harness") if p["kind"] == "boucle"]
    assert len(boucles) == 2


def test_strong_desactive_repeat_stop():
    # strong=True (modèle fort distant) : pas d'arrêt repeat_stop, la boucle continue
    # jusqu'au stop naturel.
    reg = FakeRegistry({"read_file": lambda a: "pareil"})
    same = lambda: turn_tools([("call_1", "read_file", '{"path": "x"}')])  # noqa: E731
    client, _ = make_client([same(), same(), same(), same(), turn_text("fini.")])
    events, done = run(client, registry=reg, strong=True)
    assert done["reason"] == "natural"
