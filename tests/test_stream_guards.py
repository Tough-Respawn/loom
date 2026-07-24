# Caractérisation des gardes « parle sans agir » (act-nudge / claim-audit) et de la
# compaction préventive — les zones qui n'étaient PAS couvertes avant les refactors
# P2-3/P2-4. Comportements calibrés en live (cf. mémoires projet) : à ne changer
# qu'en connaissance de cause.
from __future__ import annotations

from .fakes import FakeRegistry, collect, make_client, only, turn_text, turn_tools

USER = [{"role": "user", "content": "fais le travail"}]
SYSTEM = "tu es un agent de test"


def run(client, registry=None, **kw):
    return collect(client.stream_chat_tools(USER, SYSTEM, registry=registry, **kw))


def _reg():
    return FakeRegistry({"read_file": lambda a: "contenu"})


# ---------- act-nudge : intention annoncée sans appel d'outil ----------


def test_act_nudge_intention_relance():
    # « je vais … » sans tool_call => nudge silencieux (aucun event) + nouvel appel.
    client, fake = make_client(
        [
            turn_text("Je vais maintenant lire le fichier de configuration."),
            turn_text("Terminé."),
        ]
    )
    events, done = run(client, registry=_reg())
    assert done["reason"] == "natural"
    assert len(fake.calls) == 2
    # au 2e appel : le texte d'intention est conservé + un nudge user derrière
    msgs = fake.calls[1]["messages"]
    idx = next(
        i
        for i, m in enumerate(msgs)
        if m["role"] == "assistant" and "lire le fichier" in str(m.get("content"))
    )
    assert msgs[idx + 1]["role"] == "user"


def test_act_nudge_borne_max():
    # max_act_nudges=2 (défaut) : après 2 relances, la 3e intention passe -> stop naturel.
    intention = turn_text("Je vais maintenant lire le fichier.")
    client, fake = make_client([intention, list(intention), list(intention)])
    events, done = run(client, registry=_reg())
    assert done["reason"] == "natural"
    assert len(fake.calls) == 3


def test_claim_execution_sans_executer_relance():
    # « j'ai exécuté …, le test passe » sans AUCUNE exécution ce tour => claim-audit
    # relance ; la réponse suivante neutre passe.
    client, fake = make_client(
        [
            turn_text("J'ai exécuté le script et le test passe."),
            turn_text("D'accord."),
        ]
    )
    events, done = run(client, registry=_reg())
    assert done["reason"] == "natural"
    assert len(fake.calls) == 2


def test_verbes_de_parole_exemptes():
    # « je vais résumer » = parole, pas action outillée : AUCUNE relance.
    client, fake = make_client([turn_text("Je vais résumer la situation pour toi.")])
    events, done = run(client, registry=_reg())
    assert done["reason"] == "natural"
    assert len(fake.calls) == 1


# ---------- claim-audit ARTEFACT : resserré (2026-07-23, incident loom-amd) ----------

_ABS = "C:/tmp/loom_inexistant_xyz.py"  # chemin absolu qui n'existe pas


def test_artefact_revendique_accompli_detecte():
    # Vraie confabulation : « j'ai créé <chemin absolu> » alors qu'il n'existe pas.
    from loom.agent.client import _claims_missing_artifact

    assert _claims_missing_artifact(f"Voilà, j'ai créé {_ABS}.", set()) == _ABS
    assert _claims_missing_artifact(f"Le fichier {_ABS} a été généré.", set()) == _ABS


def test_mention_illustrative_ne_declenche_pas():
    # Le déclencheur de l'incident loom-amd : une MENTION en exemple, pas une
    # revendication d'accomplissement -> ne doit PLUS mordre.
    from loom.agent.client import _claims_missing_artifact

    assert (
        _claims_missing_artifact(f"Tu pourrais créer {_ABS}, par exemple.", set())
        is None
    )
    assert _claims_missing_artifact(f"Voici comment créer {_ABS} :", set()) is None
    # infinitif / conditionnel seul (aucun accomplissement affirmé)
    assert _claims_missing_artifact(f"Il faudrait écrire {_ABS}.", set()) is None


def test_chemin_dans_bloc_de_code_ignore():
    # Un chemin qui n'apparaît QUE dans un bloc ``` (illustration) est ignoré,
    # même si un verbe de création traîne à l'intérieur.
    from loom.agent.client import _claims_missing_artifact

    txt = f"Exemple d'agent :\n```python\n# j'ai cree {_ABS}\nopen('x')\n```\nÀ toi de jouer."
    assert _claims_missing_artifact(txt, set()) is None


def test_artefact_ecrit_ce_tour_non_flague():
    # Revendiqué ET réellement écrit ce tour -> pas une confabulation.
    from loom.agent.client import _claims_missing_artifact

    assert _claims_missing_artifact(f"j'ai créé {_ABS}", {_ABS}) is None


def test_strong_desactive_act_nudge():
    # Modèle fort distant : les gardes de comportement sont coupées.
    client, fake = make_client([turn_text("Je vais maintenant lire le fichier.")])
    events, done = run(client, registry=_reg(), strong=True)
    assert done["reason"] == "natural"
    assert len(fake.calls) == 1


# ---------- compaction préventive (chemin local) ----------


def test_compaction_preventive_force_fit_et_refocus():
    # Contexte estimé > seuil AVANT l'appel (system prompt énorme, seuil minuscule) :
    # force-fit préventif => events (compaction préventive) + context_estimate, et la
    # note de recentrage est injectée UNE fois dans la conversation.
    gros_system = "règles très détaillées. " * 60  # ~1400 chars => ~470 tokens estimés
    client, fake = make_client([turn_text("ok.")])
    events, done = collect(
        client.stream_chat_tools(
            list(USER), gros_system, registry=_reg(), compact_after_tokens=100
        )
    )
    assert done["reason"] == "natural"
    compactions = [
        p for p in only(events, "tool_result") if p["name"] == "(compaction préventive)"
    ]
    assert compactions and compactions[0]["ok"] is True
    assert only(events, "context_estimate")
    # note de recentrage présente dans les messages envoyés au modèle
    assert any(
        m["role"] == "user" and "TRONQUÉS" in str(m.get("content"))
        for m in fake.calls[0]["messages"]
    )


def test_microcompact_vide_les_vieux_resultats_outils():
    # Un GROS résultat d'outil au tour 1 ; au tour 2 le contexte dépasse le seuil =>
    # microcompact vide son contenu (keep_recent_tools=0 pour ne rien protéger).
    gros = "x" * 3000
    reg = FakeRegistry({"read_file": lambda a: gros})
    client, fake = make_client(
        [
            turn_tools([("call_1", "read_file", '{"path": "big"}')]),
            turn_text("j'ai fini."),
        ]
    )
    events, done = collect(
        client.stream_chat_tools(
            list(USER),
            SYSTEM,
            registry=reg,
            compact_after_tokens=300,
            keep_recent_tools=0,
        )
    )
    assert done["reason"] == "natural"
    assert only(events, "context_estimate")
    # au 2e appel API, le message tool ne porte plus le dump de 3000 chars
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"] != gros
    assert len(str(tool_msg["content"])) < 600
