# Contrats des gardes comportementales et de la compaction préventive.
from __future__ import annotations

from .fakes import FakeRegistry, collect, make_client, only, turn_text, turn_tools

USER = [{"role": "user", "content": "fais le travail"}]
SYSTEM = "tu es un agent de test"


def run(client, registry=None, **kw):
    return collect(client.stream_chat_tools(USER, SYSTEM, registry=registry, **kw))


def _reg():
    return FakeRegistry({"read_file": lambda a: "contenu"})




def test_act_nudge_intention_relance():
    # Une intention sans action déclenche un nouvel appel silencieux.
    client, fake = make_client(
        [
            turn_text("Je vais maintenant lire le fichier de configuration."),
            turn_text("Terminé."),
        ]
    )
    events, done = run(client, registry=_reg())
    assert done["reason"] == "natural"
    assert len(fake.calls) == 2
    # Le second appel conserve l'intention avant le nudge.
    msgs = fake.calls[1]["messages"]
    idx = next(
        i
        for i, m in enumerate(msgs)
        if m["role"] == "assistant" and "lire le fichier" in str(m.get("content"))
    )
    assert msgs[idx + 1]["role"] == "user"


def test_act_nudge_borne_max():
    # Deux nudges au maximum précèdent l'arrêt naturel.
    intention = turn_text("Je vais maintenant lire le fichier.")
    client, fake = make_client([intention, list(intention), list(intention)])
    events, done = run(client, registry=_reg())
    assert done["reason"] == "natural"
    assert len(fake.calls) == 3


def test_claim_execution_sans_executer_relance():
    # Une réussite revendiquée sans exécution doit déclencher un audit.
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
    # Une intention purement conversationnelle ne requiert aucun outil.
    client, fake = make_client([turn_text("Je vais résumer la situation pour toi.")])
    events, done = run(client, registry=_reg())
    assert done["reason"] == "natural"
    assert len(fake.calls) == 1



_ABS = "C:/tmp/loom_inexistant_xyz.py"  # chemin absolu qui n'existe pas


def test_artefact_revendique_accompli_detecte():
    # Un chemin absolu revendiqué mais absent est une vraie confabulation.
    from loom.agent.client import _claims_missing_artifact

    assert _claims_missing_artifact(f"Voilà, j'ai créé {_ABS}.", set()) == _ABS
    assert _claims_missing_artifact(f"Le fichier {_ABS} a été généré.", set()) == _ABS


def test_mention_illustrative_ne_declenche_pas():
    # Une simple mention en exemple ne constitue pas un accomplissement revendiqué.
    from loom.agent.client import _claims_missing_artifact

    assert (
        _claims_missing_artifact(f"Tu pourrais créer {_ABS}, par exemple.", set())
        is None
    )
    assert _claims_missing_artifact(f"Voici comment créer {_ABS} :", set()) is None
    assert _claims_missing_artifact(f"Il faudrait écrire {_ABS}.", set()) is None


def test_chemin_dans_bloc_de_code_ignore():
    # Ignorer les chemins présents uniquement dans un bloc de code illustratif.
    from loom.agent.client import _claims_missing_artifact

    txt = f"Exemple d'agent :\n```python\n# j'ai cree {_ABS}\nopen('x')\n```\nÀ toi de jouer."
    assert _claims_missing_artifact(txt, set()) is None


def test_artefact_ecrit_ce_tour_non_flague():
    # Une écriture réellement effectuée justifie la revendication.
    from loom.agent.client import _claims_missing_artifact

    assert _claims_missing_artifact(f"j'ai créé {_ABS}", {_ABS}) is None


def test_strong_desactive_act_nudge():
    client, fake = make_client([turn_text("Je vais maintenant lire le fichier.")])
    events, done = run(client, registry=_reg(), strong=True)
    assert done["reason"] == "natural"
    assert len(fake.calls) == 1




def test_compaction_preventive_force_fit_et_refocus():
    # Un dépassement avant appel doit compacter et injecter une seule note de recentrage.
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
    assert any(
        m["role"] == "user" and "TRONQUÉS" in str(m.get("content"))
        for m in fake.calls[0]["messages"]
    )


def test_microcompact_vide_les_vieux_resultats_outils():
    # Un gros résultat ancien doit être vidé quand aucun outil récent n'est protégé.
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
    tool_msg = next(m for m in fake.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"] != gros
    assert len(str(tool_msg["content"])) < 600
