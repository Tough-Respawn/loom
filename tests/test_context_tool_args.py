# Les arguments d'appels d'outils (un write_file porte tout le fichier) sont envoyés
# au modèle à chaque tour : l'estimation du contexte doit les compter, et la
# réduction de dernier recours doit pouvoir les raccourcir avant la tâche.
import json

from loom.agent.compaction import _ctx_estimate, _force_fit, _message_chars


def _write_call(i, size):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {"path": f"f{i}.py", "content": "x" * size}
                    ),
                },
            }
        ],
    }


def _convo(n_writes=10, size=14000):
    convo = [{"role": "user", "content": "écris le projet"}]
    for i in range(n_writes):
        convo.append(_write_call(i, size))
        convo.append({"role": "tool", "tool_call_id": f"c{i}", "content": "écrit"})
    convo.append({"role": "user", "content": "maintenant, teste-le"})
    return convo


def test_estimation_compte_les_arguments_d_outils():
    convo = _convo()
    assert _message_chars(convo[1]) > 14000
    assert _ctx_estimate("", convo) > 10 * 14000 // 3


def test_force_fit_raccourcit_les_arguments_avant_la_tache():
    convo = _convo()
    assert _force_fit(convo, "systeme", budget_chars=20000)
    # La tâche courante et la première consigne sont intactes.
    assert convo[-1]["content"] == "maintenant, teste-le"
    assert convo[0]["content"] == "écris le projet"
    # Les appels restent du JSON valide, avec leur chemin.
    for m in convo:
        for tc in m.get("tool_calls") or []:
            args = json.loads(tc["function"]["arguments"])
            assert args["path"].startswith("f")
    # Aucun message n'a été supprimé : l'appariement appel/résultat est conservé.
    assert len(convo) == 22
