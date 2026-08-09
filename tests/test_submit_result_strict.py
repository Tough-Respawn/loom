# tests/test_submit_result_strict.py
"""Validation STRICTE de submit_result (sous-ensemble JSON Schema nommé) et contrat
premier-appel-valide-gagne. Promus depuis les tests de diagnostic du 2026-08-09.

Deux niveaux :
- outil seul via ToolRegistry (la vraie frontière d'entrée : coercition PUIS
  validation stricte) — vérifie les messages d'erreur (chemins exploitables) ;
- boucle complète via SubAgentRunner + FakeOAI — vérifie que l'erreur retourne
  au sous-agent et que la correction aboutit (aucune boucle de retry dédiée).
"""

from __future__ import annotations

from loom.tools.agent import SubAgentRunner, _schema_faults, make_submit_result
from loom.tools.base import ToolRegistry

from .fakes import make_client, turn_text, turn_tools

SCHEMA = {
    "type": "object",
    "properties": {
        "bugs": {"type": "integer"},
        "level": {"type": "string", "enum": ["low", "high"]},
        "files": {"type": "array", "items": {"type": "string"}},
        "detail": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    },
    "required": ["bugs"],
}


def _reg(sink: list) -> ToolRegistry:
    return ToolRegistry([make_submit_result(SCHEMA, sink)])


def _runner(scripts):
    client, fake = make_client(scripts)
    client.save_slot = lambda *a, **k: False
    client.restore_slot = lambda *a, **k: None
    runner = SubAgentRunner(
        client, lambda: ToolRegistry([]), system_prompt="s", model=None
    )
    return runner, fake


# --- validation stricte, outil seul (messages et chemins) ---------------------


def test_entier_non_coercible_refuse_avec_chemin():
    sink: list = []
    out = _reg(sink).run("submit_result", {"bugs": "abc"})
    assert out.startswith("erreur") and "bugs" in out and "integer" in out
    assert sink == []  # rien enregistré


def test_entier_coercible_passe_toujours():
    # La coercition de premier niveau reste AVANT la validation : "2" -> 2.
    sink: list = []
    out = _reg(sink).run("submit_result", {"bugs": "2"})
    assert out.startswith("ok")
    assert sink == [{"bugs": 2}]


def test_enum_invalide_refuse():
    sink: list = []
    out = _reg(sink).run("submit_result", {"bugs": 1, "level": "banana"})
    assert out.startswith("erreur") and "level" in out and "enum" in out
    assert sink == []


def test_objet_imbrique_invalide_refuse_avec_chemin():
    sink: list = []
    out = _reg(sink).run("submit_result", {"bugs": 1, "detail": {"count": "beaucoup"}})
    assert out.startswith("erreur") and "detail.count" in out
    assert sink == []


def test_element_invalide_dans_tableau_refuse_avec_index():
    sink: list = []
    out = _reg(sink).run("submit_result", {"bugs": 1, "files": ["a.py", 2]})
    assert out.startswith("erreur") and "files[1]" in out
    assert sink == []


def test_champ_requis_imbrique_absent_refuse():
    sink: list = []
    out = _reg(sink).run("submit_result", {"bugs": 1, "detail": {}})
    assert out.startswith("erreur") and "detail.count" in out and "requis" in out
    assert sink == []


def test_champ_inconnu_permis_sans_additional_properties():
    # JSON Schema : sans additionalProperties=false, les champs inconnus passent.
    sink: list = []
    out = _reg(sink).run("submit_result", {"bugs": 1, "libre": True})
    assert out.startswith("ok")
    assert sink[-1]["libre"] is True


def test_additional_properties_false_refuse_champ_inconnu():
    schema = {
        "type": "object",
        "properties": {"bugs": {"type": "integer"}},
        "required": ["bugs"],
        "additionalProperties": False,
    }
    sink: list = []
    out = ToolRegistry([make_submit_result(schema, sink)]).run(
        "submit_result", {"bugs": 1, "intrus": 1}
    )
    assert out.startswith("erreur") and "intrus" in out
    assert sink == []


def test_faults_directs_chemins_composes():
    # Le format de chemin promis : bugs[2].confidence.
    schema = {
        "type": "object",
        "properties": {
            "bugs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"confidence": {"type": "number"}},
                },
            }
        },
    }
    faults = _schema_faults(
        {"bugs": [{"confidence": 0.9}, {}, {"confidence": "high"}]}, schema, ""
    )
    assert faults == ["bugs[2].confidence : number attendu, reçu 'high' (str)"]


# --- sémantique required / null (présence de la clé, pas sa valeur) -----------


def test_requis_type_null_accepte_none_explicite():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "null"}},
        "required": ["x"],
    }
    sink: list = []
    out = ToolRegistry([make_submit_result(schema, sink)]).run(
        "submit_result", {"x": None}
    )
    assert out.startswith("ok")
    assert sink == [{"x": None}]


def test_requis_absent_toujours_refuse():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "null"}},
        "required": ["x"],
    }
    sink: list = []
    out = ToolRegistry([make_submit_result(schema, sink)]).run("submit_result", {})
    assert out.startswith("erreur") and "x" in out
    assert sink == []


def test_non_regression_amont_none_refuse_pour_type_non_null():
    """Le contrat historique des autres outils tient : un None explicite sur un
    requis NON déclaré null reste rejeté par validate_and_coerce."""
    import pytest

    from loom.tools.base import ToolError, validate_and_coerce

    shell_like = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    with pytest.raises(ToolError):
        validate_and_coerce("run_shell", shell_like, {"command": None})
    with pytest.raises(ToolError):
        validate_and_coerce("submit_result", SCHEMA, {"bugs": None})


# --- _schema_faults défensif : jamais d'exception sur schéma malformé ---------


def test_faults_defensif_required_imbrique_non_liste():
    """La forme est refusée en amont (_validate_schema) ; si un schéma malformé
    passe quand même, _schema_faults ignore le morceau au lieu de lever."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "object", "properties": {"b": {}}, "required": 3}},
    }
    assert _schema_faults({"a": {}}, schema, "") == []


def test_faults_defensif_enum_et_items_malformes():
    schema = {
        "type": "object",
        "properties": {"a": {"enum": "pas-une-liste"}, "b": {"items": 3}},
    }
    assert _schema_faults({"a": 1, "b": [1]}, schema, "") == []


# --- contrat premier-appel-valide-gagne ---------------------------------------


def test_second_appel_refuse_sans_ecraser_le_premier():
    sink: list = []
    reg = _reg(sink)
    assert reg.run("submit_result", {"bugs": 1}).startswith("ok")
    out2 = reg.run("submit_result", {"bugs": 7})
    assert out2.startswith("erreur") and "déjà enregistré" in out2
    assert sink == [{"bugs": 1}]  # le premier fait foi


# --- boucle complète : l'erreur retourne au sous-agent, qui corrige -----------


def test_correction_apres_type_invalide():
    """Tour 1 : bugs='abc' -> erreur stricte réinjectée. Tour 2 : corrigé -> capturé."""
    runner, _ = _runner(
        [
            turn_tools([("c1", "submit_result", '{"bugs": "abc"}')]),
            turn_tools([("c2", "submit_result", '{"bugs": 3}')]),
            turn_text("fini"),
        ]
    )
    sink: list = []
    list(runner.stream("t", schema=SCHEMA, sink=sink))
    assert sink == [{"bugs": 3}]


def test_correction_apres_requis_manquant():
    runner, _ = _runner(
        [
            turn_tools([("c1", "submit_result", '{"level": "low"}')]),
            turn_tools([("c2", "submit_result", '{"bugs": 1, "level": "low"}')]),
            turn_text("fini"),
        ]
    )
    sink: list = []
    list(runner.stream("t", schema=SCHEMA, sink=sink))
    assert sink == [{"bugs": 1, "level": "low"}]


def test_zero_submit_donne_sink_vide():
    # Contrat d'échec inchangé : texte libre sans submit -> sink vide -> None amont.
    runner, _ = _runner([turn_text("rapport en texte libre")])
    sink: list = []
    list(runner.stream("t", schema=SCHEMA, sink=sink))
    assert sink == []


def test_deux_submits_dans_la_boucle_premier_gagne():
    runner, _ = _runner(
        [
            turn_tools([("c1", "submit_result", '{"bugs": 1}')]),
            turn_tools([("c2", "submit_result", '{"bugs": 7}')]),
            turn_text("fini"),
        ]
    )
    sink: list = []
    list(runner.stream("t", schema=SCHEMA, sink=sink))
    assert sink == [{"bugs": 1}]
