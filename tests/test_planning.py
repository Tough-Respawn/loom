# tests/test_planning.py
import json

from loom.parallel import FileSpec
from loom.planning import (
    UserStory,
    critique_design,
    decompose_into_stories,
    write_plan_artifacts,
)


class FakeClient:
    """complete() renvoie une réponse scriptée selon une sous-chaîne du prompt."""

    def __init__(self, by_marker):
        self.by_marker = by_marker
        self.calls = []

    def complete(self, messages, system_prompt, **kw):
        p = messages[0]["content"]
        self.calls.append(p)
        for marker, reply in self.by_marker.items():
            if marker in p:
                return reply
        return ""


def test_critique_design_appends_refinements():
    # L'auto-critique enrichit le design avec les manques/risques, sans le perdre.
    client = FakeClient(
        {"CRITIQUE": "- manque: gestion du cas vide\n- risque: pas de reset"}
    )
    refined = critique_design(client, "DESIGN initial", "fais une todo", model="m")
    assert "DESIGN initial" in refined
    assert "gestion du cas vide" in refined


def test_critique_design_robust_when_empty():
    # Critique vide -> on garde le design initial (jamais de régression).
    client = FakeClient({"CRITIQUE": "   "})
    refined = critique_design(client, "DESIGN", "t", model="m")
    assert refined == "DESIGN"


def test_decompose_into_stories_parses_json():
    payload = json.dumps(
        {
            "stories": [
                {
                    "id": "US-01",
                    "title": "Saisie",
                    "detail": "permettre de taper un chiffre",
                    "acceptance": ["taper 3 affiche 3"],
                    "files": ["app.js", "index.html"],
                }
            ]
        }
    )
    client = FakeClient({"DÉCOUPE": payload})
    specs = [FileSpec("app.js", "logique"), FileSpec("index.html", "structure")]
    stories = decompose_into_stories(client, "DESIGN", specs, "calculatrice", model="m")
    assert len(stories) == 1
    assert stories[0].id == "US-01"
    assert stories[0].acceptance == ["taper 3 affiche 3"]
    assert "app.js" in stories[0].files


def test_decompose_robust_on_invalid_json_falls_back_to_one_story_per_file():
    # Si le JSON est cassé, on ne plante pas : une US par fichier (dégradé mais utile).
    client = FakeClient({"DÉCOUPE": "pas du json"})
    specs = [FileSpec("a.js", "r1"), FileSpec("b.css", "r2")]
    stories = decompose_into_stories(client, "D", specs, "t", model="m")
    assert len(stories) == 2
    assert {s.files[0] for s in stories} == {"a.js", "b.css"}


def test_write_plan_artifacts_externalizes_md(tmp_path):
    stories = [
        UserStory("US-01", "Saisie", "taper un chiffre", ["3 -> 3"], ["app.js"]),
        UserStory("US-02", "Calcul", "additionner", ["1+1 -> 2"], ["app.js"]),
    ]
    write_plan_artifacts(str(tmp_path), "LE DESIGN", stories)
    plan_md = (tmp_path / ".loom" / "PLAN.md").read_text(encoding="utf-8")
    assert "LE DESIGN" in plan_md
    us1 = (tmp_path / ".loom" / "us" / "US-01.md").read_text(encoding="utf-8")
    assert "Saisie" in us1 and "3 -> 3" in us1
    # les deux US sont écrites
    assert (tmp_path / ".loom" / "us" / "US-02.md").exists()


def test_story_roundtrip_dict():
    s = UserStory("US-09", "t", "d", ["a", "b"], ["f.js"])
    assert UserStory.from_dict(s.to_dict()) == s
