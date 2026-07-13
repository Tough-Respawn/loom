# Caractérisation de /skill/create — identité skill = slug du dossier (fix audit :
# name du frontmatter FORCÉ au slug, y compris quand le corps arrive avec le sien).
from __future__ import annotations


def _create(web, name, body="", description="d"):
    return web.post(
        "/skill/create", data={"name": name, "description": description, "body": body}
    )


def test_nom_slugifie(web, tmp_env):
    r = _create(web, "Mon Skill Test")
    assert r.status_code == 200
    assert r.get_json()["name"] == "mon-skill-test"
    assert (tmp_env / "skills_user" / "mon-skill-test" / "SKILL.md").exists()


def test_nom_trop_court_400(web):
    assert _create(web, "ab").status_code == 400


def test_collision_409(web):
    assert _create(web, "skill-double").status_code == 200
    assert _create(web, "skill-double").status_code == 409


def test_frontmatter_divergent_force_au_slug(web, tmp_env):
    body = "---\nname: autre-nom\ndescription: desc fournie\n---\n\nCorps du skill."
    r = _create(web, "skill-fm", body=body)
    assert r.status_code == 200
    text = (tmp_env / "skills_user" / "skill-fm" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "name: skill-fm" in text
    assert "autre-nom" not in text
    assert "desc fournie" in text
    assert "Corps du skill." in text


def test_sans_frontmatter_enveloppe(web, tmp_env):
    r = _create(web, "skill-simple", body="Juste un corps.", description="ma desc")
    assert r.status_code == 200
    text = (tmp_env / "skills_user" / "skill-simple" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert text.startswith("---\nname: skill-simple\n")
    assert "ma desc" in text
    assert "Juste un corps." in text
