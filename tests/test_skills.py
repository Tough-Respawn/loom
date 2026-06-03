# tests/test_skills.py
from loom.skills import Skill, list_skills, load_skill, compose_system_prompt


def _write_skill(root, folder, text):
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def test_list_skills_parses_frontmatter(tmp_path):
    _write_skill(
        tmp_path,
        "dagster",
        "---\nname: dagster\ndescription: Mon archi\n---\nCorps de la connaissance.",
    )
    skills = list_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "dagster"
    assert skills[0].description == "Mon archi"
    assert "Corps de la connaissance." in skills[0].body
    assert "---" not in skills[0].body


def test_list_skills_fallback_name_without_frontmatter(tmp_path):
    _write_skill(tmp_path, "brut", "Juste du texte sans frontmatter.")
    skills = list_skills(tmp_path)
    assert skills[0].name == "brut"
    assert skills[0].description == ""
    assert skills[0].body.strip() == "Juste du texte sans frontmatter."


def test_list_skills_ignores_dir_without_skill_md(tmp_path):
    (tmp_path / "vide").mkdir()
    assert list_skills(tmp_path) == []


def test_list_skills_missing_dir_returns_empty(tmp_path):
    assert list_skills(tmp_path / "absent") == []


def test_compose_system_prompt_appends_active_bodies(tmp_path):
    base = "Tu es utile."
    s = Skill(name="dagster", description="d", body="ARCHI_XYZ")
    out = compose_system_prompt(base, [s])
    assert out.startswith(base)
    assert "ARCHI_XYZ" in out
    assert "# Skill : dagster" in out


def test_load_skill_by_name(tmp_path):
    _write_skill(tmp_path, "a", "---\nname: a\ndescription: x\n---\nAAA")
    assert load_skill(tmp_path, "a").body.strip() == "AAA"
    assert load_skill(tmp_path, "absent") is None
