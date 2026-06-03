# tests/test_agents.py
import pytest

from loom.agents import (
    Agent,
    AgentRun,
    RunStep,
    build_step_messages,
    compose_agent_system_prompt,
    is_blocking,
    is_reviewer,
    resolve_agents,
)


@pytest.mark.parametrize(
    "role,expected",
    [
        ("Relecteur", True),
        ("reviewer", True),
        ("rev", True),
        ("Développeur", False),
        ("Planificateur", False),
        ("code", False),
    ],
)
def test_is_reviewer(role, expected):
    assert is_reviewer(role) is expected


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Tout est bon. VERDICT: OK", False),
        ("VERDICT:OK", False),  # tolérant à l'espacement
        ("Il manque un test. VERDICT: BLOQUANT", True),
        ("verdict: bloquant", True),  # insensible à la casse
        ("Quelques remarques mais VERDICT: NON BLOQUANT", False),
        # Pas de verdict explicite => revue non concluante => on force une passe.
        ("rien à signaler", True),
        ("C'est parfait, beau travail.", True),
    ],
)
def test_is_blocking_verdict(content, expected):
    assert is_blocking(content) is expected


def _agent(id_, role, model, sys="sys", skills=None):
    return Agent(
        id=id_,
        role=role,
        model=model,
        system_prompt=sys,
        skills=list(skills or []),
    )


def test_resolve_agents_respects_pipeline_order():
    configs = [
        _agent("a", "plan", "m1"),
        _agent("b", "code", "m2"),
        _agent("c", "review", "m3"),
    ]
    resolved = resolve_agents(configs, ["c", "a", "b"])
    assert [a.id for a in resolved] == ["c", "a", "b"]
    assert [a.model for a in resolved] == ["m3", "m1", "m2"]


def test_resolve_agents_ignores_unknown_ids():
    configs = [_agent("a", "plan", "m1"), _agent("b", "code", "m2")]
    resolved = resolve_agents(configs, ["b", "zzz", "a"])
    assert [a.id for a in resolved] == ["b", "a"]


def test_build_step_messages_task_first_and_content_only():
    prior = [
        RunStep("a", "plan", "m1", reasoning="REASON_PLAN", content="CONTENT_PLAN"),
        RunStep("b", "code", "m2", reasoning="REASON_CODE", content="CONTENT_CODE"),
    ]
    msgs = build_step_messages("la tâche", prior)
    # premier message = tâche en user
    assert msgs[0] == {"role": "user", "content": "la tâche"}
    # le reasoning n'est jamais propagé
    blob = "\n".join(m["content"] for m in msgs)
    assert "REASON_PLAN" not in blob
    assert "REASON_CODE" not in blob
    # le content de chaque étape précédente est présent, préfixé par le rôle
    assert "CONTENT_PLAN" in blob
    assert "CONTENT_CODE" in blob
    assert "Étape plan" in blob
    assert "Étape code" in blob


def test_build_step_messages_empty_prior():
    msgs = build_step_messages("seule tâche", [])
    assert msgs == [{"role": "user", "content": "seule tâche"}]


def test_compose_agent_system_prompt_injects_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "dagster").mkdir(parents=True)
    (skills_dir / "dagster" / "SKILL.md").write_bytes(
        b"---\nname: dagster\ndescription: archi\n---\nARCHI_DAGSTER_XYZ"
    )
    agent = _agent("a", "code", "m1", sys="PROMPT_BASE", skills=["dagster"])
    prompt = compose_agent_system_prompt(agent, str(skills_dir))
    assert "PROMPT_BASE" in prompt
    assert "ARCHI_DAGSTER_XYZ" in prompt


def test_compose_agent_system_prompt_without_skills(tmp_path):
    agent = _agent("a", "plan", "m1", sys="JUSTE_LA_BASE")
    prompt = compose_agent_system_prompt(agent, str(tmp_path))
    assert prompt == "JUSTE_LA_BASE"


def test_agent_run_defaults():
    run = AgentRun(task="t")
    assert run.task == "t"
    assert run.steps == []


def test_agent_thinking_defaults_true():
    agent = _agent("a", "plan", "m1")
    assert agent.thinking is True


def test_agent_thinking_can_be_disabled():
    agent = Agent(id="c", role="code", model="m", system_prompt="s", thinking=False)
    assert agent.thinking is False
