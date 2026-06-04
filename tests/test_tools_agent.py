# tests/test_tools_agent.py
from types import SimpleNamespace

import pytest

from loom.client import LoomClient
from loom.tools.agent import make_dispatch_agent
from loom.tools.base import ToolError, ToolRegistry, ToolSpec


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(
        content=content, reasoning_content=None, tool_calls=tool_calls
    )


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


class FakeOpenAI:
    def __init__(self, streams):
        self._streams = list(streams)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._streams.pop(0)


def _client(streams):
    c = LoomClient(base_url="http://x/v1")
    c._client = FakeOpenAI(streams)
    return c, c._client


def _tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _read_registry():
    return ToolRegistry(
        [ToolSpec("read_file", "lit", {"type": "object"}, lambda a: "contenu")]
    )


def test_dispatch_agent_runs_subloop_and_returns_final_text():
    # le sous-agent répond directement (pas d'outil) -> sa synthèse remonte.
    stream = [_chunk(_delta(content="Synthèse : 3 fichiers."), finish_reason="stop")]
    client, fake = _client([stream])
    tool = make_dispatch_agent(
        client, _read_registry, system_prompt="sub", max_tokens=50
    )
    out = tool.run({"task": "compte les fichiers python"})
    assert out == "Synthèse : 3 fichiers."
    # le sous-agent a bien reçu SON prompt système isolé + la tâche comme message user
    msgs = fake.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "sub"}
    assert msgs[-1]["content"] == "compte les fichiers python"


def test_dispatch_agent_can_write_files():
    # un ouvrier doit pouvoir AGIR : le sous-agent écrit, l'action s'exécute.
    wrote = []

    def sub_reg():
        return ToolRegistry(
            [
                ToolSpec(
                    "write_file",
                    "w",
                    {"type": "object"},
                    lambda a: wrote.append(a) or "écrit",
                )
            ]
        )

    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(
                        0,
                        id="c1",
                        name="write_file",
                        arguments='{"path":"a.txt","content":"x"}',
                    )
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="fichier a.txt créé"), finish_reason="stop")]
    client, _ = _client([stream1, stream2])
    tool = make_dispatch_agent(client, sub_reg, system_prompt="sub")
    out = tool.run({"task": "crée a.txt"})
    assert wrote == [{"path": "a.txt", "content": "x"}]
    assert "a.txt" in out


def test_dispatch_agent_forwards_permission_deny():
    # la politique de permission du fil principal s'applique au sous-agent.
    from loom.permissions import Decision

    ran = []

    def sub_reg():
        return ToolRegistry(
            [
                ToolSpec(
                    "run_shell",
                    "s",
                    {"type": "object"},
                    lambda a: ran.append(a) or "ok",
                )
            ]
        )

    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(0, id="c1", name="run_shell", arguments='{"command":"ls"}')
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="bloqué"), finish_reason="stop")]
    client, _ = _client([stream1, stream2])
    tool = make_dispatch_agent(
        client,
        sub_reg,
        system_prompt="sub",
        permission=lambda n, a: Decision("deny", "interdit"),
    )
    tool.run({"task": "lance ls"})
    assert ran == []  # refusé par la politique -> jamais exécuté


def test_dispatch_agent_run_stream_relays_subloop_events():
    # run_stream expose l'activité de la sous-boucle EN DIRECT (pour l'UI).
    def sub_reg():
        return ToolRegistry(
            [ToolSpec("read_file", "lit", {"type": "object"}, lambda a: "contenu")]
        )

    stream1 = [
        _chunk(
            _delta(
                tool_calls=[_tc(0, id="c1", name="read_file", arguments='{"path":"x"}')]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="fini"), finish_reason="stop")]
    client, _ = _client([stream1, stream2])
    tool = make_dispatch_agent(client, sub_reg, system_prompt="sub")
    kinds = [k for k, _ in tool.run_stream({"task": "lis x"})]
    assert "tool_call" in kinds and "content" in kinds


def test_dispatch_agent_requires_task():
    client, _ = _client([])
    tool = make_dispatch_agent(client, _read_registry, system_prompt="sub")
    with pytest.raises(ToolError, match="task"):
        tool.run({"task": "  "})


def test_dispatch_agent_empty_answer_is_explicit():
    stream = [_chunk(_delta(content=""), finish_reason="stop")]
    client, _ = _client([stream])
    tool = make_dispatch_agent(client, _read_registry, system_prompt="sub")
    out = tool.run({"task": "x"})
    assert "rien" in out.lower()
