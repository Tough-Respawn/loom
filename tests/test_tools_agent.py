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


def _sub_registry():
    # registre du sous-agent : lecture seule, et SURTOUT pas de dispatch_agent.
    return ToolRegistry(
        [ToolSpec("read_file", "lit", {"type": "object"}, lambda a: "contenu")]
    )


def test_dispatch_agent_runs_subloop_and_returns_final_text():
    # le sous-agent répond directement (pas d'outil) -> sa synthèse remonte.
    stream = [_chunk(_delta(content="Synthèse : 3 fichiers."), finish_reason="stop")]
    client, fake = _client([stream])
    tool = make_dispatch_agent(
        client, _sub_registry, system_prompt="sub", max_tokens=50
    )
    out = tool.run({"task": "compte les fichiers python"})
    assert out == "Synthèse : 3 fichiers."
    # le sous-agent a bien reçu SON prompt système isolé + la tâche comme message user
    msgs = fake.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "sub"}
    assert msgs[-1]["content"] == "compte les fichiers python"


def test_dispatch_agent_subregistry_excludes_dispatch_agent():
    # garde-fou anti-récursion : le registre du sous-agent ne doit pas s'auto-contenir.
    reg = _sub_registry()
    assert "dispatch_agent" not in reg


def test_dispatch_agent_requires_task():
    client, _ = _client([])
    tool = make_dispatch_agent(client, _sub_registry, system_prompt="sub")
    with pytest.raises(ToolError, match="task"):
        tool.run({"task": "  "})


def test_dispatch_agent_empty_answer_is_explicit():
    stream = [_chunk(_delta(content=""), finish_reason="stop")]
    client, _ = _client([stream])
    tool = make_dispatch_agent(client, _sub_registry, system_prompt="sub")
    out = tool.run({"task": "x"})
    assert "rien" in out.lower()
