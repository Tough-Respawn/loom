# tests/test_client.py
from types import SimpleNamespace

from loom.client import _iter_events, LoomClient, build_create_kwargs


def _chunk(content=None, reasoning=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_iter_events_tags_reasoning_and_content():
    stream = [
        _chunk(reasoning="je "),
        _chunk(reasoning="pense"),
        _chunk(content="Bon"),
        _chunk(content="jour"),
        _chunk(content=None),
        _chunk(content=""),
    ]
    assert list(_iter_events(stream)) == [
        ("reasoning", "je "),
        ("reasoning", "pense"),
        ("content", "Bon"),
        ("content", "jour"),
    ]


def test_loom_client_builds_base_url():
    client = LoomClient(base_url="http://127.0.0.1:8080/v1")
    assert client.base_url == "http://127.0.0.1:8080/v1"
    assert client.model == "local"


def test_build_create_kwargs_includes_max_tokens_and_system():
    kw = build_create_kwargs(
        model="local",
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="SYS",
        max_tokens=777,
    )
    assert kw["model"] == "local"
    assert kw["max_tokens"] == 777
    assert kw["stream"] is True
    assert kw["messages"][0] == {"role": "system", "content": "SYS"}
    assert kw["messages"][1] == {"role": "user", "content": "hi"}


def test_loom_client_stores_resilience_params():
    c = LoomClient(base_url="http://x/v1", timeout=99, max_retries=4)
    assert c.timeout == 99
    assert c.max_retries == 4


def test_build_create_kwargs_uses_given_model():
    kw = build_create_kwargs(
        model="qwen",
        messages=[{"role": "user", "content": "x"}],
        system_prompt="s",
        max_tokens=100,
    )
    assert kw["model"] == "qwen"


def test_build_create_kwargs_thinking_on_by_default_no_extra_body():
    kw = build_create_kwargs(model="m", messages=[], system_prompt="s", max_tokens=10)
    assert "extra_body" not in kw


def test_build_create_kwargs_thinking_off_disables_via_template_kwarg():
    kw = build_create_kwargs(
        model="m", messages=[], system_prompt="s", max_tokens=10, thinking=False
    )
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_stream_chat_closes_underlying_stream_on_interrupt():
    """À l'interruption (gen.close()), la connexion HTTP au modèle est coupée."""
    closed = {"v": False}

    class FakeStream:
        def __iter__(self):
            yield _chunk(content="Hel")
            yield _chunk(content="lo")
            yield _chunk(content=" world")

        def close(self):
            closed["v"] = True

    client = LoomClient(base_url="http://x/v1")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: FakeStream())
        )
    )
    gen = client.stream_chat([{"role": "user", "content": "x"}], "sys")
    assert next(gen) == ("content", "Hel")  # un token reçu
    gen.close()  # l'utilisateur a soumis un nouveau message
    assert closed["v"] is True
