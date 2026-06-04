# tests/test_tools_image.py
from types import SimpleNamespace

import pytest

from loom.client import LoomClient
from loom.inline_image import is_inline_image, parse_inline_image, wrap_image
from loom.tools.base import ToolError, ToolRegistry, ToolSpec
from loom.tools.read import make_read_image


def test_read_image_returns_inline_image(tmp_path):
    (tmp_path / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    out = make_read_image(str(tmp_path)).run({"path": "shot.png"})
    assert is_inline_image(out)
    caption, data_url = parse_inline_image(out)
    assert caption == "shot.png"
    assert data_url.startswith("data:image/png;base64,")


def test_read_image_rejects_non_image(tmp_path):
    (tmp_path / "a.txt").write_text("juste du texte", encoding="utf-8")
    with pytest.raises(ToolError, match="format image"):
        make_read_image(str(tmp_path)).run({"path": "a.txt"})


def test_read_image_missing_file(tmp_path):
    with pytest.raises(ToolError, match="introuvable"):
        make_read_image(str(tmp_path)).run({"path": "absent.png"})


def test_read_image_too_large(tmp_path):
    (tmp_path / "big.png").write_bytes(b"\x00" * 50)
    with pytest.raises(ToolError, match="volumineuse"):
        make_read_image(str(tmp_path), max_bytes=10).run({"path": "big.png"})


# --- injection multimodale dans la boucle tool-use ---


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(
        content=content, reasoning_content=None, tool_calls=tool_calls
    )


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


def _tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
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


def test_stream_chat_tools_injects_inline_image_as_user_message():
    data_url = "data:image/png;base64,QUJD"
    registry = ToolRegistry(
        [
            ToolSpec(
                "read_image",
                "img",
                {"type": "object"},
                lambda a: wrap_image(data_url, "shot.png"),
            )
        ]
    )
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(0, id="c1", name="read_image", arguments='{"path":"shot.png"}')
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="je vois un carré rouge"), finish_reason="stop")]
    client, fake = _client([stream1, stream2])

    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "regarde shot.png"}], "s", 100, None, registry
        )
    )

    # le 2e appel modèle porte un message user multimodal avec le bloc image_url
    msgs = fake.calls[1]["messages"]
    img_parts = [
        part
        for m in msgs
        if m.get("role") == "user" and isinstance(m.get("content"), list)
        for part in m["content"]
        if part.get("type") == "image_url"
    ]
    assert any(p["image_url"]["url"] == data_url for p in img_parts)
    # le message `tool` ne contient PAS le base64 (juste un accusé)
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert tool_msgs and "base64" not in tool_msgs[0]["content"]
    # le preview UI ne fuit pas le data-url géant
    res = [p for k, p in events if k == "tool_result"]
    assert res and "base64" not in res[0]["preview"]
    assert "".join(p for k, p in events if k == "content") == "je vois un carré rouge"
