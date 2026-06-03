from loom.classify import classify_intent


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(
        self,
        messages,
        system_prompt,
        max_tokens=8,
        model=None,
        thinking=False,
        temperature=None,
    ):
        self.calls.append({"messages": messages, "thinking": thinking})
        return self.reply


def test_classify_build():
    assert (
        classify_intent(FakeClient("BUILD"), "crée un démineur", model="m") == "build"
    )


def test_classify_chat():
    assert (
        classify_intent(FakeClient("CHAT"), "explique les closures", model="m")
        == "chat"
    )


def test_classify_defaults_to_chat_when_ambiguous():
    assert classify_intent(FakeClient(""), "?", model="m") == "chat"
    assert classify_intent(FakeClient("bla bla"), "x", model="m") == "chat"


def test_classify_runs_thinking_off():
    c = FakeClient("CHAT")
    classify_intent(c, "x", model="m")
    assert c.calls[0]["thinking"] is False
