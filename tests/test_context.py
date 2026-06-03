# tests/test_context.py
from loom.context import (
    conversation_tokens,
    effective_context_budget,
    estimate_tokens,
    needs_summary,
    summarize,
)
from loom.conversation import Conversation


def test_effective_context_budget_clamps_to_window():
    # bug d'origine (ctx 4096, budget 5000, max 2048) -> clampé à 4096-2048-512 = 1536
    assert effective_context_budget(5000, 4096, 2048) == 1536
    # contexte large -> budget configuré respecté
    assert effective_context_budget(5000, 8192, 2048) == 5000
    # plancher 256 si la fenêtre est trop petite
    assert effective_context_budget(5000, 2048, 2048) == 256


def test_estimate_tokens_roughly_quarter_length():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 1


def test_conversation_tokens_counts_text_and_images():
    msgs = [
        {"role": "user", "content": "a" * 40},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "b" * 40},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
    ]
    # 10 (sys) + 10 + 10 + 1000 (image)
    assert conversation_tokens("s" * 40, msgs) == 1030


def test_needs_summary_threshold():
    msgs = [{"role": "user", "content": "a" * 40}]
    assert needs_summary("", msgs, budget=5) is True
    assert needs_summary("", msgs, budget=100) is False


class FakeClient:
    def stream_chat(self, messages, system_prompt):
        yield ("content", "RESUME_COURT")


def test_summarize_replaces_old_keeps_recent():
    conv = Conversation(system_prompt="sys")
    for i in range(10):
        conv.add("user", "x" * 40)  # 10 messages, chacun ~10 tokens
    changed = summarize(conv, FakeClient(), budget=20, keep_recent=3)
    assert changed is True
    # 1 résumé + 3 récents = 4 messages
    assert len(conv.messages) == 4
    assert "RESUME_COURT" in conv.messages[0]["content"]
    assert conv.messages[0]["role"] == "user"


def test_summarize_noop_when_under_budget():
    conv = Conversation(system_prompt="sys")
    conv.add("user", "court")
    assert summarize(conv, FakeClient(), budget=10000, keep_recent=3) is False
    assert len(conv.messages) == 1
