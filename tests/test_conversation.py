# tests/test_conversation.py
from loom.conversation import Conversation


def test_add_and_to_messages():
    conv = Conversation(system_prompt="sys")
    conv.add("user", "bonjour")
    conv.add("assistant", "salut")
    assert conv.to_messages() == [
        {"role": "user", "content": "bonjour"},
        {"role": "assistant", "content": "salut"},
    ]


def test_reset_keeps_system_prompt():
    conv = Conversation(system_prompt="sys")
    conv.add("user", "x")
    conv.reset()
    assert conv.messages == []
    assert conv.system_prompt == "sys"


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    conv.add("user", "bonjour")
    conv.save(path)

    loaded = Conversation.load(path, default_system_prompt="autre")
    assert loaded.system_prompt == "sys"
    assert loaded.messages == [{"role": "user", "content": "bonjour"}]


def test_load_missing_file_returns_empty(tmp_path):
    loaded = Conversation.load(tmp_path / "absent.json", default_system_prompt="def")
    assert loaded.system_prompt == "def"
    assert loaded.messages == []


def test_multimodal_content_roundtrip(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    parts = [
        {"type": "text", "text": "que vois-tu ?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    conv.add("user", parts)
    conv.save(path)

    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.messages[0]["content"] == parts


def test_active_skills_roundtrip(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    conv.set_skills(["dagster", "conventions"])
    conv.save(path)
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.active_skills == ["dagster", "conventions"]


def test_load_old_json_without_active_skills(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text('{"system_prompt": "s", "messages": []}', encoding="utf-8")
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.active_skills == []


def test_model_roundtrip_and_default(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys", model="gemma")
    conv.set_model("qwen")
    conv.save(path)
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.model == "qwen"


def test_load_old_json_without_model(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text('{"system_prompt": "s", "messages": []}', encoding="utf-8")
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.model == ""


def test_thinking_defaults_true_and_roundtrips(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    assert conv.thinking is True  # comportement par défaut conservé
    conv.set_thinking(False)
    conv.save(path)
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.thinking is False


def test_load_old_json_without_thinking_defaults_true(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text('{"system_prompt": "s", "messages": []}', encoding="utf-8")
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.thinking is True


def test_active_tools_default_empty_and_roundtrip(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    assert conv.active_tools == []
    conv.set_tools(["read_file", "web_search"])
    conv.save(path)
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.active_tools == ["read_file", "web_search"]


def test_load_old_json_without_tools_defaults_empty(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text('{"system_prompt": "s", "messages": []}', encoding="utf-8")
    loaded = Conversation.load(path, default_system_prompt="x")
    assert loaded.active_tools == []


def test_save_is_atomic_no_tmp_residue(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    conv.add("user", "x")
    conv.save(path)
    assert path.exists()
    # aucun fichier .tmp résiduel
    assert list(tmp_path.glob("*.tmp")) == []
    loaded = Conversation.load(path, default_system_prompt="d")
    assert loaded.messages == [{"role": "user", "content": "x"}]
