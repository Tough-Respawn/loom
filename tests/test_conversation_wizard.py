# L'état du wizard /add-model vit sur la Conversation : persisté (survit au refresh),
# None = pas de wizard actif. Round-trip to_dict/from_dict + rétro-compat ancien format.
from loom.agent.conversation import Conversation


def test_wizard_roundtrip():
    c = Conversation(system_prompt="sp")
    assert c.wizard is None
    c.set_wizard({"step": "kind"})
    c2 = Conversation.from_dict(c.to_dict(), "sp")
    assert c2.wizard == {"step": "kind"}
    c2.set_wizard(None)
    assert c2.wizard is None


def test_wizard_absent_ancien_format():
    c = Conversation.from_dict({"messages": []}, "sp")
    assert c.wizard is None


def test_reset_efface_le_wizard():
    c = Conversation(system_prompt="sp")
    c.set_wizard({"step": "kind"})
    c.reset()
    assert c.wizard is None
