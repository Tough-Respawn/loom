# infer_title : robustesse aux providers à température FIGÉE (Kimi/Moonshot renvoie
# 400 « invalid temperature: only 0.6/1 is allowed ») — on retente SANS température.
from types import SimpleNamespace

from loom.agent.client import LoomClient


class _FakeCompletions:
    """Rejette toute requête portant `temperature` (comme Moonshot), sinon répond."""

    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if "temperature" in kw:
            raise RuntimeError(
                "Error code: 400 - invalid temperature: only 1 is allowed for this model"
            )
        msg = SimpleNamespace(content="Titre Kimi")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeOAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())

    def with_options(self, **kw):
        return self


def test_infer_title_retente_sans_temperature_sur_400_temperature():
    oai = _FakeOAI()
    fake_self = SimpleNamespace(_resolve=lambda m: (oai, "kimi-k3", None))
    title = LoomClient.infer_title(fake_self, "kimi-k3", "bonjour le monde")
    assert title == "Titre Kimi"
    calls = oai.chat.completions.calls
    # 1er essai : temperature posée -> 400 ; 2e essai : MÊME variante sans temperature
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    assert len(calls) == 2  # pas de cascade sur les autres variantes
