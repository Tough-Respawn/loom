# infer_title : plus JAMAIS de `temperature` (des providers la refusent : Kimi/Moonshot
# 400 « invalid temperature: only 0.6/1 is allowed » — vécu aussi en session locale le
# 2026-09-13 avec un titre qui n'arrivait jamais), et en LOCAL le thinking est coupé dès
# le premier essai (variante llama.cpp), sinon le budget part en réflexion.
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


def _self(oai, native):
    return SimpleNamespace(
        _resolve=lambda m: (oai, "m", native),
        is_remote=lambda m: not native,
        annex_slot=lambda m: 1,
    )


def test_infer_title_n_envoie_jamais_de_temperature():
    oai = _FakeOAI()
    title = LoomClient.infer_title(_self(oai, native=False), "kimi-k3", "bonjour")
    assert title == "Titre Kimi"
    calls = oai.chat.completions.calls
    assert len(calls) == 1 and "temperature" not in calls[0]


def test_infer_title_local_coupe_le_thinking_des_le_premier_essai():
    oai = _FakeOAI()
    LoomClient.infer_title(_self(oai, native=True), "orn", "bonjour")
    first = oai.chat.completions.calls[0]
    assert first["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert first["extra_body"]["id_slot"] == 1
