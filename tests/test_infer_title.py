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


# ---- revue croisée 2026-09-13 : titrage en FLUX, interruptible via le porte-flux ----


class _ChunkStream:
    """Flux minimal : deux deltas de contenu puis fin."""

    def __init__(self, parts):
        self.parts = parts
        self.closed = False

    def __iter__(self):
        for p in self.parts:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=p))],
                usage=None,
            )

    def close(self):
        self.closed = True


def test_infer_title_avec_porte_flux_streame_et_publie_son_flux():
    stream = _ChunkStream(["Titre ", "streamé"])
    seen: list[dict] = []

    def create(**kw):
        seen.append(kw)
        return stream

    oai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    oai.with_options = lambda **kw: oai
    holder: dict = {}
    title = LoomClient.infer_title(
        _self(oai, native=True), "orn", "bonjour", stream_holder=holder
    )
    assert title == "Titre streamé"
    assert seen[0]["stream"] is True  # un flux, pour pouvoir le fermer de l'extérieur
    assert stream.closed and "stream" not in holder  # porte-flux nettoyé


def test_infer_title_abandonne_rend_vide():
    import threading

    from tests.test_warm_preemptible import _BlockingStream

    stream = _BlockingStream()
    oai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: stream))
    )
    oai.with_options = lambda **kw: oai
    holder: dict = {}
    out: list = []
    t = threading.Thread(
        target=lambda: out.append(
            LoomClient.infer_title(
                _self(oai, native=True), "orn", "x", stream_holder=holder
            )
        )
    )
    t.start()
    assert stream.started.wait(timeout=2)
    holder["abort"] = True
    holder["stream"].close()
    t.join(timeout=2)
    assert out == [""]  # abandonné : pas de titre, et pas de variante suivante tentée
    assert "stream" not in holder and holder.get("abort") is True  # signal conservé


def test_infer_title_interruptible_a_60s_de_budget():
    # Interruption en plein calcul PROUVÉE côté serveur le 2026-09-13 (cancel task,
    # cache partiel conservé) : le titrage séquentiel peut prendre 60 s sans risque,
    # un message le coupe. Sans porte-flux (distant, thread), 20 s restent la règle.
    seen: dict = {}

    def create(**kw):
        return _ChunkStream(["ok"])

    oai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    oai.with_options = lambda **kw: seen.update(kw) or oai
    LoomClient.infer_title(_self(oai, native=True), "orn", "x", stream_holder={})
    assert seen["timeout"] == 60
    LoomClient.infer_title(_self(oai, native=True), "orn", "x")
    assert seen["timeout"] == 20
