# tests/test_cache_final.py
"""Vérification du cache avec la CONFIGURATION FINALE (lot 5 du bench de placement).

La sonde d'isolation (A -> B -> A, un slot) décide s'il faut un second slot. Il
manquait la preuve que la séquence RÉELLE — conversation sur le slot 0, appel annexe
routé sur le slot final (1 s'il y en a deux), retour de la conversation — réutilise
effectivement le cache une fois les réglages choisis. C'est ce qui justifie de traiter
le gros prefill comme un coût amorti.
"""

from __future__ import annotations

import json

from loom.runtime.hardware import HardwareProfile
from loom.setup import topology as topo
from loom.setup.topology import TOPO_MOE_HYBRIDE, ServerProbe, cache_reused

_UMA = HardwareProfile(True, "Radeon 860M", 46_350, 16, vram_total_mb=48_789)


def _probe(n_parallel):
    return ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=8,
        ngl=999,
        topology=TOPO_MOE_HYBRIDE,
        profile=_UMA,
        n_parallel=n_parallel,
        kill=lambda p: None,
    )


def test_cache_reused_verdict():
    assert cache_reused(600, 4) is True
    assert cache_reused(600, 590) is False
    assert cache_reused(100, 50) is False  # même seuil que la sonde d'isolation
    assert cache_reused(100, 49) is True
    assert cache_reused(0, 0) is None  # illisible : pas de verdict


def test_verify_cache_route_l_annexe_sur_le_slot_final(monkeypatch):
    monkeypatch.setattr(topo.time, "sleep", lambda s: None)
    probe = _probe(n_parallel=2)
    appels: list[tuple[str, int | None]] = []

    def fake_completion(prompt, n_predict, cache_prompt=False, id_slot=None):
        appels.append((prompt[:20], id_slot))
        n = {1: 600, 2: 610, 3: 4}[len(appels)]
        return {"timings": {"prompt_n": n}}

    monkeypatch.setattr(probe, "_start", lambda ctx: object())
    monkeypatch.setattr(probe, "_completion", fake_completion)
    v = probe.verify_cache()
    # Conversation sur le slot 0, annexe sur le slot 1, retour sur le slot 0.
    assert [s for _, s in appels] == [0, 1, 0]
    assert v == {"first": 600, "back": 4, "annex_slot": 1, "slots": 2, "reused": True}


def test_verify_cache_un_slot_annexe_sur_le_meme_slot(monkeypatch):
    monkeypatch.setattr(topo.time, "sleep", lambda s: None)
    probe = _probe(n_parallel=1)
    appels: list[int | None] = []

    def fake_completion(prompt, n_predict, cache_prompt=False, id_slot=None):
        appels.append(id_slot)
        n = {1: 600, 2: 610, 3: 590}[len(appels)]
        return {"timings": {"prompt_n": n}}

    monkeypatch.setattr(probe, "_start", lambda ctx: object())
    monkeypatch.setattr(probe, "_completion", fake_completion)
    v = probe.verify_cache()
    assert appels == [0, 0, 0]
    assert v["reused"] is False and v["annex_slot"] == 0


def test_completion_transmet_id_slot(monkeypatch):
    probe = _probe(n_parallel=2)
    corps: list[dict] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"timings": {}}'

    def fake_urlopen(req, timeout=None):
        corps.append(json.loads(req.data))
        return _Resp()

    monkeypatch.setattr(topo.urllib.request, "urlopen", fake_urlopen)
    probe._completion("p", 8, cache_prompt=True, id_slot=1)
    probe._completion("p", 8, cache_prompt=True)
    assert corps[0]["id_slot"] == 1 and corps[0]["cache_prompt"] is True
    assert "id_slot" not in corps[1]  # sans slot explicite : le serveur choisit
