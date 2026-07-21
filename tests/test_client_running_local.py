# Sonde du serveur LOCAL : /running est un endpoint llama-swap — en mode
# MONO-MODÈLE direct, llama-server répond 404 dessus alors qu'il est VIVANT.
# Vécu 2026-07-21 (1re machine mono-modèle) : l'UI concluait « serveur éteint »
# à CHAQUE message et attendait ~90 s pour rien. Une réponse HTTP = vivant.
import io
import urllib.error
import urllib.request

from loom.agent.client import LoomClient


def _client():
    return LoomClient(base_url="http://127.0.0.1:8080/v1")


def _resp(body: bytes):
    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return R(body)


def test_running_local_swap_repond(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout=0: _resp(b'[{"model":"x"}]')
    )
    ok, txt = _client().running_local()
    assert ok is True and "x" in txt


def test_running_local_404_direct_est_vivant(monkeypatch):
    # llama-server direct : /running -> 404, /v1/models -> inventaire (le chemin
    # du GGUF contient l'id du modèle -> le test par sous-chaîne des appelants marche).
    def fake(url, timeout=0):
        if url.endswith("/running"):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
        assert url.endswith("/v1/models")
        return _resp(b'{"models":[{"name":"local/text/ornith-35b/x.gguf"}]}')

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    ok, txt = _client().running_local()
    assert ok is True and "ornith-35b" in txt


def test_running_local_injoignable(monkeypatch):
    def refuse(url, timeout=0):
        raise urllib.error.URLError("refus")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    ok, txt = _client().running_local()
    assert ok is False and txt == ""


def test_context_fingerprint_stable_et_discriminante():
    from loom.agent.client import context_fingerprint

    msgs = [
        {"role": "system", "content": "prompt système"},
        {"role": "user", "content": "bonjour"},
    ]
    a = context_fingerprint(msgs)
    assert a == context_fingerprint(list(msgs))  # déterministe
    assert a.startswith("s:") and " u:" in a
    # une mutation d'UN message change SON couple, pas les autres
    mut = [dict(msgs[0]), {"role": "user", "content": "bonsoir"}]
    b = context_fingerprint(mut)
    assert a.split()[0] == b.split()[0] and a.split()[1] != b.split()[1]
    # tool_calls comptent dans l'empreinte (une trace d'outil qui change = visible)
    tc = [dict(msgs[0]), dict(msgs[1], tool_calls=[{"id": "t1"}])]
    assert context_fingerprint(tc).split()[1] != a.split()[1]
