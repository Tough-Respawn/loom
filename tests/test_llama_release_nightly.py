"""Depuis 2026-08, llama.cpp publie des versions stables `v0.x` SANS binaires :
`/releases/latest` renvoie v0.5.0 avec un seul asset `nightly-tag.txt` (ex.
"b11146"), les exécutables vivent dans la pré-release de ce tag. Le setup doit
suivre ce pointeur, sinon il ne trouve aucun binaire à installer."""

from __future__ import annotations

from loom.setup import llama_release as lr


class _Resp:
    def __init__(self, status=200, json=None, text=""):
        self.status_code = status
        self._json = json
        self.text = text

    def json(self):
        return self._json


class _Client:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.routes[url]


_TXT = "https://github.com/ggml-org/llama.cpp/releases/download/v0.5.0/nightly-tag.txt"
_TAG = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11146"
_STABLE = {
    "tag_name": "v0.5.0",
    "assets": [{"name": "nightly-tag.txt", "browser_download_url": _TXT, "size": 7}],
}
_NIGHTLY = {
    "tag_name": "b11146",
    "assets": [
        {
            "name": "llama-b11146-bin-win-cuda-12.4-x64.zip",
            "size": 1,
            "browser_download_url": "u1",
        },
    ],
}


def test_v0x_sans_binaire_suit_le_pointeur_nightly():
    c = _Client(
        {
            lr.RELEASES_URL: _Resp(json=_STABLE),
            _TXT: _Resp(text="b11146\n"),
            _TAG: _Resp(json=_NIGHTLY),
        }
    )
    rel = lr.fetch_latest_release(c)
    assert rel["tag_name"] == "b11146"
    assert rel["stable_tag"] == "v0.5.0"
    assert c.calls == [lr.RELEASES_URL, _TXT, _TAG]


def test_release_avec_binaires_inchangee():
    c = _Client({lr.RELEASES_URL: _Resp(json=_NIGHTLY)})
    rel = lr.fetch_latest_release(c)
    assert rel["tag_name"] == "b11146" and "stable_tag" not in rel
    assert c.calls == [lr.RELEASES_URL]


def test_pointeur_illisible_rend_la_release_stable():
    # Pas de plantage : select_assets dira ensuite « aucun asset », message actionnable.
    c = _Client({lr.RELEASES_URL: _Resp(json=_STABLE), _TXT: _Resp(status=404)})
    assert lr.fetch_latest_release(c)["tag_name"] == "v0.5.0"


def test_llama_swap_non_concerne():
    swap = {
        "tag_name": "v200",
        "assets": [{"name": "nightly-tag.txt", "browser_download_url": _TXT}],
    }
    c = _Client({lr.SWAP_RELEASES_URL: _Resp(json=swap)})
    assert lr.fetch_latest_release(c, url=lr.SWAP_RELEASES_URL)["tag_name"] == "v200"
    assert c.calls == [lr.SWAP_RELEASES_URL]


# Liste RÉELLE de b11146 (ordre alphabétique de l'API GitHub : cudart en tête).
_B11146_WIN = [
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b11146-bin-win-cpu-x64.zip",
    "llama-b11146-bin-win-cuda-12.4-x64.zip",
    "llama-b11146-bin-win-cuda-13.4-arm64.zip",
    "llama-b11146-bin-win-cuda-13.4-x64.zip",
    "llama-b11146-bin-win-vulkan-x64.zip",
]


def _rel(names):
    return {
        "tag_name": "b11146",
        "assets": [{"name": n, "browser_download_url": n, "size": 1} for n in names],
    }


def test_cudart_jamais_pris_pour_le_binaire_et_meme_version():
    plan = lr.select_assets(_rel(_B11146_WIN), "windows", "x64", True)
    names = [a["name"] for a in plan.assets]
    assert names == [
        "llama-b11146-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]


def test_dll_cuda_de_la_meme_version_que_le_binaire():
    # Si seul le 13.4 existe en binaire, les DLL doivent être 13.4, pas 12.4.
    names = [n for n in _B11146_WIN if "12.4" not in n or n.startswith("cudart")]
    plan = lr.select_assets(_rel(names), "windows", "x64", True)
    assert [a["name"] for a in plan.assets] == [
        "llama-b11146-bin-win-cuda-13.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
    ]
