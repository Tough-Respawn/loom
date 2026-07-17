# Fakes injectés à la place de HfApi : les tests ne touchent JAMAIS le réseau.
from types import SimpleNamespace

import pytest

from loom.runtime.hf_catalog import HfCatalogError, list_gguf_files, search_models


class FakeApi:
    def __init__(self, models=None, info=None, err=None):
        self._models, self._info, self._err = models or [], info, err

    def list_models(self, **kw):
        if self._err:
            raise self._err
        return iter(self._models)

    def model_info(self, repo_id, files_metadata=False):
        if self._err:
            raise self._err
        return self._info


def _m(mid, dl, likes):
    return SimpleNamespace(id=mid, downloads=dl, likes=likes)


def _sib(name, size):
    return SimpleNamespace(rfilename=name, size=size)


def test_search_shortlist():
    api = FakeApi(models=[_m("org/a-GGUF", 1000, 5), _m("org/b-GGUF", 50, None)])
    hits = search_models("a", api=api)
    assert hits == [
        {"repo_id": "org/a-GGUF", "downloads": 1000, "likes": 5},
        {"repo_id": "org/b-GGUF", "downloads": 50, "likes": 0},
    ]


def test_search_erreur_reseau_actionnable():
    api = FakeApi(err=OSError("connexion perdue"))
    with pytest.raises(HfCatalogError):
        search_models("a", api=api)


def test_list_gguf_regroupe_les_parties_et_repere_mmproj():
    mb = 1024 * 1024
    info = SimpleNamespace(
        siblings=[
            _sib("m.Q4_K_M-00001-of-00002.gguf", 10 * mb),
            _sib("m.Q4_K_M-00002-of-00002.gguf", 5 * mb),
            _sib("m.Q8_0.gguf", 30 * mb),
            _sib("mmproj-F16.gguf", 2 * mb),
            _sib("README.md", 1),
        ]
    )
    files = list_gguf_files("org/a", api=FakeApi(info=info))
    by_name = {f["filename"]: f for f in files}
    grp = by_name["m.Q4_K_M-00001-of-00002.gguf"]
    assert grp["size_mb"] == 15
    assert grp["part_files"] == [
        "m.Q4_K_M-00001-of-00002.gguf",
        "m.Q4_K_M-00002-of-00002.gguf",
    ]
    assert by_name["mmproj-F16.gguf"]["is_mmproj"] is True
    assert by_name["m.Q8_0.gguf"]["size_mb"] == 30
    # tri par taille croissante
    assert [f["size_mb"] for f in files] == sorted(f["size_mb"] for f in files)
