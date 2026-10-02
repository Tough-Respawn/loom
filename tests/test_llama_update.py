"""Veille des mises à jour llama.cpp (loom/runtime/llama_update.py) : lecture de la
build du binaire, dernier build nocturne, PR suivies, bandeau. Rien n'est installé."""

from __future__ import annotations

import json
import time

from loom.runtime import llama_update as lu


# -- build du binaire ------------------------------------------------------------
def test_parse_build_nouveau_et_ancien_format():
    assert lu.parse_build("version: 0.5.0-dev (build 11364, commit 3631aa30d)") == 11364
    assert lu.parse_build("version: 9442 (d4c8e2c29)") == 9442
    assert lu.parse_build("") is None
    assert lu.parse_build(None) is None


# -- bandeau (pur) ---------------------------------------------------------------
_LATEST = {"tag": "b11400", "build": 11400, "published_at": "2026-10-05T10:00:00Z"}
_OPEN = {
    "number": 26004,
    "title": "checkpoints",
    "merged": False,
    "state": "open",
    "merged_at": None,
    "url": "u26004",
}
_MERGED = {
    "number": 29600,
    "title": "Bonsai",
    "merged": True,
    "state": "closed",
    "merged_at": "2026-10-04T08:00:00Z",
    "url": "u29600",
}


def test_rien_a_signaler():
    assert lu.build_notice(11400, _LATEST, [_OPEN], "rebuild.bat") is None


def test_nouvelle_build_avec_pr_ouverte_donne_la_commande_de_recompilation():
    n = lu.build_notice(11364, _LATEST, [_OPEN], "C:/llama.cpp/build-loom-cuda.bat")
    assert n["kind"] == "rebuild"
    assert "b11400" in n["message"] and "b11364" in n["message"]
    assert "#26004" in n["message"]  # dit POURQUOI on recompile au lieu de télécharger
    assert n["command"] == "C:/llama.cpp/build-loom-cuda.bat"


def test_toutes_les_pr_mergees_le_binaire_officiel_suffit():
    n = lu.build_notice(11364, _LATEST, [_MERGED], "rebuild.bat")
    assert n["kind"] == "official"
    assert "#29600" in n["message"]
    assert n["command"] == "uv run loom-setup"


def test_pr_mergee_signalee_meme_sans_nouvelle_build():
    n = lu.build_notice(11400, _LATEST, [_OPEN, _MERGED], "rebuild.bat")
    assert n is not None and "#29600" in n["message"]
    assert (
        n["kind"] == "info"
    )  # #26004 encore ouverte : pas encore de passage au binaire officiel


def test_sans_pr_suivie_nouvelle_build_propose_le_setup():
    n = lu.build_notice(11364, _LATEST, [], "")
    assert n["kind"] == "official" and n["command"] == "uv run loom-setup"


def test_build_inconnue_ou_reseau_absent_rien():
    assert lu.build_notice(None, _LATEST, [_OPEN], "x") is None
    assert lu.build_notice(11364, None, [_OPEN], "x") is None


def test_cle_de_fermeture_change_avec_le_contenu():
    a = lu.build_notice(11364, _LATEST, [_OPEN], "x")
    b = lu.build_notice(
        11364, {**_LATEST, "tag": "b11401", "build": 11401}, [_OPEN], "x"
    )
    assert a["key"] != b["key"]  # une build plus récente rouvre un bandeau fermé


# -- réseau (client injecté) -----------------------------------------------------
class _Resp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


class _Client:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return self.routes.get(url, _Resp(404))


def test_latest_nightly_ignore_les_versions_v0x():
    c = _Client(
        {
            lu.RELEASES_LIST_URL: _Resp(
                data=[
                    {"tag_name": "v0.5.0", "published_at": "x"},
                    {"tag_name": "b11400", "published_at": "2026-10-05T10:00:00Z"},
                ]
            )
        }
    )
    assert lu.latest_nightly(c) == _LATEST


def test_pr_states():
    c = _Client(
        {
            lu.PR_URL.format(n=29600): _Resp(
                data={
                    "number": 29600,
                    "title": "Bonsai",
                    "state": "closed",
                    "merged": True,
                    "merged_at": "2026-10-04T08:00:00Z",
                    "html_url": "u29600",
                }
            )
        }
    )
    assert lu.pr_states(c, [29600]) == [_MERGED]


def test_check_met_en_cache_et_ne_rappelle_pas_github_avant_24h(tmp_path, monkeypatch):
    monkeypatch.setattr(lu, "binary_build", lambda b: 11364)
    c = _Client(
        {
            lu.RELEASES_LIST_URL: _Resp(
                data=[{"tag_name": "b11400", "published_at": "2026-10-05T10:00:00Z"}]
            ),
            lu.PR_URL.format(n=26004): _Resp(
                data={
                    "number": 26004,
                    "title": "checkpoints",
                    "state": "open",
                    "merged": False,
                    "merged_at": None,
                    "html_url": "u26004",
                }
            ),
        }
    )
    cache = tmp_path / "llama_update.json"
    r1 = lu.check("bin.exe", [26004], "rebuild.bat", cache, client=c)
    assert r1["notice"]["kind"] == "rebuild" and r1["current_build"] == 11364
    n_calls = len(c.calls)
    r2 = lu.check("bin.exe", [26004], "rebuild.bat", cache, client=c)
    assert len(c.calls) == n_calls and r2["notice"] == r1["notice"]
    # Cache périmé -> nouvel appel.
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["checked_at"] = time.time() - 25 * 3600
    cache.write_text(json.dumps(data), encoding="utf-8")
    lu.check("bin.exe", [26004], "rebuild.bat", cache, client=c)
    assert len(c.calls) > n_calls


def test_check_hors_ligne_ne_leve_jamais(tmp_path, monkeypatch):
    monkeypatch.setattr(lu, "binary_build", lambda b: 11364)

    class _Down:
        def get(self, *a, **k):
            raise ConnectionError("hors ligne")

    r = lu.check("bin.exe", [26004], "x", tmp_path / "c.json", client=_Down())
    assert r["notice"] is None and "hors ligne" in r["error"]


def test_binaire_change_invalide_le_cache(tmp_path, monkeypatch):
    builds = iter([11364, 11400])
    monkeypatch.setattr(lu, "binary_build", lambda b: next(builds))
    c = _Client(
        {
            lu.RELEASES_LIST_URL: _Resp(
                data=[{"tag_name": "b11400", "published_at": "2026-10-05T10:00:00Z"}]
            )
        }
    )
    cache = tmp_path / "c.json"
    assert lu.check("a.exe", [], "", cache, client=c)["notice"] is not None
    # Recompilé : la build lue au binaire a changé -> le bandeau disparaît sans attendre 24 h.
    assert lu.check("a.exe", [], "", cache, client=c)["notice"] is None


# -- branchement web -------------------------------------------------------------
def test_route_rend_le_dernier_resultat_sans_reseau(web, app):
    assert web.get("/llama/update").get_json() == {
        "notice": None,
        "current_build": None,
    }
    app.S.llama_update = {
        "current_build": 11364,
        "notice": {"kind": "rebuild", "key": "k"},
    }
    d = web.get("/llama/update").get_json()
    assert d["current_build"] == 11364 and d["notice"]["kind"] == "rebuild"


def test_passe_de_veille_respecte_update_check(monkeypatch):
    from types import SimpleNamespace

    from loom.web.routes import misc

    seen = {}

    def fake_check(bin_, prs, hint, cache):
        seen.update(bin=bin_, prs=prs, hint=hint)
        return {"notice": {"kind": "info"}}

    monkeypatch.setattr(lu, "check", fake_check)
    cfg = SimpleNamespace(
        update_check=True, server_bin="b.exe", track_prs=[26004], rebuild_hint="r.bat"
    )
    monkeypatch.setattr("loom.config.load_config", lambda defaults, local: cfg)
    S = SimpleNamespace(
        config_defaults_path="d", config_local_path="l", llama_update=None
    )
    misc.llama_update_once(S)
    assert S.llama_update == {"notice": {"kind": "info"}}
    assert seen == {"bin": "b.exe", "prs": [26004], "hint": "r.bat"}
    cfg.update_check = False
    misc.llama_update_once(S)
    assert S.llama_update is None


def test_build_maison_compare_sa_base_officielle(tmp_path, monkeypatch):
    # Le numéro de build llama.cpp = nombre de commits : un build maison compte aussi
    # ses commits de PR (11364 = base 11362 + 2). BUILD.txt donne la vraie base.
    exe = tmp_path / "llama-server.exe"
    exe.write_text("")
    monkeypatch.setattr(
        "loom.setup.llama_release.verify_binary",
        lambda b: "version: 0.5.0-dev (build 11364, commit 3631aa30d)",
    )
    assert lu.binary_build(str(exe)) == 11364
    (tmp_path / "BUILD.txt").write_text(
        "loom/runtime 3631aa30d\nupstream_build=11362\n"
    )
    assert lu.binary_build(str(exe)) == 11362


def test_accord_pluriel_des_pr_ouvertes():
    two = [_OPEN, {**_OPEN, "number": 29600}]
    assert "encore ouvertes" in lu.build_notice(11364, _LATEST, two, "x")["message"]
    assert (
        "encore ouverte :" in lu.build_notice(11364, _LATEST, [_OPEN], "x")["message"]
    )
