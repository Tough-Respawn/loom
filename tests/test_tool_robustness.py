# Robustesse de la frontière d'outils face aux formes naturellement produites par un modèle.
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from loom.tools.base import (
    ToolError,
    _resolve_in_root,
    coerce_enum,
    validate_and_coerce,
)
from loom.tools.browser import _looks_like_host
from loom.tools.calc import calculate
from loom.tools.fs import make_edit_file
from loom.tools.read import make_read_file
from loom.tools.todo import make_manage_todos




def test_array_json_string_parsee():
    sch = {"properties": {"xs": {"type": "array"}}, "required": []}
    assert validate_and_coerce("t", sch, {"xs": "[1,2,3]"}) == {"xs": [1, 2, 3]}


def test_array_scalaire_seul_enveloppe():
    sch = {"properties": {"xs": {"type": "array"}}, "required": []}
    assert validate_and_coerce("t", sch, {"xs": "foo"}) == {"xs": ["foo"]}


def test_object_json_string_parsee():
    sch = {"properties": {"w": {"type": "object"}}, "required": []}
    assert validate_and_coerce("t", sch, {"w": '{"a":1}'}) == {"w": {"a": 1}}


def test_string_recoit_liste_jointe():
    sch = {"properties": {"cmd": {"type": "string"}}, "required": []}
    out = validate_and_coerce("t", sch, {"cmd": ["git", "status"]})
    assert out == {"cmd": "git status"}




@pytest.mark.parametrize(
    "value,expected",
    [
        ("done", "done"),
        ("Done", "done"),  # casse
        ("DONE", "done"),
        ("in-progress", "in_progress"),  # tiret
        ("in progress", "in_progress"),  # espace
    ],
)
def test_coerce_enum_mecanique(value, expected):
    assert coerce_enum(value, ["pending", "in_progress", "done"]) == expected


def test_coerce_enum_alias():
    aliases = {"completed": "done", "wip": "in_progress"}
    allowed = ["pending", "in_progress", "done"]
    assert coerce_enum("completed", allowed, aliases) == "done"
    assert coerce_enum("wip", allowed, aliases) == "in_progress"


def test_coerce_enum_inconnu_reste_tel_quel():
    # Ne pas deviner une valeur enum non résolue.
    assert coerce_enum("banane", ["pending", "done"]) == "banane"


def test_coerce_enum_top_level_via_schema():
    sch = {
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["episodic", "memory", "profile", "soul"],
                "x_aliases": {"fact": "memory", "user": "profile"},
            }
        },
        "required": [],
    }
    assert validate_and_coerce("remember", sch, {"kind": "fact"})["kind"] == "memory"
    assert validate_and_coerce("remember", sch, {"kind": "USER"})["kind"] == "profile"




class _Conv:
    def __init__(self):
        self.todos = []


@pytest.mark.parametrize(
    "status,mark",
    [
        ("completed", "[x]"),
        ("Done", "[x]"),
        ("fini", "[x]"),
        ("in-progress", "[~]"),
        ("in progress", "[~]"),
        ("wip", "[~]"),
        ("todo", "[ ]"),
    ],
)
def test_todo_statuts_synonymes(status, mark):
    todo = make_manage_todos(_Conv())
    out = todo.run({"todos": [{"content": "x", "status": status}]})
    assert mark in out


def test_todo_statut_vraiment_invalide_rejete():
    todo = make_manage_todos(_Conv())
    with pytest.raises(ToolError):
        todo.run({"todos": [{"content": "x", "status": "zzz"}]})




@pytest.mark.parametrize(
    "expr,expected",
    [
        ("√16", 4),
        ("2√9", 6),
        ("3(4+5)", 27),
        ("2pi", pytest.approx(6.283185307, rel=1e-6)),
        ("200*20%", pytest.approx(40)),
        ("20%", pytest.approx(0.2)),
        ("10%3", 1),  # modulo PRÉSERVÉ (pas pourcentage)
        ("1e3", 1000),  # notation scientifique PRÉSERVÉE
        ("1.5e-3", pytest.approx(0.0015)),
        ("log10(1000)", 3),  # identifiant à chiffre non cassé
    ],
)
def test_calc_ecritures_modele(expr, expected):
    out = calculate(expr)
    val = float(out.split(" = ", 1)[1].split(" (~", 1)[0])
    assert val == expected


def test_calc_exposant_borne_ne_gele_pas():
    with pytest.raises(ToolError):
        calculate("9^9^9")


def test_calc_virgule_message_actionnable():
    with pytest.raises(ToolError) as e:
        calculate("1,5*2")
    assert "POINT décimal" in str(e.value)




def test_edit_file_utf16():
    d = Path(tempfile.mkdtemp())
    f = d / "ps.txt"
    f.write_text("bonjour monde\ndeux\n", encoding="utf-16")
    ef = make_edit_file(str(d))
    out = ef.run({"path": "ps.txt", "old_string": "bonjour", "new_string": "salut"})
    assert "modifié" in out
    assert "salut monde" in f.read_text(encoding="utf-16")




def test_read_file_offset_limit_alias():
    d = Path(tempfile.mkdtemp())
    f = d / "big.txt"
    f.write_text("\n".join(f"L{i}" for i in range(1, 101)), encoding="utf-8")
    rf = make_read_file(str(d), 60000)
    out = rf.run({"path": "big.txt", "offset": 50, "limit": 3})
    assert "L50" in out and "L52" in out
    assert "L1\n" not in out  # ne repart PAS du début




def test_resolve_expanduser():
    root = Path(tempfile.gettempdir())
    got = _resolve_in_root(root, "~/sous/fichier.txt")
    assert "~" not in str(got)
    assert got == (Path.home() / "sous" / "fichier.txt").resolve()


def test_resolve_strip_guillemets():
    root = Path(tempfile.gettempdir())
    got = _resolve_in_root(root, '"C:/Users/x.txt"')
    assert '"' not in str(got)




@pytest.mark.parametrize(
    "target,is_host",
    [
        ("localhost:3000", True),
        ("127.0.0.1:8080", True),
        ("example.com", True),
        ("www.a.org/foo", True),
        ("plain_word", False),  # pas de point ni de port -> pas un hôte
    ],
)
def test_looks_like_host(target, is_host):
    assert bool(_looks_like_host(target)) is is_host




def test_fetch_url_domaine_nu_et_schema(monkeypatch):
    import loom.tools.web as web

    seen = {}
    # Pas de DNS réel dans un test (réseau de l'hôte = variable, ex. DNS64).
    monkeypatch.setattr(web, "_blocked_host_reason", lambda url: None)
    monkeypatch.setattr(
        web,
        "fetch_page",
        lambda url, cfg, snippet="", raise_status=False, impersonate=None: (
            seen.setdefault("url", url) or "OK"
        ),
    )
    fu = web.make_fetch_url(web.WebSearchConfig())
    fu.run({"url": "example.com"})
    assert seen["url"] == "https://example.com"
    with pytest.raises(ToolError):
        fu.run({"url": "ftp://x"})




def test_find_files_recursif_par_defaut():
    from loom.tools.search import make_find_files

    d = Path(tempfile.mkdtemp())
    (d / "src").mkdir()
    (d / "src" / "a.py").write_text("x", encoding="utf-8")
    (d / "top.py").write_text("y", encoding="utf-8")
    ff = make_find_files(str(d))
    out = ff.run({"pattern": "*.py"})
    assert "src/a.py" in out and "top.py" in out  # récursif, pas seulement la racine


def test_read_line_count_zero_rejete():
    d = Path(tempfile.mkdtemp())
    (d / "f.txt").write_text("a\nb\nc\n", encoding="utf-8")
    rf = make_read_file(str(d), 60000)
    with pytest.raises(ToolError):
        rf.run({"path": "f.txt", "start_line": 2, "line_count": 0})


def test_edit_file_prefixe_numero_ligne_detecte():
    d = Path(tempfile.mkdtemp())
    (d / "e.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    ef = make_edit_file(str(d))
    with pytest.raises(ToolError) as e:
        ef.run({"path": "e.py", "old_string": "  2→    return 1", "new_string": "x"})
    assert "N→" in str(e.value) or "préfixe" in str(e.value)  # pointe le piège N→




def test_fetch_url_remonte_statut_http_et_url_origine(monkeypatch):
    # Un 403 doit conserver le statut et l'URL lisible d'origine.
    import httpx

    import loom.tools.web as web

    def fake_get(
        url, params=None, headers=None, timeout=None, pin_ip=None, impersonate=None
    ):
        req = httpx.Request("GET", "https://52.222.201.14/liste.htm")  # épinglée
        return httpx.Response(403, request=req)

    monkeypatch.setattr(web, "_resolve_validated", lambda url: (None, "52.222.201.14"))
    monkeypatch.setattr(web, "_http_get", fake_get)
    fu = web.make_fetch_url(web.WebSearchConfig())
    with pytest.raises(ToolError) as e:
        fu.run({"url": "https://www.seloger.com/liste.htm"})
    msg = str(e.value)
    assert "403" in msg
    assert "www.seloger.com" in msg
    assert "52.222.201.14" not in msg


def test_fetch_page_web_search_garde_le_repli_snippet(monkeypatch):
    # Une page de résultat en 403 ne doit pas casser toute la recherche.
    import httpx

    import loom.tools.web as web

    def fake_get(
        url, params=None, headers=None, timeout=None, pin_ip=None, impersonate=None
    ):
        req = httpx.Request("GET", url)
        return httpx.Response(403, request=req)

    monkeypatch.setattr(web, "_resolve_validated", lambda url: (None, "1.2.3.4"))
    monkeypatch.setattr(web, "_http_get", fake_get)
    out = web.fetch_page("https://x.example/a", web.WebSearchConfig(), snippet="RÉSUMÉ")
    assert out == "RÉSUMÉ"


def test_curl_alias_powershell_hint(monkeypatch):
    # Simuler PowerShell 5.1 où curl est un alias et échoue au binding.
    from types import SimpleNamespace

    import loom.tools.shell as shell
    from loom.tools.shell import _unix_ism_hint

    monkeypatch.setattr(
        shell, "detect", lambda: SimpleNamespace(shell_kind="powershell")
    )
    stderr = (
        "Invoke-WebRequest : Impossible de traiter la commande, car un ou "
        "plusieurs paramètres obligatoires sont absents : Uri."
    )
    hint = _unix_ism_hint("curl -s https://exemple.fr/api", stderr)
    assert "curl.exe" in hint


def test_unix_ism_hint_stderr_francais(monkeypatch):
    # Le diagnostic doit reconnaître aussi le message Windows français.
    from types import SimpleNamespace

    import loom.tools.shell as shell
    from loom.tools.shell import _unix_ism_hint

    monkeypatch.setattr(
        shell, "detect", lambda: SimpleNamespace(shell_kind="powershell")
    )
    stderr = (
        "grep : Le terme «grep» n'est pas reconnu comme nom d'applet de commande, "
        "fonction, fichier de script ou programme exécutable."
    )
    hint = _unix_ism_hint("grep -r motif .", stderr)
    assert "Select-String" in hint


def test_use_skill_nom_d_outil_redirige():
    # Distinguer une confusion outil/skill d'un skill inconnu.
    from loom.tools.skills import make_use_skill

    us = make_use_skill(lambda: [])
    with pytest.raises(ToolError) as e:
        us.run({"name": "dispatch_agent"})
    msg = str(e.value).lower()
    assert "outil" in msg and "dispatch_agent" in msg




def test_fetch_url_params_objet_encode(monkeypatch):
    # L'outil doit encoder lui-même les paramètres structurés.
    import loom.tools.web as web

    seen = {}
    # Pas de DNS réel dans un test (réseau de l'hôte = variable, ex. DNS64).
    monkeypatch.setattr(web, "_blocked_host_reason", lambda url: None)
    monkeypatch.setattr(
        web,
        "fetch_page",
        lambda url, cfg, snippet="", raise_status=False, impersonate=None: (
            seen.setdefault("url", url) or "OK"
        ),
    )
    fu = web.make_fetch_url(web.WebSearchConfig())
    fu.run(
        {
            "url": "https://www.bienici.com/realEstateAds.json",
            "params": {"filters": {"size": 2, "filterType": "buy"}, "page": 1},
        }
    )
    assert seen["url"].startswith("https://www.bienici.com/realEstateAds.json?")
    assert "filters=%7B%22size%22" in seen["url"]  # JSON compact url-encodé
    assert "page=1" in seen["url"]




def test_check_page_accepte_steps(monkeypatch):
    import loom.tools.browser as br

    seen = {}

    def fake_interactive(ws, target, steps):
        seen["steps"] = steps
        return {
            "url": target,
            "ok": True,
            "console_errors": [],
            "steps": [
                {"op": "click", "selector": "#b", "ok": True, "observed": "1 match"}
            ],
        }

    monkeypatch.setattr(br, "run_interactive", fake_interactive)
    d = Path(tempfile.mkdtemp())
    (d / "p.html").write_text("<html></html>", encoding="utf-8")
    cp = br.make_check_page(str(d))
    out = cp.run({"url": "p.html", "steps": [{"op": "click", "selector": "#b"}]})
    assert seen["steps"] == [{"op": "click", "selector": "#b"}]
    assert "VERDICT" in out


def test_check_page_sans_steps_comportement_actuel(monkeypatch):
    import loom.tools.browser as br

    monkeypatch.setattr(br, "_render_page", lambda *a: "console : 0 erreur(s)")
    d = Path(tempfile.mkdtemp())
    (d / "p.html").write_text("<html></html>", encoding="utf-8")
    cp = br.make_check_page(str(d))
    out = cp.run({"url": "p.html"})
    assert "0 erreur" in out


def test_check_interactive_retire_du_catalogue():
    from loom.tools.base import AVAILABLE_TOOLS

    names = [t["name"] for t in AVAILABLE_TOOLS]
    assert "check_interactive" not in names
    import loom.tools.browser as br

    spec = br.make_check_page(".")
    assert "steps" in spec.parameters["properties"]




def test_plugins_hors_du_set_par_defaut():
    # Les outils d'administration de plugins restent activables mais hors du défaut.
    import tomllib

    with open("config/defaults.toml", "rb") as f:
        cfg = tomllib.load(f)
    enabled = cfg["tools"]["enabled"]
    for name in ("list_plugins", "add_marketplace", "install_plugin"):
        assert name not in enabled, name


def test_read_image_absent_sans_vision():
    # Masquer le schéma vision sans basculer implicitement vers un autre modèle.
    from loom.tools import build_registry

    reg = build_registry(".", 1000, ["read_file", "read_image"], active_is_vision=False)
    names = [t["function"]["name"] for t in reg.openai_tools()]
    assert "read_image" not in names
    assert "read_file" in names
    reg2 = build_registry(".", 1000, ["read_file", "read_image"], active_is_vision=True)
    assert "read_image" in [t["function"]["name"] for t in reg2.openai_tools()]


def test_read_image_gate_erreur_franche():
    # Un outil vision masqué reste connu afin de fournir une erreur explicite.
    from loom.tools import build_registry

    reg = build_registry(".", 1000, ["read_file", "read_image"], active_is_vision=False)
    out = reg.run("read_image", {"path": "photo.jpg"})
    assert "outil inconnu" not in out
    assert "N'A PAS la vision" in out and "VISION" in out
    assert "outil inconnu" in reg.run("grep", {})


def test_x_aliases_pas_serialises_au_modele():
    # Les métadonnées `x_*` servent à la validation mais restent hors du schéma modèle.
    from loom.tools.base import ToolSpec

    spec = ToolSpec(
        name="t",
        description="d",
        parameters={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["a", "b"],
                    "x_aliases": {"c": "a"},
                }
            },
        },
        run=lambda a: "",
    )
    out = spec.to_openai()
    assert "x_aliases" not in str(out)
    assert spec.parameters["properties"]["kind"]["x_aliases"] == {"c": "a"}
