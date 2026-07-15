# Robustesse des outils face aux écritures « naturelles » d'un modèle (audit 2026-07-14).
# Cause racine commune au bug `^` du calc : la frontière d'entrée (base.py) coerçait les
# scalaires mais PAS les conteneurs JSON-string ni les enums. Ces tests figent les
# corrections (coercition centrale + enum tolérant + calc/edit_file/read robustes).
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


# ---------- base.py : coercition des conteneurs ----------


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


# ---------- base.py : coerce_enum ----------


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
    # On ne DEVINE pas : une valeur non résolue reste inchangée (l'outil rejettera).
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


# ---------- manage_todos : statuts tolérants + plan préservé ----------


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


# ---------- calc : cas de l'audit ----------


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


# ---------- edit_file : fichier UTF-16 (défaut PowerShell) ----------


def test_edit_file_utf16():
    d = Path(tempfile.mkdtemp())
    f = d / "ps.txt"
    f.write_text("bonjour monde\ndeux\n", encoding="utf-16")
    ef = make_edit_file(str(d))
    out = ef.run({"path": "ps.txt", "old_string": "bonjour", "new_string": "salut"})
    assert "modifié" in out
    # Le fichier reste lisible et l'édition a pris.
    assert "salut monde" in f.read_text(encoding="utf-16")


# ---------- read_file : alias offset/limit ----------


def test_read_file_offset_limit_alias():
    d = Path(tempfile.mkdtemp())
    f = d / "big.txt"
    f.write_text("\n".join(f"L{i}" for i in range(1, 101)), encoding="utf-8")
    rf = make_read_file(str(d), 60000)
    out = rf.run({"path": "big.txt", "offset": 50, "limit": 3})
    assert "L50" in out and "L52" in out
    assert "L1\n" not in out  # ne repart PAS du début


# ---------- Vague 2 : chemins (~/guillemets) ----------


def test_resolve_expanduser():
    root = Path(tempfile.gettempdir())
    got = _resolve_in_root(root, "~/sous/fichier.txt")
    assert "~" not in str(got)
    assert got == (Path.home() / "sous" / "fichier.txt").resolve()


def test_resolve_strip_guillemets():
    root = Path(tempfile.gettempdir())
    got = _resolve_in_root(root, '"C:/Users/x.txt"')
    assert '"' not in str(got)


# ---------- Vague 2 : détection d'hôte réseau (check_page sans schéma) ----------


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


# ---------- Vague 2 : fetch_url domaine nu / schéma ----------


def test_fetch_url_domaine_nu_et_schema(monkeypatch):
    import loom.tools.web as web

    seen = {}
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


# ---------- Vague 3 : finitions (search récursif, read line_count, edit N→) ----------


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


# ---------- Vague 4 (session chasse-invest 14/07) : erreurs web/shell/skill actionnables ----------


def test_fetch_url_remonte_statut_http_et_url_origine(monkeypatch):
    # Un 403 anti-bot doit remonter le STATUT et l'URL D'ORIGINE — pas « page
    # indisponible (hors-ligne) », et jamais l'URL à IP épinglée (session 14/07 :
    # 100+ 403 SeLoger/Leboncoin illisibles, affichés https://52.222.201.14/...).
    import httpx

    import loom.tools.web as web

    def fake_get(url, params=None, headers=None, timeout=None, pin_ip=None, impersonate=None):
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
    # Le chemin web_search (raise_status par défaut) garde le repli silencieux
    # sur snippet : un résultat de recherche qui 403 ne casse pas la recherche.
    import httpx

    import loom.tools.web as web

    def fake_get(url, params=None, headers=None, timeout=None, pin_ip=None, impersonate=None):
        req = httpx.Request("GET", url)
        return httpx.Response(403, request=req)

    monkeypatch.setattr(web, "_resolve_validated", lambda url: (None, "1.2.3.4"))
    monkeypatch.setattr(web, "_http_get", fake_get)
    out = web.fetch_page("https://x.example/a", web.WebSearchConfig(), snippet="RÉSUMÉ")
    assert out == "RÉSUMÉ"


def test_curl_alias_powershell_hint():
    # `curl -s <url>` sous PowerShell 5.1 = alias d'Invoke-WebRequest -> erreur de
    # binding « paramètres obligatoires absents » (18× dans la session du 14/07).
    # La table unix-ismes ne couvrait que « commande inconnue ».
    from loom.tools.shell import _unix_ism_hint

    stderr = (
        "Invoke-WebRequest : Impossible de traiter la commande, car un ou "
        "plusieurs paramètres obligatoires sont absents : Uri."
    )
    hint = _unix_ism_hint("curl -s https://exemple.fr/api", stderr)
    assert "curl.exe" in hint


def test_unix_ism_hint_stderr_francais():
    # Windows FR : « n'est pas reconnu » (pas « not recognized ») — le hint doit
    # quand même se déclencher.
    from loom.tools.shell import _unix_ism_hint

    stderr = (
        "grep : Le terme «grep» n'est pas reconnu comme nom d'applet de commande, "
        "fonction, fichier de script ou programme exécutable."
    )
    hint = _unix_ism_hint("grep -r motif .", stderr)
    assert "Select-String" in hint


def test_use_skill_nom_d_outil_redirige():
    # use_skill('dispatch_agent') (vu en session) : dire que c'est un OUTIL à
    # appeler directement, pas seulement « skill inconnu ».
    from loom.tools.skills import make_use_skill

    us = make_use_skill(lambda: [])
    with pytest.raises(ToolError) as e:
        us.run({"name": "dispatch_agent"})
    msg = str(e.value).lower()
    assert "outil" in msg and "dispatch_agent" in msg


# ---------- Rationalisation 2026-07-15 : fetch_url params ----------


def test_fetch_url_params_objet_encode(monkeypatch):
    # Le modèle passe les query params en OBJET, l'outil encode (les valeurs
    # dict/list sont JSON-sérialisées) — fini l'encodage %7B%22 à la main en
    # PowerShell (44+ échecs dans la session chasse-invest sur l'API Bien'ici).
    import loom.tools.web as web

    seen = {}
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


# ---------- Rationalisation 2026-07-15 : check_page absorbe steps ----------


def test_check_page_accepte_steps(monkeypatch):
    # check_page avec `steps` joue le chemin interactif (ex-check_interactive).
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
    # et check_page expose bien `steps` dans son schéma
    import loom.tools.browser as br

    spec = br.make_check_page(".")
    assert "steps" in spec.parameters["properties"]


# ---------- Rationalisation 2026-07-15 : B2 plugins hors défaut, B4 vision ----------


def test_plugins_hors_du_set_par_defaut():
    # Les 3 outils plugins (opérations de setup, jamais appelés par le modèle en
    # usage réel) ne sont plus dans le set par défaut — restent activables.
    import tomllib

    with open("config/defaults.toml", "rb") as f:
        cfg = tomllib.load(f)
    enabled = cfg["tools"]["enabled"]
    for name in ("list_plugins", "add_marketplace", "install_plugin"):
        assert name not in enabled, name


def test_read_image_absent_sans_vision():
    # Modèle texte pur : read_image n'occupe pas 344 tokens de schéma pour
    # répondre « je ne vois pas ». La décision 2026-07-09 (pas de repli vers un
    # autre modèle) est préservée : l'outil disparaît, il ne bascule pas.
    from loom.tools import build_registry

    reg = build_registry(
        ".", 1000, ["read_file", "read_image"], active_is_vision=False
    )
    names = [t["function"]["name"] for t in reg.openai_tools()]
    assert "read_image" not in names
    assert "read_file" in names
    # Modèle vision : l'outil est là.
    reg2 = build_registry(
        ".", 1000, ["read_file", "read_image"], active_is_vision=True
    )
    assert "read_image" in [t["function"]["name"] for t in reg2.openai_tools()]


def test_x_aliases_pas_serialises_au_modele():
    # Les clés x_* (métadonnées de coercition) restent fonctionnelles côté
    # validation mais ne partent PAS dans le schéma envoyé au modèle (bruit).
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
