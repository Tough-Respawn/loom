# tests/test_parallel.py
from loom.parallel import (
    FileSpec,
    compute_budget,
    extract_code,
    fix_files,
    generate_files,
    plan_files,
)


class FakeClient:
    """complete() renvoie une réponse scriptée par mot-clé du prompt."""

    def __init__(self, by_keyword=None, default="contenu"):
        self.by_keyword = by_keyword or {}
        self.default = default
        self.calls = []

    def complete(
        self,
        messages,
        system_prompt,
        max_tokens=2048,
        model=None,
        thinking=False,
        temperature=None,
    ):
        prompt = messages[0]["content"]
        self.calls.append({"prompt": prompt, "thinking": thinking, "model": model})
        for kw, resp in self.by_keyword.items():
            if kw in prompt:
                return resp
        return self.default


def test_extract_code_strips_markdown_fence():
    assert extract_code("```js\nconst a = 1;\n```").strip() == "const a = 1;"
    assert extract_code("```\nplain\n```").strip() == "plain"


def test_extract_code_passthrough_without_fence():
    assert extract_code("body{}\n").strip() == "body{}"
    assert extract_code("") == ""


def test_plan_files_parses_json_with_surrounding_text():
    raw = (
        'Voici le plan:\n{"design": "ids: #board", "files": '
        '[{"path": "index.html", "role": "structure"}, '
        '{"path": "app.js", "role": "logique"}]} merci'
    )
    client = FakeClient(default=raw)
    design, specs = plan_files(client, "fais un jeu", model="m")
    assert design == "ids: #board"
    assert [s.path for s in specs] == ["index.html", "app.js"]
    assert client.calls[0]["thinking"] is False  # pas de reasoning au plan


def test_parse_plan_delimited_tolerates_code_snippets():
    from loom.parallel import _parse_plan

    # format délimité : le design contient du code (accolades) qui CASSERAIT du JSON
    raw = (
        "===DESIGN===\n"
        "ETAT: snake=[{x:5,y:5}]; snippet: function tick(){ snake.unshift({x:0,y:0}); }\n"
        "===FILES===\n"
        "- index.html | structure\n"
        "game.js | boucle + etat\n"
        "style.css | styles\n"
    )
    design, specs = _parse_plan(raw)
    assert "snake=[{x:5,y:5}]" in design and "function tick()" in design
    assert [s.path for s in specs] == ["index.html", "game.js", "style.css"]
    assert specs[1].role == "boucle + etat"


def test_parse_plan_regex_fallback_on_malformed():
    from loom.parallel import _parse_plan

    # ni format délimité ni JSON valide -> extraction des noms de fichiers du texte
    raw = "Le projet contient index.html, puis style.css et game.js (boucle {x,y})."
    design, specs = _parse_plan(raw)
    assert [s.path for s in specs] == ["index.html", "style.css", "game.js"]


def test_generate_files_runs_each_spec_with_brut_output():
    specs = [FileSpec("index.html", "structure"), FileSpec("app.js", "logique")]
    # clé = le path ENTRE BACKTICKS (cible unique du prompt ; all_paths est en clair)
    client = FakeClient(
        by_keyword={"`index.html`": "```html\n<h1>hi</h1>\n```", "`app.js`": "let x=1;"}
    )
    out = dict(generate_files(client, "design", specs, model="m"))
    assert "<h1>hi</h1>" in out["index.html"]
    assert "let x=1;" in out["app.js"]
    # thinking OFF pour la génération de code (pas de sur-raisonnement)
    assert all(c["thinking"] is False for c in client.calls)


def test_generate_files_empty_specs():
    assert generate_files(FakeClient(), "d", [], model="m") == []


# --- compute_budget : tailles DÉRIVÉES du budget serveur (P1, anti-overflow) --------


def test_compute_budget_derives_safe_values_for_shared_pool():
    # contexte serveur 24576 partagé entre 4 slots (kv_unified), 5 fichiers à générer.
    workers, gen, cap = compute_budget(24576, 4, 5)
    assert (workers, gen, cap) == (3, 4096, 8192)
    # invariant clé : workers × (reserve + gen) <= 0.9 · context (jamais de débordement)
    assert workers * (2048 + gen) <= int(24576 * 0.9)


def test_compute_budget_single_slot_serializes():
    workers, gen, cap = compute_budget(8192, 1, 2)
    assert (workers, gen, cap) == (1, 4096, 8192)


def test_compute_budget_never_exceeds_parallel_or_files():
    # quelle que soit la combinaison, workers <= n_parallel ET <= n_files, gen >= 1024
    for ctx in (2048, 8192, 24576, 65536):
        for npar in (1, 2, 4, 8):
            for nf in (1, 3, 7):
                w, g, _ = compute_budget(ctx, npar, nf)
                assert 1 <= w <= min(npar, nf)
                assert g >= 1024
                assert w * (2048 + g) <= int(ctx * 0.9) or w == 1


def test_compute_budget_guards_degenerate_inputs():
    # entrées nulles/None -> valeurs plancher sûres, jamais d'exception ni de 0.
    w, g, cap = compute_budget(0, 0, 0)
    assert w == 1 and g >= 1024 and cap > 0


def test_fix_prompt_clips_to_file_char_cap():
    from loom.parallel import _fix_prompt

    spec = FileSpec("app.js", "logique")
    big_design = "D" * 50_000
    current = [("app.js", "T" * 50_000), ("other.js", "S" * 50_000)]
    prompt = _fix_prompt(spec, big_design, current, "VERIFY: bug", file_char_cap=8192)
    # le contexte embarqué est BORNÉ (sinon N fichiers entiers × N requêtes saturent -c)
    assert "…[tronqué]" in prompt
    assert prompt.count("S" * 1000) <= 1  # le voisin est clippé court
    assert len(prompt) < 8192 * 2  # ordre de grandeur du budget, pas 150k


def test_fix_files_injects_current_files_and_defects():
    specs = [FileSpec("app.js", "logique")]
    current = [("app.js", "OLD"), ("index.html", "<html>")]
    client = FakeClient(default="NEW")
    out = fix_files(client, "design", specs, current, "VERIFY: bug X", model="m")
    assert len(out) == 1 and out[0][0] == "app.js" and out[0][1].strip() == "NEW"
    p = client.calls[0]["prompt"]
    assert "OLD" in p and "<html>" in p and "VERIFY: bug X" in p  # contexte + défauts


def test_derive_modes_create_when_file_absent(tmp_path):
    from loom.parallel import FileSpec, derive_modes

    specs = [FileSpec("new.js", "logique")]
    # verifier ne devrait même pas être appelé pour un fichier absent
    planned = derive_modes(specs, str(tmp_path), verifier=lambda paths: 1 / 0)
    assert [(p.spec.path, p.mode) for p in planned] == [("new.js", "create")]


def test_derive_modes_patch_when_existing_and_verify_ok(tmp_path):
    from loom.parallel import FileSpec, derive_modes
    from loom.verify import VerifyReport

    (tmp_path / "ok.js").write_text("let x = 1;\n", encoding="utf-8")
    specs = [FileSpec("ok.js", "logique")]
    planned = derive_modes(
        specs, str(tmp_path), verifier=lambda paths: VerifyReport(ok=True)
    )
    assert planned[0].mode == "patch"


def test_derive_modes_rewrite_when_existing_and_verify_fails(tmp_path):
    from loom.parallel import FileSpec, derive_modes
    from loom.verify import Defect, VerifyReport

    (tmp_path / "broken.js").write_text("let x = ;\n", encoding="utf-8")
    specs = [FileSpec("broken.js", "logique")]
    report = VerifyReport(
        ok=False, defects=[Defect("broken.js:1", "syntax", "Unexpected")]
    )
    planned = derive_modes(specs, str(tmp_path), verifier=lambda paths: report)
    assert planned[0].mode == "rewrite"


def test_derive_modes_patch_when_verifier_returns_none(tmp_path):
    # fichier non-vérifiable (ex. .css) -> verifier renvoie None -> patch (sûr)
    from loom.parallel import FileSpec, derive_modes

    (tmp_path / "style.css").write_text("body{}\n", encoding="utf-8")
    specs = [FileSpec("style.css", "styles")]
    planned = derive_modes(specs, str(tmp_path), verifier=lambda paths: None)
    assert planned[0].mode == "patch"


def test_edit_one_applies_targeted_edit_and_returns_full_content(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "app.js").write_text("let x = 1;\nlet y = 2;\n", encoding="utf-8")
    client = FakeClient(
        default='{"old_string": "let x = 1;", "new_string": "let x = 9;"}'
    )
    path, content = edit_one(
        client,
        "design",
        FileSpec("app.js", "logique"),
        str(tmp_path),
        model="m",
        max_tokens=512,
        file_char_cap=8192,
    )
    assert path == "app.js"
    assert content == "let x = 9;\nlet y = 2;\n"
    assert (tmp_path / "app.js").read_text(
        encoding="utf-8"
    ) == "let x = 9;\nlet y = 2;\n"


def test_edit_one_falls_back_to_rewrite_when_old_string_absent(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "app.js").write_text("let x = 1;\n", encoding="utf-8")
    client = FakeClient(
        by_keyword={
            "JSON": '{"old_string": "ABSENT", "new_string": "z"}',
            "COMPLET et FINAL": "let x = 1;\nlet z = 3;\n",
        }
    )
    path, content = edit_one(
        client,
        "design",
        FileSpec("app.js", "logique"),
        str(tmp_path),
        model="m",
        max_tokens=512,
        file_char_cap=8192,
    )
    assert path == "app.js"
    assert content.strip() == "let x = 1;\nlet z = 3;"


def test_edit_one_falls_back_on_invalid_json(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "app.js").write_text("let x = 1;\n", encoding="utf-8")
    client = FakeClient(
        by_keyword={
            "JSON": "ceci n'est pas du json",
            "COMPLET et FINAL": "REWRITTEN\n",
        }
    )
    _, content = edit_one(
        client,
        "design",
        FileSpec("app.js", "logique"),
        str(tmp_path),
        model="m",
        max_tokens=512,
        file_char_cap=8192,
    )
    assert content.strip() == "REWRITTEN"


def test_edit_one_caps_injected_content_to_half_file_char_cap(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "big.js").write_text("X" * 50_000, encoding="utf-8")
    client = FakeClient(default='{"old_string": "XX", "new_string": "Y"}')
    edit_one(
        client,
        "design",
        FileSpec("big.js", "logique"),
        str(tmp_path),
        model="m",
        max_tokens=512,
        file_char_cap=8192,
    )
    injected = client.calls[0]["prompt"]
    assert "…[tronqué]" in injected
    assert len(injected) < 8192


def test_cap_rewrites_downgrades_large_files_to_patch(tmp_path):
    from loom.parallel import FileSpec, PlannedFile, cap_rewrites

    (tmp_path / "huge.js").write_text("x\n" * 300, encoding="utf-8")  # 300 lignes
    planned = [PlannedFile(FileSpec("huge.js", "logique"), "rewrite")]
    out = cap_rewrites(planned, str(tmp_path), max_lines=200)
    assert out[0].mode == "patch"


def test_cap_rewrites_keeps_small_rewrites(tmp_path):
    from loom.parallel import FileSpec, PlannedFile, cap_rewrites

    (tmp_path / "small.js").write_text("x\n" * 50, encoding="utf-8")
    planned = [PlannedFile(FileSpec("small.js", "logique"), "rewrite")]
    out = cap_rewrites(planned, str(tmp_path), max_lines=200)
    assert out[0].mode == "rewrite"


def test_cap_rewrites_ignores_non_rewrite_modes(tmp_path):
    from loom.parallel import FileSpec, PlannedFile, cap_rewrites

    (tmp_path / "huge.js").write_text("x\n" * 300, encoding="utf-8")
    planned = [
        PlannedFile(FileSpec("huge.js", "logique"), "create"),
        PlannedFile(FileSpec("huge.js", "logique"), "patch"),
    ]
    out = cap_rewrites(planned, str(tmp_path), max_lines=200)
    assert [p.mode for p in out] == ["create", "patch"]


def test_file_prompt_clips_design_to_file_char_cap():
    from loom.parallel import FileSpec, _file_prompt

    spec = FileSpec("app.js", "logique")
    big_design = "D" * 50_000
    prompt = _file_prompt(spec, big_design, ["app.js"], file_char_cap=8192)
    assert "…[tronqué]" in prompt
    assert len(prompt) < 8192 * 2  # borné, pas 50k


def test_file_prompt_no_clip_by_default():
    from loom.parallel import FileSpec, _file_prompt

    spec = FileSpec("app.js", "logique")
    big_design = "D" * 50_000
    prompt = _file_prompt(spec, big_design, ["app.js"])  # pas de cap -> pas de clip
    assert "…[tronqué]" not in prompt
    assert big_design in prompt


def test_generate_one_passes_file_char_cap_to_prompt():
    from loom.parallel import FileSpec, generate_one

    spec = FileSpec("app.js", "logique")
    client = FakeClient(default="let x=1;")
    generate_one(
        client,
        "D" * 50_000,
        spec,
        ["app.js"],
        model="m",
        max_tokens=512,
        file_char_cap=8192,
    )
    assert "…[tronqué]" in client.calls[0]["prompt"]


def test_compute_budget_measured_reserve_shrinks_gen():
    from loom.parallel import compute_budget

    _, gen_small, _ = compute_budget(8192, 1, 1, reserve_prompt_tokens=1024)
    _, gen_big, _ = compute_budget(8192, 1, 1, reserve_prompt_tokens=4096)
    assert (
        gen_big < gen_small
    )  # un prompt mesuré plus gros laisse moins pour la génération


def test_review_semantic_returns_semantic_defects():
    from loom.parallel import review_semantic

    raw = '{"defects": [{"location": "game.js", "evidence": "cliquer une mine ne fait rien"}]}'
    client = FakeClient(default=raw)
    defects = review_semantic(client, "design", [("game.js", "code")], model="m")
    assert len(defects) == 1
    assert defects[0].location == "game.js"
    assert defects[0].kind == "semantic"
    assert "mine" in defects[0].evidence


def test_review_semantic_empty_when_no_defects():
    from loom.parallel import review_semantic

    client = FakeClient(default='{"defects": []}')
    assert review_semantic(client, "design", [("a.js", "x")], model="m") == []


def test_review_semantic_empty_on_invalid_json():
    from loom.parallel import review_semantic

    client = FakeClient(default="pas du json du tout")
    assert review_semantic(client, "design", [("a.js", "x")], model="m") == []


def test_review_semantic_skips_malformed_items():
    from loom.parallel import review_semantic

    # un item sans 'location' est ignoré ; l'autre est gardé
    raw = (
        '{"defects": [{"evidence": "no loc"}, {"location": "b.js", "evidence": "bug"}]}'
    )
    client = FakeClient(default=raw)
    defects = review_semantic(client, "design", [("b.js", "x")], model="m")
    assert [d.location for d in defects] == ["b.js"]
    assert all(d.kind == "semantic" for d in defects)


def test_review_semantic_runs_thinking_off():
    from loom.parallel import review_semantic

    client = FakeClient(default='{"defects": []}')
    review_semantic(client, "design", [("a.js", "x")], model="m")
    assert client.calls[0]["thinking"] is False


# --- best_of : best-of-N en réparation (garde le 1er candidat valide) ----------------


def test_best_of_keeps_first_valid_candidate(tmp_path):
    from loom.parallel import best_of

    calls = {"n": 0}

    def make():
        calls["n"] += 1
        # 1er candidat: Python invalide ; 2e: valide
        return ("m.py", "x = " if calls["n"] == 1 else "x = 1\n")

    path, content = best_of(make, 3)
    assert path == "m.py"
    assert content == "x = 1\n"
    assert calls["n"] == 2  # s'est arrêté dès le 1er valide


def test_best_of_returns_last_when_none_valid():
    from loom.parallel import best_of

    calls = {"n": 0}

    def make():
        calls["n"] += 1
        return ("m.py", "x = (")  # toujours invalide

    path, content = best_of(make, 3)
    assert calls["n"] == 3
    assert content == "x = ("  # le dernier candidat (faute de mieux)


def test_best_of_n1_single_call_even_if_invalid():
    from loom.parallel import best_of

    calls = {"n": 0}

    def make():
        calls["n"] += 1
        return ("m.py", "x = (")

    best_of(make, 1)
    assert calls["n"] == 1


def test_best_of_no_checker_extension_keeps_first(tmp_path):
    from loom.parallel import best_of

    calls = {"n": 0}

    def make():
        calls["n"] += 1
        return ("page.html", "<html>")  # .html n'a pas de checker -> ok -> 1 seul appel

    best_of(make, 3)
    assert calls["n"] == 1
