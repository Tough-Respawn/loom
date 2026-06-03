# tests/test_orchestrator.py
from loom.agents import Agent
from loom.orchestrator import run_pipeline
from loom.tools import ToolRegistry, ToolSpec


def _drive(gen):
    """Consomme le générateur : renvoie (events, AgentRun)."""
    events, run = [], None
    for ev in gen:
        if ev["type"] == "run_done":
            run = ev["run"]
        else:
            events.append(ev)
    return events, run


def _agent(id_, role, model, sys="sys", tools=None):
    return Agent(
        id=id_, role=role, model=model, system_prompt=sys, tools=list(tools or [])
    )


class FakeClient:
    """Rejoue des events par model et enregistre les appels."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.calls = []

    def stream_chat(
        self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
    ):
        self.calls.append((model, list(messages), system_prompt))
        yield from self._scripts.get(model, [("content", f"out-{model}")])


class SeqClient:
    """Rejoue une séquence de réponses, un appel après l'autre (ordre du pipeline)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.models = []

    def stream_chat(
        self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
    ):
        self.models.append(model)
        yield from self._responses.pop(0)


def test_run_pipeline_order_and_model_per_agent(tmp_path):
    agents = [_agent("a", "plan", "m1"), _agent("b", "code", "m2")]
    client = FakeClient(
        {
            "m1": [("reasoning", "r1"), ("content", "PLAN1")],
            "m2": [("content", "CODE2")],
        }
    )
    _events, run = _drive(
        run_pipeline(agents, "ma tâche", client, str(tmp_path), max_tokens=1024)
    )
    assert [s.agent_id for s in run.steps] == ["a", "b"]
    assert [s.model for s in run.steps] == ["m1", "m2"]
    assert run.steps[0].content == "PLAN1"
    assert run.steps[0].reasoning == "r1"
    assert [c[0] for c in client.calls] == ["m1", "m2"]


def test_run_pipeline_propagates_content_not_reasoning(tmp_path):
    agents = [_agent("a", "plan", "m1"), _agent("b", "code", "m2")]
    client = FakeClient(
        {
            "m1": [("reasoning", "SECRET_REASON"), ("content", "VISIBLE_PLAN")],
            "m2": [("content", "CODE2")],
        }
    )
    _drive(run_pipeline(agents, "tâche", client, str(tmp_path), max_tokens=512))
    second_msgs = client.calls[1][1]
    blob = "\n".join(m["content"] for m in second_msgs)
    assert "VISIBLE_PLAN" in blob
    assert "SECRET_REASON" not in blob
    assert second_msgs[0] == {"role": "user", "content": "tâche"}


def test_run_pipeline_emits_events_in_order(tmp_path):
    agents = [_agent("a", "plan", "m1"), _agent("b", "code", "m2")]
    client = FakeClient(
        {"m1": [("reasoning", "r1"), ("content", "P1")], "m2": [("content", "C2")]}
    )
    events, _ = _drive(
        run_pipeline(agents, "tâche", client, str(tmp_path), max_tokens=256)
    )
    types = [e["type"] for e in events]
    assert types[0] == "agent_start"
    assert events[0]["agent"] == "a" and events[0]["model"] == "m1"
    streamed = [e for e in events if e["type"] in ("reasoning", "content")]
    assert all("agent" in e for e in streamed)
    i_a_done = types.index("agent_done")
    i_b_start = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "agent_start" and e["agent"] == "b"
    )
    assert i_a_done < i_b_start


def test_run_pipeline_uses_agent_max_tokens_override(tmp_path):
    captured = {}

    class CapClient:
        def stream_chat(
            self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
        ):
            captured[model] = max_tokens
            yield ("content", "x")

    agents = [
        Agent(id="a", role="plan", model="m1", system_prompt="s", max_tokens=99),
        Agent(id="b", role="code", model="m2", system_prompt="s"),
    ]
    _drive(run_pipeline(agents, "t", CapClient(), str(tmp_path), max_tokens=777))
    assert captured["m1"] == 99
    assert captured["m2"] == 777


def test_run_pipeline_uses_tools_for_agent_with_tools(tmp_path):
    seen = {}

    def factory(active):
        seen["active"] = list(active)
        return ToolRegistry(
            [ToolSpec("write_file", "w", {"type": "object"}, lambda a: "ok")]
        )

    class ToolClient:
        def stream_chat_tools(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            registry=None,
            thinking=True,
            permission=None,
            confirm=None,
        ):
            seen["tools_used"] = True
            yield ("content", "CODE")

        def stream_chat(self, *a, **k):  # ne doit pas être appelé
            seen["plain_used"] = True
            yield ("content", "x")

    agent = _agent("coder", "dev", "m", tools=["write_file"])
    _drive(
        run_pipeline(
            [agent],
            "t",
            ToolClient(),
            str(tmp_path),
            max_tokens=100,
            tool_factory=factory,
        )
    )
    assert seen["active"] == ["write_file"]
    assert seen.get("tools_used") is True
    assert "plain_used" not in seen


def test_run_pipeline_passes_agent_thinking(tmp_path):
    captured = {}

    class ThinkClient:
        def stream_chat(
            self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
        ):
            captured[model] = thinking
            yield ("content", "x")

    agents = [
        Agent(id="a", role="plan", model="m1", system_prompt="s", thinking=True),
        Agent(id="b", role="code", model="m2", system_prompt="s", thinking=False),
    ]
    _drive(run_pipeline(agents, "t", ThinkClient(), str(tmp_path), max_tokens=100))
    assert captured["m1"] is True
    assert captured["m2"] is False


def test_run_pipeline_passes_thinking_to_tools(tmp_path):
    captured = {}

    def factory(active):
        return ToolRegistry(
            [ToolSpec("write_file", "w", {"type": "object"}, lambda a: "ok")]
        )

    class ToolThinkClient:
        def stream_chat_tools(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            registry=None,
            thinking=True,
            permission=None,
            confirm=None,
        ):
            captured["thinking"] = thinking
            yield ("content", "CODE")

    agent = _agent("coder", "dev", "m", tools=["write_file"])
    agent.thinking = False
    _drive(
        run_pipeline(
            [agent],
            "t",
            ToolThinkClient(),
            str(tmp_path),
            max_tokens=100,
            tool_factory=factory,
        )
    )
    assert captured["thinking"] is False


def test_run_pipeline_verify_gate_overrides_reviewer_text(tmp_path):
    """Le reviewer dit 'VERDICT: OK' mais le Vérificateur déterministe bloque ->
    révision forcée + rapport de défauts injecté au développeur (P0.4)."""
    from loom.tools import ToolRegistry, ToolSpec
    from loom.verify import Defect, VerifyReport

    reports = [
        VerifyReport(ok=False, defects=[Defect("game.js:1", "syntax", "boom")]),
        VerifyReport(ok=True),
    ]

    def verifier(paths):
        assert paths == ["game.js"]
        return reports.pop(0)

    seen = {"diag": False}

    class WClient:
        def stream_chat_tools(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            registry=None,
            thinking=True,
            permission=None,
            confirm=None,
        ):
            if any("VÉRIFICATEUR" in m.get("content", "") for m in messages):
                seen["diag"] = True
            yield (
                "tool_result",
                {
                    "name": "write_file",
                    "ok": True,
                    "preview": "écrit",
                    "path": "game.js",
                },
            )
            yield ("content", "fait")

        def stream_chat(
            self, messages, system_prompt, max_tokens=2048, model=None, thinking=True
        ):
            yield ("content", "VERDICT: OK")  # texte OK, mais verify bloque

    def factory(active):
        return ToolRegistry(
            [ToolSpec("write_file", "w", {"type": "object"}, lambda a: "écrit")]
        )

    coder = Agent(
        id="coder", role="code", model="m", system_prompt="s", tools=["write_file"]
    )
    reviewer = Agent(id="reviewer", role="rev", model="m", system_prompt="s")
    events, _run = _drive(
        run_pipeline(
            [coder, reviewer],
            "t",
            WClient(),
            str(tmp_path),
            max_tokens=100,
            tool_factory=factory,
            verifier=verifier,
            max_revisions=2,
        )
    )
    assert any(e["type"] == "revision" for e in events)  # révision malgré 'OK' texte
    verifs = [e for e in events if e["type"] == "verify"]
    assert verifs and verifs[0]["ok"] is False and verifs[-1]["ok"] is True
    assert seen["diag"] is True  # diagnostic injecté au développeur


def test_run_pipeline_review_loop_on_blocking(tmp_path):
    agents = [
        _agent("planner", "plan", "m"),
        _agent("coder", "code", "m"),
        _agent("reviewer", "rev", "m"),
    ]
    responses = [
        [("content", "PLAN")],
        [("content", "CODE v1")],
        [("content", "il manque un test. VERDICT: BLOQUANT")],
        [("content", "CODE v2 corrigé")],
        [("content", "tout est bon. VERDICT: OK")],
    ]
    events, run = _drive(
        run_pipeline(agents, "t", SeqClient(responses), str(tmp_path), max_tokens=100)
    )
    # le développeur + le relecteur repassent une fois
    assert [s.agent_id for s in run.steps] == [
        "planner",
        "coder",
        "reviewer",
        "coder",
        "reviewer",
    ]
    assert any(e["type"] == "revision" for e in events)


def test_run_pipeline_no_loop_when_ok(tmp_path):
    agents = [
        _agent("planner", "plan", "m"),
        _agent("coder", "code", "m"),
        _agent("reviewer", "rev", "m"),
    ]
    responses = [
        [("content", "PLAN")],
        [("content", "CODE")],
        [("content", "parfait. VERDICT: OK")],
    ]
    events, run = _drive(
        run_pipeline(agents, "t", SeqClient(responses), str(tmp_path), max_tokens=100)
    )
    assert [s.agent_id for s in run.steps] == ["planner", "coder", "reviewer"]
    assert not any(e["type"] == "revision" for e in events)


def test_run_build_fanout_plan_generate_verify(tmp_path):
    from loom.orchestrator import run_build
    from loom.verify import VerifyReport

    plan = (
        '{"design": "#board, div.cell", "files": '
        '[{"path": "index.html", "role": "structure"}, '
        '{"path": "app.js", "role": "logique"}]}'
    )

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            if "`index.html`" in p:
                return "<div id='board'></div>"
            return "console.log('app');"

    written = []

    def write(path, content):
        fp = tmp_path / path
        fp.write_text(content, encoding="utf-8")
        written.append(str(fp))
        return str(fp)

    def verifier(paths):
        assert set(paths) == set(written)  # vérifie EXACTEMENT les fichiers écrits
        return VerifyReport(ok=True)

    events, _run = _drive(
        run_build("fais un jeu", C(), model="m", write=write, verifier=verifier)
    )
    types = [e["type"] for e in events]
    assert types[0] == "agent_start"  # planificateur d'abord
    assert "tool_begin" in types and "tool_result" in types
    assert any(e["type"] == "verify" and e["ok"] for e in events)
    assert (tmp_path / "index.html").exists() and (tmp_path / "app.js").exists()
    assert not any(e["type"] == "revision" for e in events)  # OK -> aucune correction


def test_run_build_fix_loop_on_defect(tmp_path):
    from loom.orchestrator import run_build
    from loom.verify import Defect, VerifyReport

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "y"}]}'
    reports = [
        VerifyReport(ok=False, defects=[Defect("app.js:1", "syntax", "boom")]),
        VerifyReport(ok=True),
    ]
    seen = {"fix": False}

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            if "DEFAUTS detectes" in p:  # prompt de correction (boucle fermée)
                seen["fix"] = True
            return "ok();"

    def write(path, content):
        return str(tmp_path / path)

    def verifier(paths):
        return reports.pop(0)

    events, _run = _drive(
        run_build("t", C(), model="m", write=write, verifier=verifier, max_rounds=2)
    )
    assert any(e["type"] == "revision" for e in events)  # correction déclenchée
    verifs = [e for e in events if e["type"] == "verify"]
    assert verifs[0]["ok"] is False and verifs[-1]["ok"] is True
    assert seen["fix"] is True  # le fix a bien reçu le rapport de défauts


def test_run_build_stops_when_defects_do_not_shrink(tmp_path):
    """Anti-divergence : si l'ensemble des défauts ne décroît pas d'un round à l'autre,
    on arrête tôt (un 4B oscille : corrige A, casse B, à len égal indéfiniment)."""
    from loom.orchestrator import run_build
    from loom.verify import Defect, VerifyReport

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "y"}]}'

    class StatefulVerifier:
        """Renvoie TOUJOURS le même défaut non vide : aucun rétrécissement → stop tôt."""

        def __init__(self):
            self.calls = 0

        def __call__(self, paths):
            self.calls += 1
            return VerifyReport(ok=False, defects=[Defect("a.js:1", "syntax", "x")])

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            return "ok();"  # contenu non vide → fichier peuplé, _incomplete() False

    def write(path, content):
        return str(tmp_path / path)

    events, _run = _drive(
        run_build(
            "t", C(), model="m", write=write, verifier=StatefulVerifier(), max_rounds=3
        )
    )
    revisions = [e for e in events if e["type"] == "revision"]
    # Round 1 établit la baseline, round 2 détecte le non-rétrécissement → stop.
    # On s'arrête TÔT (2 < max_rounds=3) au lieu d'osciller jusqu'au bout.
    assert len(revisions) == 2
    assert len(revisions) < 3


def test_run_build_keeps_fixing_while_defects_shrink(tmp_path):
    """Tant que l'ensemble des défauts DÉCROÎT strictement, la boucle continue jusqu'à
    convergence (ok=True)."""
    from loom.orchestrator import run_build
    from loom.verify import Defect, VerifyReport

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "y"}]}'

    class StatefulVerifier:
        """Trajectoire scriptée décroissante : 2 défauts → 1 défaut → ok."""

        def __init__(self):
            self.reports = [
                VerifyReport(
                    ok=False,
                    defects=[
                        Defect("a.js:1", "syntax", "x"),
                        Defect("b.js:2", "syntax", "y"),
                    ],
                ),
                VerifyReport(ok=False, defects=[Defect("a.js:1", "syntax", "x")]),
                VerifyReport(ok=True),
            ]

        def __call__(self, paths):
            return self.reports.pop(0) if self.reports else VerifyReport(ok=True)

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            return "ok();"

    def write(path, content):
        return str(tmp_path / path)

    events, _run = _drive(
        run_build(
            "t", C(), model="m", write=write, verifier=StatefulVerifier(), max_rounds=3
        )
    )
    revisions = [e for e in events if e["type"] == "revision"]
    assert len(revisions) >= 2  # la boucle a continué tant que ça rétrécissait
    verifs = [e for e in events if e["type"] == "verify"]
    assert verifs[-1]["ok"] is True  # terminé sur un succès


def test_run_build_retries_transient_then_succeeds(tmp_path):
    """Une erreur TRANSITOIRE (timeout) sur la génération d'un fichier est ré-essayée
    (borné) ; le fichier finit écrit et le run converge (P3)."""
    import httpx
    from openai import APITimeoutError

    from loom.orchestrator import run_build
    from loom.verify import VerifyReport

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "y"}]}'
    attempts = {"app.js": 0}

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            attempts["app.js"] += 1
            if attempts["app.js"] == 1:  # 1er essai : transitoire -> doit ré-essayer
                raise APITimeoutError(request=httpx.Request("POST", "http://loom"))
            return "ok();"

    written = []

    def write(path, content):
        fp = tmp_path / path
        fp.write_text(content, encoding="utf-8")
        written.append(str(fp))
        return str(fp)

    events, _run = _drive(
        run_build(
            "t",
            C(),
            model="m",
            write=write,
            verifier=lambda paths: VerifyReport(ok=True),
        )
    )
    # le fichier a bien été écrit après ré-essai, et le tool_result final est OK
    assert (tmp_path / "app.js").exists()
    results = [e for e in events if e["type"] == "tool_result"]
    assert any(e["ok"] for e in results)
    assert attempts["app.js"] == 2  # un échec puis un succès


def test_run_build_keeps_last_good_on_fix_failure(tmp_path):
    """Si une CORRECTION échoue (erreur non transitoire), on NE détruit PAS la version
    précédente : last-good conservée, le run ne crashe pas, et l'event le signale (P3)."""
    import httpx
    from openai import APIError

    from loom.orchestrator import run_build
    from loom.verify import Defect, VerifyReport

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "y"}]}'

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            if "DEFAUTS detectes" in p:  # phase de fix : échec dur (overflow simulé)
                raise APIError(
                    "context overflow",
                    request=httpx.Request("POST", "http://loom"),
                    body=None,
                )
            return "good();"  # 1re génération : bonne version

    saved = {}

    def write(path, content):
        saved[path] = content
        return str(tmp_path / path)

    # verify : défaut au 1er passage -> déclenche un round de fix (qui va échouer)
    reports = [VerifyReport(ok=False, defects=[Defect("app.js:1", "syntax", "x")])]

    def verifier(paths):
        return reports.pop(0) if reports else VerifyReport(ok=False)

    events, _run = _drive(
        run_build("t", C(), model="m", write=write, verifier=verifier, max_rounds=1)
    )
    # la bonne version reste sur disque (le fix raté ne l'a pas écrasée)
    assert saved["app.js"].strip() == "good();"
    fix_results = [
        e
        for e in events
        if e["type"] == "tool_result" and not e["ok"] and "app.js" in e.get("path", "")
    ]
    assert fix_results and "version précédente conservée" in fix_results[0]["preview"]


def test_run_build_patch_mode_routes_to_edit_one(tmp_path):
    """Brownfield : un fichier EXISTANT dont le verify passe => mode 'patch' => il est
    ÉDITÉ par edit_one (remplacement ciblé old/new), PAS régénéré intégralement.
    Stratégie : test d'intégration de bout en bout via run_build (le routage est
    interne). On prouve l'édition par (a) le prompt d'édition émis ("Renvoie le JSON")
    et (b) le fichier final identique sauf la ligne ciblée."""
    from loom.orchestrator import run_build
    from loom.verify import VerifyReport

    original = "const a = 1;\nlet x = 1;\nconst b = 2;\n"
    (tmp_path / "app.js").write_text(original, encoding="utf-8")

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "logique"}]}'
    edit_json = '{"old_string": "let x = 1;", "new_string": "let x = 42;"}'
    seen = {"edit": False, "regen": False}

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            if "Renvoie le JSON" in p:  # prompt d'ÉDITION (edit_one)
                seen["edit"] = True
                return edit_json
            seen["regen"] = True  # prompt de génération complète (generate_one)
            return "// REGENERATED\n"

    def write(path, content):
        fp = tmp_path / path
        fp.write_text(content, encoding="utf-8")
        return str(fp)

    events, _run = _drive(
        run_build(
            "t",
            C(),
            model="m",
            write=write,
            workspace=str(tmp_path),
            verifier=lambda paths: VerifyReport(ok=True),
        )
    )

    assert seen["edit"] is True  # edit_one a bien été utilisé pour le fichier existant
    assert seen["regen"] is False  # PAS de régénération complète
    final = (tmp_path / "app.js").read_text(encoding="utf-8")
    # remplacement ciblé : seule la ligne x change, les autres lignes sont intactes
    assert final == "const a = 1;\nlet x = 42;\nconst b = 2;\n"
    assert any(e["type"] == "verify" and e["ok"] for e in events)


def test_run_build_semantic_review_emits_advisory_when_enabled(tmp_path):
    """review_semantic activé + verify vert -> un event content "review" est émis
    avec le défaut sémantique (advisory, non bloquant, aucune correction)."""
    from loom.orchestrator import run_build
    from loom.verify import VerifyReport

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "logique"}]}'
    review = '{"defects": [{"location": "app.js", "evidence": "compteur jamais incrémenté"}]}'

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            if "SÉMANTIQUES" in p:  # prompt de relecture sémantique
                return review
            return "let x = 1;\n"

    def write(path, content):
        return str(tmp_path / path)

    events, _run = _drive(
        run_build(
            "t",
            C(),
            model="m",
            write=write,
            workspace=str(tmp_path),
            verifier=lambda paths: VerifyReport(ok=True),
            semantic_review=True,
        )
    )
    review_contents = [
        e for e in events if e["type"] == "content" and e.get("agent") == "review"
    ]
    assert len(review_contents) == 1
    assert "compteur jamais incrémenté" in review_contents[0]["text"]
    assert any(
        e["type"] == "agent_start" and e.get("agent") == "review" for e in events
    )


def test_run_build_no_semantic_review_by_default(tmp_path):
    """Sans semantic_review (défaut off) -> aucun agent "review" : no-op total."""
    from loom.orchestrator import run_build
    from loom.verify import VerifyReport

    plan = '{"design": "x", "files": [{"path": "app.js", "role": "logique"}]}'

    class C:
        def complete(
            self,
            messages,
            system_prompt,
            max_tokens=2048,
            model=None,
            thinking=False,
            temperature=None,
        ):
            p = messages[0]["content"]
            if "PLAN D'IMPLEMENTATION" in p:
                return plan
            return "let x = 1;\n"

    def write(path, content):
        return str(tmp_path / path)

    events, _run = _drive(
        run_build(
            "t",
            C(),
            model="m",
            write=write,
            workspace=str(tmp_path),
            verifier=lambda paths: VerifyReport(ok=True),
        )
    )
    assert not any(e.get("agent") == "review" for e in events)


def test_run_pipeline_revision_bounded_to_one(tmp_path):
    agents = [
        _agent("planner", "plan", "m"),
        _agent("coder", "code", "m"),
        _agent("reviewer", "rev", "m"),
    ]
    # le relecteur bloque DEUX fois : une seule révision doit avoir lieu (max=1)
    responses = [
        [("content", "PLAN")],
        [("content", "CODE v1")],
        [("content", "VERDICT: BLOQUANT")],
        [("content", "CODE v2")],
        [("content", "VERDICT: BLOQUANT encore")],
    ]
    _events, run = _drive(
        run_pipeline(agents, "t", SeqClient(responses), str(tmp_path), max_tokens=100)
    )
    assert [s.agent_id for s in run.steps] == [
        "planner",
        "coder",
        "reviewer",
        "coder",
        "reviewer",
    ]
