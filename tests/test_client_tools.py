# tests/test_client_tools.py
from types import SimpleNamespace

from loom.client import LoomClient, _iter_turn, build_create_kwargs
from loom.tools import ToolRegistry, ToolSpec


def _delta(content=None, reasoning=None, tool_calls=None):
    return SimpleNamespace(
        content=content, reasoning_content=reasoning, tool_calls=tool_calls
    )


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


def _tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _usage_chunk(completion=0, prompt=0, total=0):
    """Chunk final d'include_usage : choices vide + usage (tokens réels)."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            completion_tokens=completion, prompt_tokens=prompt, total_tokens=total
        ),
    )


class FakeOpenAI:
    """Mock du SDK openai : rend un stream différent à chaque create()."""

    def __init__(self, streams):
        self._streams = list(streams)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._streams.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(streams):
    c = LoomClient(base_url="http://x/v1")
    c._client = FakeOpenAI(streams)
    return c, c._client


def test_build_create_kwargs_adds_tools_when_present():
    kw = build_create_kwargs("m", [], "s", 10, tools=[{"type": "function"}])
    assert kw["tools"] == [{"type": "function"}]


def test_build_create_kwargs_no_tools_key_when_empty():
    kw = build_create_kwargs("m", [], "s", 10)
    assert "tools" not in kw


def test_iter_turn_accumulates_fragmented_tool_calls():
    stream = [
        _chunk(
            _delta(tool_calls=[_tc(0, id="c1", name="read_file", arguments='{"pa')])
        ),
        _chunk(_delta(tool_calls=[_tc(0, arguments='th":"a.md')])),
        _chunk(_delta(tool_calls=[_tc(0, arguments='"}')]), finish_reason="tool_calls"),
    ]
    collector = {}
    events = list(_iter_turn(stream, collector))
    # aucun reasoning/content ; un tool_begin émis dès id+name connus (UI "en cours")
    assert events == [("tool_begin", {"id": "c1", "name": "read_file"})]
    assert collector["tool_calls"] == [
        {"id": "c1", "name": "read_file", "arguments": '{"path":"a.md"}'}
    ]
    assert collector["finish_reason"] == "tool_calls"


def test_iter_turn_yields_reasoning_and_content():
    stream = [
        _chunk(_delta(reasoning="je ")),
        _chunk(_delta(content="Bonjour"), finish_reason="stop"),
    ]
    collector = {}
    events = list(_iter_turn(stream, collector))
    assert events == [("reasoning", "je "), ("content", "Bonjour")]
    assert collector["tool_calls"] == []


def test_complete_non_streaming_returns_message_content():
    captured = {}

    class FakeCreate:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="RESULTAT"))]
            )

    c = LoomClient(base_url="http://x/v1")
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCreate()))
    out = c.complete(
        [{"role": "user", "content": "hi"}], "sys", max_tokens=50, model="m"
    )
    assert out == "RESULTAT"
    assert captured["stream"] is False  # non-streamé
    assert "stream_options" not in captured  # usage retiré hors streaming


def test_build_create_kwargs_requests_usage():
    kw = build_create_kwargs("m", [], "s", 10)
    assert kw["stream_options"] == {"include_usage": True}


def test_iter_turn_emits_usage_and_skips_empty_choices():
    stream = [
        _chunk(_delta(content="Hi"), finish_reason="stop"),
        _usage_chunk(completion=7, prompt=20, total=27),
    ]
    collector = {}
    events = list(_iter_turn(stream, collector))
    assert ("content", "Hi") in events
    usage = [p for k, p in events if k == "usage"]
    assert usage == [{"completion_tokens": 7, "prompt_tokens": 20, "total_tokens": 27}]
    assert collector["tool_calls"] == []  # le chunk d'usage ne casse rien


def test_iter_events_emits_usage():
    from loom.client import _iter_events

    stream = [_chunk(_delta(content="yo")), _usage_chunk(completion=3)]
    events = list(_iter_events(stream))
    assert ("content", "yo") in events
    assert (
        "usage",
        {"completion_tokens": 3, "prompt_tokens": 0, "total_tokens": 0},
    ) in events


def test_stream_chat_tools_executes_tool_and_loops():
    calls_seen = []

    def run(args):
        calls_seen.append(args)
        return f"CONTENU de {args['path']}"

    registry = ToolRegistry([ToolSpec("read_file", "lit", {"type": "object"}, run)])
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(0, id="c1", name="read_file", arguments='{"path":"x.md"}')
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [
        _chunk(_delta(content="Voici ")),
        _chunk(_delta(content="le résumé."), finish_reason="stop"),
    ]
    client, fake = _client([stream1, stream2])

    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "lis x.md"}], "sys", 100, None, registry
        )
    )
    kinds = [e[0] for e in events]
    assert "tool_call" in kinds and "tool_result" in kinds and "content" in kinds
    # l'outil a bien été exécuté avec les bons arguments
    assert calls_seen == [{"path": "x.md"}]
    # le résultat de l'outil est réinjecté (role=tool) au 2e appel du modèle
    second_msgs = fake.calls[1]["messages"]
    assert any(
        m.get("role") == "tool" and "CONTENU de x.md" in m["content"]
        for m in second_msgs
    )
    # le texte final est streamé
    text = "".join(e[1] for e in events if e[0] == "content")
    assert text == "Voici le résumé."


def test_stream_chat_tools_no_tool_calls_is_plain_answer():
    registry = ToolRegistry(
        [ToolSpec("read_file", "lit", {"type": "object"}, lambda a: "x")]
    )
    stream = [_chunk(_delta(content="Réponse directe."), finish_reason="stop")]
    client, fake = _client([stream])
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "salut"}], "s", 50, None, registry
        )
    )
    assert [e for e in events if e[0] == "tool_call"] == []
    assert "".join(e[1] for e in events if e[0] == "content") == "Réponse directe."
    assert len(fake.calls) == 1


def test_stream_chat_tools_ask_approved_executes():
    from loom.permissions import Decision

    ran = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "write_file",
                "w",
                {"type": "object"},
                lambda a: ran.append(a) or "écrit",
            )
        ]
    )
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(
                        0,
                        id="c1",
                        name="write_file",
                        arguments='{"path":"x.txt","content":"y"}',
                    )
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="fait"), finish_reason="stop")]
    client, fake = _client([stream1, stream2])
    seen = []

    def confirm(tid, name, args):
        seen.append((tid, name, args))
        return True

    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "écris"}],
            "s",
            50,
            None,
            registry,
            permission=lambda n, a: Decision("ask"),
            confirm=confirm,
        )
    )
    kinds = [e[0] for e in events]
    assert "tool_request" in kinds  # l'UI a été sollicitée
    assert ran == [{"path": "x.txt", "content": "y"}]  # exécuté car approuvé
    assert seen and seen[0][1] == "write_file"


def test_stream_chat_tools_ask_refused_blocks_execution():
    from loom.permissions import Decision

    ran = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "run_shell", "s", {"type": "object"}, lambda a: ran.append(a) or "ok"
            )
        ]
    )
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(0, id="c1", name="run_shell", arguments='{"command":"ls"}')
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="ok"), finish_reason="stop")]
    client, fake = _client([stream1, stream2])
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "x"}],
            "s",
            50,
            None,
            registry,
            permission=lambda n, a: Decision("ask"),
            confirm=lambda *a: False,
        )
    )
    assert ran == []  # refusé -> jamais exécuté
    results = [p for k, p in events if k == "tool_result"]
    assert results and results[0]["ok"] is False
    # le summary du tool_request expose la commande à l'utilisateur
    reqs = [p for k, p in events if k == "tool_request"]
    assert reqs and reqs[0]["summary"] == "ls"


def test_stream_chat_tools_max_iters_guard():
    registry = ToolRegistry(
        [ToolSpec("loop", "boucle", {"type": "object"}, lambda a: "encore")]
    )

    def always_tool():
        return [
            _chunk(
                _delta(tool_calls=[_tc(0, id="c", name="loop", arguments="{}")]),
                finish_reason="tool_calls",
            )
        ]

    client, fake = _client([always_tool() for _ in range(20)])
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "x"}], "s", 50, None, registry, max_iters=3
        )
    )
    # s'arrête à max_iters, ne consomme pas les 20 streams
    assert len(fake.calls) <= 3
    # un signal de fin est émis (pas de boucle infinie silencieuse)
    assert events  # au moins le tool_call/tool_result des itérations


def test_stream_chat_tools_stops_on_no_progress():
    """Non-progrès : le modèle réémet le MÊME appel tour après tour -> on coupe avant
    d'épuiser le plafond d'itérations (détecteur de répétition)."""
    registry = ToolRegistry(
        [ToolSpec("read_file", "lit", {"type": "object"}, lambda a: "toujours pareil")]
    )

    def same_call():
        return [
            _chunk(
                _delta(
                    tool_calls=[
                        _tc(0, id="c", name="read_file", arguments='{"path":"x.md"}')
                    ]
                ),
                finish_reason="tool_calls",
            )
        ]

    # 10 tours identiques disponibles, mais le détecteur doit couper bien avant.
    client, fake = _client([same_call() for _ in range(10)])
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "lis x.md"}],
            "s",
            50,
            None,
            registry,
            max_iters=10,
            repeat_limit=3,
        )
    )
    # coupé au 3e tour identique : 3 appels modèle, pas 10
    assert len(fake.calls) == 3
    text = "".join(p for k, p in events if k == "content")
    assert "progress" in text.lower()


def test_stream_chat_tools_stops_on_wall_clock():
    """Mur de temps : si le budget est dépassé, la boucle s'arrête avec un message."""
    registry = ToolRegistry(
        [ToolSpec("loop", "x", {"type": "object"}, lambda a: "encore")]
    )
    stream = [
        _chunk(
            _delta(tool_calls=[_tc(0, id="c", name="loop", arguments="{}")]),
            finish_reason="tool_calls",
        )
    ]
    client, fake = _client([stream for _ in range(5)])
    # max_seconds=0 -> dépassé dès la 1re vérif (en tête de boucle) : aucun appel modèle.
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "x"}],
            "s",
            50,
            None,
            registry,
            max_seconds=0.0,
        )
    )
    assert fake.calls == []  # coupé avant le 1er appel
    text = "".join(p for k, p in events if k == "content")
    assert "temps" in text.lower()


def test_stream_chat_tools_default_iter_cap_is_not_eight():
    """Garde-fou non déterministe : le plafond par défaut suit la best practice (>= 15),
    pas l'ancien 8 jugé trop bas."""
    import inspect

    sig = inspect.signature(LoomClient.stream_chat_tools)
    assert sig.parameters["max_iters"].default >= 15


def test_stream_chat_tools_permission_deny_blocks_execution():
    calls_seen = []

    def run(args):
        calls_seen.append(args)
        return "JAMAIS"

    registry = ToolRegistry([ToolSpec("write_file", "écrit", {"type": "object"}, run)])
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(0, id="c1", name="write_file", arguments='{"path":"a.txt"}')
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="ok"), finish_reason="stop")]
    client, fake = _client([stream1, stream2])

    def permission(name, args):
        from loom.permissions import Decision

        return Decision("deny", "interdit")

    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "écris"}],
            "s",
            50,
            None,
            registry,
            permission=permission,
        )
    )
    # l'outil n'est JAMAIS exécuté
    assert calls_seen == []
    # un tool_result de refus est émis (ok=False)
    results = [e[1] for e in events if e[0] == "tool_result"]
    assert results and results[0]["ok"] is False
    assert "refus" in results[0]["preview"].lower()
    # le message de refus est réinjecté comme résultat d'outil au 2e appel
    second_msgs = fake.calls[1]["messages"]
    assert any(
        m.get("role") == "tool" and "refusé" in m["content"] for m in second_msgs
    )


def _api_error(message="Failed to parse tool call arguments as JSON"):
    import httpx
    from openai import APIError

    req = httpx.Request("POST", "http://x/v1/chat/completions")
    return APIError(message, request=req, body=None)


def test_stream_chat_tools_skips_truncated_tool_call_args():
    """Args JSON tronqués (réponse coupée) -> non exécuté, signalé pour réémission."""
    ran = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "write_file", "w", {"type": "object"}, lambda a: ran.append(a) or "ok"
            )
        ]
    )
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(
                        0,
                        id="c1",
                        name="write_file",
                        arguments='{"path":"a.js","conten',
                    )
                ]
            ),
            finish_reason="length",
        )
    ]
    stream2 = [_chunk(_delta(content="ok"), finish_reason="stop")]
    client, fake = _client([stream1, stream2])
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "x"}], "s", 50, None, registry
        )
    )
    assert ran == []  # JAMAIS exécuté avec des args vides
    res = [p for k, p in events if k == "tool_result"]
    assert res and res[0]["ok"] is False and "tronqué" in res[0]["preview"].lower()


def test_stream_chat_tools_serializes_writes_one_per_turn():
    """2 write_file batchés dans UN tour -> seul le 1er s'exécute, le 2e est différé."""
    ran = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "write_file",
                "w",
                {"type": "object"},
                lambda a: ran.append(a["path"]) or "écrit",
            )
        ]
    )
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(
                        0,
                        id="c1",
                        name="write_file",
                        arguments='{"path":"a.js","content":"A"}',
                    ),
                    _tc(
                        1,
                        id="c2",
                        name="write_file",
                        arguments='{"path":"b.js","content":"B"}',
                    ),
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="fini"), finish_reason="stop")]
    client, fake = _client([stream1, stream2])
    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "écris 2 fichiers"}], "s", 100, None, registry
        )
    )
    assert ran == ["a.js"]  # seul le 1er écrit ; le 2e différé
    results = [p for k, p in events if k == "tool_result"]
    assert any("différé" in r["preview"].lower() for r in results)


def test_stream_chat_tools_recovers_from_api_error():
    """Un 500 (tool_call tronqué) ne crashe pas : on relance la passe en corrigeant."""
    registry = ToolRegistry(
        [ToolSpec("write_file", "w", {"type": "object"}, lambda a: "écrit")]
    )
    stream_ok = [_chunk(_delta(content="ok, plus court."), finish_reason="stop")]
    client, fake = _client([_api_error(), stream_ok])

    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "écris un gros fichier"}],
            "s",
            100,
            None,
            registry,
        )
    )
    # erreur signalée de façon récupérable (pas d'exception remontée)
    results = [p for k, p in events if k == "tool_result"]
    assert results and results[0]["ok"] is False
    # la passe a été relancée et la réponse finale est streamée
    assert "".join(p for k, p in events if k == "content").endswith("plus court.")
    assert len(fake.calls) == 2
    # le message correctif (découper en plus petit) a été injecté avant la relance
    retry_msgs = fake.calls[1]["messages"]
    assert any(
        "plus petit" in m.get("content", "").lower()
        or "découpe" in m.get("content", "").lower()
        for m in retry_msgs
    )


def test_stream_chat_tools_gives_up_after_repeated_api_errors():
    """Après N reprises infructueuses, on s'arrête proprement (pas de crash)."""
    registry = ToolRegistry(
        [ToolSpec("write_file", "w", {"type": "object"}, lambda a: "x")]
    )
    client, fake = _client([_api_error() for _ in range(5)])

    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "x"}],
            "s",
            50,
            None,
            registry,
            max_overflow_retries=2,
        )
    )
    text = "".join(p for k, p in events if k == "content")
    assert "interrompue" in text.lower()  # message d'erreur final lisible
    assert len(fake.calls) == 3  # 1 essai initial + 2 reprises


def test_stream_chat_tools_permission_allow_executes():
    calls_seen = []
    registry = ToolRegistry(
        [
            ToolSpec(
                "write_file",
                "écrit",
                {"type": "object"},
                lambda a: calls_seen.append(a) or "écrit ok",
            )
        ]
    )
    stream1 = [
        _chunk(
            _delta(
                tool_calls=[
                    _tc(0, id="c1", name="write_file", arguments='{"path":"a.txt"}')
                ]
            ),
            finish_reason="tool_calls",
        )
    ]
    stream2 = [_chunk(_delta(content="fini"), finish_reason="stop")]
    client, fake = _client([stream1, stream2])

    def permission(name, args):
        from loom.permissions import Decision

        return Decision("allow")

    events = list(
        client.stream_chat_tools(
            [{"role": "user", "content": "écris"}],
            "s",
            50,
            None,
            registry,
            permission=permission,
        )
    )
    assert calls_seen == [{"path": "a.txt"}]
    results = [e[1] for e in events if e[0] == "tool_result"]
    assert results and results[0]["ok"] is True
