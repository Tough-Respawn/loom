# Faux client SDK openai + faux registry pour scripter stream_chat_tools sans modèle.
# Formats calqués sur ce que _iter_turn consomme (client.py:382-444) : SimpleNamespace
# suffit, tout est lu en getattr.
from __future__ import annotations

from types import SimpleNamespace as NS


def chunk(content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    """Un chunk streamé façon openai. tool_calls: list[(index, id, name, args_fragment)]."""
    tcs = None
    if tool_calls is not None:
        tcs = [
            NS(index=ix, id=cid, function=NS(name=name, arguments=args))
            for ix, cid, name, args in tool_calls
        ]
    delta = NS(reasoning_content=reasoning, content=content, tool_calls=tcs)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=usage)


def usage_chunk(completion=5, prompt=100, cached=0):
    u = NS(
        completion_tokens=completion,
        prompt_tokens=prompt,
        total_tokens=completion + prompt,
        prompt_tokens_details={"cached_tokens": cached},
    )
    return NS(choices=[], usage=u)


def turn_text(text, finish="stop", with_usage=True):
    """Script d'un tour : texte simple puis stop."""
    out = [chunk(content=text, finish=finish)]
    if with_usage:
        out.append(usage_chunk())
    return out


def turn_tools(calls, with_usage=True):
    """Script d'un tour finissant en tool_calls. calls: list[(id, name, args_json)]."""
    out = [
        chunk(tool_calls=[(i, cid, name, args)])
        for i, (cid, name, args) in enumerate(calls)
    ]
    out.append(chunk(finish="tool_calls"))
    if with_usage:
        out.append(usage_chunk())
    return out


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


class FakeOAI:
    """Rejoue un script par appel : scripts[i] = liste de chunks du i-ème appel API.
    Script épuisé -> RuntimeError (la boucle la classe en api_error, le test échoue
    sur la reason attendue). Enregistre les kwargs de chaque appel pour inspection."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.scripts:
            raise RuntimeError("script épuisé (appel API de trop)")
        return _FakeStream(self.scripts.pop(0))


class FakeRegistry:
    """Registry duck-typé minimal : handlers[name] = callable(args)->str.
    Convention Loom : résultat commençant par 'erreur' => ok=False."""

    def __init__(self, handlers=None):
        self.handlers = handlers or {}
        self.calls: list[tuple[str, dict]] = []

    def openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"outil de test {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in self.handlers
        ]

    def run(self, name, args):
        self.calls.append((name, args))
        h = self.handlers.get(name)
        return h(args) if h else f"erreur: outil inconnu {name}"

    def is_streaming(self, name):
        return False

    def __len__(self):
        # create_app teste `len(registry)` pour décider d'activer les outils.
        return len(self.handlers)


def make_client(scripts, remote=False):
    """LoomClient branché sur un FakeOAI. remote=True monte une route 'remote-x'
    (chemin parallèle possible) ; sinon chemin local (model=None)."""
    from loom.agent.client import LoomClient

    client = LoomClient("http://127.0.0.1:9/v1")
    fake = FakeOAI(scripts)
    if remote:
        client.add_remote_route(
            "remote-x",
            {"base_url": "http://127.0.0.1:9/v1", "api_key": "k", "model": "remote/x"},
        )
        client._routes["remote-x"]["client"] = fake
    else:
        client._client = fake
    return client, fake


def collect(gen):
    """Déroule le générateur d'events ; renvoie (events, done_payload)."""
    events = list(gen)
    assert events, "aucun event yieldé"
    kind, payload = events[-1]
    assert kind == "done", (
        f"dernier event {kind!r}, attendu 'done' — events: {events[-6:]}"
    )
    return events, payload


def kinds(events):
    return [k for k, _ in events]


def only(events, kind):
    return [p for k, p in events if k == kind]
