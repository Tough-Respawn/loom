# loom/client.py
"""Client modèle : parle à l'endpoint OpenAI-compatible de Loom via le SDK openai."""

from __future__ import annotations

import json
from collections.abc import Iterator

from openai import APIError, OpenAI

# Outils mutateurs à gros contenu : sérialisés (1/tour) pour qu'un batch de N
# write_file ne sature pas max_tokens et ne tronque pas les derniers (P1.1).
_SERIAL_WRITE = frozenset({"write_file", "edit_file"})


def _usage_dict(usage) -> dict:
    """Normalise l'usage (tokens réels) renvoyé par le serveur en fin de stream."""
    return {
        "completion_tokens": getattr(usage, "completion_tokens", None) or 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) or 0,
        "total_tokens": getattr(usage, "total_tokens", None) or 0,
    }


def _iter_events(stream) -> Iterator[tuple[str, object]]:
    """Yield ('reasoning'|'content', txt) par delta, et ('usage', dict) en fin de
    stream si le serveur renvoie l'usage (stream_options.include_usage)."""
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield ("usage", _usage_dict(usage))
        # Le chunk final d'include_usage porte `choices == []` : on le saute.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("reasoning", reasoning)
        content = delta.content
        if content:
            yield ("content", content)


def build_create_kwargs(
    model: str,
    messages: list[dict],
    system_prompt: str,
    max_tokens: int,
    thinking: bool = True,
    tools: list[dict] | None = None,
) -> dict:
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": True,
        "max_tokens": max_tokens,
        # Demande l'usage réel (tokens) dans un chunk final ; ignoré si non supporté.
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = tools
    if not thinking:
        # Désactive la réflexion préalable du modèle (chat template). Vérifié
        # empiriquement sur Gemma : réponse directe au lieu d'un long "Thinking
        # Process". Passe par extra_body car c'est un champ non-standard OpenAI.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


def _iter_turn(stream, collector: dict) -> Iterator[tuple[str, str]]:
    """Yield ('reasoning'|'content', txt) ET accumule les tool_calls streamés.

    Les tool_calls arrivent fragmentés : chaque morceau porte un `.index`, et
    `function.arguments` est une chaîne concaténée morceau par morceau. On les
    regroupe par index, puis on les expose dans `collector["tool_calls"]`.
    """
    acc: dict[int, dict] = {}
    announced: set[int] = set()
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield ("usage", _usage_dict(usage))
        # Chunk final d'include_usage : `choices == []`, rien d'autre à lire.
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("reasoning", reasoning)
        content = getattr(delta, "content", None)
        if content:
            yield ("content", content)
        for tc in getattr(delta, "tool_calls", None) or []:
            slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments
            # Annonce le DÉBUT de l'appel dès id+name connus (avant que les arguments
            # — le contenu du fichier — finissent de streamer) : l'UI montre alors
            # l'écriture EN COURS au lieu d'apparaître seulement une fois terminée.
            if tc.index not in announced and slot["id"] and slot["name"]:
                announced.add(tc.index)
                yield ("tool_begin", {"id": slot["id"], "name": slot["name"]})
        if getattr(choice, "finish_reason", None):
            collector["finish_reason"] = choice.finish_reason
    collector["tool_calls"] = [acc[i] for i in sorted(acc)]


class LoomClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "loom-local",
        model: str = "local",
        timeout: int = 120,
        max_retries: int = 6,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def stream_chat(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 2048,
        model: str | None = None,
        thinking: bool = True,
    ) -> Iterator[tuple[str, str]]:
        """Yield les events (reasoning|content), system prompt injecté en tête."""
        kwargs = build_create_kwargs(
            model or self.model, messages, system_prompt, max_tokens, thinking
        )
        stream = self._client.chat.completions.create(**kwargs)
        try:
            yield from _iter_events(stream)
        finally:
            _close(stream)

    def complete(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 2048,
        model: str | None = None,
        thinking: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Complétion NON-streamée : renvoie le texte final d'un coup.

        Pour la génération PARALLÈLE de fichiers (plusieurs appels concurrents que
        llama-server batche en continu -> GPU saturé). `thinking=False` par défaut :
        on veut du code direct, pas du raisonnement (qui explose le temps/les tokens).
        `temperature` basse (ex 0.2) pour les sorties à FORMAT strict (plan).
        """
        kwargs = build_create_kwargs(
            model or self.model, messages, system_prompt, max_tokens, thinking
        )
        kwargs["stream"] = False
        kwargs.pop("stream_options", None)  # usage non pertinent hors streaming
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def stream_chat_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 2048,
        model: str | None = None,
        registry=None,
        thinking: bool = True,
        max_iters: int = 8,
        permission=None,
        confirm=None,
        max_overflow_retries: int = 2,
    ) -> Iterator[tuple[str, object]]:
        """Boucle tool-use : relaie le texte, exécute les outils, relance le modèle.

        Yield les mêmes tuples que stream_chat — ('reasoning'|'content', str) —
        plus ('tool_call', {id,name,arguments}) et ('tool_result', {id,name,ok,
        preview}). S'arrête quand le modèle ne demande plus d'outil, ou au bout de
        `max_iters` (garde-fou anti-boucle).
        """
        convo = list(messages)
        tools = registry.openai_tools() if registry else None
        overflow_retries = 0
        for _ in range(max_iters):
            kwargs = build_create_kwargs(
                model or self.model,
                convo,
                system_prompt,
                max_tokens,
                thinking,
                tools=tools,
            )
            collector: dict = {"tool_calls": [], "finish_reason": None}
            text = ""
            try:
                stream = self._client.chat.completions.create(**kwargs)
                try:
                    for kind, chunk in _iter_turn(stream, collector):
                        if kind == "content":
                            text += chunk
                        yield (kind, chunk)
                finally:
                    _close(stream)
            except APIError as exc:
                # Le modèle a vraisemblablement dépassé max_tokens AU MILIEU d'un
                # tool_call → JSON tronqué → llama-server renvoie 500. On NE crashe
                # PAS : on demande de découper et on relance la passe (reprise
                # bornée par max_overflow_retries), sinon on s'arrête proprement.
                if overflow_retries >= max_overflow_retries:
                    yield (
                        "content",
                        f"\n[génération interrompue : {str(exc)[:160]}. "
                        "Fichiers déjà écrits conservés.]",
                    )
                    return
                overflow_retries += 1
                note = (
                    "Ta réponse précédente était trop longue et a été tronquée par "
                    "la limite de tokens. Écris des fichiers PLUS PETITS : un seul "
                    "fichier par appel write_file, et découpe tout contenu volumineux "
                    "en plusieurs fichiers/appels successifs. Reprends, en plus court."
                )
                convo.append({"role": "user", "content": note})
                yield (
                    "tool_result",
                    {"name": "(génération)", "ok": False, "preview": note},
                )
                continue

            tool_calls = collector["tool_calls"]
            if not tool_calls:
                return  # réponse finale déjà streamée

            convo.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            wrote_this_turn = False  # P1.1 : un seul write/edit par tour (anti-batch)
            for tc in tool_calls:
                name = tc["name"]
                yield ("tool_call", {"id": tc["id"], "name": name})
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    # Arguments tronqués (réponse coupée par max_tokens). NE PAS exécuter
                    # avec des args vides (erreur trompeuse 'path manquant') : signaler la
                    # troncature pour que le modèle réémette l'appel en plus court.
                    result = (
                        "erreur: arguments tronqués (réponse coupée). "
                        "Réémets cet appel d'outil, en plus court."
                    )
                    convo.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result}
                    )
                    yield (
                        "tool_result",
                        {"id": tc["id"], "name": name, "ok": False, "preview": result},
                    )
                    continue
                # P1.1 : sérialiser les écritures (1 write/edit par tour) -> évite le
                # batch de N gros write_file qui sature max_tokens et tronque.
                if name in _SERIAL_WRITE and wrote_this_turn:
                    result = (
                        "différé : un seul write_file/edit_file par tour. Réémets "
                        "cet appel (seul) au prochain tour."
                    )
                    convo.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result}
                    )
                    yield (
                        "tool_result",
                        {
                            "id": tc["id"],
                            "name": name,
                            "ok": False,
                            "preview": result,
                            "path": args.get("path"),
                        },
                    )
                    continue
                decision = permission(name, args) if permission else None
                action = decision.action if decision else "allow"

                if action == "deny":
                    # Garde-fou non contournable : jamais exécuté.
                    result = f"refusé par la politique de sécurité: {decision.reason}"
                    ok = False
                elif action == "ask":
                    # Confirmation interactive : on signale l'UI puis on ATTEND la
                    # décision via `confirm` (bloquant). Refus par défaut sans confirm.
                    summary = str(args.get("command") or args.get("path") or "")
                    yield (
                        "tool_request",
                        {"id": tc["id"], "name": name, "summary": summary},
                    )
                    if confirm and confirm(tc["id"], name, args):
                        result = (
                            registry.run(name, args)
                            if registry
                            else "erreur: pas d'outils"
                        )
                        ok = not result.startswith("erreur")
                    else:
                        result = "refusé par l'utilisateur"
                        ok = False
                else:  # allow
                    result = (
                        registry.run(name, args) if registry else "erreur: pas d'outils"
                    )
                    ok = not result.startswith("erreur")

                convo.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )
                # Détail dépliable côté UI : pour les écritures, le contenu RÉELLEMENT
                # écrit (et non le message de retour) ; pour edit, le diff old/new ;
                # sinon le résultat complet de l'outil. Borné pour ne pas gonfler le SSE.
                if name == "write_file":
                    detail = args.get("content") or ""
                elif name == "edit_file":
                    detail = f"- {args.get('old_string', '')}\n+ {args.get('new_string', '')}"
                else:
                    detail = result
                yield (
                    "tool_result",
                    {
                        "id": tc["id"],
                        "name": name,
                        "ok": ok,
                        "preview": result[:300],
                        "path": args.get("path"),
                        "detail": detail[:4000] if detail else None,
                    },
                )
                if name in _SERIAL_WRITE:
                    wrote_this_turn = True
        yield ("content", "\n(arrêt : trop d'appels d'outils successifs)")


def _close(stream) -> None:
    """Coupe la connexion HTTP au modèle (interruption ou fin de tour)."""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
