"""Trace de TOUS les appels modèle : rôle, slot, empreintes par section, divergence
de préfixe, premier octet, timings serveur et issue.

Post-mortem du 2026-09-30 (Bonsai 2, modèle hybride, un seul slot) : la maintenance
(ré-amorçage, reflect) était invisible dans le journal de session, et il a fallu
comparer des empreintes à la main pour découvrir que le system prompt changeait en
cours de session. Chaque appel passe désormais par `traced_create`, qui écrit :

- `call.request` : rôle (`purpose`), modèle, slot, nombre de messages, et la
  position de la première différence avec l'appel précédent sur le même slot ;
- `prefix.diff` : le détail de cette différence quand le préfixe a changé
  (quelle section du system prompt, quel message) — sur un modèle hybride, toute
  différence avant le premier checkpoint impose un recalcul complet ;
- `call.end` : issue (ok / annule / erreur), premier octet, durée, et les timings
  llama-server (tokens réutilisés, recalculés, débits).

Le rôle vient d'un contexte (`call_purpose`) posé par l'appelant, sinon du défaut du
point d'appel. Tout est best-effort : la trace ne doit jamais casser un appel.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from typing import Any

from loom.agent.debuglog import log_event, tools_fingerprint

_local = threading.local()
_lock = threading.Lock()
# Dernière liste d'éléments envoyée par (modèle, slot) : le cache llama-server est
# propre à un slot, c'est donc par slot qu'une divergence de préfixe a un coût.
_last_elements: dict[tuple[str, Any], list[tuple[str, str]]] = {}


@contextmanager
def call_purpose(purpose: str):
    """Étiquette les appels modèle faits dans ce bloc (thread courant)."""
    prev = getattr(_local, "purpose", None)
    _local.purpose = purpose
    try:
        yield
    finally:
        _local.purpose = prev


def current_purpose(default: str) -> str:
    return getattr(_local, "purpose", None) or default


def _h(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "replace")).hexdigest()[:8]


def system_sections(text: str) -> list[tuple[str, int, str]]:
    """Découpe le system prompt sur ses titres markdown de niveau 1 (`# …`) :
    [(titre, longueur, empreinte)]. Le texte avant le premier titre est « (début) »."""
    sections: list[tuple[str, list[str]]] = [("(début)", [])]
    for line in text.splitlines(keepends=True):
        if line.startswith("# "):
            sections.append((line[2:].strip()[:60], [line]))
        else:
            sections[-1][1].append(line)
    out = []
    for title, lines in sections:
        body = "".join(lines)
        if body or title != "(début)":
            out.append((title, len(body), _h(body)))
    return out


def _content_str(message: dict) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    content = content or ""
    if message.get("tool_calls"):
        content += json.dumps(message["tool_calls"], ensure_ascii=False, default=str)
    return content


def request_elements(kwargs: dict) -> list[tuple[str, str]]:
    """Éléments du prompt dans l'ordre de rendu du chat template (outils, puis texte
    système par section, puis messages) : [(étiquette, empreinte)]."""
    elements: list[tuple[str, str]] = [
        ("outils", tools_fingerprint(kwargs.get("tools")))
    ]
    messages = kwargs.get("messages") or []
    for i, message in enumerate(messages):
        if i == 0 and message.get("role") == "system":
            for title, length, h in system_sections(_content_str(message)):
                elements.append((f"system « {title} »", f"{length}:{h}"))
            continue
        content = _content_str(message)
        role = message.get("role") or "?"
        elements.append((f"message {i} ({role})", f"{len(content)}:{_h(content)}"))
    return elements


def prefix_diff(
    prev: list[tuple[str, str]] | None, cur: list[tuple[str, str]]
) -> tuple[str, int | None]:
    """Compare deux listes d'éléments : ('premier', None) sans appel précédent,
    ('identique', None), ('ajout', None) si `cur` prolonge `prev`, sinon
    ('diverge', index du premier élément différent)."""
    if prev is None:
        return "premier", None
    for i, (a, b) in enumerate(zip(prev, cur)):
        if a != b:
            return "diverge", i
    if len(cur) == len(prev):
        return "identique", None
    if len(cur) > len(prev):
        return "ajout", None
    return "diverge", len(cur)


class CallTrace:
    """Un appel modèle : `call.request` à la création, `call.end` à la clôture."""

    def __init__(self, purpose: str, kwargs: dict) -> None:
        self.purpose = purpose
        self.model = str(kwargs.get("model") or "")
        extra = kwargs.get("extra_body") or {}
        self.slot = extra.get("id_slot")
        self.t0 = time.monotonic()
        self.first_byte_ms: float | None = None
        self.timings: dict | None = None
        self.usage: dict | None = None
        self.done = False
        try:
            self._log_request(kwargs)
        except Exception:  # noqa: BLE001 - la trace ne casse jamais un appel
            pass

    def _log_request(self, kwargs: dict) -> None:
        elements = request_elements(kwargs)
        # Distant (pas de slot) : le cache du fournisseur est par préfixe, un titre ou un
        # résumé n'évince pas la conversation — ne comparer qu'à la même famille d'appels
        # (fausse alerte « outils » sur GLM-5.3 après un titre, 2026-09-30).
        family = (
            "conv" if self.purpose in ("turn", "prime", "keepwarm") else self.purpose
        )
        key = (self.model, self.slot) if self.slot is not None else (self.model, family)
        with _lock:
            prev = _last_elements.get(key)
            _last_elements[key] = elements
        verdict, idx = prefix_diff(prev, elements)
        fields: dict[str, Any] = {
            "purpose": self.purpose,
            "model": self.model,
            "slot": "auto" if self.slot is None else self.slot,
            "msgs": len(kwargs.get("messages") or []),
            "prefixe": verdict,
        }
        if idx is not None:
            fields["diverge_a"] = (elements[idx] if idx < len(elements) else prev[idx])[
                0
            ]
        log_event("call.request", **fields)
        if verdict == "diverge" and prev is not None:
            before = prev[idx][1] if idx < len(prev) else "(absent)"
            after = elements[idx][1] if idx < len(elements) else "(absent)"
            log_event(
                "prefix.diff",
                level="INFO",
                purpose=self.purpose,
                model=self.model,
                slot=fields["slot"],
                element=fields["diverge_a"],
                index=idx,
                avant=before,
                apres=after,
                elements_communs=idx,
            )

    def on_chunk(self, chunk: Any) -> None:
        if self.first_byte_ms is None:
            self.first_byte_ms = (time.monotonic() - self.t0) * 1000
        tim = getattr(chunk, "timings", None)
        if isinstance(tim, dict) and tim:
            self.timings = tim
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.usage = {
                "prompt": getattr(usage, "prompt_tokens", None),
                "completion": getattr(usage, "completion_tokens", None),
            }

    def end(self, outcome: str, error: str | None = None) -> None:
        if self.done:
            return
        self.done = True
        try:
            from loom.agent.streaming import _turn_timing_fields

            fields: dict[str, Any] = {
                "purpose": self.purpose,
                "model": self.model,
                "slot": "auto" if self.slot is None else self.slot,
                "issue": outcome,
                "duree_s": round(time.monotonic() - self.t0, 1),
            }
            if self.first_byte_ms is not None:
                fields["premier_octet_s"] = round(self.first_byte_ms / 1000, 1)
            if self.timings:
                fields.update(_turn_timing_fields(self.timings, self.first_byte_ms))
            elif self.usage:
                fields["prompt_tok"] = self.usage.get("prompt")
                fields["completion_tok"] = self.usage.get("completion")
            if error:
                fields["erreur"] = error[:200]
            log_event(
                "call.end", level="WARN" if outcome == "erreur" else "DEBUG", **fields
            )
        except Exception:  # noqa: BLE001 - best-effort
            pass


class TracedStream:
    """Enveloppe un stream OpenAI : relaie les chunks à l'identique, note le premier
    octet et les timings, et clôt la trace à la fin, à l'annulation ou sur erreur."""

    def __init__(self, stream: Any, trace: CallTrace) -> None:
        self._stream = stream
        self._trace = trace

    def __iter__(self):
        try:
            for chunk in self._stream:
                self._trace.on_chunk(chunk)
                yield chunk
        except Exception as exc:
            self._trace.end("erreur", str(exc))
            raise
        self._trace.end("ok")

    def close(self) -> None:
        # Fermé avant la fin du flux = annulé (message arrivé, /cancel…).
        self._trace.end("annule")
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


_model_switch_hook = None
_last_model: str | None = None


def set_model_switch_hook(hook) -> None:
    """`hook(model)` est appelé AVANT le premier appel à un modèle différent du
    précédent — llama-swap va (re)lancer son llama-server, qui écrasera son journal :
    c'est le moment de l'archiver."""
    global _model_switch_hook
    _model_switch_hook = hook


def _note_model(model: str) -> None:
    global _last_model
    with _lock:
        switched = bool(model) and model != _last_model
        _last_model = model or _last_model
    if switched and _model_switch_hook is not None:
        try:
            _model_switch_hook(model)
        except Exception:  # noqa: BLE001 - best-effort
            pass


def traced_create(client: Any, default_purpose: str, **kwargs: Any) -> Any:
    """`client.chat.completions.create(**kwargs)` tracé. Rend un `TracedStream` pour un
    appel streamé, la réponse telle quelle sinon."""
    _note_model(str(kwargs.get("model") or ""))
    trace = CallTrace(current_purpose(default_purpose), kwargs)
    try:
        result = client.chat.completions.create(**kwargs)
    except Exception as exc:
        trace.end("erreur", str(exc))
        raise
    if kwargs.get("stream"):
        return TracedStream(result, trace)
    trace.on_chunk(result)
    trace.end("ok")
    return result
