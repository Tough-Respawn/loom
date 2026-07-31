from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

from loom.agent.compaction import _msg_chars
from loom.agent.debuglog import _debug, log_event


def _usage_dict(usage: Any) -> dict:
    """Normalise l'usage (tokens réels) renvoyé par le serveur en fin de stream.

    Capture aussi `cached_tokens` (sous-ensemble de prompt_tokens facturé ~÷5 quand le
    provider a un cache hit de préfixe) : c'est LA mesure qui dit si le prompt caching
    marche. Il vit dans `prompt_tokens_details.cached_tokens` (OpenAI/GLM/DeepSeek…) ou
    parfois à plat ; 0 si le provider ne le renvoie pas (aucun hit ou pas de cache)."""
    details = getattr(usage, "prompt_tokens_details", None)
    cached = 0
    if isinstance(details, dict):
        cached = details.get("cached_tokens", 0) or 0
    elif details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    if not cached:
        cached = getattr(usage, "cached_tokens", 0) or 0
    return {
        "completion_tokens": getattr(usage, "completion_tokens", None) or 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) or 0,
        "total_tokens": getattr(usage, "total_tokens", None) or 0,
        "cached_tokens": int(cached or 0),
    }


def _estimate_usage(system_prompt, messages, text, reasoning, tool_calls) -> dict:
    """Usage ESTIMÉ (≈3 car./token) pour un provider qui n'honore pas include_usage : sans
    lui, le compteur ↑/↓ resterait figé à 0 sur une API distante. Approximatif mais bien
    mieux que zéro ; le local (llama.cpp) renvoie toujours l'usage réel -> jamais utilisé là.
    Marqué estimated=True (les compteurs restent fonctionnels, l'UI peut le nuancer)."""
    prompt_chars = len(system_prompt or "") + sum(
        _msg_chars(m.get("content")) for m in messages
    )
    out_chars = (
        len(text or "")
        + len(reasoning or "")
        + sum(len(tc.get("arguments") or "") for tc in (tool_calls or []))
    )
    p, c = prompt_chars // 3, out_chars // 3
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "estimated": True,
    }


def _iter_events(stream: Any) -> Iterator[tuple[str, object]]:
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
    max_tokens: int | None,
    thinking: bool = True,
    tools: list[dict] | None = None,
    native_extras: bool = True,
) -> dict:
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": True,
        # Demande l'usage réel (tokens) dans un chunk final ; ignoré si non supporté.
        "stream_options": {"include_usage": True},
    }
    # max_tokens=None -> on l'OMET : le provider applique SA propre limite. Sert aux modèles
    # DISTANTS sans cap explicite (leur machine n'a pas les contraintes du local 6 Go).
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    # extra_body = paramètres spécifiques au backend llama.cpp LOCAL. On ne les envoie PAS à
    # une API distante (native_extras=False) : une API hébergée OpenAI-compatible rejette
    # souvent un extra_body inconnu (400 sur repeat_penalty / chat_template_kwargs).
    if native_extras:
        # Anti-boucle de dégénérescence : llama.cpp ne pénalise PAS la répétition par défaut,
        # un modèle peut se verrouiller à répéter le même paragraphe à l'infini (observé :
        # « Je vais créer les fichiers… » en boucle, ~30k tokens gaspillés). repeat_penalty /
        # repeat_last_n sont natifs llama.cpp. Valeurs modérées (1.1 sur les 64 derniers
        # tokens) : assez pour casser un cycle, pas pour gêner la répétition LÉGITIME du code.
        extra_body: dict = {"repeat_penalty": 1.1, "repeat_last_n": 64}
        if not thinking:
            # Désactive la réflexion préalable du modèle (chat template). Vérifié sur Gemma :
            # réponse directe au lieu d'un long "Thinking Process". Champ non-standard OpenAI.
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        kwargs["extra_body"] = extra_body
    return kwargs


# Coupe-circuit anti-boucle (filet par-dessus le sampling) : une ligne « longue » répétée
# autant de fois = dégénérescence du décodage. Seuils prudents pour ne pas couper du code
# légitime : on ne compte que les lignes >= _LOOP_MIN_LEN car. (les phrases en boucle font
# 30-40 car. ; les `},` / `</div>` du code sont ignorés), et il en faut _LOOP_THRESHOLD.
_LOOP_THRESHOLD = 10
_LOOP_MIN_LEN = 24


def _scan_repeat(buf: str, counts: dict[str, int]) -> tuple[str, str | None]:
    """Découpe `buf` sur les sauts de ligne, met à jour `counts` pour chaque ligne normalisée
    assez longue, et renvoie (reste_incomplet, ligne_en_boucle|None). La ligne en boucle est
    renvoyée dès qu'elle atteint _LOOP_THRESHOLD occurrences."""
    if "\n" not in buf:
        return buf, None
    *complete, rest = buf.split("\n")
    for ln in complete:
        norm = ln.strip()
        if len(norm) >= _LOOP_MIN_LEN:
            counts[norm] = counts.get(norm, 0) + 1
            if counts[norm] >= _LOOP_THRESHOLD:
                return rest, norm
    return rest, None


def _turn_timing_fields(tim: dict, first_byte_ms: float | None) -> dict:
    """Champs de la ligne `turn.timing` depuis l'extension `timings` de llama-server :
    prefill et génération MESURÉS côté serveur ; `chargement_s` = attente avant le
    1er octet MOINS le prefill (= chargement du modèle / file llama-swap devant).
    Pure (testable) ; les providers distants n'émettent pas `timings` -> pas de ligne."""
    pp_ms = float(tim.get("prompt_ms") or 0.0)
    pp_n = int(tim.get("prompt_n") or 0)
    tg_ms = float(tim.get("predicted_ms") or 0.0)
    tg_n = int(tim.get("predicted_n") or 0)
    out = {
        # Tokens RÉUTILISÉS du cache KV : 0 = prefill entier repayé (préfixe qui a
        # glissé ?), élevé = cache au travail — le diagnostic re-prefill en un chiffre.
        "cache_tok": int(tim.get("cache_n") or 0),
        "prefill_s": round(pp_ms / 1000, 1),
        "prefill_tok": pp_n,
        "prefill_tps": round(pp_n / (pp_ms / 1000), 1) if pp_ms > 0 else 0.0,
        "generation_s": round(tg_ms / 1000, 1),
        "generation_tok": tg_n,
        "generation_tps": round(tg_n / (tg_ms / 1000), 1) if tg_ms > 0 else 0.0,
        "total_s": round((pp_ms + tg_ms) / 1000, 1),
    }
    if first_byte_ms is not None:
        out["chargement_s"] = round(max(0.0, first_byte_ms - pp_ms) / 1000, 1)
    return out


def _iter_turn(stream: Any, collector: dict) -> Iterator[tuple[str, str]]:
    """Yield ('reasoning'|'content', txt) ET accumule les tool_calls streamés.

    Les tool_calls arrivent fragmentés : chaque morceau porte un `.index`, et
    `function.arguments` est une chaîne concaténée morceau par morceau. On les
    regroupe par index, puis on les expose dans `collector["tool_calls"]`.

    Filet anti-boucle : si la sortie (réflexion + contenu) répète la même ligne, on coupe
    le flux et on pose `collector["looped"]` — la boucle d'outils relancera avec un ordre
    ferme d'agir, au lieu de laisser brûler max_tokens sur un cycle.
    """
    acc: dict[int, dict] = {}
    announced: set[int] = set()
    rep_counts: dict[str, int] = {}
    rep_buf = ""
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield ("usage", _usage_dict(usage))
        # Extension llama-server : le chunk FINAL porte `timings` (prompt_ms/predicted_ms
        # mesurés serveur) -> décomposition chiffrée du tour (turn.timing). Les providers
        # distants ne l'émettent pas (champ absent, sans effet).
        _tim = getattr(chunk, "timings", None)
        if isinstance(_tim, dict) and _tim:
            collector["timings"] = _tim
        # Chunk final d'include_usage : `choices == []`, rien d'autre à lire.
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ("reasoning", reasoning)
            rep_buf += reasoning
        content = getattr(delta, "content", None)
        if content:
            yield ("content", content)
            rep_buf += content
        # Détection de dégénérescence (réflexion ET contenu) : on coupe net si une ligne
        # se répète. Pas de tool_call dans ce cas -> on peut sortir sans rien perdre.
        if "looped" not in collector:
            rep_buf, looped = _scan_repeat(rep_buf, rep_counts)
            if looped:
                collector["looped"] = looped[:80]
                break
        for tc in getattr(delta, "tool_calls", None) or []:
            slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None and getattr(fn, "name", None):
                slot["name"] = fn.name
            # Annonce le DÉBUT de l'appel dès id+name connus, AVANT de streamer les
            # arguments : la pastille existe déjà quand ses deltas d'arguments arrivent.
            if tc.index not in announced and slot["id"] and slot["name"]:
                announced.add(tc.index)
                yield ("tool_begin", {"id": slot["id"], "name": slot["name"]})
            # Arguments streamés morceau par morceau (pour write_file le CONTENU du
            # fichier ; pour tout outil ses paramètres). Chaque fragment est un vrai
            # token généré par le modèle -> on le remonte (tool_args) pour que le
            # compteur live avance et que la pastille montre la taille qui grossit, au
            # lieu de rester muette pendant la génération de l'appel.
            if fn is not None and getattr(fn, "arguments", None):
                slot["arguments"] += fn.arguments
                if slot["id"]:
                    yield ("tool_args", {"id": slot["id"], "n": len(fn.arguments)})
        if getattr(choice, "finish_reason", None):
            collector["finish_reason"] = choice.finish_reason
    collector["tool_calls"] = [acc[i] for i in sorted(acc)]


# --- Filet : récupérer les appels d'outil émis en TEXTE ------------------------------
# Certains modèles (surtout après une erreur d'outil) sortent l'appel DANS le texte au
# lieu du canal structuré. Sans filet, tool_calls reste vide -> la boucle s'arrête sur un
# appel "raté". On reconstruit depuis deux formats connus : Hermes/JSON et XML-ish.
_TOOLCALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_FUNC_XML = re.compile(
    r"<function=([\w.\-]+)\s*>(.*?)</function>", re.DOTALL | re.IGNORECASE
)
_PARAM_XML = re.compile(
    r"<parameter=([\w.\-]+)\s*>(.*?)</parameter>", re.DOTALL | re.IGNORECASE
)


def _salvage_tool_calls(text: str, reasoning: str) -> list[dict]:
    """Reconstruit des appels d'outil émis en TEXTE (channel structuré vide). Renvoie la
    MÊME forme que les tool_calls structurés : [{'id','name','arguments'(JSON str)}].
    Vide si rien d'exploitable. L'exécution en aval reste soumise aux permissions."""
    blob = f"{text}\n{reasoning}"
    calls: list[dict] = []

    def _emit(name: str, args) -> None:
        name = (name or "").strip()
        if not name:
            return
        if isinstance(args, dict):
            arguments = json.dumps(args, ensure_ascii=False)
        else:
            arguments = str(args)
        calls.append(
            {"id": f"salvage-{len(calls)}", "name": name, "arguments": arguments}
        )

    for inner in _TOOLCALL_BLOCK.findall(blob):
        inner = inner.strip()
        # Hermes/JSON : {"name": "...", "arguments": {...}}
        try:
            obj = json.loads(inner)
        except (json.JSONDecodeError, ValueError):
            obj = None
        if isinstance(obj, dict) and obj.get("name"):
            _emit(obj["name"], obj.get("arguments", {}))
            continue
        # XML-ish : <function=nom> ... <parameter=clé>valeur</parameter> ...
        m = re.search(r"<function=([\w.\-]+)", inner)
        if m:
            params = {k: v.strip() for k, v in _PARAM_XML.findall(inner)}
            _emit(m.group(1), params)

    # Fallback : <function=...>...</function> hors de tout <tool_call>.
    if not calls:
        for name, body in _FUNC_XML.findall(blob):
            params = {k: v.strip() for k, v in _PARAM_XML.findall(body)}
            _emit(name, params)

    return calls


def _stream_model_turn(
    oai,
    api_model: str,
    kwargs: dict,
    system_prompt: str,
    convo: list[dict],
    collector: dict,
    tools,
    thinking: bool,
    st: dict,
    stream_holder: dict | None = None,
) -> Iterator[tuple[str, object]]:
    """Un appel modèle streamé : relaie les events tels quels, remplit `collector`
    (tool_calls, finish_reason, looped) et pose le texte/raisonnement accumulés
    dans st["text"] / st["reasoning"]. Les APIError/httpx.HTTPError remontent à
    l'appelant (traitées par _handle_stream_api_error)."""
    text = ""
    reasoning = ""
    saw_usage = False
    _first_byte = True
    log_event(
        "turn.request",
        model=api_model,
        msgs=len(convo),
        tools=bool(tools),
        thinking=thinking,
    )
    _t_req = time.monotonic()
    _first_ms: float | None = None
    stream = oai.chat.completions.create(**kwargs)
    # Expose le stream à /cancel : il pourra le fermer pour débloquer une itération
    # figée (modèle distant lent/bloqué). close() lève httpx.ReadError, rattrapée par
    # l'appelant -> le verrou de session est libéré de façon bornée.
    if stream_holder is not None:
        stream_holder["stream"] = stream
    try:
        for kind, chunk in _iter_turn(stream, collector):
            if _first_byte:
                _first_byte = False
                _first_ms = (time.monotonic() - _t_req) * 1000
                log_event("stream.first_byte", ms=round(_first_ms))
            if kind == "content":
                text += chunk
            elif kind == "reasoning":
                reasoning += chunk
            elif kind == "usage":
                saw_usage = True
                log_event(
                    "usage",
                    prompt=chunk.get("prompt_tokens"),
                    completion=chunk.get("completion_tokens"),
                    total=chunk.get("total_tokens"),
                )
            yield (kind, chunk)
    finally:
        if stream_holder is not None:
            stream_holder["stream"] = None
        _close(stream)
    # Décomposition CHIFFRÉE du tour (llama-server local seulement — `timings` absent
    # chez les providers distants) : chargement / prefill / génération, chacun avec sa
    # durée et son débit. Demande user 2026-07-19 : plus d'attente opaque.
    _tim = collector.get("timings")
    if isinstance(_tim, dict) and _tim:
        log_event("turn.timing", **_turn_timing_fields(_tim, _first_ms))
    if not saw_usage:
        # Provider sans include_usage : estimation pour garder ↑/↓ vivants.
        yield (
            "usage",
            _estimate_usage(
                system_prompt,
                convo,
                text,
                reasoning,
                collector["tool_calls"],
            ),
        )
    _debug(
        "REPONSE <- modele",
        {
            "reasoning": reasoning,
            "content": text,
            "tool_calls": collector["tool_calls"],
            "finish_reason": collector["finish_reason"],
        },
    )
    st["text"] = text
    st["reasoning"] = reasoning


def _close(stream) -> None:
    """Coupe la connexion HTTP au modèle (interruption ou fin de tour)."""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
