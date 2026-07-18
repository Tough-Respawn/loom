# loom/agent/client.py
"""Client modèle : parle à l'endpoint OpenAI-compatible de Loom via le SDK openai."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import unicodedata
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from loom.agent.inline_image import (
    image_user_message,
    is_inline_image,
    parse_inline_image,
)

# Écritures à GROS contenu intégral : sérialisées (1/tour) pour qu'un batch de N ne
# sature pas max_tokens et ne tronque pas les derniers (P1.1). On NE sérialise QUE
# celles-ci : les éditions par bloc (edit_file) écrivent peu
# -> pas de risque d'overflow, et les laisser passer ensemble réduit le nombre de tours
# d'un refactor multi-fichiers (cf. plafond max_iters).
_SERIAL_WRITE = frozenset({"write_file", "append_file"})

# Outils d'EXÉCUTION / VÉRIFICATION : relancer LE MÊME appel N fois est légitime (« relance
# jusqu'à 3 runs verts », re-tester après un fix, confirmer une stabilité). Le détecteur de
# non-progrès les EXCLUT donc de sa signature : sinon il coupe un modèle qui fait exactement
# ce qu'on lui demande (observé sur le test LRU). Les vraies boucles à attraper sont les
# re-edit_file / re-write_file / re-read_file à l'identique — elles, restent comptées.
_VERIFY_TOOLS = frozenset({"run_shell", "check_page", "serve_and_check"})


def _safe_args(raw: str) -> str:
    """Renvoie des arguments JSON VALIDES pour l'historique. Si l'appel a été tronqué
    (réponse coupée par max_tokens -> JSON cassé), on remet `{}` : sans ça, le JSON
    invalide reste dans la conversation et CHAQUE requête suivante échoue (500 'parse
    error') -> cascade infinie. Le message d'erreur d'outil signale déjà la troncature."""
    raw = raw or "{}"
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return "{}"


def _classify_api_error(exc: APIError) -> str:
    """Range une erreur du SDK openai en catégorie d'ACTION (pas en code HTTP brut).

    Le piège historique : tout `APIError` était traité comme un overflow (« écris plus
    petit »), y compris un 404 « modèle inconnu » ou un serveur éteint -> diagnostic
    trompeur + retries inutiles. On discrimine :
    - 'timeout' / 'connection' : transport (serveur lent ou pas lancé) -> stop, pas de retry ;
    - 'model_not_found' : 404 (llama-swap n'a pas le modèle demandé) -> stop ;
    - 'other' : erreur cliente 4xx (auth, requête invalide) -> stop, on remonte la cause ;
    - 'overflow' : 5xx OU erreur sans statut (tool_call vraisemblablement tronqué par
      max_tokens) -> seul cas où « écris plus court » + retry borné a un sens.
    """
    if isinstance(exc, APITimeoutError):  # sous-classe d'APIConnectionError -> avant
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return "connection"
    status = getattr(exc, "status_code", None)
    if status == 404:
        return "model_not_found"
    # Débordement de la FENÊTRE DE CONTEXTE (entrée) : llama.cpp/llama-swap le renvoie en
    # 400 « request (N tokens) exceeds the available context size ». C'est RÉCUPÉRABLE (on
    # compacte l'historique et on relance), à ne pas confondre avec une vraie 4xx cliente.
    msg = str(getattr(exc, "message", "") or exc).lower()
    if "context" in msg and (
        "exceed" in msg or "size" in msg or "length" in msg or "too long" in msg
    ):
        return "context_overflow"
    if status is not None and status < 500:
        return "other"
    return "overflow"


def _classify_stream_error(exc: Exception) -> str:
    """Erreur pendant le STREAM : le SDK openai n'enrobe que la phase de requête ; en
    pleine itération, httpx fuit À NU (ReadTimeout vécu en prod : prefill post-compaction
    plus long que le timeout de lecture -> traceback brut au lieu du message propre).
    On range ces exceptions dans les mêmes catégories d'action que les APIError."""
    if isinstance(exc, APIError):
        return _classify_api_error(exc)
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return "connection"


# --- Mode debug : trace l'échange avec le modèle (terminal + debug.log) ----------------
# ACTIF PAR DÉFAUT : le trace par session est le premier outil de diagnostic de Loom, et
# son coût est négligeable (fichiers texte, images masquées). LOOM_DEBUG=0 pour couper.
_B64_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")


def _debug_on() -> bool:
    return os.environ.get("LOOM_DEBUG", "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _trunc(text: str, limit: int) -> str:
    """Tronque + masque les images base64 (illisibles, énormes) pour un log propre."""
    text = _B64_RE.sub("data:image/...;base64,<masque>", text)
    return (
        text
        if len(text) <= limit
        else text[:limit] + f" ...[+{len(text) - limit} car.]"
    )


# Fichier de log debug : permet d'inspecter l'échange modèle APRÈS coup (le terminal
# n'est pas lisible à distance). Écrit en plus de stderr. Cible PARAMÉTRABLE : la web app
# la pointe sur sessions/<id>/debug.log à chaque tour pour un trace PAR SESSION (au même
# titre que session.json). Défaut global tant qu'aucune session n'est désignée.
# client.py vit dans loom/agent/ : remonter de TROIS niveaux (loom/agent -> loom -> repo)
# pour viser l'état machine sous var/logs/ (gitignored).
_DEBUG_LOG_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent / "var" / "logs" / "loom-debug.log"
)
# Chemin du log debug PAR THREAD (donc par génération) : avec plusieurs sessions qui
# génèrent EN PARALLÈLE, un chemin global entremêlerait leurs traces. Chaque requête /chat
# (un thread) pose le sien via set_debug_log_path -> les logs restent séparés par session.
_debug_local = threading.local()


def set_debug_log_path(path) -> None:
    """Redirige le trace debug de CE thread vers `path` (ex. sessions/<id>/debug.log). Le
    dossier est créé à l'écriture. Appelé par la web app au début de chaque tour."""
    _debug_local.path = Path(path)


def _current_debug_log() -> Path:
    return getattr(_debug_local, "path", None) or _DEBUG_LOG_DEFAULT


def _emit(text: str, terminal: bool = True) -> None:
    """Écrit dans le fichier de log, et sur stderr si `terminal` (défaut), sans JAMAIS
    lever (un crash d'encodage ne doit pas casser la génération) : encodage tolérant.
    `terminal=False` = détail réservé au fichier — le terminal reste tenable."""
    if terminal:
        try:
            enc = getattr(sys.stderr, "encoding", None) or "utf-8"
            buf = getattr(sys.stderr, "buffer", None)
            if buf is not None:
                buf.write(text.encode(enc, "replace") + b"\n")
                buf.flush()
            else:
                sys.stderr.write(text + "\n")
                sys.stderr.flush()
        except Exception:  # noqa: BLE001 - le debug est best-effort, jamais bloquant
            pass
    try:
        path = _current_debug_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(text + "\n")
    except Exception:  # noqa: BLE001 - best-effort
        pass


def _debug(label: str, payload: Any, limit: int = 4000, terminal: bool = True) -> None:
    """Imprime un bloc de debug (fichier debug.log, + stderr si `terminal`), no-op si
    désactivé. Labels ASCII volontairement (pas d'accents/flèches) pour rester lisible
    sur tout terminal Windows. Les blocs VOLUMINEUX (dump de requête, slots KV) passent
    terminal=False : le détail vit dans le fichier, le terminal garde les lignes
    compactes de log_event."""
    if not _debug_on():
        return
    body = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
    _emit(f"\n===== [LOOM_DEBUG] {label} =====", terminal)
    _emit(_trunc(body, limit), terminal)


# --- Flux d'événements STRUCTURÉ (façon Claude Code) : une ligne par événement, horodatée --
# `<ISO-Z> [LEVEL] event key=val …`. Complète les blocs _debug (dump complet requête/réponse)
# par une trace fine et lisible : timing (1er byte, durée d'outil), tokens, garde-fous, erreurs.
# Même gate LOOM_DEBUG, même fichier (sessions/<id>/debug.log), best-effort.
def _ts() -> str:
    """Horodatage ISO 8601 UTC à la milliseconde, suffixe Z (comme Claude Code)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _fmt_val(v) -> str:
    """Rend une valeur de champ compacte : chaîne tronquée+échappée (quotée si espace),
    base64 masqué. Nombres/bools tels quels."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        s = _trunc(v.replace("\n", "\\n").replace("\r", ""), 140)
        return f'"{s}"' if (not s or " " in s or "=" in s) else s
    return str(v)


def log_event(event: str, level: str = "DEBUG", **fields) -> None:
    """Écrit une ligne d'événement structurée. No-op si LOOM_DEBUG désactivé ; ne lève jamais."""
    if not _debug_on():
        return
    line = f"{_ts()} [{level}] {event}"
    if fields:
        line += " " + " ".join(f"{k}={_fmt_val(val)}" for k, val in fields.items())
    _emit(line)


def _debug_messages(model: str, messages: list[dict]) -> None:
    """Trace la requête : modèle ciblé + chaque message (rôle + contenu tronqué)."""
    if not _debug_on():
        return
    lines = [f"model={model}  ({len(messages)} messages)"]
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        extra = ""
        if m.get("tool_calls"):
            extra = " +tool_calls=" + json.dumps(m["tool_calls"], ensure_ascii=False)
        # System prompt affiché EN ENTIER (on veut pouvoir l'inspecter : catalogue, dossier
        # de travail, identité SOUL/USER/MEMORY) ; les autres messages (contenus de fichiers,
        # sorties d'outils) restent bornés pour ne pas noyer le log.
        cap = 40000 if m.get("role") == "system" else 1500
        lines.append(f"  [{m.get('role')}] {_trunc((content or '') + extra, cap)}")
    # Dump complet RÉSERVÉ AU FICHIER : dans le terminal, la ligne compacte
    # `turn.request model=… msgs=…` (log_event) suffit — le system prompt entier
    # à chaque tour rendait la console intenable.
    _debug("REQUETE -> modele", "\n".join(lines), limit=60000, terminal=False)


def _sub_activity_line(kind: str, payload) -> str:
    """Rend un event de sous-agent en ligne lisible pour le flux live de sa pastille.
    Le détail (synthèse) reste dans le résultat final ; ici on veut juste VOIR l'ouvrier
    agir : quel outil il appelle, s'il réussit, et sa synthèse au fil de l'eau."""
    if kind == "tool_call":
        return f"\n→ {payload.get('name', 'outil')}"
    if kind == "tool_result":
        mark = "✓" if payload.get("ok") else "✕"
        loc = f" {payload['path']}" if payload.get("path") else ""
        head = (payload.get("preview") or "").split("\n")[0][:80]
        return f"  {mark}{loc} {head}\n"
    if kind == "content":
        return payload if isinstance(payload, str) else ""
    return ""


def _stream_tool_events(registry, tc_id: str, name: str, args: dict):
    """Exécute un outil STREAMANT (dispatch_agent) en relayant son activité live.

    Générateur partagé par les chemins parallèle et séquentiel : yield les events
    à relayer (tool_stream, sub_usage) puis, en dernier, ("__result__", synthèse).
    L'appelant route les events vers son canal (yield direct ou queue de thread)."""
    parts: list[str] = []
    for sub_kind, sub_payload in registry.run_stream(name, args):
        line = _sub_activity_line(sub_kind, sub_payload)
        if line:
            yield ("tool_stream", {"id": tc_id, "text": line})
        if sub_kind == "content" and isinstance(sub_payload, str):
            parts.append(sub_payload)
        elif sub_kind == "usage":  # conso du sous-agent -> totaux de session
            yield ("sub_usage", sub_payload)
    yield ("__result__", "".join(parts).strip() or "(le sous-agent n'a rien renvoyé)")


def _tool_result_payload(
    tc_id: str, name: str, ok: bool, tool_content, args: dict
) -> dict:
    """Payload de l'event tool_result (pastille UI), UNIQUE pour les chemins parallèle
    et séquentiel. detail/in_full sont SPÉCIALISÉS pour les outils d'écriture/shell —
    jamais parallel-safe, donc le chemin parallèle retombe toujours sur le cas générique.
    - preview (1 ligne) : état replié ; detail conservé pour rétro-compat ;
    - in_full = ce que l'outil a REÇU (commande, contenu écrit, diff, chemin/args) ;
    - out_full = ce qu'il a RENVOYÉ. Le tout borné."""
    if name == "write_file":
        detail = args.get("content") or ""
    elif name == "edit_file":
        detail = f"- {args.get('old_string', '')}\n+ {args.get('new_string', '')}"
    else:
        detail = tool_content
    if name == "run_shell":
        in_full = args.get("command") or ""
    elif name in ("write_file", "append_file"):
        in_full = f"{args.get('path', '')}\n{args.get('content', '')}"
    elif name == "edit_file":
        in_full = (
            f"{args.get('path', '')}\n- {args.get('old_string', '')}"
            f"\n+ {args.get('new_string', '')}"
        )
    else:
        in_full = args.get("path") or json.dumps(args, ensure_ascii=False)
    return {
        "id": tc_id,
        "name": name,
        "ok": ok,
        "preview": str(tool_content)[:300],
        "path": args.get("path"),
        # Commande réellement lancée par run_shell : pour la VOIR dans la pastille
        # (sinon on ne voit que le résultat, pas ce qui a tourné).
        "cmd": args.get("command"),
        "detail": detail[:4000] if detail else None,
        "in_full": str(in_full)[:8000],
        "out_full": str(tool_content)[:8000],
    }


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


# --- Microcompact INTERNE à la boucle d'outils ---------------------------------------
# Sur une chaîne longue (refactor multi-fichiers, exploration), `convo` accumule TOUS les
# messages role:tool et finit par approcher la fenêtre du modèle -> overflow. CC règle ça
# par un "microcompact" SANS LLM : on vide le CONTENU des plus vieux résultats d'outils
# (gros stdout, gros read périmés) en gardant les N derniers + toute la STRUCTURE (chaque
# tool_call garde son tool_result). Non destructif pour le raisonnement (user/assistant
# intacts) et bien moins risqué qu'un résumé généré par un 4B.
# Le stub NE dit PAS « relis le fichier » : ça poussait le modèle à re-lire en boucle ce
# qui venait d'être purgé (thrash observé en session). Il l'oriente vers write_note —
# consigner l'essentiel AVANT que le résultat ne soit effacé, puis relire sa note (durable)
# au lieu du fichier entier.
_CLEARED_TOOL = (
    "[résultat d'outil ancien allégé pour tenir dans le contexte — son contenu n'est plus "
    "ici. NE refais PAS le travail déjà fait (ne re-liste pas, ne re-lis pas en boucle) : "
    "reprends depuis tes NOTES (read_note) et ton PLAN (manage_todos). Et désormais, dès "
    "qu'un résultat te servira à une étape ULTÉRIEURE, consigne-le avec write_note PENDANT "
    "que tu l'as encore.]"
)

# Sur-vérification compulsive (observée en éval : 14 check_interactive verts d'affilée
# sur une page qui marchait -> 20 tours, 230k tokens, arrêt max_iters). Les checks sont
# volontairement EXCLUS du détecteur de non-progrès (re-prouver est légitime, cf.
# _VERIFY_TOOLS) : le remède n'est donc PAS une coupe, c'est un SIGNAL dans le résultat.
_BROWSER_CHECKS = frozenset({"check_page", "serve_and_check"})
_STATE_CHANGERS = frozenset(
    {"write_file", "append_file", "edit_file", "run_shell", "format_code"}
)
_VERIFY_STREAK_NOTE = 3  # nb de checks verts consécutifs avant d'annoter le résultat


def _verify_streak_update(name: str, ok: bool, streak: int) -> int:
    """Nouveau compteur de checks navigateur VERTS consécutifs : un outil qui change
    l'état (écriture, shell) remet à zéro (la preuve précédente est périmée), un check
    raté aussi (échec = information nouvelle) ; les lectures n'y touchent pas."""
    if name in _STATE_CHANGERS:
        return 0
    if name in _BROWSER_CHECKS:
        return streak + 1 if ok else 0
    return streak


# Note de RECENTRAGE après un force-fit : un historique tronqué mais encore répétitif
# induit l'IMITATION (observé en éval : le modèle a « continué » la série de vieux tours
# archivés au lieu d'exécuter la tâche). La note casse le motif et repointe la demande.
# Préfixe '[harnais' = reconnue par _force_fit (jamais prise pour la tâche courante).
# Reformulée le 2026-07-10 : l'ancienne version ordonnait « reprends la DERNIÈRE
# demande telle quelle et exécute-la » — injectée EN PLEIN TOUR (le force-fit
# préventif tourne avant chaque appel), elle pouvait faire repartir de zéro un
# modèle qui avançait bien, et insinuait une dérive là où la troncature n'est
# qu'une opération de routine. On garde UNIQUEMENT l'anti-imitation (le cœur
# validé en éval, cas context_squeeze) : informatif, jamais directif.
_REFOCUS_NOTE = (
    "[harnais : des tours anciens ci-dessus ont été TRONQUÉS pour tenir dans la "
    "fenêtre — opération de routine, rien d'anormal. Ce contenu tronqué est du "
    "contexte ARCHIVÉ : ne l'imite pas, ne le continue pas. Ta tâche en cours et "
    "la dernière demande utilisateur restent INCHANGÉES : poursuis ton travail "
    "normalement, sans repartir de zéro.]"
)


# --- Anti « parle sans agir » --------------------------------------------------------
# Échec central d'un petit modèle : il ÉCRIT l'intention (« je vais lire X ») ou AFFIRME
# le résultat (« j'ai créé le fichier ») SANS émettre l'appel d'outil. Comme un tour sans
# tool_call termine la boucle, la tâche s'arrête inachevée (ou sur une affirmation fausse).
# On détecte ce cas au stop et on RELANCE le modèle pour qu'il exécute réellement (nudge
# borné). Ce n'est pas un orchestrateur : on ne décide pas QUOI faire, on force juste le
# passage de la parole à l'acte. dispatch_agent reste l'autre garde-fou (exécution réelle
# par un sous-agent). On exclut les verbes de PAROLE (résumer/expliquer) qui ne sont pas
# des actions outillées, pour ne pas harceler un vrai message final. Marqueurs SANS accents
# ni apostrophe courbe : la comparaison normalise le texte (un modèle quantifié laisse
# parfois tomber les accents -> on veut quand même détecter).
_ACT_INTENT = (
    "je vais ",
    "laisse-moi ",
    "permets-moi ",
    "je commence par ",
    "je dois d'abord ",
    "il faut que je ",
    "je m'occupe ",
    "commencons par ",
    "je vais maintenant ",
    "je vais d'abord ",
)
_ACT_CLAIM = (
    "j'ai cree",
    "j'ai ecrit",
    "j'ai modifie",
    "j'ai ajoute",
    "j'ai lance",
    "j'ai execute",
    "j'ai teste",
    "j'ai corrige",
    "j'ai supprime",
    "le test passe",
)
_TALK_VERBS = ("resum", "expliqu", "montr", "repond", "decri", "te dire", "vous dire")


def _norm(text: str) -> str:
    """Minuscule, sans accents, apostrophe courbe -> droite : pour comparer aux marqueurs
    quel que soit l'encodage exact produit par le modèle."""
    text = text.lower().replace("’", "'")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _intends_to_act(text: str, executed: bool) -> bool:
    """Vrai si `text` ANNONCE une prochaine action outillée non appelée, ou AFFIRME avoir
    agi alors qu'AUCUNE exécution réelle n'a eu lieu ce tour-ci (confabulation)."""
    low = _norm(text.strip())
    if not low:
        return False
    for marker in _ACT_INTENT:
        pos = low.find(marker)
        if pos != -1:
            tail = low[pos : pos + 60]  # « je vais RESUMER » = parole, pas action outil
            if not any(v in tail for v in _TALK_VERBS):
                return True
    if not executed and any(m in low for m in _ACT_CLAIM):
        return True
    return False


# --- Audit de claim déterministe (anti-confabulation, couches A et B) -----------------
# Le modèle décide TOUT ; on l'empêche seulement de PRÉTENDRE un résultat qu'il n'a pas
# produit. Garde de vérité, pas orchestrateur. Deux vérifs déterministes au stop :
#   A. il revendique un FICHIER (créé/contient) qui n'existe pas -> artefact inventé ;
#   B. il rapporte un RÉSULTAT D'EXÉCUTION sans avoir lancé run_shell ni dispatch_agent.
_WRITE_TOOLS = frozenset({"write_file", "append_file", "edit_file"})
# Outils dont un ECHEC = signal de BUG (execution / verification), par opposition aux erreurs
# d'usage d'outil (ligne hors limite, etc.). Une cascade ici impose la methode debug.
_BUG_SIGNAL_TOOLS = frozenset({"run_shell", "check_page", "format_code"})
# Outils PARALLEL-SAFE : lecture seule / indépendants, sans effet de bord, sans confirmation,
# sans ordre entre eux. Pour un modèle DISTANT, un tour n'appelant QUE ceux-ci s'exécute en
# CONCURRENCE (règle Loom : local = inline/1 slot ; distant = on exploite le parallélisme).
# Exclus : écritures, run_shell, todos, notes-écriture, vérifs (check_*), use_skill, install,
# et read_image (accusé + message user multimodal différé -> ordre sensible, on le sérialise).
_PARALLEL_SAFE = frozenset(
    {
        "dispatch_agent",
        "find_files",
        "search_text",
        "list_dir",
        "read_file",
        "web_search",
        "fetch_url",
        "recall",
        "read_note",
        "list_plugins",
    }
)
_DEBUG_FORCE = (
    "STOP — plusieurs erreurs s'enchainent et corriger au coup par coup ne regle pas la "
    "cause. Methode debug OBLIGATOIRE maintenant, ne patche plus au hasard :\n"
    "1. REPRODUIRE : relance la commande/page qui echoue, lis l'erreur EN ENTIER (fichier, ligne, code).\n"
    "2. LOCALISER avec les outils : read_file l'etat reel, search_text la definition, run_shell/check_page "
    "pour VOIR — remonte jusqu'a la SOURCE de la mauvaise valeur, pas la ou elle explose.\n"
    "3. CAUSE RACINE unique : formule UNE hypothese verifiable (X echoue PARCE QUE...).\n"
    "4. UN seul changement minimal a la cause, puis RELANCE la repro et CONSTATE la preuve.\n"
    "Un fix qui ne regle pas -> retour a LOCALISER, jamais un autre patch au hasard. Si chaque "
    "fix en revele un autre, la base est pourrie : reecris proprement le fichier en cause."
)
_EXEC_CLAIM = (
    "a affiche",
    "a retourne",
    "resultat :",
    "sortie :",
    "exit=",
    "j'ai lance",
    "j'ai execute",
    "j'ai teste",
    "le test passe",
    "-> success",
    "preuve :",
    "preuve de sortie",
    "j'ai simule",  # aveu typique de confabulation
)
# Chemin ABSOLU de fichier (avec extension), Windows ou POSIX, cité dans le texte.
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[\w./\\-]+\.[A-Za-z0-9]{1,6}")
_ARTIFACT_VERBS = ("cree", "ecrit", "genere", "produit", "contient", "preuve")


def _claims_execution(text: str) -> bool:
    """Vrai si `text` rapporte une sortie / un résultat d'exécution."""
    low = _norm(text)
    return any(m in low for m in _EXEC_CLAIM)


def _claims_missing_artifact(text: str, files_written: set) -> str | None:
    """Renvoie le 1er chemin ABSOLU revendiqué (créé/contient) qui n'existe PAS et n'a pas
    été écrit ce tour — artefact inventé. Sinon None. Chemins absolus uniquement (vérif
    fiable sans connaître le workspace)."""
    low = _norm(text)
    for match in _PATH_RE.finditer(text):
        path = match.group(0).strip("`\"'")
        idx = low.find(_norm(path))
        if idx == -1:
            continue
        window = low[max(0, idx - 60) : idx + len(path) + 60]
        if not any(v in window for v in _ARTIFACT_VERBS):
            continue
        if path in files_written:
            continue
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
        except OSError:
            pass
        return path
    return None


def _msg_chars(content) -> int:
    """Taille approx. d'un contenu de message (str ou liste de parts multimodales)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    return 0


def _microcompact_tools(
    convo: list[dict], keep_recent_tools: int, min_clear_chars: int = 400
) -> int:
    """Vide le CONTENU des plus vieux messages role:tool (garde les `keep_recent_tools`
    derniers intacts), en place. Renvoie le nb de messages allégés.

    SÉLECTIF : un petit résultat (accusé « modifié : x.py », code retour, message
    d'erreur court) est une PREUVE dense — le vider ne libère presque rien (le
    placeholder fait ~370 car.) et détruit de l'information que le modèle re-paierait
    en re-lecture. On ne vide que les GROS résultats (dumps de fichiers, stdouts
    longs) : le volumineux part, les preuves techniques restent."""
    idx = [i for i, m in enumerate(convo) if m.get("role") == "tool"]
    older = idx[:-keep_recent_tools] if keep_recent_tools else idx
    n = 0
    for i in older:
        content = convo[i].get("content")
        if content == _CLEARED_TOOL:
            continue
        if isinstance(content, str) and len(content) <= min_clear_chars:
            continue
        convo[i] = {**convo[i], "content": _CLEARED_TOOL}
        n += 1
    return n


# --- Compaction par RÉSUMÉ (dernier étage, avec LLM) ---------------------------------
# Le microcompact ne touche QUE les résultats d'outils. Quand ce sont les TOURS du modèle
# (assistant/reasoning, contenu écrit inline) qui saturent, vider les tool results ne
# suffit plus -> autrefois on abandonnait. Ici on RÉSUME les vieux tours en un bloc dense
# et on poursuit. Le résumé est en ANGLAIS TÉLÉGRAPHIQUE : le plus dense en tokens à
# fidélité égale (le français coûte ~15-20 % de tokens en plus), et il colle aux
# identifiants de code déjà anglais. On préserve les littéraux (chemins, noms, valeurs).
_SUMMARY_MARKER = "[SESSION SUMMARY — older turns compacted to fit the context window]"

_SUMMARY_SYSTEM = (
    "You compact a coding agent's own conversation so it fits the model's context "
    "window. Output a DENSE summary in terse English bullet points — no prose, no "
    "preamble, never restate these instructions. Preserve VERBATIM every file path, "
    "identifier, function/variable name, shell command, URL and numeric value. Capture, "
    "in order: GOAL, what was DONE, what was LEARNED/DECIDED — including approaches "
    "REJECTED and why (so they are not retried) — ERRORS hit with their exact messages, "
    "current STATE of the code, and what remains TODO. Stay faithful; when unsure, keep "
    "the literal token."
)


def _flatten_msg(content) -> str:
    """Texte brut d'un contenu de message (str ou liste de parts multimodales)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(str(p.get("text") or p.get("content") or ""))
            else:
                out.append(str(p))
        return "\n".join(x for x in out if x)
    return "" if content is None else str(content)


def _flatten_for_summary(old: list[dict], budget_chars: int) -> str:
    """Aplatit les vieux tours en UN texte borné à envoyer au résumeur. Le convo déborde
    déjà la fenêtre -> on ne peut pas tout renvoyer : on garde le 1er message (le BUT) puis
    on remplit depuis la FIN (le plus récent = le plus utile pour l'état courant) jusqu'au
    budget ; le milieu ancien saute. '' si rien à aplatir."""
    if not old:
        return ""
    head = f"[{old[0].get('role', '?')}] {_flatten_msg(old[0].get('content'))}"
    tail_parts: list[str] = []
    used = len(head)
    for m in reversed(old[1:]):
        seg = f"[{m.get('role', '?')}] {_flatten_msg(m.get('content'))}"
        if used + len(seg) + 40 > budget_chars:
            tail_parts.append("[...older turns elided...]")
            break
        tail_parts.append(seg)
        used += len(seg) + 2
    return head + "\n\n" + "\n\n".join(reversed(tail_parts))


def _drop_orphan_tools(convo: list[dict]) -> None:
    """Retire tout message role:tool ORPHELIN — dont le plus proche message non-tool qui
    précède n'est pas un assistant porteur de tool_calls. Un tool orphelin (son appel a
    été droppé par le force-fit) fait échouer le rendu du chat template (400 llama.cpp) ;
    même règle que summarize_old_turns (« ne pas orpheliner un résultat d'outil »)."""
    i = 0
    while i < len(convo):
        if convo[i].get("role") == "tool":
            j = i - 1
            while j >= 0 and convo[j].get("role") == "tool":
                j -= 1
            anchored = (
                j >= 0
                and convo[j].get("role") == "assistant"
                and convo[j].get("tool_calls")
            )
            if not anchored:
                convo.pop(i)
                continue
        i += 1


def _force_fit(convo: list[dict], system_prompt: str, budget_chars: int) -> bool:
    """Réduction DÉTERMINISTE de dernier recours (AUCUN LLM) : clippe les contenus les plus
    gros — et, à défaut, drope les messages les plus anciens (garde toujours les 2 derniers)
    — jusqu'à passer sous `budget_chars`. GARANTIT un fit tant que `system_prompt` seul tient
    dans le budget. Sert à ne JAMAIS s'arrêter pour saturation : on tronque plutôt qu'on
    abandonne. Mute `convo` en place ; renvoie True si on tient le budget après réduction."""

    def _total() -> int:
        return len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)

    _CLIP_FLOOR = 200

    def _longest(skip: int) -> tuple[int, int]:
        """(index, taille) du message au contenu le plus long, hors `skip`."""
        idx, longest = -1, 0
        for i, m in enumerate(convo):
            if i == skip:
                continue
            c = m.get("content")
            n = len(c) if isinstance(c, str) else _msg_chars(c)
            if n > longest:
                longest, idx = n, i
        return idx, longest

    guard = 0
    while _total() > budget_chars and guard < 5000:
        guard += 1
        # SÉLECTIF : la TÂCHE COURANTE (dernier message user) est exemptée du clip tant
        # qu'il reste autre chose à réduire — c'est elle que le modèle doit exécuter
        # après la coupe (une grosse spec collée par l'utilisateur EST la tâche).
        task_idx = max(
            (
                i
                for i, m in enumerate(convo)
                if m.get("role") == "user"
                and not str(m.get("content", "")).startswith("[harnais")
            ),
            default=-1,
        )
        # Un message est « épuisé » sous _CLIP_FLOOR : le clip tête+queue produit
        # ~120 car. + marqueur (~50) — re-clipper sous ce plancher ne décroît PLUS
        # (boucle sans progrès, vue au self-test). > _CLIP_FLOOR garantit la
        # décroissance stricte : len/2 + marqueur < len dès que len > 2×marqueur.
        idx, longest = _longest(skip=task_idx)
        if longest <= _CLIP_FLOOR:  # plus rien d'autre : la tâche en dernier recours
            idx, longest = _longest(skip=-1)
        if idx >= 0 and longest > _CLIP_FLOOR:
            c = convo[idx].get("content")
            if isinstance(c, str):
                # Clip TÊTE + QUEUE (pas tête seule) : la fin d'un long contenu porte
                # souvent la conclusion/l'erreur — la preuve — plus que son milieu.
                keep = max(120, len(c) // 2)
                head = keep * 2 // 3
                tail = keep - head
                convo[idx] = {
                    **convo[idx],
                    "content": c[:head]
                    + " …[milieu tronqué pour tenir dans le contexte]… "
                    + c[len(c) - tail :],
                }
            elif isinstance(c, list):
                parts = []
                for p in c:
                    t = p.get("text") if isinstance(p, dict) else None
                    if isinstance(t, str) and len(t) > 120:
                        parts.append(
                            {**p, "text": t[: max(120, len(t) // 2)] + " …[tronqué]"}
                        )
                    else:
                        parts.append(p)
                convo[idx] = {**convo[idx], "content": parts}
            else:
                convo[idx] = {**convo[idx], "content": "…[tronqué]"}
        elif len(convo) > 2:
            # Plus rien de clippable mais encore trop : on drope le plus ancien, SAUF la
            # tâche courante (vécu en éval : le pop l'emportait -> conversation réduite à
            # [assistant, tool] SANS message user -> 400 « Unable to generate parser for
            # this template » côté llama.cpp, et un modèle sans but même sinon).
            victim = 1 if task_idx == 0 else 0
            convo.pop(victim)
            _drop_orphan_tools(convo)
        else:
            break
    return _total() <= budget_chars


def _ctx_estimate(system_prompt: str, convo: list[dict]) -> int:
    # Estimation ~3 car./token du contexte VIVANT (prompt + convo courant). Sert à
    # rafraîchir la jauge IMMÉDIATEMENT après une compaction, sans attendre l'usage
    # réel du prochain appel (sinon la jauge reste au pic pendant tout l'appel suivant).
    return (len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)) // 3


def _inject_notes(notes_provider, convo: list[dict]) -> Iterator[tuple[str, object]]:
    """Draine les notes en vol : chacune est injectée dans `convo` (role user,
    préfixe explicite) et ré-émise en event ('note', texte injecté)."""
    # Notes en vol : les remarques utilisateur arrivées pendant le tour sont
    # injectées MAINTENANT (juste avant l'appel modèle = le point d'arrêt),
    # sans interrompre quoi que ce soit. L'appelant reçoit l'event 'note'
    # pour persister/afficher exactement ce qui a été injecté.
    if notes_provider is None:
        return
    try:
        for _raw_note in notes_provider() or []:
            _wrapped = (
                "[User note received mid-turn — take it into account "
                f"and continue the task] {_raw_note}"
            )
            convo.append({"role": "user", "content": _wrapped})
            yield ("note", _wrapped)
    except Exception as _e:  # noqa: BLE001 - notes best-effort
        _debug("NOTES_ERR", str(_e))


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
    stream = oai.chat.completions.create(**kwargs)
    try:
        for kind, chunk in _iter_turn(stream, collector):
            if _first_byte:
                _first_byte = False
                log_event(
                    "stream.first_byte",
                    ms=round((time.monotonic() - _t_req) * 1000),
                )
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
        _close(stream)
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


def _dispatch_no_tool_calls(
    collector: dict,
    text: str,
    convo: list[dict],
    strong: bool,
    max_loop_breaks: int,
    max_length_continues: int,
    max_empty_retries: int,
    max_act_nudges: int,
    st: dict,
) -> Iterator[tuple[str, object]]:
    """Fin de tour SANS appel d'outil : boucle dégénérée -> continuation 'length' ->
    réponse vide -> audit de claim / act-nudge -> stop naturel. Issue via
    st["action"] : "continue" (relancer le tour) ou "done" (l'event terminal
    ('done', …) a déjà été yieldé)."""
    # BOUCLE DE DÉGÉNÉRESCENCE (détectée au streaming) : le modèle a répété la
    # même phrase sans agir. À traiter AVANT la continuation 'length' : lui dire
    # « continue où tu t'es arrêté » ne ferait qu'alimenter le cycle. On coupe et
    # on relance avec un ordre FERME d'émettre un appel d'outil, borné.
    if collector.get("looped"):
        if st["loop_breaks"] >= max_loop_breaks:
            yield (
                "content",
                "\n[génération interrompue : le modèle tournait en boucle (même "
                "phrase répétée) sans agir. Reformule ou découpe la demande.]",
            )
            yield ("done", {"reason": "loop_degenerate"})
            st["action"] = "done"
            return
        st["loop_breaks"] += 1
        nudge = (
            "Tu répètes la même phrase en boucle sans rien faire. ARRÊTE de "
            "planifier en prose. Émets MAINTENANT un seul appel d'outil — "
            "manage_todos pour poser le plan, OU directement le premier write_file "
            "— sans aucun texte avant. Un seul outil, tout de suite."
        )
        convo.append({"role": "user", "content": nudge})
        _debug(
            "LOOP_BREAK",
            f"boucle détectée ({collector.get('looped')!r}), "
            f"relance {st['loop_breaks']}/{max_loop_breaks}",
        )
        yield (
            "tool_result",
            {"name": "(boucle)", "ok": False, "preview": nudge},
        )
        st["action"] = "continue"
        return
    # CONTINUATION sur troncature : la réponse texte/raisonnement a été coupée
    # par la limite de tokens (finish_reason == "length") sans appel d'outil.
    # Plutôt que de rendre une réponse tronquée, on relance le modèle pour qu'il
    # POURSUIVE là où il s'est arrêté. Autant de fois que nécessaire (cap dur
    # max_length_continues, anti-runaway). Le texte continue d'être streamé à
    # l'UI tour après tour (le web app concatène). Cas des tool_calls tronqués
    # NON concerné (géré par 'arguments tronqués' / overflow).
    if (
        collector["finish_reason"] == "length"
        and st["length_continues"] < max_length_continues
    ):
        st["length_continues"] += 1
        if text:
            convo.append({"role": "assistant", "content": text})
            nudge = (
                "Ta réponse a été coupée par la limite de tokens. CONTINUE "
                "exactement là où tu t'es arrêté, sans répéter ce qui précède."
            )
        else:
            nudge = (
                "Ta réflexion a été coupée par la limite de tokens. Termine et "
                "DONNE ta réponse (ou émets l'appel d'outil) MAINTENANT, plus "
                "direct."
            )
        convo.append({"role": "user", "content": nudge})
        _debug(
            "CONTINUATION(length)",
            f"relance {st['length_continues']}/{max_length_continues}",
        )
        st["action"] = "continue"
        return
    # RÉPONSE VIDE (EOS immédiat : 0 texte, 0 tool call — vécu en éval,
    # cas context_squeeze) : le stop naturel serait un SILENCE total pour
    # l'utilisateur. On relance, borné. Filet de FONCTIONNEMENT (pas une
    # garde de comportement) -> actif aussi pour un modèle fort (strong).
    if not text.strip():
        if st["empty_retries"] < max_empty_retries:
            st["empty_retries"] += 1
            nudge = (
                "Ta réponse est arrivée VIDE (aucun texte, aucun appel "
                "d'outil). Réponds MAINTENANT : donne le résultat demandé, "
                "ou émets l'appel d'outil nécessaire."
            )
            convo.append({"role": "user", "content": nudge})
            log_event(
                "guard",
                level="WARN",
                kind="empty_response",
                retry=st["empty_retries"],
            )
            yield (
                "tool_result",
                {"name": "(réponse vide)", "ok": False, "preview": nudge},
            )
            st["action"] = "continue"
            return
        yield (
            "content",
            "\n[génération interrompue : le modèle a rendu une réponse "
            "vide malgré les relances.]",
        )
        yield ("done", {"reason": "empty_response"})
        st["action"] = "done"
        return
    # Audit de claim au stop : le modèle prétend-il un résultat qu'il n'a pas
    # produit ? (A) artefact fichier inventé, (B) résultat d'exécution sans
    # run_shell/dispatch, ou intention/affirmation sans exécution réelle. On le
    # relance pour qu'il FASSE vraiment (borné). Garde de vérité, pas orchestrateur.
    # COUPÉ pour un modèle FORT (distant) : ces relances de comportement, utiles à
    # un petit modèle qui confabule, ne font que sur-piloter un modèle qui se vérifie
    # déjà seul (cf. GLM qui doutait de sa propre preuve correcte).
    missing = _claims_missing_artifact(text, st["files_written"])
    exec_confab = not st["executed"] and _claims_execution(text)
    if (
        not strong
        and st["act_nudges"] < max_act_nudges
        and (missing or exec_confab or _intends_to_act(text, st["executed"]))
    ):
        st["act_nudges"] += 1
        convo.append({"role": "assistant", "content": text or "..."})
        if missing:
            nudge = (
                f"Tu affirmes avoir produit « {missing} » mais ce fichier "
                "n'existe pas (ou est vide). Crée-le RÉELLEMENT avec un outil "
                "puis vérifie-le — n'invente pas d'artefact ni de preuve."
            )
            label = "CLAIM_AUDIT(artefact)"
        elif exec_confab:
            nudge = (
                "Tu rapportes un résultat d'exécution (sortie, « ça marche », "
                "preuve) mais tu n'as lancé AUCUNE commande ce tour (ni run_shell "
                "ni dispatch_agent). Lance-la RÉELLEMENT et rapporte la VRAIE "
                "sortie — n'invente pas de résultat."
            )
            label = "CLAIM_AUDIT(exécution)"
        else:
            nudge = (
                "Tu as annoncé/affirmé une action mais tu n'as rien exécuté : "
                "rien n'a été réellement fait. Émets MAINTENANT l'appel d'outil "
                "directement (aucune phrase avant). Si la tâche est vraiment "
                "terminée ET vérifiée, dis seulement le résultat constaté."
            )
            label = "ACT_NUDGE"
        convo.append({"role": "user", "content": nudge})
        _debug(label, nudge)
        log_event("guard", kind=label)
        st["action"] = "continue"
        return
    yield ("done", {"reason": "natural"})
    st["action"] = "done"  # réponse finale déjà streamée (stop naturel du modèle)


def _check_no_progress(
    tool_calls: list[dict], strong: bool, repeat_limit: int, st: dict
) -> Iterator[tuple[str, object]]:
    """Détecteur de non-progrès (mêmes appels que le tour précédent). Issue via
    st["action"] : "proceed" (on exécute les outils) ou "done" (repeat_stop,
    l'event terminal a déjà été yieldé). Met à jour st["repeat_streak"] /
    st["prev_sig_set"]."""
    st["action"] = "proceed"
    # Non-progrès : même jeu d'appels (outils+args) que le tour précédent ? On EXCLUT
    # les outils d'exécution/vérification (_VERIFY_TOOLS) : re-lancer la même preuve est
    # légitime. Un tour PUREMENT de vérif -> signature vide -> compté comme progrès (on
    # ne coupe pas, on remet le compteur à zéro). Backstop ultime contre le vrai runaway :
    # max_iters. Les boucles dégénérées (re-edit/re-write/re-read identiques) restent prises.
    sig_set = frozenset(
        f"{tc['name']}\x00{tc['arguments']}"
        for tc in tool_calls
        if tc["name"] not in _VERIFY_TOOLS
    )
    if not sig_set:
        # Tour purement exécution/vérif : progrès légitime, on ne coupe pas et on
        # laisse passer vers l'exécution des outils (surtout PAS de continue ici).
        st["repeat_streak"] = 0
        st["prev_sig_set"] = None
        return
    st["repeat_streak"] = (
        st["repeat_streak"] + 1 if sig_set == st["prev_sig_set"] else 0
    )
    st["prev_sig_set"] = sig_set
    # COUPÉ pour un modèle FORT (distant) : le seul backstop reste max_iters. Sur un
    # petit modèle, la répétition = dégénérescence ; sur un fort, c'est presque
    # toujours du légitime (le juger « bloqué » l'interrompt à tort).
    if not strong and st["repeat_streak"] >= repeat_limit - 1:
        log_event("guard", level="WARN", kind="repeat_stop")
        yield (
            "content",
            "\n(arrêt : le modèle réémet les mêmes appels sans progresser).",
        )
        yield ("done", {"reason": "repeat_stop"})
        st["action"] = "done"


def _run_tools_parallel(
    registry, tool_calls: list[dict], convo: list[dict], st: dict
) -> Iterator[tuple[str, object]]:
    """Exécute un tour d'outils PARALLEL-SAFE concurremment (1 thread par outil),
    relaie l'activité live au fil de l'eau et recolle les résultats DANS L'ORDRE
    dans `convo`. Met à jour st["executed"] / st["fail_count"]."""
    import queue as _queue

    _q: _queue.Queue = _queue.Queue()
    _res: dict = {}  # id -> (result, ok, args)
    # Signale à l'UI un GROUPE parallèle -> rendu en « arène » côté à côté (animation
    # des agents qui tournent en même temps), pas des pastilles empilées.
    yield (
        "parallel",
        {
            "ids": [tc["id"] for tc in tool_calls],
            "names": [tc["name"] for tc in tool_calls],
        },
    )
    for _tc in tool_calls:  # pastilles d'abord : elles se rempliront en parallèle
        yield ("tool_call", {"id": _tc["id"], "name": _tc["name"]})
        log_event(
            "tool.call",
            name=_tc["name"],
            args_len=len(_tc["arguments"] or ""),
        )

    def _pworker(tc):
        name = tc["name"]
        try:
            pargs = json.loads(tc["arguments"] or "{}")
        except json.JSONDecodeError:
            _res[tc["id"]] = (
                "erreur: arguments tronqués (réponse coupée).",
                False,
                {},
            )
            _q.put((tc["id"], "__done__", None))
            return
        try:
            if registry.is_streaming(
                name
            ):  # dispatch_agent : activité relayée live (helper partagé)
                r = "(le sous-agent n'a rien renvoyé)"
                for ek, ep in _stream_tool_events(registry, tc["id"], name, pargs):
                    if ek == "__result__":
                        r = ep
                    else:
                        _q.put((tc["id"], ek, ep))
            else:  # lecture/recherche : exécution directe
                r = registry.run(name, pargs)
            okp = not r.startswith("erreur")
        except Exception as e:  # noqa: BLE001 - un outil qui casse n'arrête pas les autres
            r, okp = f"erreur: {e}", False
        _res[tc["id"]] = (r, okp, pargs)
        _q.put((tc["id"], "__done__", None))

    _threads = [
        threading.Thread(target=_pworker, args=(tc,), daemon=True, name="loom-parallel")
        for tc in tool_calls
    ]
    for _t in _threads:
        _t.start()
    _left = len(_threads)
    while _left > 0:  # relaie l'activité live des N outils au fil de l'eau
        _tid, _kind, _payload = _q.get()
        if _kind == "__done__":
            _left -= 1
        else:
            yield (_kind, _payload)
    for _t in _threads:
        _t.join()
    for (
        tc
    ) in tool_calls:  # résultats DANS L'ORDRE (messages `tool` cohérents pour l'API)
        name = tc["name"]
        r, okp, pargs = _res.get(tc["id"], ("(vide)", False, {}))
        # Streak de troncature tenu ICI (mono-thread) et pas dans _pworker : st
        # n'est pas protégé contre les écritures concurrentes des workers.
        if str(r).startswith("erreur: arguments tronqués"):
            st["truncated_streak"] = st.get("truncated_streak", 0) + 1
        else:
            st["truncated_streak"] = 0
        convo.append({"role": "tool", "tool_call_id": tc["id"], "content": r})
        if name == "dispatch_agent" and not str(r).startswith("refusé"):
            st["executed"] = True
        if not okp and name in _BUG_SIGNAL_TOOLS:
            st["fail_count"] += 1
        yield (
            "tool_result",
            _tool_result_payload(tc["id"], name, okp, r, pargs),
        )


def _run_tools_sequential(
    tool_calls: list[dict],
    registry,
    permission,
    confirm,
    convo: list[dict],
    strong: bool,
    st: dict,
) -> Iterator[tuple[str, object]]:
    """Exécute les appels d'outils d'un tour, SÉQUENTIELLEMENT : permission/confirm,
    garde-fou P1.1 (1 write par tour), images inline différées, anti sur-vérification.
    Met à jour convo et st (executed, files_written, fail_count, verify_streak)."""
    wrote_this_turn = False  # P1.1 : un seul write_file/append_file par tour
    # Images inline (read_image) à faire VOIR au modèle : différées après TOUS
    # les résultats d'outils (les messages `tool` doivent rester contigus).
    image_followups: list[dict] = []
    for tc in tool_calls:
        name = tc["name"]
        yield ("tool_call", {"id": tc["id"], "name": name})
        log_event("tool.call", name=name, args_len=len(tc["arguments"] or ""))
        _t_tool = time.monotonic()
        try:
            args = json.loads(tc["arguments"] or "{}")
            st["truncated_streak"] = 0
        except json.JSONDecodeError:
            # Arguments tronqués (réponse coupée par max_tokens). NE PAS exécuter
            # avec des args vides (erreur trompeuse 'path manquant') : signaler la
            # troncature pour que le modèle réémette l'appel en plus court.
            # Le streak est lu par _preventive_compaction : à saturation de fenêtre
            # (completion étranglée), « plus court » ne suffit JAMAIS — 2 troncatures
            # de suite déclenchent une compaction forcée avant l'appel suivant.
            st["truncated_streak"] = st.get("truncated_streak", 0) + 1
            result = (
                "erreur: arguments tronqués (réponse coupée). "
                "Réémets cet appel d'outil, en plus court."
            )
            convo.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            yield (
                "tool_result",
                {"id": tc["id"], "name": name, "ok": False, "preview": result},
            )
            continue
        # P1.1 : sérialiser les écritures à gros contenu (1 par tour) -> évite le
        # batch de N gros write_file/append_file qui sature max_tokens et tronque.
        # Les éditions par bloc (edit/replace/insert) ne passent PAS par ici.
        if name in _SERIAL_WRITE and wrote_this_turn:
            result = (
                "différé : un seul write_file/append_file par tour. Réémets "
                "cet appel (seul) au prochain tour."
            )
            convo.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
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
                    registry.run(name, args) if registry else "erreur: pas d'outils"
                )
                ok = not result.startswith("erreur")
            else:
                result = "refusé par l'utilisateur"
                ok = False
        elif registry and registry.is_streaming(name):  # allow + streamant
            # Outil streamant (dispatch_agent) : activité relayée EN DIRECT dans
            # sa pastille, synthèse reconstruite (helper partagé avec le parallèle).
            result = "(le sous-agent n'a rien renvoyé)"
            for ek, ep in _stream_tool_events(registry, tc["id"], name, args):
                if ek == "__result__":
                    result = ep
                else:
                    yield (ek, ep)
            ok = not result.startswith("erreur")
        else:  # allow
            result = registry.run(name, args) if registry else "erreur: pas d'outils"
            ok = not result.startswith("erreur")

        # read_image renvoie une image inline encodée : on ne met qu'un accusé
        # TEXTE dans le message `tool` (pas de base64 géant), et on diffère le
        # message `user` multimodal qui fera réellement VOIR l'image au modèle.
        if is_inline_image(result):
            caption, data_url = parse_inline_image(result)
            tool_content = f"[image « {caption} » chargée — fournie ci-dessous]"
            image_followups.append(image_user_message(caption, data_url))
            ok = True
        else:
            tool_content = result
        # Anti SUR-VÉRIFICATION (informatif, jamais bloquant) : au-delà de
        # _VERIFY_STREAK_NOTE checks navigateur VERTS d'affilée sans changement
        # d'état entre-temps, le résultat le dit au modèle. On n'empêche RIEN
        # (les checks restent exclus du non-progrès : re-prouver est légitime) ;
        # on nomme la preuve déjà faite. Coupé pour un modèle fort (strong).
        st["verify_streak"] = _verify_streak_update(name, ok, st["verify_streak"])
        if (
            not strong
            and ok
            and name in _BROWSER_CHECKS
            and st["verify_streak"] >= _VERIFY_STREAK_NOTE
        ):
            tool_content += (
                f"\n[harnais : {st['verify_streak']} vérifications vertes d'affilée "
                "sans changement d'état entre-temps — la preuve est faite. "
                "Conclus MAINTENANT ; ne re-vérifie que si tu modifies "
                "quelque chose.]"
            )
        convo.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": tool_content}
        )
        log_event(
            "tool.result",
            name=name,
            ok=ok,
            ms=round((time.monotonic() - _t_tool) * 1000),
            preview=str(tool_content)[:90],
        )
        yield (
            "tool_result",
            _tool_result_payload(tc["id"], name, ok, tool_content, args),
        )
        if name in _SERIAL_WRITE:
            wrote_this_turn = True
        # Suivi pour l'audit de claim : une EXÉCUTION réelle (run_shell/dispatch,
        # même en échec mais hors refus de permission) et les FICHIERS écrits.
        if name in ("run_shell", "dispatch_agent") and not str(result).startswith(
            "refusé"
        ):
            st["executed"] = True
        if ok and name in _WRITE_TOOLS and args.get("path"):
            st["files_written"].add(args["path"])
        # Cascade de bugs : on compte les échecs des outils d'EXÉCUTION/VÉRIF (pas les
        # erreurs d'usage type ligne hors limite). Au 2e échec, on IMPOSE la méthode debug.
        if not ok and name in _BUG_SIGNAL_TOOLS:
            st["fail_count"] += 1
    convo.extend(image_followups)  # images vues au tour suivant


class LoomClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "loom-local",
        model: str = "local",
        timeout: int = 120,
        max_retries: int = 6,
        routes: dict | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # Serveur LOCAL : timeout de LECTURE long. Pendant un gros prefill (contexte
        # recalculé après compaction : plusieurs minutes à ~200 t/s) ou un chargement de
        # modèle, llama-server n'émet RIEN — c'est du travail légitime, pas une panne.
        # Un read=120s coupait ces phases (ReadTimeout vécu). Connexion/écriture restent
        # au timeout court : un serveur éteint doit échouer vite.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(float(timeout), read=max(600.0, float(timeout))),
            max_retries=max_retries,
        )
        # Routes vers des modèles DISTANTS (API OpenAI-compatible) : id de modèle ->
        # {base_url, api_key, model, enable_thinking_param}. Tout modèle absent de cette
        # table part vers l'endpoint LOCAL (_client). Un client openai par endpoint, monté
        # une fois ici. Le reste du code appelle toujours `model=<id>` : le routage est interne.
        self._routes: dict[str, dict] = {}
        # Disjoncteur slot KV : modèles dont le save/restore a PENDU (hang serveur,
        # cf. _slot_action) -> on n'essaie plus jusqu'au restart du process.
        self._slot_broken: set[str] = set()
        for rid, spec in (routes or {}).items():
            self._routes[rid] = {
                "client": OpenAI(
                    base_url=spec["base_url"],
                    api_key=spec.get("api_key") or "none",
                    timeout=timeout,
                    max_retries=max_retries,
                ),
                "base_url": spec["base_url"],
                "api_key": spec.get("api_key") or "",
                "model": spec.get("model") or rid,
                "enable_thinking_param": bool(spec.get("enable_thinking_param", False)),
            }
        # Cache de la fenêtre de contexte découverte par provider (id Loom -> int|None).
        # None mémorisé = provider interrogé mais muet -> on ne re-frappe pas l'API.
        self._ctx_cache: dict[str, int | None] = {}

    def _resolve(self, model: str | None):
        """(client_openai, model_api, native_extras) pour le modèle demandé. Un modèle
        distant (présent dans routes) part vers son endpoint, SANS les extra_body llama.cpp ;
        sinon l'endpoint local avec les extras natifs."""
        if model and model in self._routes:
            r = self._routes[model]
            return r["client"], r["model"], r["enable_thinking_param"]
        return self._client, (model or self.model), True

    def summarize_slice(
        self,
        old_messages: list[dict],
        model: str | None = None,
        budget_chars: int = 30000,
    ) -> str:
        """PRIMITIVE UNIQUE de résumé, partagée par TOUS les chemins de compaction : l'étage
        de la boucle d'outils, le résumé pré-tour (context.summarize) et le bouton manuel.
        Aplatit les vieux tours (borné), appelle le modèle en NON-stream, retire le <think>.
        Renvoie le texte du résumé, ou '' si rien à résumer / réponse vide / appel en échec.
        FAIL-SOFT : ne lève jamais (un résumé raté ne doit jamais crasher l'appelant)."""
        body = _flatten_for_summary(list(old_messages), budget_chars)
        if not body.strip():
            return ""
        oai, api_model, _ = self._resolve(model)
        try:
            resp = oai.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": body},
                ],
                # 700 (et non 2000) : un résumé dense/télégraphique n'a pas besoin de plus,
                # et sur un modèle local lent (~8 tok/s) 2000 tokens = plusieurs MINUTES.
                max_tokens=700,
                temperature=0.2,
            )
            summary = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - best-effort : jamais crasher l'appelant
            log_event(
                "summary.error",
                level="WARN",
                msg=f"{type(exc).__name__}: {str(exc)[:120]}",
            )
            return ""
        # Modèle « thinking » (Qwen/local) : le raisonnement peut précéder le contenu -> on
        # ne garde que l'après-</think>.
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()
        return summary.strip()

    def summarize_old_turns(
        self,
        convo: list[dict],
        model: str | None = None,
        keep_recent: int = 6,
        budget_chars: int = 30000,
    ) -> int:
        """Résume les vieux tours EN PLACE via `summarize_slice` (garde les `keep_recent`
        derniers intacts). Renvoie le nb de messages remplacés (0 = trop peu à résumer,
        réponse vide, ou appel en échec — rien n'est touché)."""
        cut = len(convo) - keep_recent
        if cut < 2:
            return 0  # trop peu de vieux tours (ou convo plus courte que keep_recent)
        # Ne pas orpheliner un résultat d'outil : si la queue conservée débute par un
        # role:tool dont l'appel part au résumé, on pousse ces tool vers le bloc résumé
        # (certains providers rejettent un message tool sans tool_calls le précédant).
        while cut < len(convo) and convo[cut].get("role") == "tool":
            cut += 1
        summary = self.summarize_slice(convo[:cut], model, budget_chars)
        if not summary:
            return 0
        convo[:cut] = [{"role": "user", "content": f"{_SUMMARY_MARKER}\n{summary}"}]
        return cut

    def compact_conversation(
        self,
        messages: list[dict],
        system_prompt: str = "",
        target_chars: int | None = None,
        keep_recent_tools: int = 2,
    ) -> tuple[list[dict], int]:
        """Compaction MANUELLE (bouton UI) : DÉTERMINISTE et INSTANTANÉE sur une COPIE.

        AUCUN appel modèle : un résumé LLM sur un modèle local lent (~8 tok/s × 2000 tokens)
        bloquerait le bouton PLUSIEURS MINUTES (observé live : /compact figé à « … », verrou
        tenu, 429 sur les clics suivants). Ici c'est purement local/instantané :
        1. vide les vieux résultats d'outils (`_microcompact_tools`) ;
        2. si `target_chars` est donné, CLIPPE le contexte vivant pour tenir dessous
           (`_force_fit`) — libère la conversation jusqu'au plancher. Le prompt système +
           les schémas d'outils, eux, sont INCOMPRESSIBLES (souvent l'essentiel du ctx).
        Renvoie (messages, tokens_libérés_estimés) ; 0 = déjà au plus bas."""
        convo = list(messages)

        def _tot() -> int:
            return len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)

        before = _tot()
        _microcompact_tools(convo, keep_recent_tools)
        if target_chars:
            _force_fit(convo, system_prompt, target_chars)
        return convo, max(0, (before - _tot()) // 3)

    def local_server_root(self) -> str:
        """Racine du serveur LOCAL (llama-swap) : la base_url SANS le suffixe `/v1`.
        Sert à joindre l'API de management (`/running`, `/api/models/unload`)."""
        b = self.base_url
        return b[:-3] if b.endswith("/v1") else b.rstrip("/")

    def is_remote(self, model: str | None) -> bool:
        """Vrai si le modèle est servi par une API DISTANTE (route montée), pas en local."""
        return bool(model and model in self._routes)

    def add_remote_route(self, model_id: str, spec: dict) -> None:
        """Monte (ou remplace) À CHAUD la route d'un modèle distant : un client OpenAI de plus,
        sans redémarrer. Un distant = juste une URL + une clé, rien à charger en VRAM -> l'ajout
        est immédiat. Invalide le cache de contexte pour ce modèle."""
        self._routes[model_id] = {
            "client": OpenAI(
                base_url=spec["base_url"],
                api_key=spec.get("api_key") or "none",
                timeout=self.timeout,
                max_retries=self.max_retries,
            ),
            "base_url": spec["base_url"],
            "api_key": spec.get("api_key") or "",
            "model": spec.get("model") or model_id,
            "enable_thinking_param": bool(spec.get("enable_thinking_param", False)),
        }
        self._ctx_cache.pop(model_id, None)

    def remove_remote_route(self, model_id: str) -> None:
        """Démonte à chaud la route d'un modèle distant (l'id disparaît du sélecteur)."""
        self._routes.pop(model_id, None)
        self._ctx_cache.pop(model_id, None)

    def remote_api_key(self, model_id: str) -> str:
        """Clé brute d'une route distante — USAGE SERVEUR uniquement (préserver la clé lors
        d'une édition sans la re-saisir, indice masqué). NE JAMAIS renvoyer telle quelle au client."""
        return (self._routes.get(model_id) or {}).get("api_key", "")

    def remote_route_info(self, model_id: str) -> dict:
        """Infos SÛRES d'une route distante pour l'UI (jamais la clé en clair) : base_url,
        modèle côté provider, présence d'une clé."""
        r = self._routes.get(model_id) or {}
        return {
            "base_url": r.get("base_url", ""),
            "model": r.get("model", model_id),
            "has_key": bool(r.get("api_key")),
            "enable_thinking_param": bool(r.get("enable_thinking_param", False)),
        }

    def ping_remote(
        self, base_url: str, api_key: str, model: str, timeout: float = 15.0
    ) -> tuple[bool, str]:
        """Test de connexion RÉEL d'un endpoint distant : 1 requête 1-token, non-stream,
        sans retry. (ok, message). Sert au bouton « Tester » avant d'enregistrer un modèle."""
        try:
            oai = OpenAI(
                base_url=base_url,
                api_key=api_key or "none",
                timeout=timeout,
                max_retries=0,
            )
            oai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True, "OK"
        except Exception as e:  # noqa: BLE001 - remonte un message clair à l'UI
            return False, f"{type(e).__name__}: {str(e)[:160]}"

    def remote_context(self, model: str | None) -> int | None:
        """Fenêtre de contexte RÉELLE d'un modèle distant, lue DU PROVIDER (`GET /models`).

        C'est « le modèle lui-même » qui répond, pas la config Loom. Best-effort : renvoie
        l'entier si le provider publie un champ de contexte (OpenRouter `context_length`,
        vLLM `max_model_len`, `max_input_tokens`…), sinon None — beaucoup d'API (Z.ai, OpenAI)
        renvoient le schéma nu (id/object/created/owned_by) et NE publient rien : l'appelant
        retombe alors sur la valeur déclarée en config. Résultat mis en cache (l'absence aussi)
        pour ne pas re-frapper l'API à chaque page."""
        if not model or model not in self._routes:
            return None
        if model in self._ctx_cache:
            return self._ctx_cache[model]
        import json as _json
        import urllib.error
        import urllib.request

        r = self._routes[model]
        base = str(r.get("base_url") or "").rstrip("/")
        api_model = r.get("model") or model
        key = r.get("api_key") or ""
        keys = (
            "context_length",
            "context_window",
            "max_context_length",
            "max_model_len",
            "max_input_tokens",
        )
        found: int | None = None
        try:
            req = urllib.request.Request(
                base + "/models", headers={"Authorization": "Bearer " + key}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = _json.loads(resp.read().decode("utf-8", "replace"))
            items = data.get("data") if isinstance(data, dict) else data
            for m in items or []:
                if not isinstance(m, dict) or m.get("id") != api_model:
                    continue
                for k in keys:
                    v = m.get(k)
                    if isinstance(v, (int, float)) and v > 0:
                        found = int(v)
                        break
                tp = m.get("top_provider")  # OpenRouter niche le contexte ici
                if found is None and isinstance(tp, dict):
                    v = tp.get("context_length")
                    if isinstance(v, (int, float)) and v > 0:
                        found = int(v)
                break
        except (urllib.error.URLError, OSError, ValueError):
            found = None
        self._ctx_cache[model] = found
        return found

    def unload_local(self, model: str | None = None, timeout: float = 30.0) -> bool:
        """Décharge le(s) modèle(s) LOCAL(aux) via l'API llama-swap (libère la VRAM). Model
        None = tous. Best-effort : False si le serveur local est injoignable (non lancé, ou
        llama-server direct sans API de swap). Cf. `POST /api/models/unload[/:model]`."""
        import urllib.error
        import urllib.request

        root = self.local_server_root()
        path = f"/api/models/unload/{model}" if model else "/api/models/unload"
        try:
            req = urllib.request.Request(root + path, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def running_local(self, timeout: float = 5.0) -> tuple[bool, str]:
        """(joignable, texte brut JSON) de llama-swap `GET /running`. Best-effort : le texte
        brut suffit pour tester par sous-chaîne quel modèle est chargé, sans coupler au schéma."""
        import urllib.error
        import urllib.request

        root = self.local_server_root()
        try:
            with urllib.request.urlopen(root + "/running", timeout=timeout) as resp:
                return True, resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError):
            return False, ""

    def warmup_local(self, model: str) -> None:
        """Charge un modèle LOCAL en envoyant un ping 1-token : llama-swap charge à la 1re
        requête (et swap l'ancien si besoin). BLOQUANT le temps du chargement -> appeler dans
        un thread. Best-effort (avale toute erreur : serveur non lancé, etc.)."""
        try:
            for _ in self.stream_chat(
                [{"role": "user", "content": "ping"}],
                "",
                1,
                model=model,
                thinking=False,
            ):
                pass
        except Exception as e:  # noqa: BLE001 - warmup best-effort, jamais bloquant
            _debug("WARMUP_ERR", str(e))
            pass

    def _slot_action(self, model: str | None, action: str, name: str) -> bool:
        """POST /slots/0?action=save|restore sur le serveur LOCAL (via la route
        llama-swap /upstream/<modèle>/, repli /slots direct pour un llama-server
        sans swap). Nécessite --slot-save-path côté serveur. Best-effort.

        DISJONCTEUR : constaté le 2026-07-10 sur ornith q8 (CUDA + mmproj), le save
        PEND côté llama-server (502 après ~60 s) alors qu'il marche en CPU pur ->
        timeout court (20 s) et, au premier échec, on ARRÊTE d'essayer pour ce
        modèle (jusqu'au restart) : l'appelant retombe sur le ré-amorçage par
        re-prefill. Sans ça, chaque fin de tour perdait ~1 min à attendre le hang."""
        import urllib.request

        if self.is_remote(model):
            return False  # distant : cache géré par le provider, pas de slot local
        key = model or "(local)"
        if key in self._slot_broken:
            return False
        root = self.local_server_root()
        payload = json.dumps({"filename": name}).encode()
        paths = [f"/upstream/{model}/slots/0?action={action}"] if model else []
        paths.append(f"/slots/0?action={action}")
        for path in paths:
            req = urllib.request.Request(
                root + path,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    body = json.loads(resp.read().decode() or "{}")
                _debug(f"SLOT_{action.upper()}", {"name": name, **body}, terminal=False)
                return True
            except Exception as e:  # noqa: BLE001 - slot KV best-effort, jamais bloquant
                # DISJONCTE (plus d'essais pour ce modèle) sur les échecs DURABLES :
                # - timeout = hang serveur (vécu : ~60 s perdus par tour) ;
                # - HTTP 501 = llama-server refuse (constaté 2026-07-10 : slot save NON
                #   SUPPORTÉ quand un projecteur multimodal --mmproj est chargé — les
                #   modèles vision retombent définitivement sur le ré-amorçage).
                code = getattr(e, "code", None)
                if (
                    isinstance(e, TimeoutError)
                    or "timed out" in str(e).lower()
                    or code == 501
                ):
                    self._slot_broken.add(key)
                    if code == 501:
                        # Cause racine CONFIRMÉE (2026-07-13, A/B même binaire b9442 :
                        # qwen avec --mmproj -> 501, sans -> 200) : llama-server refuse
                        # le save/restore de slot quand un projecteur MULTIMODAL est
                        # chargé (« This feature is not supported by multimodal »).
                        # Limitation upstream — pas un flag manquant (--slot-save-path
                        # est bien passé). Le corps du 501 est joint pour distinguer
                        # les causes si le serveur change.
                        detail = ""
                        try:
                            detail = (e.read() or b"").decode("utf-8", "replace")[:120]
                        except Exception:  # noqa: BLE001 - diagnostic best-effort
                            pass
                        cause = (
                            "501 : mmproj (multimodal) chargé — limitation llama.cpp"
                            + (f" [{detail}]" if detail else "")
                        )
                    else:
                        cause = "hang serveur"
                    _debug(
                        f"SLOT_{action.upper()}_ERR",
                        f"{path} : {cause} -> save/restore DÉSACTIVÉ pour {key}",
                    )
                    print(
                        f"[slot] save/restore KV désactivé pour {key} ({cause}) — "
                        "repli sur le ré-amorçage par re-prefill",
                        flush=True,
                    )
                    return False
                _debug(f"SLOT_{action.upper()}_ERR", f"{path} : {e}", terminal=False)
        return False

    def save_slot(self, model: str | None, name: str) -> bool:
        """Sauve le cache KV du slot local dans <slot-save-path>/<name> (~ms). À faire
        PENDANT que le slot contient la conversation, AVANT tout appel qui l'écrase
        (titre, reflect, sous-agent). Cf. restore_slot."""
        return self._slot_action(model, "save", name)

    def restore_slot(self, model: str | None, name: str) -> bool:
        """Restaure un cache KV sauvé par save_slot (~ms au lieu de re-préfiller des
        minutes) : le prochain tour de la conversation ne préfille que son delta."""
        return self._slot_action(model, "restore", name)

    def warm_context(
        self,
        messages: list[dict],
        system_prompt: str,
        model: str | None = None,
        registry=None,
        thinking: bool = True,
    ) -> bool:
        """Ré-amorce le cache KV du slot LOCAL : re-prefill silencieux (1 token) du
        MÊME préfixe que le prochain tour — system prompt + messages + schémas
        d'outils, car le chat template rend le tout : un écart dans N'IMPORTE quel
        élément = zéro réutilisation. Le slot llama-server est UNIQUE : tout appel
        intermédiaire (titre, reflect, ping) écrase le cache de la conversation ->
        sans ré-amorçage le message suivant re-préfillerait TOUT (des minutes en
        local). Best-effort (avale toute erreur), False si échec."""
        try:
            oai, api_model, native = self._resolve(model)
            kwargs = build_create_kwargs(
                api_model,
                messages,
                system_prompt,
                1,
                thinking,
                tools=registry.openai_tools() if registry else None,
                native_extras=native,
            )
            stream = oai.chat.completions.create(**kwargs)
            try:
                for _ in _iter_events(stream):
                    pass
            finally:
                _close(stream)
            return True
        except Exception as e:  # noqa: BLE001 - amorçage best-effort, jamais bloquant
            _debug("WARM_CTX_ERR", str(e))
            return False

    def infer_title(self, model: str | None, message: str) -> str:
        """Titre COURT (3-5 mots) d'une conversation, inféré par le modèle. NON streamé, tout
        petit budget.

        Point clé : on COUPE le raisonnement — sinon un modèle « thinking » (glm, Qwen…)
        épuise le budget en réflexion et ne sort jamais le titre (d'où le repli sur le début
        du message). AGNOSTIQUE au provider ET au modèle : on essaie les conventions connues
        pour désactiver le thinking, chacune ignorée/rejetée SANS CASSE si le backend ne la
        connaît pas ; on garde la 1re qui produit un vrai titre. Couvre le DISTANT (Z.ai/GLM,
        OpenRouter…) comme le LOCAL (llama.cpp/Qwen). Renvoie "" si rien d'exploitable ->
        l'appelant gère le repli (début du message)."""
        oai, api_model, _ = self._resolve(model)
        prompt = (
            "Donne un titre TRÈS court (3 à 5 mots) résumant cette demande, en français, "
            "sans guillemets ni ponctuation finale. Réponds UNIQUEMENT par le titre.\n\n"
            "Demande : " + (message or "")[:500]
        )
        base = {
            "model": api_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Tu génères des titres de conversation courts et clairs.",
                },
                {"role": "user", "content": prompt},
            ],
            # Petite marge : si un backend ne sait pas couper le thinking, il a quand même la
            # place de finir un raisonnement trivial ET d'émettre le titre.
            "max_tokens": 96,
            "temperature": 0.3,
        }
        # Conventions anti-thinking connues, dans l'ordre (chacune best-effort), puis appel nu.
        attempts = (
            {"extra_body": {"thinking": {"type": "disabled"}}},  # Z.ai / GLM
            {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            },  # llama.cpp / Qwen (local)
            {"extra_body": {"reasoning": {"enabled": False}}},  # OpenRouter
            {},  # modèle sans raisonnement / provider strict
        )
        # Titre = cosmétique : ÉCHEC RAPIDE si le backend est lent/éteint (pas de retries en
        # cascade qui feraient traîner la fin du tour). On retombe alors sur le repli message.
        fast = oai.with_options(max_retries=0, timeout=20)
        for extra in attempts:
            try:
                resp = fast.chat.completions.create(**base, **extra)
                txt = (resp.choices[0].message.content or "").strip()
                txt = txt.strip('"').strip("'").strip()
                if txt:
                    return txt.splitlines()[0][:60].strip()
            except (APIConnectionError, APITimeoutError):
                # Backend down/lent : inutile de tenter les autres variantes de param (elles
                # échoueront pareil) -> on abandonne vite, l'appelant fait le repli message.
                break
            except Exception as e:  # noqa: BLE001 - param rejeté par ce backend : variante suivante
                _debug("TITLE_ERR", str(e))
                continue
        return ""

    def describe_image(self, data_uri: str, question: str, model: str) -> str:
        """Fait décrire une image par un modèle VISION (`model`) pour un modèle de raisonnement
        qui ne voit pas (ex. glm-5.2). Appel court NON streamé. Renvoie une description texte
        (exhaustive, structurée). Sert au routage de read_image (approche « VLM comme outil » :
        le raisonneur interroge l'image à la demande). En cas d'erreur : message clair, jamais
        d'exception qui casserait la boucle."""
        oai, api_model, _ = self._resolve(model)
        sys_p = (
            "Tu décris une image pour un AUTRE modèle qui ne la voit pas. Sois exhaustif, "
            "structuré et FIDÈLE : transcris le texte lisible tel quel, décris le layout, la "
            "hiérarchie, les couleurs, les composants et leur position. Pas d'interprétation "
            "gratuite ni de conseil — juste ce qui est réellement dans l'image."
        )
        q = (question or "").strip() or (
            "Décris cette image exhaustivement (texte, layout, couleurs, éléments)."
        )
        content = [
            {"type": "text", "text": q},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
        try:
            resp = oai.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": content},
                ],
                max_tokens=1500,
                stream=False,
            )
            return (
                resp.choices[0].message.content or ""
            ).strip() or "(le VLM n'a rien renvoyé)"
        except Exception as exc:  # noqa: BLE001 - décrire une image ne doit jamais crasher
            _debug("DESCRIBE_IMAGE_ERR", str(exc))
            return f"[description d'image indisponible via le modèle vision : {str(exc)[:160]}]"

    def stream_chat(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 2048,
        model: str | None = None,
        thinking: bool = True,
    ) -> Iterator[tuple[str, str]]:
        """Yield les events (reasoning|content), system prompt injecté en tête."""
        oai, api_model, native = self._resolve(model)
        kwargs = build_create_kwargs(
            api_model,
            messages,
            system_prompt,
            max_tokens,
            thinking,
            native_extras=native,
        )
        _debug_messages(kwargs["model"], kwargs["messages"])
        stream = oai.chat.completions.create(**kwargs)
        reasoning, content = "", ""
        saw_usage = False
        try:
            for kind, chunk in _iter_events(stream):
                if kind == "reasoning":
                    reasoning += chunk
                elif kind == "content":
                    content += chunk
                elif kind == "usage":
                    saw_usage = True
                yield (kind, chunk)
            if not saw_usage:
                yield (
                    "usage",
                    _estimate_usage(system_prompt, messages, content, reasoning, []),
                )
        finally:
            _close(stream)
            _debug("REPONSE <- modele", {"reasoning": reasoning, "content": content})

    def _preventive_compaction(
        self,
        convo: list[dict],
        system_prompt: str,
        model: str | None,
        compact_after_tokens: int | None,
        keep_recent_tools: int,
        refocus_note: bool,
        st: dict,
    ) -> Iterator[tuple[str, object]]:
        """Compaction PRÉVENTIVE avant l'appel modèle : microcompact des vieux
        résultats d'outils, puis force-fit si un résultat RÉCENT est géant (local).
        Ne stoppe jamais le tour ; met à jour st["refocus_done"]."""
        # BACKSTOP TRONCATURE (session 2026-07-14) : 2 arguments d'outil tronqués de
        # suite = la génération est étranglée par la fenêtre (prompt+completion ≈
        # contexte), pas par la verbosité — « réémets plus court » ne débloquera
        # jamais. On compacte de FORCE, même sans seuil configuré et même si
        # l'estimation en caractères (qui peut sous-compter) reste sous le seuil.
        if st.get("truncated_streak", 0) >= 2 and not self.is_remote(model):
            st["truncated_streak"] = 0
            _microcompact_tools(convo, keep_recent_tools)
            _force_fit(
                convo,
                system_prompt,
                max((compact_after_tokens or 0) * 3, len(system_prompt) + 4000),
            )
            _debug(
                "COMPACT_TRONCATURE",
                "2 tool calls tronqués de suite -> compaction forcée "
                f"(~{_ctx_estimate(system_prompt, convo)} tokens).",
            )
            yield (
                "tool_result",
                {
                    "name": "(compaction sur troncature)",
                    "ok": True,
                    "preview": (
                        "Appels d'outil coupés deux fois de suite : contexte "
                        "compacté pour redonner de la place à la génération. "
                        "Je réémets l'appel."
                    ),
                },
            )
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
        # Microcompact : si le contexte vivant approche la fenêtre, vider les vieux
        # résultats d'outils AVANT d'appeler le modèle (évite l'overflow sur une
        # chaîne longue). Estimation grossière ~4 car./token, comme loom.context.
        if not compact_after_tokens:
            return
        # ~3 car./token (et non 4) : code/TSX/JSON tokenise plus dense que de la prose.
        # Surestimer fait déclencher la compaction PLUS TÔT — biais voulu (on ne vide
        # que les vieux résultats), pour ne pas heurter la fenêtre par sous-comptage.
        approx = (
            len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)
        ) // 3
        if approx <= compact_after_tokens:
            return
        cleared = _microcompact_tools(convo, keep_recent_tools)
        if cleared:
            _debug(
                "MICROCOMPACT",
                f"{cleared} résultat(s) d'outil allégé(s) (~{approx} tokens "
                f"> seuil {compact_after_tokens}).",
            )
            # Jauge à jour TOUT DE SUITE (sinon elle reste au pic tant que
            # l'appel suivant n'a pas rendu son usage réel).
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
        # ESCALADE PRÉVENTIVE : vider les vieux résultats ne suffit pas quand UN
        # résultat RÉCENT est géant (ex. read_file d'un gros JSON minifié -> 74k
        # car.) — le microcompact le GARDE (il fait partie des récents). Résultat :
        # avant, on touchait quand même l'overflow (400 / pic à 100 %) et on ne
        # rattrapait qu'en réactif. Ici, si ça déborde ENCORE, on FORCE-FIT AVANT
        # l'appel -> plus jamais de 400. Local uniquement (le distant gère sa
        # fenêtre lui-même).
        if (
            not self.is_remote(model)
            and _ctx_estimate(system_prompt, convo) > compact_after_tokens
        ):
            # Budget PLANCHER = prompt système (incompressible) + un jeu de
            # travail minimal. Sans lui, un seuil plus petit que le prompt
            # système rend le budget INATTEIGNABLE : le force-fit détruit
            # alors tout le travail récent À CHAQUE tour (résultat du dernier
            # read compris) -> le modèle relit en boucle -> repeat_stop
            # (vécu en éval, cas context_squeeze).
            _force_fit(
                convo,
                system_prompt,
                max(compact_after_tokens * 3, len(system_prompt) + 4000),
            )
            if refocus_note and not st["refocus_done"]:
                st["refocus_done"] = True
                convo.append({"role": "user", "content": _REFOCUS_NOTE})
            _debug(
                "FORCE_FIT_PREVENTIF",
                f"un résultat récent trop gros -> clip avant l'appel "
                f"(~{_ctx_estimate(system_prompt, convo)} tokens <= seuil "
                f"{compact_after_tokens}).",
            )
            yield (
                "tool_result",
                {
                    "name": "(compaction préventive)",
                    "ok": True,
                    "preview": (
                        "Contexte réduit AVANT saturation (un résultat récent "
                        "trop gros pour la fenêtre). Je continue."
                    ),
                },
            )
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})

    def _handle_stream_api_error(
        self,
        exc: Exception,
        convo: list[dict],
        system_prompt: str,
        model: str | None,
        compact_after_tokens: int | None,
        max_overflow_retries: int,
        max_summaries: int,
        refocus_note: bool,
        st: dict,
    ) -> Iterator[tuple[str, object]]:
        """Erreur pendant l'appel modèle : échelle context_overflow (étages 1-4),
        output_overflow, puis erreurs non récupérables. Issue via st["action"] :
        "continue" (relancer le même tour) ou "done" (l'event terminal ('done', …)
        a déjà été yieldé). Met à jour overflow_retries / summary_retries /
        force_fits / refocus_done dans st."""
        kind = _classify_stream_error(exc)
        log_event("api.error", level="WARN", kind=kind, msg=str(exc)[:140])
        # DÉBORDEMENT D'ENTRÉE : la requête (prompt + historique + résultats d'outils
        # accumulés) dépasse la fenêtre de contexte. On NE crashe PAS et on ne demande
        # PAS « écris plus court » (ça vise la sortie) : on COMPACTE DUR — on vide TOUS
        # les vieux résultats d'outils (pas seulement au-delà des 4 derniers), de plus
        # en plus agressivement à chaque retry — puis on RELANCE le même tour. Le modèle
        # reprend avec SES messages (ce qu'il a déjà fait) intacts ; les gros résultats
        # d'outils deviennent le placeholder _CLEARED_TOOL (qui dit de ne pas refaire).
        if kind == "context_overflow":
            # Jauge HONNÊTE : la requête qui vient d'échouer a dépassé la fenêtre,
            # mais un 400 ne rend pas d'usage -> la jauge serait restée au dernier
            # appel réussi (ex. 66 %), donnant l'impression qu'on compacte trop tôt.
            # On pousse l'estimation du contexte VIVANT (qui a débordé) : la jauge
            # monte au pic (~100 %) AVANT la compaction, puis redescend après.
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
            # ÉTAGE 1-2 : vider les vieux résultats d'outils (sans LLM, sûr, gratuit).
            if st["overflow_retries"] < max_overflow_retries:
                st["overflow_retries"] += 1
                keep = (
                    1 if st["overflow_retries"] == 1 else 0
                )  # 2e retry : on vide tout
                cleared = _microcompact_tools(convo, keep)
                log_event(
                    "guard",
                    level="WARN",
                    kind="context_overflow",
                    retry=st["overflow_retries"],
                    cleared=cleared,
                )
                _debug(
                    "CONTEXT_OVERFLOW",
                    f"compaction dure (keep={keep}) : {cleared} résultat(s) d'outil "
                    f"vidé(s), retry {st['overflow_retries']}/{max_overflow_retries}.",
                )
                yield (
                    "tool_result",
                    {
                        "name": "(compaction)",
                        "ok": True,
                        "preview": (
                            f"Contexte saturé : {cleared} ancien(s) résultat(s) "
                            "d'outil allégé(s) pour libérer de la place. Je reprends "
                            "où j'en étais."
                        ),
                    },
                )
                yield (
                    "context_estimate",
                    {"tokens": _ctx_estimate(system_prompt, convo)},
                )
                st["action"] = "continue"
                return
            # ÉTAGE 3 : les tool results sont tous vidés et ça déborde encore -> ce
            # sont les TOURS du modèle qui saturent. On RÉSUME les vieux tours en un
            # bloc dense (anglais) et on poursuit, au lieu d'abandonner. Une seule
            # fois (max_summaries) ; borné pour ne pas boucler.
            # LOCAL UNIQUEMENT : un distant a une grande fenêtre + gère son propre
            # contexte/cache ; on ne réécrit pas son historique (cache-bust). S'il
            # déborde vraiment (rare), on tombe sur l'arrêt propre ci-dessous.
            if not self.is_remote(model) and st["summary_retries"] < max_summaries:
                st["summary_retries"] += 1
                # VISIBLE pendant le résumé (appel modèle bloquant, muet) : label
                # d'activité « compaction… » comme « le modèle tourne », sinon
                # l'UI paraît figée. Effacé dès que le tool_result/texte reprend.
                yield ("status", {"label": "compaction du contexte (résumé)…"})
                collapsed = self.summarize_old_turns(
                    convo, model, keep_recent=6, budget_chars=30000
                )
                if collapsed:
                    st["overflow_retries"] = (
                        0  # convo réduit : le microcompact peut resservir
                    )
                    log_event(
                        "guard",
                        level="WARN",
                        kind="context_summarized",
                        collapsed=collapsed,
                    )
                    _debug(
                        "CONTEXT_SUMMARY",
                        f"{collapsed} ancien(s) tour(s) résumé(s) en un bloc, "
                        "reprise à partir du résumé.",
                    )
                    yield (
                        "tool_result",
                        {
                            "name": "(résumé de session)",
                            "ok": True,
                            "preview": (
                                f"Contexte saturé : {collapsed} anciens tours "
                                "résumés en un bloc dense pour libérer de la place. "
                                "Je reprends à partir du résumé."
                            ),
                        },
                    )
                    yield (
                        "context_estimate",
                        {"tokens": _ctx_estimate(system_prompt, convo)},
                    )
                    st["action"] = "continue"
                    return
            # ÉTAGE 4 : FORCE-FIT déterministe (AUCUN LLM) — on ne s'arrête JAMAIS
            # pour saturation. Le résumé a gardé des messages récents encore trop
            # gros ? On CLIPPE le contexte vivant sous un budget qui RÉTRÉCIT à chaque
            # passe (géométrique : contre l'erreur d'estimation chars/token et le cas
            # où même clippé ça re-déborde), puis on relance. Converge toujours.
            st["force_fits"] += 1
            shrink = max(0.12, 0.7 ** st["force_fits"])
            base = compact_after_tokens or _ctx_estimate(system_prompt, convo) or 8000
            # Même plancher « système + minimal » qu'au préventif : la pression
            # géométrique s'applique au CONVO, pas à l'incompressible. Si le
            # prompt système seul dépasse la vraie fenêtre, on finit sur l'arrêt
            # context_irreducible (cas anormal), pas sur une destruction stérile.
            budget = max(len(system_prompt) + 1500, int(base * 3 * shrink))
            _force_fit(convo, system_prompt, budget)
            if refocus_note and not st["refocus_done"]:
                st["refocus_done"] = True
                convo.append({"role": "user", "content": _REFOCUS_NOTE})
            log_event(
                "guard",
                level="WARN",
                kind="context_force_fit",
                force_fit=st["force_fits"],
                est_tokens=_ctx_estimate(system_prompt, convo),
            )
            _debug(
                "FORCE_FIT",
                f"passe {st['force_fits']} : contexte clippé sous ~{budget} car. "
                f"(~{_ctx_estimate(system_prompt, convo)} tokens), reprise.",
            )
            yield (
                "tool_result",
                {
                    "name": "(compaction forcée)",
                    "ok": True,
                    "preview": (
                        f"Contexte réduit de force (passe {st['force_fits']}) pour tenir "
                        "dans la fenêtre. Je continue."
                    ),
                },
            )
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
            if st["force_fits"] < 8:
                st["action"] = "continue"
                return
            # Garde-fou ANTI-RUNAWAY : après 8 réductions géométriques (budget ~12 %
            # du seuil), si ça déborde ENCORE, c'est dégénéré (prompt système ~ la
            # fenêtre entière) — là seulement, on s'arrête pour ne pas boucler.
            yield (
                "content",
                "\n[génération interrompue : contexte irréductible même après "
                "compaction forcée — cas anormal (prompt système trop grand pour la "
                "fenêtre ?). Le travail déjà écrit est conservé.]",
            )
            yield ("done", {"reason": "context_irreducible"})
            st["action"] = "done"
            return
        # OVERFLOW : tool_call vraisemblablement tronqué par max_tokens (5xx ou
        # erreur sans statut). On NE crashe PAS : on demande de découper et on
        # relance (reprise bornée par max_overflow_retries), sinon stop propre.
        if kind == "overflow":
            if st["overflow_retries"] >= max_overflow_retries:
                yield (
                    "content",
                    f"\n[génération interrompue : {str(exc)[:160]}. "
                    "Fichiers déjà écrits conservés.]",
                )
                yield ("done", {"reason": "output_overflow"})
                st["action"] = "done"
                return
            st["overflow_retries"] += 1
            log_event(
                "guard",
                level="WARN",
                kind="output_overflow",
                retry=st["overflow_retries"],
            )
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
            st["action"] = "continue"
            return
        # Erreurs NON récupérables : pas un overflow -> message clair et stop net,
        # PAS de « écris plus court » trompeur ni de retry voué à re-échouer.
        reason = {
            "timeout": (
                "le serveur a mis trop de temps à répondre (timeout) — souvent "
                "un long recalcul de contexte (après compaction) ; relance, le "
                "cache rend la reprise plus rapide."
            ),
            # « injoignable » : dire QUOI lancer — loom.web tourne forcément (il
            # affiche ce message), c'est le serveur MODÈLE qui manque (ou l'API).
            "connection": (
                "API distante injoignable (réseau ou base_url à vérifier)."
                if self.is_remote(model or self.model)
                else "serveur de modèle local injoignable — lance la stack "
                "modèle (llama-swap / serve) ou choisis un modèle distant."
            ),
            "model_not_found": (
                f"modèle « {model or self.model} » introuvable ou non chargé "
                "(vérifie le modèle sélectionné)."
            ),
            "other": f"erreur du serveur de modèle : {str(exc)[:160]}",
        }[kind]
        yield ("content", f"\n[génération interrompue : {reason}]")
        yield ("done", {"reason": "api_error", "kind": kind})
        st["action"] = "done"

    def stream_chat_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 2048,
        model: str | None = None,
        registry=None,
        thinking: bool = True,
        max_iters: int = 500,
        permission=None,
        confirm=None,
        max_overflow_retries: int = 2,
        max_summaries: int = 1,
        repeat_limit: int = 3,
        compact_after_tokens: int | None = None,
        keep_recent_tools: int = 4,
        max_act_nudges: int = 2,
        max_length_continues: int = 30,
        max_loop_breaks: int = 2,
        max_empty_retries: int = 2,
        strong: bool = False,
        notes_provider=None,
        refocus_note: bool = True,
    ) -> Iterator[tuple[str, object]]:
        """Boucle tool-use : relaie le texte, exécute les outils, relance le modèle.

        `notes_provider` (optionnel) : callable sans argument renvoyant les REMARQUES
        de l'utilisateur arrivées PENDANT le tour (« notes en vol », façon Claude
        Code). Drainées avant CHAQUE appel modèle : chacune est injectée dans la
        conversation (role user, préfixe explicite) et ré-émise en event ('note',
        texte injecté) pour que l'appelant la persiste/affiche. Une note ne stoppe
        jamais le tour — elle l'infléchit au prochain point d'arrêt.

        Yield les mêmes tuples que stream_chat — ('reasoning'|'content', str) —
        plus ('tool_call', {id,name,arguments}) et ('tool_result', {id,name,ok,
        preview}).

        L'ARRÊT est piloté par le modèle (stop naturel) : dès qu'il répond SANS
        tool_call, on sort. Par-dessus, deux garde-fous non-négociables (best
        practice agentic : le modèle, surtout petit, ne sait pas toujours s'arrêter) :
        - `max_iters` : backstop ANTI-RUNAWAY tres haut (defaut 500) — PAS un cap de
          progression ; l'arret normal vient du non-progres (repeat_limit) et du coupe-circuit anti-boucle ;
        - `repeat_limit` : non-progrès — si le modèle réémet `repeat_limit` fois de
          suite EXACTEMENT le même jeu d'appels (mêmes outils + mêmes args), il
          tourne en rond, on coupe. Chaque garde-fou émet un message d'arrêt EXPLICITE
          (on sait que c'est la sécurité, pas une fin normale).

        PAS de mur de temps : sur un modèle local lent, un chrono global décapitait
        la boucle en plein travail (cf. session démineur). Les bornes sont le NOMBRE
        de tours et le NON-PROGRÈS, jamais l'horloge.

        Chaque sortie émet un event terminal ('done', {'reason': ...}) : 'natural'
        (stop du modèle), 'repeat_stop', 'loop_degenerate', 'max_iters',
        'context_irreducible', 'output_overflow', 'api_error', 'empty_response'
        (réponse vide malgré les relances). Les consommateurs
        qui ne s'en servent pas l'ignorent (dispatch if/elif) ; les évals s'en
        servent comme stop_reason mesurable au lieu de pattern-matcher les textes.
        """
        convo = list(messages)
        # Résolu une fois : le modèle est fixe pour tout l'appel. Route vers l'endpoint
        # local ou distant selon l'id, et coupe les extra_body llama.cpp si distant.
        oai, api_model, native = self._resolve(model)
        tools = registry.openai_tools() if registry else None
        # État PARTAGÉ de la boucle : compteurs/garde-fous mutés par les sous-
        # générateurs privés. "action" porte l'issue posée par chaque helper :
        # "continue" (relancer le tour), "done" (sortie — l'event terminal
        # ('done', …) a déjà été yieldé) ou "proceed" (on poursuit le tour).
        st: dict = {
            "overflow_retries": 0,
            "summary_retries": 0,  # nb de compactions PAR RÉSUMÉ déjà tentées
            "force_fits": 0,  # nb de réductions DÉTERMINISTES forcées (dernier recours, jamais d'arrêt)
            "prev_sig_set": None,  # jeu d'appels du tour précédent (détecteur de non-progrès)
            "repeat_streak": 0,
            "executed": False,  # un run_shell / dispatch_agent a-t-il réellement tourné ce tour ?
            "files_written": set(),  # chemins écrits avec succès ce tour (couche A)
            "act_nudges": 0,  # nb de relances « passe de la parole à l'acte » déjà émises
            "length_continues": 0,  # nb de relances « continue » sur troncature max_tokens
            "loop_breaks": 0,  # nb de coupes « tu répètes la même phrase, agis » déjà émises
            "fail_count": 0,  # échecs cumulés d'outils d'exécution/vérif ce tour (cascade de bugs)
            "debug_forced": False,  # méthode debug déjà imposée ce tour ? (anti-nag)
            "refocus_done": False,  # note de recentrage post-force-fit déjà émise ? (une seule)
            "empty_retries": 0,  # nb de relances sur réponse VIDE (0 texte, 0 tool call)
            "truncated_streak": 0,  # troncatures d'arguments d'outil CONSÉCUTIVES
            "verify_streak": 0,  # checks navigateur verts consécutifs (anti sur-vérification)
            "text": "",  # texte accumulé du dernier appel modèle
            "reasoning": "",  # raisonnement accumulé du dernier appel modèle
            "action": "",  # issue posée par le dernier sous-générateur
        }

        for _ in range(max_iters):
            # Compaction préventive (microcompact + force-fit) AVANT l'appel modèle.
            yield from self._preventive_compaction(
                convo,
                system_prompt,
                model,
                compact_after_tokens,
                keep_recent_tools,
                refocus_note,
                st,
            )
            # Notes en vol injectées au point d'arrêt (juste avant l'appel modèle).
            yield from _inject_notes(notes_provider, convo)
            kwargs = build_create_kwargs(
                api_model,
                convo,
                system_prompt,
                max_tokens,
                thinking,
                tools=tools,
                native_extras=native,
            )
            _debug_messages(kwargs["model"], kwargs["messages"])
            collector: dict = {"tool_calls": [], "finish_reason": None}
            try:
                yield from _stream_model_turn(
                    oai,
                    api_model,
                    kwargs,
                    system_prompt,
                    convo,
                    collector,
                    tools,
                    thinking,
                    st,
                )
            except (APIError, httpx.HTTPError) as exc:
                yield from self._handle_stream_api_error(
                    exc,
                    convo,
                    system_prompt,
                    model,
                    compact_after_tokens,
                    max_overflow_retries,
                    max_summaries,
                    refocus_note,
                    st,
                )
                if st["action"] == "done":
                    return
                continue
            text, reasoning = st["text"], st["reasoning"]

            tool_calls = collector["tool_calls"]
            # FILET : appel d'outil émis en TEXTE (channel structuré vide) ? On le récupère
            # et on l'exécute, au lieu de s'arrêter sur un appel "raté".
            if not tool_calls:
                salvaged = _salvage_tool_calls(text, reasoning)
                if salvaged:
                    tool_calls = salvaged
                    _debug(
                        "SALVAGE",
                        f"{len(salvaged)} appel(s) d'outil récupéré(s) du texte.",
                    )
            if not tool_calls:
                # Fin de tour SANS outil : boucle dégénérée / continuation length /
                # réponse vide / audit de claim, sinon stop naturel du modèle.
                yield from _dispatch_no_tool_calls(
                    collector,
                    text,
                    convo,
                    strong,
                    max_loop_breaks,
                    max_length_continues,
                    max_empty_retries,
                    max_act_nudges,
                    st,
                )
                if st["action"] == "done":
                    return
                continue

            yield from _check_no_progress(tool_calls, strong, repeat_limit, st)
            if st["action"] == "done":
                return

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
                                # JSON sain dans l'historique : un appel tronqué (JSON
                                # cassé) provoquerait un 500 'parse error' à CHAQUE tour
                                # suivant -> cascade. _safe_args remet {} si invalide.
                                "arguments": _safe_args(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            # PARALLÉLISME (distant uniquement). Règle Loom : local = inline / 1 slot llama-swap
            # -> on sérialise (un sous-agent local = même slot = zéro gain) ; distant = machine du
            # provider -> on EXPLOITE la concurrence. Cas sûr : un tour n'appelant QUE des outils
            # PARALLEL-SAFE (lectures, recherches, dispatch_agent) -> on les lance concurremment
            # (1 thread chacun, activité relayée en direct via une file, résultats recollés DANS
            # L'ORDRE). Dès qu'un outil non-safe est présent (écriture, shell, todo, vérif, image),
            # ou en local, tout reste SÉQUENTIEL (garde-fous P1.1, confirm, ordre... préservés).
            _seq_tool_calls = tool_calls
            _parallel = (
                registry is not None
                and self.is_remote(model)
                and len(tool_calls) >= 2
                and all(tc.get("name") in _PARALLEL_SAFE for tc in tool_calls)
            )
            if _parallel:
                _seq_tool_calls = []  # la boucle séquentielle ne fait rien ce tour
                yield from _run_tools_parallel(registry, tool_calls, convo, st)
            yield from _run_tools_sequential(
                _seq_tool_calls,
                registry,
                permission,
                confirm,
                convo,
                strong,
                st,
            )
            # Forçage debugging (déterministe) : le modèle n'appelle jamais use_skill seul ; à
            # la 2e erreur d'exécution on injecte la méthode systématique, une seule fois par tour.
            if st["fail_count"] >= 2 and not st["debug_forced"]:
                st["debug_forced"] = True
                convo.append({"role": "user", "content": _DEBUG_FORCE})
                yield (
                    "tool_result",
                    {
                        "name": "(debug)",
                        "ok": True,
                        "preview": "Cascade d'erreurs — méthode debug imposée (cause racine, pas symptôme).",
                    },
                )
        yield (
            "content",
            f"\n(arrêt : backstop anti-runaway atteint après {max_iters} tours d'outils — "
            "cas anormal ; relance pour reprendre là où ça s'est arrêté).",
        )
        yield ("done", {"reason": "max_iters"})


def _close(stream) -> None:
    """Coupe la connexion HTTP au modèle (interruption ou fin de tour)."""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
