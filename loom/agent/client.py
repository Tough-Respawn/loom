# loom/agent/client.py
"""Client modèle : parle à l'endpoint OpenAI-compatible de Loom via le SDK openai."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

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
_VERIFY_TOOLS = frozenset(
    {"run_shell", "check_page", "serve_and_check", "check_interactive"}
)


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


# --- Mode debug (LOOM_DEBUG=1) : trace l'échange avec le modèle dans le terminal -------
_B64_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")


def _debug_on() -> bool:
    return os.environ.get("LOOM_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
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
_DEBUG_LOG = (
    Path(__file__).resolve().parent.parent.parent / "var" / "logs" / "loom-debug.log"
)


def set_debug_log_path(path) -> None:
    """Redirige le trace debug vers `path` (ex. sessions/<id>/debug.log). Le dossier est
    créé à l'écriture. Appelé par la web app au début de chaque tour."""
    global _DEBUG_LOG
    _DEBUG_LOG = Path(path)


def _emit(text: str) -> None:
    """Écrit sur stderr ET dans le fichier de log, sans JAMAIS lever (un crash d'encodage
    ne doit pas casser la génération) : encodage tolérant, caractères non gérés remplacés."""
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
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(text + "\n")
    except Exception:  # noqa: BLE001 - best-effort
        pass


def _debug(label: str, payload, limit: int = 4000) -> None:
    """Imprime un bloc de debug sur stderr (terminal de loom.web), no-op si désactivé.
    Labels ASCII volontairement (pas d'accents/flèches) pour rester lisible sur tout
    terminal Windows."""
    if not _debug_on():
        return
    body = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
    _emit(f"\n===== [LOOM_DEBUG] {label} =====")
    _emit(_trunc(body, limit))


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
    _debug("REQUETE -> modele", "\n".join(lines), limit=60000)


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


def _usage_dict(usage) -> dict:
    """Normalise l'usage (tokens réels) renvoyé par le serveur en fin de stream."""
    return {
        "completion_tokens": getattr(usage, "completion_tokens", None) or 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) or 0,
        "total_tokens": getattr(usage, "total_tokens", None) or 0,
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
    native_extras: bool = True,
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


def _iter_turn(stream, collector: dict) -> Iterator[tuple[str, str]]:
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
_BUG_SIGNAL_TOOLS = frozenset(
    {"run_shell", "check_page", "check_interactive", "format_code"}
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


def _microcompact_tools(convo: list[dict], keep_recent_tools: int) -> int:
    """Vide le CONTENU des plus vieux messages role:tool (garde les `keep_recent_tools`
    derniers intacts), en place. Renvoie le nb de messages allégés."""
    idx = [i for i, m in enumerate(convo) if m.get("role") == "tool"]
    older = idx[:-keep_recent_tools] if keep_recent_tools else idx
    n = 0
    for i in older:
        if convo[i].get("content") != _CLEARED_TOOL:
            convo[i] = {**convo[i], "content": _CLEARED_TOOL}
            n += 1
    return n


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
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        # Routes vers des modèles DISTANTS (API OpenAI-compatible) : id de modèle ->
        # {base_url, api_key, model, enable_thinking_param}. Tout modèle absent de cette
        # table part vers l'endpoint LOCAL (_client). Un client openai par endpoint, monté
        # une fois ici. Le reste du code appelle toujours `model=<id>` : le routage est interne.
        self._routes: dict[str, dict] = {}
        for rid, spec in (routes or {}).items():
            self._routes[rid] = {
                "client": OpenAI(
                    base_url=spec["base_url"],
                    api_key=spec.get("api_key") or "none",
                    timeout=timeout,
                    max_retries=max_retries,
                ),
                "model": spec.get("model") or rid,
                "enable_thinking_param": bool(spec.get("enable_thinking_param", False)),
            }

    def _resolve(self, model: str | None):
        """(client_openai, model_api, native_extras) pour le modèle demandé. Un modèle
        distant (présent dans routes) part vers son endpoint, SANS les extra_body llama.cpp ;
        sinon l'endpoint local avec les extras natifs."""
        if model and model in self._routes:
            r = self._routes[model]
            return r["client"], r["model"], r["enable_thinking_param"]
        return self._client, (model or self.model), True

    def local_server_root(self) -> str:
        """Racine du serveur LOCAL (llama-swap) : la base_url SANS le suffixe `/v1`.
        Sert à joindre l'API de management (`/running`, `/api/models/unload`)."""
        b = self.base_url
        return b[:-3] if b.endswith("/v1") else b.rstrip("/")

    def is_remote(self, model: str | None) -> bool:
        """Vrai si le modèle est servi par une API DISTANTE (route montée), pas en local."""
        return bool(model and model in self._routes)

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
        except Exception:  # noqa: BLE001 - warmup best-effort, jamais bloquant
            pass

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
        repeat_limit: int = 3,
        compact_after_tokens: int | None = None,
        keep_recent_tools: int = 4,
        max_act_nudges: int = 2,
        max_length_continues: int = 30,
        max_loop_breaks: int = 2,
        strong: bool = False,
    ) -> Iterator[tuple[str, object]]:
        """Boucle tool-use : relaie le texte, exécute les outils, relance le modèle.

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
        """
        convo = list(messages)
        # Résolu une fois : le modèle est fixe pour tout l'appel. Route vers l'endpoint
        # local ou distant selon l'id, et coupe les extra_body llama.cpp si distant.
        oai, api_model, native = self._resolve(model)
        tools = registry.openai_tools() if registry else None
        overflow_retries = 0
        prev_sig_set = None  # jeu d'appels du tour précédent (détecteur de non-progrès)
        repeat_streak = 0
        executed = (
            False  # un run_shell / dispatch_agent a-t-il réellement tourné ce tour ?
        )
        files_written: set[str] = set()  # chemins écrits avec succès ce tour (couche A)
        act_nudges = 0  # nb de relances « passe de la parole à l'acte » déjà émises
        length_continues = 0  # nb de relances « continue » sur troncature max_tokens
        loop_breaks = 0  # nb de coupes « tu répètes la même phrase, agis » déjà émises
        fail_count = (
            0  # échecs cumulés d'outils d'exécution/vérif ce tour (cascade de bugs)
        )
        debug_forced = False  # méthode debug déjà imposée ce tour ? (anti-nag)
        for _ in range(max_iters):
            # Microcompact : si le contexte vivant approche la fenêtre, vider les vieux
            # résultats d'outils AVANT d'appeler le modèle (évite l'overflow sur une
            # chaîne longue). Estimation grossière ~4 car./token, comme loom.context.
            if compact_after_tokens:
                # ~3 car./token (et non 4) : code/TSX/JSON tokenise plus dense que de la prose.
                # Surestimer fait déclencher la compaction PLUS TÔT — biais voulu (on ne vide
                # que les vieux résultats), pour ne pas heurter la fenêtre par sous-comptage.
                approx = (
                    len(system_prompt)
                    + sum(_msg_chars(m.get("content")) for m in convo)
                ) // 3
                if approx > compact_after_tokens:
                    cleared = _microcompact_tools(convo, keep_recent_tools)
                    if cleared:
                        _debug(
                            "MICROCOMPACT",
                            f"{cleared} résultat(s) d'outil allégé(s) (~{approx} tokens "
                            f"> seuil {compact_after_tokens}).",
                        )
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
            try:
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
            except APIError as exc:
                kind = _classify_api_error(exc)
                log_event("api.error", level="WARN", kind=kind, msg=str(exc)[:140])
                # DÉBORDEMENT D'ENTRÉE : la requête (prompt + historique + résultats d'outils
                # accumulés) dépasse la fenêtre de contexte. On NE crashe PAS et on ne demande
                # PAS « écris plus court » (ça vise la sortie) : on COMPACTE DUR — on vide TOUS
                # les vieux résultats d'outils (pas seulement au-delà des 4 derniers), de plus
                # en plus agressivement à chaque retry — puis on RELANCE le même tour. Le modèle
                # reprend avec SES messages (ce qu'il a déjà fait) intacts ; les gros résultats
                # d'outils deviennent le placeholder _CLEARED_TOOL (qui dit de ne pas refaire).
                if kind == "context_overflow":
                    if overflow_retries >= max_overflow_retries:
                        yield (
                            "content",
                            "\n[génération interrompue : contexte saturé même après "
                            "compaction. Le travail déjà écrit est conservé ; relance une "
                            "demande plus ciblée pour continuer.]",
                        )
                        return
                    overflow_retries += 1
                    keep = 1 if overflow_retries == 1 else 0  # 2e retry : on vide tout
                    cleared = _microcompact_tools(convo, keep)
                    log_event(
                        "guard",
                        level="WARN",
                        kind="context_overflow",
                        retry=overflow_retries,
                        cleared=cleared,
                    )
                    _debug(
                        "CONTEXT_OVERFLOW",
                        f"compaction dure (keep={keep}) : {cleared} résultat(s) d'outil vidé(s), "
                        f"retry {overflow_retries}/{max_overflow_retries}.",
                    )
                    yield (
                        "tool_result",
                        {
                            "name": "(compaction)",
                            "ok": True,
                            "preview": (
                                f"Contexte saturé : {cleared} ancien(s) résultat(s) d'outil "
                                "résumé(s) pour libérer de la place. Je reprends où j'en étais."
                            ),
                        },
                    )
                    continue
                # OVERFLOW : tool_call vraisemblablement tronqué par max_tokens (5xx ou
                # erreur sans statut). On NE crashe PAS : on demande de découper et on
                # relance (reprise bornée par max_overflow_retries), sinon stop propre.
                if kind == "overflow":
                    if overflow_retries >= max_overflow_retries:
                        yield (
                            "content",
                            f"\n[génération interrompue : {str(exc)[:160]}. "
                            "Fichiers déjà écrits conservés.]",
                        )
                        return
                    overflow_retries += 1
                    log_event(
                        "guard",
                        level="WARN",
                        kind="output_overflow",
                        retry=overflow_retries,
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
                    continue
                # Erreurs NON récupérables : pas un overflow -> message clair et stop net,
                # PAS de « écris plus court » trompeur ni de retry voué à re-échouer.
                reason = {
                    "timeout": "le serveur a mis trop de temps à répondre (timeout).",
                    "connection": "serveur de modèle injoignable (Loom est-il lancé ?).",
                    "model_not_found": (
                        f"modèle « {model or self.model} » introuvable ou non chargé "
                        "(vérifie le modèle sélectionné)."
                    ),
                    "other": f"erreur du serveur de modèle : {str(exc)[:160]}",
                }[kind]
                yield ("content", f"\n[génération interrompue : {reason}]")
                return

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
                # BOUCLE DE DÉGÉNÉRESCENCE (détectée au streaming) : le modèle a répété la
                # même phrase sans agir. À traiter AVANT la continuation 'length' : lui dire
                # « continue où tu t'es arrêté » ne ferait qu'alimenter le cycle. On coupe et
                # on relance avec un ordre FERME d'émettre un appel d'outil, borné.
                if collector.get("looped"):
                    if loop_breaks >= max_loop_breaks:
                        yield (
                            "content",
                            "\n[génération interrompue : le modèle tournait en boucle (même "
                            "phrase répétée) sans agir. Reformule ou découpe la demande.]",
                        )
                        return
                    loop_breaks += 1
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
                        f"relance {loop_breaks}/{max_loop_breaks}",
                    )
                    yield (
                        "tool_result",
                        {"name": "(boucle)", "ok": False, "preview": nudge},
                    )
                    continue
                # CONTINUATION sur troncature : la réponse texte/raisonnement a été coupée
                # par la limite de tokens (finish_reason == "length") sans appel d'outil.
                # Plutôt que de rendre une réponse tronquée, on relance le modèle pour qu'il
                # POURSUIVE là où il s'est arrêté. Autant de fois que nécessaire (cap dur
                # max_length_continues, anti-runaway). Le texte continue d'être streamé à
                # l'UI tour après tour (le web app concatène). Cas des tool_calls tronqués
                # NON concerné (géré par 'arguments tronqués' / overflow).
                if (
                    collector["finish_reason"] == "length"
                    and length_continues < max_length_continues
                ):
                    length_continues += 1
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
                        f"relance {length_continues}/{max_length_continues}",
                    )
                    continue
                # Audit de claim au stop : le modèle prétend-il un résultat qu'il n'a pas
                # produit ? (A) artefact fichier inventé, (B) résultat d'exécution sans
                # run_shell/dispatch, ou intention/affirmation sans exécution réelle. On le
                # relance pour qu'il FASSE vraiment (borné). Garde de vérité, pas orchestrateur.
                # COUPÉ pour un modèle FORT (distant) : ces relances de comportement, utiles à
                # un petit modèle qui confabule, ne font que sur-piloter un modèle qui se vérifie
                # déjà seul (cf. GLM qui doutait de sa propre preuve correcte).
                missing = _claims_missing_artifact(text, files_written)
                exec_confab = not executed and _claims_execution(text)
                if (
                    not strong
                    and act_nudges < max_act_nudges
                    and (missing or exec_confab or _intends_to_act(text, executed))
                ):
                    act_nudges += 1
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
                    continue
                return  # réponse finale déjà streamée (stop naturel du modèle)

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
                repeat_streak = 0
                prev_sig_set = None
            else:
                repeat_streak = repeat_streak + 1 if sig_set == prev_sig_set else 0
                prev_sig_set = sig_set
                # COUPÉ pour un modèle FORT (distant) : le seul backstop reste max_iters. Sur un
                # petit modèle, la répétition = dégénérescence ; sur un fort, c'est presque
                # toujours du légitime (le juger « bloqué » l'interrompt à tort).
                if not strong and repeat_streak >= repeat_limit - 1:
                    log_event("guard", level="WARN", kind="repeat_stop")
                    yield (
                        "content",
                        "\n(arrêt : le modèle réémet les mêmes appels sans progresser).",
                    )
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
                # P1.1 : sérialiser les écritures à gros contenu (1 par tour) -> évite le
                # batch de N gros write_file/append_file qui sature max_tokens et tronque.
                # Les éditions par bloc (edit/replace/insert) ne passent PAS par ici.
                if name in _SERIAL_WRITE and wrote_this_turn:
                    result = (
                        "différé : un seul write_file/append_file par tour. Réémets "
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
                elif registry and registry.is_streaming(name):  # allow + streamant
                    # Outil streamant (dispatch_agent) : on relaie son activité EN DIRECT
                    # dans sa pastille (tool_stream) et on reconstruit la synthèse finale.
                    parts: list[str] = []
                    for sub_kind, sub_payload in registry.run_stream(name, args):
                        line = _sub_activity_line(sub_kind, sub_payload)
                        if line:
                            yield ("tool_stream", {"id": tc["id"], "text": line})
                        if sub_kind == "content" and isinstance(sub_payload, str):
                            parts.append(sub_payload)
                    result = (
                        "".join(parts).strip() or "(le sous-agent n'a rien renvoyé)"
                    )
                    ok = not result.startswith("erreur")
                else:  # allow
                    result = (
                        registry.run(name, args) if registry else "erreur: pas d'outils"
                    )
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
                convo.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": tool_content}
                )
                # Détail dépliable côté UI : pour les écritures, le contenu RÉELLEMENT
                # écrit (et non le message de retour) ; pour edit, le diff old/new ;
                # sinon le résultat (l'accusé pour une image, pas son base64). Borné.
                if name == "write_file":
                    detail = args.get("content") or ""
                elif name == "edit_file":
                    detail = f"- {args.get('old_string', '')}\n+ {args.get('new_string', '')}"
                else:
                    detail = tool_content
                # Vue IN/OUT de la pastille : in_full = ce que l'outil a REÇU (commande shell,
                # contenu écrit, diff, ou chemin/args), out_full = ce qu'il a RENVOYÉ. La preview
                # (1 ligne) reste pour l'état replié ; detail conservé pour rétro-compat.
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
                    import json as _json

                    in_full = args.get("path") or _json.dumps(args, ensure_ascii=False)
                log_event(
                    "tool.result",
                    name=name,
                    ok=ok,
                    ms=round((time.monotonic() - _t_tool) * 1000),
                    preview=str(tool_content)[:90],
                )
                yield (
                    "tool_result",
                    {
                        "id": tc["id"],
                        "name": name,
                        "ok": ok,
                        "preview": tool_content[:300],
                        "path": args.get("path"),
                        # Commande réellement lancée par run_shell : pour la VOIR dans la
                        # pastille (sinon on ne voit que le résultat, pas ce qui a tourné).
                        "cmd": args.get("command"),
                        "detail": detail[:4000] if detail else None,
                        "in_full": str(in_full)[:8000],
                        "out_full": str(tool_content)[:8000],
                    },
                )
                if name in _SERIAL_WRITE:
                    wrote_this_turn = True
                # Suivi pour l'audit de claim : une EXÉCUTION réelle (run_shell/dispatch,
                # même en échec mais hors refus de permission) et les FICHIERS écrits.
                if name in ("run_shell", "dispatch_agent") and not str(
                    result
                ).startswith("refusé"):
                    executed = True
                if ok and name in _WRITE_TOOLS and args.get("path"):
                    files_written.add(args["path"])
                # Cascade de bugs : on compte les échecs des outils d'EXÉCUTION/VÉRIF (pas les
                # erreurs d'usage type ligne hors limite). Au 2e échec, on IMPOSE la méthode debug.
                if not ok and name in _BUG_SIGNAL_TOOLS:
                    fail_count += 1
            convo.extend(image_followups)  # images vues au tour suivant
            # Forçage debugging (déterministe) : le modèle n'appelle jamais use_skill seul ; à
            # la 2e erreur d'exécution on injecte la méthode systématique, une seule fois par tour.
            if fail_count >= 2 and not debug_forced:
                debug_forced = True
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


def _close(stream) -> None:
    """Coupe la connexion HTTP au modèle (interruption ou fin de tour)."""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
