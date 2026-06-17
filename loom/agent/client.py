# loom/agent/client.py
"""Client modèle : parle à l'endpoint OpenAI-compatible de Loom via le SDK openai."""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from loom.agent.inline_image import (
    image_user_message,
    is_inline_image,
    parse_inline_image,
)

# Écritures à GROS contenu intégral : sérialisées (1/tour) pour qu'un batch de N ne
# sature pas max_tokens et ne tronque pas les derniers (P1.1). On NE sérialise QUE
# celles-ci : les éditions par bloc (edit_file/replace_lines/insert_lines) écrivent peu
# -> pas de risque d'overflow, et les laisser passer ensemble réduit le nombre de tours
# d'un refactor multi-fichiers (cf. plafond max_iters).
_SERIAL_WRITE = frozenset({"write_file", "append_file"})


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
# client.py vit dans loom/agent/ : remonter de DEUX niveaux pour viser loom/data/
# (et non loom/agent/data/, créé par erreur après la réorg en sous-paquets).
_DEBUG_LOG = Path(__file__).resolve().parent.parent / "data" / "loom-debug.log"


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
        lines.append(f"  [{m.get('role')}] {_trunc((content or '') + extra, 1500)}")
    _debug("REQUETE -> modele", "\n".join(lines), limit=20000)


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
    "[résultat d'outil ancien retiré pour garder de la place dans le contexte. "
    "Ne re-lis pas le fichier en boucle : si une info t'est encore utile plus tard, "
    "consigne-la avec write_note pendant que tu l'as, puis relis ta note (read_note).]"
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
_WRITE_TOOLS = frozenset(
    {"write_file", "append_file", "edit_file", "replace_lines", "insert_lines"}
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
        _debug_messages(kwargs["model"], kwargs["messages"])
        stream = self._client.chat.completions.create(**kwargs)
        reasoning, content = "", ""
        try:
            for kind, chunk in _iter_events(stream):
                if kind == "reasoning":
                    reasoning += chunk
                elif kind == "content":
                    content += chunk
                yield (kind, chunk)
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
        max_iters: int = 30,
        permission=None,
        confirm=None,
        max_overflow_retries: int = 2,
        repeat_limit: int = 3,
        compact_after_tokens: int | None = None,
        keep_recent_tools: int = 4,
        max_act_nudges: int = 2,
        max_length_continues: int = 30,
    ) -> Iterator[tuple[str, object]]:
        """Boucle tool-use : relaie le texte, exécute les outils, relance le modèle.

        Yield les mêmes tuples que stream_chat — ('reasoning'|'content', str) —
        plus ('tool_call', {id,name,arguments}) et ('tool_result', {id,name,ok,
        preview}).

        L'ARRÊT est piloté par le modèle (stop naturel) : dès qu'il répond SANS
        tool_call, on sort. Par-dessus, deux garde-fous non-négociables (best
        practice agentic : le modèle, surtout petit, ne sait pas toujours s'arrêter) :
        - `max_iters` : plafond dur de tours d'outils (anti-runaway) ;
        - `repeat_limit` : non-progrès — si le modèle réémet `repeat_limit` fois de
          suite EXACTEMENT le même jeu d'appels (mêmes outils + mêmes args), il
          tourne en rond, on coupe. Chaque garde-fou émet un message d'arrêt EXPLICITE
          (on sait que c'est la sécurité, pas une fin normale).

        PAS de mur de temps : sur un modèle local lent, un chrono global décapitait
        la boucle en plein travail (cf. session démineur). Les bornes sont le NOMBRE
        de tours et le NON-PROGRÈS, jamais l'horloge.
        """
        convo = list(messages)
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
        for _ in range(max_iters):
            # Microcompact : si le contexte vivant approche la fenêtre, vider les vieux
            # résultats d'outils AVANT d'appeler le modèle (évite l'overflow sur une
            # chaîne longue). Estimation grossière ~4 car./token, comme loom.context.
            if compact_after_tokens:
                approx = (
                    len(system_prompt)
                    + sum(_msg_chars(m.get("content")) for m in convo)
                ) // 4
                if approx > compact_after_tokens:
                    cleared = _microcompact_tools(convo, keep_recent_tools)
                    if cleared:
                        _debug(
                            "MICROCOMPACT",
                            f"{cleared} résultat(s) d'outil allégé(s) (~{approx} tokens "
                            f"> seuil {compact_after_tokens}).",
                        )
            kwargs = build_create_kwargs(
                model or self.model,
                convo,
                system_prompt,
                max_tokens,
                thinking,
                tools=tools,
            )
            _debug_messages(kwargs["model"], kwargs["messages"])
            collector: dict = {"tool_calls": [], "finish_reason": None}
            text = ""
            reasoning = ""
            try:
                stream = self._client.chat.completions.create(**kwargs)
                try:
                    for kind, chunk in _iter_turn(stream, collector):
                        if kind == "content":
                            text += chunk
                        elif kind == "reasoning":
                            reasoning += chunk
                        yield (kind, chunk)
                finally:
                    _close(stream)
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
                missing = _claims_missing_artifact(text, files_written)
                exec_confab = not executed and _claims_execution(text)
                if act_nudges < max_act_nudges and (
                    missing or exec_confab or _intends_to_act(text, executed)
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
                    continue
                return  # réponse finale déjà streamée (stop naturel du modèle)

            # Non-progrès : même jeu d'appels (outils+args) que le tour précédent ?
            sig_set = frozenset(
                f"{tc['name']}\x00{tc['arguments']}" for tc in tool_calls
            )
            repeat_streak = repeat_streak + 1 if sig_set == prev_sig_set else 0
            prev_sig_set = sig_set
            if repeat_streak >= repeat_limit - 1:
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
            convo.extend(image_followups)  # images vues au tour suivant
        yield (
            "content",
            f"\n(arrêt : garde-fou anti-boucle après {max_iters} tours d'outils — "
            "la tâche n'est peut-être pas terminée).",
        )


def _close(stream) -> None:
    """Coupe la connexion HTTP au modèle (interruption ou fin de tour)."""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass
