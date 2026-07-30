from __future__ import annotations

from __future__ import annotations
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any




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


def context_fingerprint(messages: list[dict]) -> str:
    """Empreinte compacte du contexte envoyé : `rôle:longueur:md5-8` par message.

    Sert à DIFFER deux tours pour localiser LA mutation de préfixe qui casse le
    cache KV (vécu 2026-07-21 : cache_tok=0 -> re-prefill complet de 62 s au
    milieu d'une session ; le préfixe avait rétréci de ~270 tokens entre deux
    tours — quelle section a muté reste à identifier, c'est le rôle de ce log).
    Le premier couple qui diffère entre deux lignes = le point de divergence.
    `tools` : les SCHÉMAS d'outils sont rendus par le chat template EN TÊTE de
    prompt — une variation (ordre, champ dynamique) casse le préfixe au token ~0
    en laissant les messages byte-identiques (signature : sim élevée côté
    serveur, cached_tokens=0). Ils font donc partie de l'empreinte."""
    import hashlib

    parts = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        content = content or ""
        if m.get("tool_calls"):
            content += json.dumps(m["tool_calls"], ensure_ascii=False, default=str)
        h = hashlib.md5(content.encode("utf-8", "replace")).hexdigest()[:8]
        parts.append(f"{(m.get('role') or '?')[:1]}:{len(content)}:{h}")
    return " ".join(parts)


def tools_fingerprint(tools: list | None) -> str:
    """Empreinte des schémas d'outils tels qu'envoyés (ordre PRÉSERVÉ : c'est
    l'ordre rendu par le template, donc l'ordre qui compte pour le cache)."""
    import hashlib

    if not tools:
        return "T:0:-"
    blob = json.dumps(tools, ensure_ascii=False, default=str)
    return f"T:{len(tools)}:{len(blob)}:{hashlib.md5(blob.encode()).hexdigest()[:8]}"


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
    # Empreinte par message (rôle:longueur:md5-8) : diffable entre tours pour
    # localiser une mutation de préfixe qui invalide le cache KV.
    _debug("CTX_EMPREINTE", context_fingerprint(messages), terminal=False)
