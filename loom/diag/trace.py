"""Retrace une session Loom : journal client + journal llama-server sur une seule
ligne de temps, résumé par appel et alertes.

    uv run python -m loom.diag.trace <id_session> [--out rapport.md]

Né du post-mortem Bonsai 2 (2026-09-30), où six relectures ont été nécessaires
pour reconstruire à la main ce que cet outil calcule :

- le journal de session (var/sessions/<id>/debug.log) donne les appels modèle
  (`call.request` / `call.end` / `prefix.diff`), la maintenance (`maint.*`) et les
  actions de slot (`slot.action`), horodatés en UTC ;
- le journal llama-server (var/logs/llama/<modèle>.log et archive/) donne les
  tâches et les décisions de cache, horodatées depuis le démarrage du serveur.

Les horloges sont recalées en appariant les fins d'appel client (`call.end`) et les
fins de tâche serveur (`release`) : on retient le décalage qui en apparie le plus.
Lecture seule.

Piège connu (cf. revue CR3) : llama-server soumet un lot de prompt au GPU sans
attendre la fin du calcul, ses lignes « prompt processing » sont donc décalées d'un
lot. Seuls les totaux par tâche (et les timings renvoyés au client) sont fiables ;
l'outil ne tire aucune conclusion des lignes de progression.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS_ROOT = REPO_ROOT / "var" / "sessions"
LLAMA_LOGS = REPO_ROOT / "var" / "logs" / "llama"

_CLIENT_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z) \[(\w+)\] (\S+)(.*)$"
)
_FIELD = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
_SERVER_LINE = re.compile(r"^(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) ([A-Z]) (.*)$")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_TASK = re.compile(r"\| task (\d+) \|")

# Lignes serveur retenues sur la ligne de temps (le reste est du bruit).
_SERVER_KEYS = (
    "launch_slot_",
    "release:",
    "stop processing",
    "save_slot_ch",
    "load_slot_ch",
    "restored",
    "erased invalidated",
    "forcing full prompt",
    "saving prompt",
    "looking for better prompt",
    "found better prompt",
    "cancel task",
    "created context checkpoint",
    "model loaded",
    "exiting",
)
# Lignes émises par les mêmes fonctions mais sans intérêt pour le diagnostic.
_SERVER_NOISE = ("sampler chain", "sampler params")
# Événements client de l'ancien format, redondants avec call.request / call.end.
_CLIENT_LEGACY = ("turn.request", "turn.timing", "usage", "stream.first_byte")


@dataclass
class Event:
    t: datetime
    source: str  # "client" | "serveur"
    name: str
    fields: dict = field(default_factory=dict)
    text: str = ""


def _parse_value(raw: str):
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"')
    if raw in ("true", "false"):
        return raw == "true"
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        return raw


def parse_client_log(path: Path) -> list[Event]:
    """Événements `log_event` d'un debug.log (les blocs de dump sont ignorés)."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _CLIENT_LINE.match(line)
        if not m:
            continue
        ts, _level, name, rest = m.groups()
        t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        fields = {k: _parse_value(v) for k, v in _FIELD.findall(rest)}
        out.append(Event(t, "client", name, fields))
    return out


def parse_server_log(path: Path) -> list[tuple[float, str]]:
    """[(secondes depuis le démarrage du serveur, texte)] d'un journal llama-server."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _SERVER_LINE.match(_ANSI.sub("", line))
        if not m:
            continue
        mi, se, ms, us, level, text = m.groups()
        rel = int(mi) * 60 + int(se) + int(ms) / 1000 + int(us) / 1e6
        out.append((rel, f"{level} {text}"))
    return out


def estimate_offset(
    client_ends: list[datetime], server_releases: list[float], tol: float = 0.5
) -> tuple[datetime | None, int]:
    """Décalage (instant UTC du démarrage serveur) qui apparie le plus de fins
    d'appel client avec des fins de tâche serveur, à `tol` secondes près."""
    best, best_n = None, 0
    for c in client_ends:
        for s in server_releases:
            origin = c - timedelta(seconds=s)
            n = sum(
                1
                for c2 in client_ends
                if any(
                    abs((c2 - origin).total_seconds() - s2) <= tol
                    for s2 in server_releases
                )
            )
            if n > best_n:
                best, best_n = origin, n
    return best, best_n


def _server_candidates(model: str) -> list[Path]:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
    cands = [LLAMA_LOGS / f"{safe}.log"]
    cands += sorted((LLAMA_LOGS / "archive").glob(f"{safe}-*.log"))
    return [p for p in cands if p.is_file()]


def load_server_events(client: list[Event], model: str, explicit: Path | None):
    """Choisit le journal serveur qui s'apparie le mieux au client, et le recale."""
    ends = [
        e.t for e in client if e.name == "call.end" and e.fields.get("model") == model
    ]
    paths = [explicit] if explicit else _server_candidates(model)
    best = (None, None, 0, [])
    for path in paths:
        lines = parse_server_log(path)
        releases = [rel for rel, text in lines if "stop processing" in text]
        origin, n = estimate_offset(ends, releases)
        if origin is not None and n > best[2]:
            best = (path, origin, n, lines)
    path, origin, n, lines = best
    if origin is None:
        return None, 0, len(ends), []
    events = []
    for rel, text in lines:
        keep = text.startswith("E ") or any(k in text for k in _SERVER_KEYS)
        if keep and not any(n in text for n in _SERVER_NOISE):
            events.append(
                Event(origin + timedelta(seconds=rel), "serveur", "srv", text=text)
            )
    return path, n, len(ends), events


def _fmt_t(t: datetime) -> str:
    return t.strftime("%H:%M:%S.") + f"{t.microsecond // 1000:03d}"


def _describe(e: Event) -> str:
    if e.source == "serveur":
        m = _TASK.search(e.text)
        task = f"[tâche {m.group(1)}] " if m else ""
        body = re.sub(
            r"^\w\s+(slot|srv)\s+\S+\s*(id\s+\d+\s*\|\s*task\s+-?\d+\s*\|\s*)?",
            "",
            e.text,
        )
        return f"{task}{body.strip()}"
    f = e.fields
    if e.name == "call.request":
        extra = f" — diverge à {f['diverge_a']}" if f.get("diverge_a") else ""
        return f"appel {f.get('purpose')} (slot {f.get('slot')}, {f.get('msgs')} msg) préfixe={f.get('prefixe')}{extra}"
    if e.name == "call.end":
        if "cache_tok" in f:
            return (
                f"fin {f.get('purpose')} [{f.get('issue')}] : {f.get('cache_tok')} tok réutilisés, "
                f"{f.get('prefill_tok')} recalculés en {f.get('prefill_s')} s, "
                f"{f.get('generation_tok')} générés ({f.get('duree_s')} s au total)"
            )
        return f"fin {f.get('purpose')} [{f.get('issue')}] ({f.get('duree_s')} s)"
    if e.name == "prefix.diff":
        return f"PRÉFIXE MODIFIÉ ({f.get('purpose')}, slot {f.get('slot')}) : {f.get('element')} {f.get('avant')} -> {f.get('apres')}"
    return f"{e.name} " + " ".join(f"{k}={v}" for k, v in f.items())


def alerts(client: list[Event]) -> list[str]:
    """Constats automatiques sur les appels et la maintenance."""
    out = []
    refused: dict[str, list[str]] = {}
    last_turn_slot_user: dict = {}
    for e in client:
        f = e.fields
        if e.name == "prefix.diff":
            out.append(
                f"{_fmt_t(e.t)} préfixe modifié avant l'appel {f.get('purpose')} (slot {f.get('slot')}) : "
                f"{f.get('element')} a changé — sur un modèle hybride, tout ce qui suit est recalculé"
            )
        if e.name == "call.request" and f.get("slot") not in (None, "auto"):
            slot = f.get("slot")
            prev = last_turn_slot_user.get(slot)
            if f.get("purpose") not in ("turn", "prime", "keepwarm") and prev in (
                "turn",
                "prime",
            ):
                out.append(
                    f"{_fmt_t(e.t)} le slot {slot} de la conversation est pris par un appel "
                    f"{f.get('purpose')} : le tour suivant devra être ré-amorcé"
                )
            last_turn_slot_user[slot] = f.get("purpose")
        if e.name == "call.end" and isinstance(f.get("prefill_tok"), int):
            if f["prefill_tok"] >= 1000 and f.get("purpose") in ("turn", "prime"):
                out.append(
                    f"{_fmt_t(e.t)} recalcul important ({f.get('purpose')}) : {f['prefill_tok']} tokens "
                    f"en {f.get('prefill_s')} s, {f.get('cache_tok')} réutilisés"
                )
        if e.name == "call.end" and f.get("issue") in ("annule", "erreur"):
            out.append(
                f"{_fmt_t(e.t)} appel {f.get('purpose')} {f.get('issue')} ({f.get('erreur', '')})".rstrip(
                    " ()"
                )
            )
        if e.name == "slot.action" and str(f.get("resultat", "")).startswith("refuse"):
            key = (
                f"{f.get('action')} de {f.get('fichier')} refusée : {f.get('resultat')}"
            )
            refused.setdefault(key, []).append(_fmt_t(e.t))
        if e.name == "maint.end" and f.get("interrompue"):
            out.append(
                f"{_fmt_t(e.t)} maintenance interrompue par un message ({f.get('duree_s')} s)"
            )
    # Un refus répété à chaque tour tient en une ligne.
    for key, times in refused.items():
        out.append(f"{key} ({len(times)} fois, dès {times[0]})")
    return out


def build_report(
    session_id: str, server_log: Path | None = None, sessions_root: Path = SESSIONS_ROOT
) -> str:
    sdir = sessions_root / session_id
    dlog = sdir / "debug.log"
    if not dlog.is_file():
        raise FileNotFoundError(f"journal de session introuvable : {dlog}")
    client = parse_client_log(dlog)
    models = sorted(
        {
            e.fields.get("model")
            for e in client
            if e.name == "call.request" and e.fields.get("model")
        }
    )
    lines = [f"# Trace de la session {session_id}", ""]
    if not client:
        lines.append(
            "Aucun événement structuré dans le journal (session antérieure à la journalisation ?)."
        )
        return "\n".join(lines)
    server_events: list[Event] = []
    for model in models:
        path, n, total, events = load_server_events(client, model, server_log)
        if path is None:
            lines.append(
                f"- {model} : aucun journal serveur apparié ({total} fins d'appel côté client)."
            )
        else:
            lines.append(
                f"- {model} : journal serveur `{path}` recalé sur {n}/{total} fins d'appel."
            )
        server_events += events
    lines.append("")

    calls = [e for e in client if e.name == "call.end"]
    lines += [
        "## Appels modèle",
        "",
        "| fin | rôle | slot | issue | réutilisés | recalculés | prefill s | générés | durée s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in calls:
        f = e.fields
        lines.append(
            f"| {_fmt_t(e.t)} | {f.get('purpose')} | {f.get('slot')} | {f.get('issue')} | {f.get('cache_tok', '')} | "
            f"{f.get('prefill_tok', '')} | {f.get('prefill_s', '')} | {f.get('generation_tok', '')} | {f.get('duree_s')} |"
        )
    lines += ["", "## Alertes", ""]
    found = alerts(client)
    lines += [f"- {a}" for a in found] or ["- aucune"]
    lines += [
        "",
        "## Ligne de temps",
        "",
        "| heure | source | événement |",
        "|---|---|---|",
    ]
    timeline = [e for e in client if e.name not in _CLIENT_LEGACY] + server_events
    for e in sorted(timeline, key=lambda ev: ev.t):
        lines.append(
            f"| {_fmt_t(e.t)} | {e.source} | {_describe(e).replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", help="identifiant de session (dossier de var/sessions)")
    ap.add_argument("--server-log", type=Path, help="journal llama-server à utiliser")
    ap.add_argument("--out", type=Path, help="écrire le rapport dans ce fichier")
    args = ap.parse_args(argv)
    try:
        report = build_report(args.session, args.server_log)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"rapport écrit : {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
