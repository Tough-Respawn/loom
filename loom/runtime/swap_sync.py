"""Application de la config llama-swap sans interrompre le modèle qui tourne.

llama-swap (`--watch-config`) recharge sa config de façon GLOBALE : il construit un
serveur neuf et arrête l'ancien avec TOUS ses modèles, y compris ceux dont l'entrée n'a
pas changé (vérifié sur v241 avec des serveurs factices, et dans le code de v264). Une
écriture du yaml qui ne touche pas le modèle actif coûterait donc son processus et son
cache KV (vécu le 2026-10-14 : supprimer un autre modèle a coupé Bonsai, 10 258 tokens
de cache perdus, 210 s avant le premier token).

Règle : on n'écrit le yaml surveillé que si rien ne tourne, ou si l'entrée d'un modèle
en cours change (il doit être relancé de toute façon). Sinon la nouvelle version est
mise EN ATTENTE (`<yaml>.pending`) et appliquée à la prochaine transition — une requête
vers un modèle qui ne tourne pas, que llama-swap aurait de toute façon fait permuter."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import yaml

from loom.utils import atomic_write_text

# Écrit par llama-swap APRÈS l'arrêt de l'ancien serveur (llama-swap.go, reload) : à ce
# moment le port de l'ancien modèle ne répond déjà plus.
RELOADED_MARK = "configuration reloaded"


def pending_path(yaml_path: str | Path) -> Path:
    p = Path(yaml_path)
    return p.with_name(p.name + ".pending")


def running_models(text: str) -> set[str] | None:
    """Ids des modèles listés par `GET /running` de llama-swap ; None si la réponse
    n'est pas celle de llama-swap (vide, illisible, ou /v1/models d'un llama-server
    direct). Égalité exacte : « bonsai-2-27b » n'est pas « bonsai-2-27b-q2_0 »."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    running = data.get("running") if isinstance(data, dict) else None
    if not isinstance(running, list):
        return None
    return {
        str(item["model"])
        for item in running
        if isinstance(item, dict) and item.get("model")
    }


def _models(text: str) -> dict | None:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else None


def decide(current: str | None, new: str, running: set[str] | None) -> str:
    """'unchanged' | 'write' | 'defer'. `running` = modèles en cours (set vide : rien ne
    tourne ; None : inconnu -> on ne prend pas le risque d'interrompre)."""
    if current is not None and current == new:
        return "unchanged"
    if current is None:
        return "write"
    if running is not None and not running:
        return "write"
    if running is None:
        return "defer"
    cur_models, new_models = _models(current), _models(new)
    if cur_models is None or new_models is None:
        return "write"  # yaml appliqué illisible : rien de sûr à préserver
    if any(cur_models.get(m) != new_models.get(m) for m in running):
        return "write"
    return "defer"


def write_or_defer(
    new_text: str, yaml_path: str | Path, running: set[str] | None
) -> str:
    """Écrit le yaml surveillé ou met la nouvelle version en attente.
    Renvoie 'unchanged', 'written' ou 'deferred'."""
    y = Path(yaml_path)
    pend = pending_path(y)
    current = y.read_text(encoding="utf-8") if y.exists() else None
    decision = decide(current, new_text, running)
    if decision == "defer":
        atomic_write_text(pend, new_text)
        return "deferred"
    # Écrit ou déjà appliqué : une version en attente est dépassée.
    pend.unlink(missing_ok=True)
    if decision == "unchanged":
        return "unchanged"
    y.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(y, new_text)
    return "written"


def drop_pending(yaml_path: str | Path) -> None:
    pending_path(yaml_path).unlink(missing_ok=True)


def fetch_logs(root: str) -> str | None:
    """Journal tamponné de llama-swap (`GET /logs`), None si injoignable."""
    try:
        with urllib.request.urlopen(root.rstrip("/") + "/logs", timeout=3) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - signal best-effort, l'appelant a un délai
        return None


def _reloaded_since(before: str, after: str) -> bool:
    """Un « configuration reloaded » est-il apparu APRÈS le texte `before` ? Le tampon
    de llama-swap est borné : si l'ancre a disparu, on compare les décomptes."""
    tail = before[-400:]
    if not tail:
        return RELOADED_MARK in after
    i = after.rfind(tail)
    if i >= 0:
        return RELOADED_MARK in after[i + len(tail) :]
    return after.count(RELOADED_MARK) > before.count(RELOADED_MARK)


def apply_pending(
    yaml_path: str | Path,
    root: str | None,
    timeout: float = 20.0,
    poll: float = 0.2,
    fetch_logs=fetch_logs,
) -> str | None:
    """Applique la version en attente et attend que llama-swap l'ait rechargée, pour
    qu'une requête envoyée ensuite ne démarre pas un modèle sous l'ANCIENNE config (le
    rechargement le tuerait aussitôt). None si rien n'attendait ; 'applied' (rechargement
    vu dans /logs) ; 'applied_unconfirmed' (llama-swap injoignable, ou délai dépassé)."""
    y = Path(yaml_path)
    pend = pending_path(y)
    if not pend.exists():
        return None
    before = fetch_logs(root) if root else None
    os.replace(pend, y)
    if before is None:
        return "applied_unconfirmed"
    deadline = time.monotonic() + timeout
    while True:
        after = fetch_logs(root)
        if after is not None and _reloaded_since(before, after):
            return "applied"
        if time.monotonic() >= deadline:
            return "applied_unconfirmed"
        time.sleep(poll)
