"""Archive DURABLE et automatique des benchs : var/bench/<modèle>/<horodatage>.json.

Jusqu'au 2026-10-10 les mesures ne vivaient que dans [bench] de local.toml (écrasé au
bench suivant), dans des commentaires de model.toml et dans l'état d'application d'une
session /rebench — consommé par « oui » ou effacé par « annuler » (le 3e run Ornith a
perdu ses échantillons bruts ainsi). Chaque bench écrit désormais un JSON complet :
matériel, build, flags, profil GGUF, isolation, placement (candidats, non explorés,
présélection, mesures avec échantillons), calibration, réglage final, cache, verdict ;
puis la trace de l'APPLICATION quand elle a lieu. Lecture : n'importe quel outil JSON.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime
from pathlib import Path

from loom.utils import atomic_write_text

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "var" / "bench"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _json_safe(obj):
    """Copie sérialisable : dataclasses -> dict (+ `key` quand elle existe), Path -> str,
    ensembles -> listes triées ; None conservé (JSON le permet)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {f.name: _json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        key = getattr(obj, "key", None)
        if isinstance(key, str):
            d["key"] = key
        return d
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, Path):
        return (
            obj.as_posix()
        )  # séparateurs POSIX : lisible et stable d'une machine à l'autre
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    return obj


#: Schéma COMMUN du compte rendu d'un bench (loom-setup et /rebench) : les clés qui
#: permettent de reproduire chaque mesure. Une trace progressive peut en porter
#: d'autres (couples, étape courante…) : conservées.
BENCH_SCHEMA = (
    "source",
    "etape",
    "gguf",
    "server_bin",
    "build",
    "materiel",
    "flags",
    "profil",
    "contexte_utile",
    "kv_estime_mb",
    "llama_bench",
    "isolation",
    "plan",
    "placement",
    "placement_avant",
    "calibration",
    "ubatch",
    "final",
    "cache",
    "ecrit",
    "verdict_texte",
    "verdict",
    "echec",
)


def bench_payload(**sections) -> dict:
    """Compte rendu au schéma commun : toutes les clés de BENCH_SCHEMA présentes (None
    si absentes), plus les sections supplémentaires passées."""
    out = {k: None for k in BENCH_SCHEMA}
    out.update(sections)
    return out


def archive_bench(
    model_id: str,
    payload: dict,
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Écrit `payload` (complété de version, date, model_id) dans
    <root>/<model_id>/<AAAAMMJJ-HHMMSS>.json, sans jamais écraser un fichier existant
    (suffixe -2, -3… à la même seconde). Renvoie le chemin."""
    now = now or datetime.now()
    folder = Path(root or BENCH_DIR) / (_SAFE.sub("-", str(model_id)) or "modele")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}.json"
    n = 2
    while path.exists():
        path = folder / f"{stamp}-{n}.json"
        n += 1
    data = {
        "version": 1,
        "date": now.isoformat(timespec="seconds"),
        "model_id": str(model_id),
    }
    data.update(_json_safe(payload))
    atomic_write_text(
        path, json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    return path


def note_application(
    path: Path | str, applied: dict, *, now: datetime | None = None
) -> None:
    """Complète une archive par la trace de ce qui a été APPLIQUÉ (date + réglages).
    Best-effort : une archive absente ou illisible n'empêche jamais l'application."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    data["application"] = {
        "date": (now or datetime.now()).isoformat(timespec="seconds"),
        **_json_safe(applied),
    }
    try:
        atomic_write_text(
            p, json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n"
        )
    except OSError:
        return
