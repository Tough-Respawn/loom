"""Local task evaluations. Scores belong to exact artifacts, engines and harnesses."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

from loom.utils import atomic_write_text

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "var" / "model-evaluations"


@functools.lru_cache(maxsize=32)
def _hash_file(path: str, size: int, mtime: int) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_hash(path) -> str:
    p = Path(path).resolve()
    st = p.stat()
    return _hash_file(str(p), st.st_size, st.st_mtime_ns)


def environment_key(server_bin) -> str:
    """Invalidate stale results after an engine, harness or machine change."""
    digest = hashlib.sha256()
    digest.update(file_hash(server_bin).encode())
    digest.update(
        (platform.node() + platform.platform() + platform.processor()).encode()
    )
    paths = [
        *(ROOT / "loom" / "agent").glob("*.py"),
        *(ROOT / "loom" / "tools").rglob("*.py"),
        *(ROOT / "loom" / "prompts").rglob("*"),
        ROOT / "evals" / "cases.py",
        ROOT / "evals" / "run_eval.py",
        ROOT / "evals" / "recommendation_cases.py",
    ]
    for path in sorted(
        p for p in paths if p.is_file() and "__pycache__" not in p.parts
    ):
        digest.update(file_hash(path).encode())
    return digest.hexdigest()


def summarize_runs(results: dict) -> dict:
    if not results or any(len(runs) < 3 for runs in results.values()):
        raise ValueError("au moins trois répétitions par cas sont nécessaires")
    if len({len(runs) for runs in results.values()}) != 1:
        raise ValueError("même nombre de répétitions pour chaque cas")
    runs = [r for group in results.values() for r in group]
    passed = sum(r.get("passed") is True and not r.get("error") for r in runs)
    n = len(runs)
    p = passed / n
    z = 1.96
    lower = (
        p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    ) / (1 + z * z / n)
    durations = [float(r["duration_s"]) for r in runs]
    return {
        "passed": passed,
        "total": n,
        "success_rate": p,
        "lower_bound": lower,
        "median_seconds": statistics.median(durations),
        "cases": sorted(results),
    }


def publish(
    *, repo, filename, sha256, environment, settings, results, source, store=STORE
):
    summary = summarize_runs(results)
    record = dict(
        summary,
        repo=repo,
        filename=filename,
        sha256=sha256,
        environment=environment,
        settings=settings,
        source=str(source),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    # Keep distinct configurations/campaigns; never overwrite another experiment.
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path = Path(store) / (key + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2))
    return path


def campaign_identity(cfg, model):
    """Identity recorded before a campaign, with the actual configured artifact."""
    spec = next((m for m in cfg.models if m.id == model), None)
    if spec is None or not spec.dir:
        raise ValueError("La publication nécessite un modèle local installé")
    artifact = Path(spec.dir) / spec.filename
    if "-of-" in spec.filename:
        raise ValueError("Publication des GGUF découpés non prise en charge")
    return dict(
        repo=spec.repo,
        filename=spec.filename,
        sha256=file_hash(artifact),
        environment=environment_key(spec.server_bin or cfg.server_bin),
        settings={
            "context": spec.context or cfg.context,
            "slots": max(cfg.n_parallel, 2 if spec.cache_isolation else 1),
            "thinking": False,
            "max_tokens": cfg.chat.max_tokens,
            "n_gpu_layers": spec.n_gpu_layers,
            "cpu_moe": spec.cpu_moe,
            "n_cpu_moe": spec.n_cpu_moe,
        },
    )


def lookup(repo, file, environment, store=STORE, *, settings=None):
    if not environment or not file.get("sha256"):
        return None
    found = []
    for path in Path(store).glob("*.json"):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
            age = datetime.now(timezone.utc) - datetime.fromisoformat(r["created_at"])
            if (
                r["repo"] == repo
                and r["filename"] == file["filename"]
                and r["sha256"] == file["sha256"]
                and r["environment"] == environment
                and 0 <= age.days <= 90
                and r["total"] >= 3
                and all(
                    r.get("settings", {}).get(k) == v
                    for k, v in (settings or {}).items()
                )
            ):
                found.append(r)
        except (ValueError, KeyError, TypeError, OSError):
            continue
    return max(found, key=lambda r: r["created_at"], default=None)


def rank_candidates(candidates):
    """Only compare measured success for the same cases; unknown is not a zero."""
    suites = {tuple(c["quality"]["cases"]) for c in candidates if c.get("quality")}
    comparable = len(suites) == 1

    def key(c):
        q = c.get("quality") if comparable else None
        return (
            q is not None,
            q["lower_bound"] if q else 0,
            -q["median_seconds"] if q else 0,
            c.get("params", 0),
        )

    return sorted(candidates, key=key, reverse=True)
