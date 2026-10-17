"""Shared recommendation entry point for setup and the web wizard."""

from __future__ import annotations

from loom.runtime import gguf_remote, model_evidence, model_install
from loom.runtime.server_args import GPU_KV_BYTES, MMPROJ_OFFLOAD


def recommend_repository(
    repo,
    files,
    *,
    vram_mb,
    ram_mb,
    server_bin=None,
    context=16384,
    slots=1,
    max_tokens=8192,
    gpu_tuning=None,
    margin_mb=6144,
    supported=None,
    analysis=None,
):
    if supported is None:
        supported = gguf_remote.supported_formats(
            str(server_bin) if server_bin else None
        )
    auxiliary_mb = min((f["size_mb"] for f in files if f.get("is_mmproj")), default=0)
    files = [f for f in files if not f.get("is_aux", f.get("is_mmproj", False))]
    if analysis is None:
        analysis = gguf_remote.analyze_repo(repo, files, supported)
    # A recurrent model gets an isolated second slot when installed by Loom.
    effective_slots = max(slots, 2 if analysis and analysis.get("recurrent") else 1)
    out = model_install.recommend_quant(
        files,
        vram_mb,
        max(0, ram_mb - (0 if MMPROJ_OFFLOAD else auxiliary_mb)),
        margin_mb=margin_mb,
        analysis=analysis,
        supported=supported,
        context=context,
        slots=effective_slots,
        cache_bytes=GPU_KV_BYTES if (vram_mb > 0 if gpu_tuning is None else gpu_tuning) else 2.0,
        extra_mb=auxiliary_mb if MMPROJ_OFFLOAD else 0,
    )
    expected_params, _ = model_install._hit_size({"repo_id": repo or ""})
    if (analysis and not analysis.get("split") and expected_params
            and analysis["n_params"] < expected_params / 2):
        # A readable header can belong to a drafter, not the advertised model.
        for f in out:
            f.update(recommended=False, status="mismatch")
    try:
        environment = model_evidence.environment_key(server_bin) if server_bin else None
    except (OSError, TypeError):
        environment = None
    for f in out:
        f["quality"] = model_evidence.lookup(
            repo,
            f,
            environment,
            settings={
                "context": context,
                "slots": effective_slots,
                "thinking": False,
                "max_tokens": max_tokens,
            },
        )
        f["context"] = context
        f["slots"] = effective_slots
        f["compatibility"] = (
            "estimated" if supported is not None and analysis else "unknown"
        )
    eligible = [
        f
        for f in out
        if f["fits"] and f["status"] == "ok" and (f["full_gpu"] or f["recommended"])
    ]
    measured = [dict(pick=f, quality=f["quality"]) for f in eligible if f["quality"]]
    if measured and len({tuple(c["quality"]["cases"]) for c in measured}) == 1:
        winner = model_evidence.rank_candidates(measured)[0]["pick"]
        for f in out:
            f["recommended"] = f is winner
        winner["evidence"] = "evaluated"
    return out, analysis
