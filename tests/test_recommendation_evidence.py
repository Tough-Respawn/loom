import json
from types import SimpleNamespace

import pytest

from evals.recommendation_cases import TRACKS, check_blocked, check_catalogue
from loom.agent.transcript import render_transcript
from loom.runtime import gguf_remote, model_evidence
from loom.runtime.model_install import recommend_quant, vram_reserve_mb
from loom.runtime.recommendation import recommend_repository


def test_shared_recommender_uses_scores_after_memory_constraints(monkeypatch):
    monkeypatch.setattr(model_evidence, "environment_key", lambda path: "engine")

    def measured(repo, file, environment, **kwargs):
        return model_evidence.summarize_runs(
            {
                "task": [
                    {
                        "passed": i < (3 if file["size_mb"] == 2500 else 1),
                        "duration_s": 10,
                    }
                    for i in range(3)
                ]
            }
        )

    monkeypatch.setattr(model_evidence, "lookup", measured)
    files = [
        {"filename": f"m-{label}.gguf", "size_mb": size}
        for label, size in (("Q4_K_M", 2500), ("Q8_0", 4000), ("F16", 8000))
    ]
    out, _ = recommend_repository(
        "org/model",
        files,
        vram_mb=6144,
        ram_mb=24000,
        server_bin="engine",
        supported={"Q4_K_M", "Q8_0", "F16"},
        analysis={},
    )
    assert [f["size_mb"] for f in out if f["recommended"]] == [2500]
    assert out[0]["evidence"] == "evaluated"


def test_projector_stays_in_ram_like_the_launcher():
    files = [
        {"filename": "m-Q4_K_M.gguf", "size_mb": 4300},
        {"filename": "mmproj-F16.gguf", "size_mb": 700, "is_mmproj": True},
    ]
    out, _ = recommend_repository(
        "org/model",
        files,
        vram_mb=6144,
        ram_mb=24000,
        supported={"Q4_K_M"},
        analysis={},
    )
    assert len(out) == 1
    assert out[0]["full_gpu"] and out[0]["recommended"]


def test_recommendation_matches_gpu_cache_launch_flags(monkeypatch):
    from loom.runtime import server_args
    monkeypatch.setattr(server_args, "no_mmap_args", lambda _: [])
    args = server_args.build_server_args("server", "model", 8080, 65536, 99, 8,
        gpu_tuning=True, n_parallel=2, mmproj_path="vision")
    assert args[args.index("-ctk") + 1] == "q8_0"
    assert args[args.index("-ctv") + 1] == "q8_0"
    assert "--no-mmproj-offload" in args
    assert args[args.index("-c") + 1] == "131072"
    analysis = dict(moe=False, n_params=9e9, f_emb=0, f_exp=0,
        expert_count=0, expert_used=0, unknown_types=[], kv_per_token=15360, recurrent=True)
    files = [{"filename": "m-Q2_0.gguf", "size_mb": 2438},
             {"filename": "mmproj-F16.gguf", "size_mb": 595, "is_mmproj": True}]
    out, _ = recommend_repository("org/Model-9B", files, vram_mb=6143, ram_mb=64000,
        context=65536, slots=1, supported={"Q2_0"}, analysis=analysis)
    assert out[0]["recommended"] and out[0]["full_gpu"]


def test_configured_context_can_rule_out_bf16_even_with_q8_cache():
    analysis = dict(moe=False, n_params=1942653248, f_emb=0, f_exp=0,
        expert_count=0, expert_used=0, unknown_types=[], kv_per_token=7168, recurrent=True)
    files = [{"filename": "m-BF16.gguf", "size_mb": 3716},
             {"filename": "m-Q8_0.gguf", "size_mb": 1980}]
    for context, expected in ((16384, "BF16"), (65536, "Q8_0")):
        out, _ = recommend_repository("org/Model-2B", files, vram_mb=6143, ram_mb=64000,
            context=context, slots=1, supported={"BF16", "Q8_0"}, analysis=analysis)
        assert [f["label"] for f in out if f["recommended"]] == [expected]


def test_separate_input_embedding_is_host_resident():
    analysis = dict(moe=False, n_params=9e9, f_emb=.15, f_exp=0,
                    expert_count=0, expert_used=0, unknown_types=[])
    out = recommend_quant([{"filename": "m-Q4_K_M.gguf", "size_mb": 5000}],
                          6144, 24000, analysis=analysis)
    assert out[0]["full_gpu"] and out[0]["recommended"]


def test_drafter_header_does_not_become_a_27b_recommendation():
    analysis = dict(moe=False, n_params=1.8e9, f_emb=0, f_exp=0,
                    expert_count=0, expert_used=0, unknown_types=[])
    out, _ = recommend_repository("org/Model-27B", [{"filename": "m-Q4_1.gguf", "size_mb": 1856}],
        vram_mb=6144, ram_mb=64000, supported={"Q4_1"}, analysis=analysis)
    assert not out[0]["recommended"] and out[0]["status"] == "mismatch"


def test_web_wizard_only_searches_alternatives_if_no_quant_fits(monkeypatch):
    from loom.runtime import hardware, hf_catalog, recommendation
    from loom.web.routes import models, model_admin
    calls = []
    monkeypatch.setattr(model_admin, "_configured_server_bin", lambda: None)
    monkeypatch.setattr(hardware, "detect_hardware", lambda _: hardware.HardwareProfile(True, "GPU", 6000, 8))
    monkeypatch.setattr(hardware, "ram_total_mb", lambda: 24000)
    monkeypatch.setattr(hf_catalog, "search_family", lambda q: calls.append(q) or [])
    monkeypatch.setattr(recommendation, "recommend_repository",
        lambda repo, files, **kw: ([dict(f, recommended=f["fits"]) for f in files], None))
    state = SimpleNamespace(config_defaults_path=None, config_local_path=None, settings={}, models=[])
    deps = models._wizard_deps(state)
    deps.recommend([{"fits": True}], "org/Family-4B")
    assert calls == []
    deps.recommend([{"fits": False}], "org/Family-27B")
    assert calls == ["Family"]


def test_tied_embeddings_count_toward_full_gpu():
    a = dict(
        moe=False,
        n_params=9e9,
        f_emb=0,
        f_exp=0,
        expert_count=0,
        expert_used=0,
        unknown_types=[],
    )
    out = recommend_quant(
        [{"filename": "model-Q4_K_M.gguf", "size_mb": 5000}], 6144, 24000, analysis=a
    )
    assert not out[0]["full_gpu"]
    assert not out[0]["recommended"]
    assert out[0]["ram_share"] > 0


def test_context_slots_recurrent_and_projector_consume_budget():
    a = {"kv_per_token": 32768, "recurrent": True}
    small = vram_reserve_mb(a, context=4096, slots=1, cache_bytes=2)
    large = vram_reserve_mb(a, context=16384, slots=2, cache_bytes=2, extra_mb=600)
    assert large > small + 1000


def test_engine_capabilities_never_extended_from_python_enum(tmp_path, monkeypatch):
    (tmp_path / "llama-quantize.exe").touch()
    monkeypatch.setattr(
        gguf_remote.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="  15 or Q4_K_M : medium\n", stderr=""),
    )
    gguf_remote.supported_formats.cache_clear()
    caps = gguf_remote.supported_formats(str(tmp_path / "llama-server.exe"))
    assert "Q2_0" not in caps
    assert gguf_remote.label_supported("Q4_K_XL", caps)


def test_measured_quality_requires_repetitions_exact_artifact_and_config(tmp_path):
    results = {"task": [{"passed": True, "duration_s": 10, "error": None}] * 3}
    with pytest.raises(ValueError):
        model_evidence.summarize_runs({"task": results["task"][:1]})
    model_evidence.publish(
        repo="org/repo",
        filename="m.gguf",
        sha256="abc",
        environment="engine",
        settings={"context": 8192},
        results=results,
        source="campaign",
        store=tmp_path,
    )
    file = {"filename": "m.gguf", "sha256": "abc"}
    assert (
        model_evidence.lookup(
            "org/repo", file, "engine", tmp_path, settings={"context": 8192}
        )["passed"]
        == 3
    )
    assert model_evidence.lookup("org/repo", file, "other-engine", tmp_path) is None
    assert (
        model_evidence.lookup(
            "org/repo", dict(file, sha256="changed"), "engine", tmp_path
        )
        is None
    )
    assert (
        model_evidence.lookup(
            "org/repo", file, "engine", tmp_path, settings={"context": 16384}
        )
        is None
    )


def test_task_success_beats_parameters_or_speed():
    def candidate(passed, seconds, size):
        summary = model_evidence.summarize_runs(
            {"task": [{"passed": i < passed, "duration_s": seconds} for i in range(3)]}
        )
        return {"quality": summary, "pick": {"full_gpu": True, "size_mb": size}}

    good, bad = candidate(3, 40, 1500), candidate(1, 2, 5000)
    assert model_evidence.rank_candidates([bad, good])[0] is good


def test_journal_keeps_full_tool_evidence_and_unfinished_calls():
    journal = render_transcript(
        [
            {
                "event": "tool_result",
                "data": {
                    "name": "fetch_url",
                    "ok": True,
                    "in_full": '{"url":"https://example.test"}',
                    "preview": "short",
                    "out_full": "HTTP 403\n```\nJavaScript required",
                },
            },
            {"event": "tool_call", "data": {"id": "pending", "name": "read_file"}},
        ]
    )
    assert "https://example.test" in journal and "HTTP 403" in journal
    assert "JavaScript required" in journal and "aucun résultat enregistré" in journal
    assert "ne garantit pas" in journal


def test_grader_rejects_tool_activity_without_the_requested_tracks():
    traj = SimpleNamespace(
        tool_calls=[("read_file", {"path": t["source"]}) for t in TRACKS],
        final_text="Titres inaccessibles sans Discogs Premium.",
        error=None,
    )
    assert not all(check_catalogue(traj, None).values())
    traj.final_text = json.dumps({"tracks": TRACKS})
    assert all(check_catalogue(traj, None).values())
    traj.final_text = json.dumps({"tracks": TRACKS[:-1]})
    assert not all(check_catalogue(traj, None).values())
    traj.tool_calls = [("read_file", {"path": "blocked.txt"})]
    traj.final_text = json.dumps(
        {
            "tracks": [],
            "status": "unverified",
            "reason": "HTTP 403; JavaScript required",
        }
    )
    assert all(check_blocked(traj, None).values())
