"""ST-05 — banc A/B d'intelligence de code (HORS PRODUCTION).

Compare, sur les MÊMES cas et le MÊME modèle local :
- variante « base »  : les outils actuels de Loom, tels quels ;
- variante « proto » : les mêmes + code_outline et code_diagnostics
  (evals/code_tools.py — jamais dans le registre de production).

Mesures par run : réussite (graders code), appels d'outils, échecs d'outils
(retries), tokens in/out, CARACTÈRES LUS via les résultats d'outils, durée,
raison d'arrêt. Artefacts dans un dossier de campagne non écrasable
(campaign.json + transcripts + report.json).

Usage :
  uv run python -m evals.run_code_eval --self-test          # graders, hors-ligne
  uv run python -m evals.run_code_eval --runs 3 --model <id-local>
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from evals.code_cases import CODE_CASES
from evals.harness import (
    git_dirty,
    git_head_sha,
    load_eval_config,
    make_client,
    make_perm,
)
from evals.run_eval import (
    Trajectory,
    _save_transcript,
    case_passed,
    load_variants,
    new_campaign_dir,
    write_campaign_meta,
)
from loom.agent.conversation import Conversation
from loom.tools import AVAILABLE_TOOLS, build_registry

# « prod » (ST-06) : les VRAIS outils de production (loom/tools/code.py) via le
# VRAI registre et la VRAIE politique de permissions — plus aucun wrapper de banc :
# les outils sont classés READ_TOOLS (la leçon des 17/17 refus de ST-05 est dans
# la production, pas dans le banc). L'ancienne variante « proto » (evals/code_tools)
# n'existe plus ici ; ses artefacts ST-05 restent la référence historique.
VARIANTS = ("base", "prod")
_CODE_TOOLS = ("code_outline", "code_diagnostics")


def run_one(
    client,
    model,
    chat_prompt,
    sub_prompt,
    case,
    ws: Path,
    cfg,
    perm,
    max_iters,
    variant: str,
    discovery: bool = False,
):
    """Un run. base = outils standards SANS les outils code ; prod = registre de
    production complet, outils code PRÉCHARGÉS (mesure de l'outil, comparable à
    ST-05) sauf `discovery=True` (le modèle doit les découvrir via tool_search)."""

    import loom.prompts as _p

    _p.SUBAGENT_SYSTEM = sub_prompt
    enabled = [t["name"] for t in AVAILABLE_TOOLS]
    if variant == "base":
        enabled = [n for n in enabled if n not in _CODE_TOOLS]
    convo = Conversation(system_prompt=chat_prompt)
    if variant == "prod" and not discovery:
        # Pré-charger les schémas différés isole la valeur de L'OUTIL du saut de
        # découverte ; le mode --discovery mesure ce saut séparément.
        convo.deferred_loaded = list(_CODE_TOOLS)
    registry = build_registry(
        workspace_dir=str(ws),
        max_bytes=cfg.chat.read_file_max_bytes,
        enabled=enabled,
        web_cfg=cfg.chat.web_search,
        client=client,
        conversation=convo,
        model=model,
        sub_max_tokens=cfg.chat.max_tokens,
        permission=perm,
        active_model=model,
    )
    # Caractères réellement RAMENÉS au modèle par les outils (frontière registry.run).
    # Limite assumée : un outil streamant (dispatch) passe par run_stream, non compté.
    chars = {"n": 0}
    orig_run = registry.run

    def counting_run(name, args):
        r = orig_run(name, args)
        chars["n"] += len(r or "")
        return r

    registry.run = counting_run
    traj = Trajectory()
    t0 = time.monotonic()
    try:
        for kind, payload in client.stream_chat_tools(
            [{"role": "user", "content": case.prompt}],
            chat_prompt,
            max_tokens=cfg.chat.max_tokens,
            model=model,
            registry=registry,
            thinking=False,
            max_iters=max_iters,
            permission=perm,
        ):
            if kind == "content":
                traj.final_text += payload
            elif kind == "usage":
                traj.model_turns += 1
                traj.prompt_tokens += payload.get("prompt_tokens") or 0
                traj.completion_tokens += payload.get("completion_tokens") or 0
            elif kind == "done":
                traj.stop_reason = payload.get("reason") or ""
            elif kind == "tool_result":
                args = {}
                if payload.get("path") is not None:
                    args["path"] = payload["path"]
                if payload.get("cmd") is not None:
                    args["command"] = payload["cmd"]
                traj.tool_calls.append((payload.get("name"), args))
                traj.tool_results.append(
                    {
                        "name": payload.get("name"),
                        "ok": payload.get("ok"),
                        "preview": str(payload.get("preview", ""))[:300],
                    }
                )
    except Exception as e:  # un run qui plante = donnée, pas un crash du banc
        traj.error = f"{type(e).__name__}: {e}"
        traj.stop_reason = traj.stop_reason or "crash"
    traj.duration_s = round(time.monotonic() - t0, 1)
    traj.chars_read = chars["n"]
    return traj


def _record(traj, checks, model) -> dict:
    return {
        "checks": checks,
        "passed": case_passed(checks),
        "model": model,
        "n_model_turns": traj.model_turns,
        "n_tool_calls": traj.n_tool_calls,
        "tool_failures": sum(1 for r in traj.tool_results if not r.get("ok")),
        "prompt_tokens": traj.prompt_tokens,
        "completion_tokens": traj.completion_tokens,
        "chars_read": getattr(traj, "chars_read", 0),
        "stop_reason": traj.stop_reason,
        "duration_s": traj.duration_s,
        "error": traj.error,
        "tools": [n for n, _ in traj.tool_calls],
        "final": (traj.final_text or "")[:800],
    }


def _mean(xs) -> float:
    return round(sum(xs) / len(xs), 1) if xs else 0.0


def report(results: dict, run_dir: Path) -> None:
    import json as _json

    print("\n" + "=" * 78)
    print("BANC A/B INTELLIGENCE DE CODE — réussite et coûts par cas")
    print("=" * 78)
    actives = [v for v in VARIANTS if v in results]
    for cid in [c.id for c in CODE_CASES if any(c.id in results[v] for v in actives)]:
        for v in actives:
            rd = results[v].get(cid, [])
            if not rd:
                continue
            p = sum(1 for r in rd if r["passed"])
            print(
                f"  {cid.ljust(18)} [{v.ljust(5)}] {p}/{len(rd)}  "
                f"outils={_mean([r['n_tool_calls'] for r in rd])} "
                f"échecs={_mean([r['tool_failures'] for r in rd])} "
                f"tok={_mean([r['prompt_tokens'] for r in rd])}/"
                f"{_mean([r['completion_tokens'] for r in rd])} "
                f"lus={_mean([r['chars_read'] for r in rd])}c "
                f"durée={_mean([r['duration_s'] for r in rd])}s"
            )
    for v in actives:
        allr = [r for rd in results[v].values() for r in rd]
        if allr:
            print(
                f"TOTAL [{v}] : {sum(1 for r in allr if r['passed'])}/{len(allr)}  "
                f"lus moy={_mean([r['chars_read'] for r in allr])}c  "
                f"tok in moy={_mean([r['prompt_tokens'] for r in allr])}"
            )
    (run_dir / "report.json").write_text(
        _json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDétail : {run_dir}\\report.json + transcripts par variante.")


def self_test() -> bool:
    """Chaque grader s'exécute et rend un dict[str, bool] sur workspace seedé."""
    ok = True
    traj = Trajectory(final_text="rien")
    for case in CODE_CASES:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            case.setup(ws)
            try:
                checks = case.check(traj, ws)
                assert isinstance(checks, dict) and all(
                    isinstance(v, bool) for v in checks.values()
                )
                print(f"  [ok] {case.id}: {len(checks)} checks")
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"  [XX] {case.id}: {type(e).__name__}: {e}")
    print("SELF-TEST", "VERT" if ok else "ROUGE")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--cases", default=None, help="ids séparés par des virgules")
    ap.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    ap.add_argument(
        "--discovery",
        action="store_true",
        help="prod SANS préchargement : le modèle doit découvrir via tool_search",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    cfg = load_eval_config()
    model = args.model or cfg.default_model
    client, base_url = make_client(cfg, model)
    perm = make_perm(cfg)
    only = set(args.cases.split(",")) if args.cases else None
    chat_p, sub_p = load_variants("new")["new"]

    from loom.runtime.stay_awake import StayAwake

    _awake = StayAwake()
    _awake.acquire()

    sha = git_head_sha()
    run_dir = new_campaign_dir(sha or None)
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    write_campaign_meta(
        run_dir,
        sha=sha,
        dirty=git_dirty(),
        model=model,
        variants=list(variants),
        runs=args.runs,
        cases=only,
        judge=False,
    )
    print(f"Modèle : {model} @ {base_url} | runs={args.runs}\nCampagne : {run_dir}\n")
    results: dict = {v: {} for v in variants}
    for case in CODE_CASES:
        if only and case.id not in only:
            continue
        for v in variants:
            runs_data = []
            for k in range(args.runs):
                with tempfile.TemporaryDirectory(
                    prefix=f"loom_code_{case.id}_", ignore_cleanup_errors=True
                ) as tmp:
                    ws = Path(tmp)
                    case.setup(ws)
                    traj = run_one(
                        client,
                        model,
                        chat_p,
                        sub_p,
                        case,
                        ws,
                        cfg,
                        perm,
                        args.max_iters,
                        variant=v,
                        discovery=args.discovery,
                    )
                    checks = case.check(traj, ws)
                    runs_data.append(_record(traj, checks, model))
                    _save_transcript(
                        run_dir, v, case.id, k, traj, checks, None, model=model
                    )
                    mark = "ok" if runs_data[-1]["passed"] else "XX"
                    print(
                        f"  [{v}] {case.id} run{k + 1}/{args.runs} [{mark}] "
                        f"stop={traj.stop_reason or '?'} outils={traj.n_tool_calls} "
                        f"tok={traj.prompt_tokens}/{traj.completion_tokens} "
                        f"lus={getattr(traj, 'chars_read', 0)}c {traj.duration_s}s "
                        f"{runs_data[-1]['tools']}"
                        + (f" ERREUR={traj.error}" if traj.error else "")
                    )
            results[v][case.id] = runs_data
    report(results, run_dir)


if __name__ == "__main__":
    main()
