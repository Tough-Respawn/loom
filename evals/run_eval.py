"""Harnais d'éval des prompts de Loom, à la Anthropic : jeu de cas figé + grader.

Compare DEUX variantes du system prompt (par défaut : ancien = git HEAD, nouveau = disque)
sur le même eval set (evals/cases.py), N runs par cas (le petit modèle est stochastique),
et grade chaque run par :
  1. un grader CODE déterministe (marqueurs objectifs depuis la trajectoire d'outils) ;
  2. un juge LLM (le modèle lui-même) qui note « la tâche est-elle accomplie » contre la
     rubrique du cas — model-graded eval, comme recommandé pour les sorties ouvertes.

Sortie : tableau comparatif (taux de réussite par check, ancien vs nouveau) + JSON +
transcripts. Le serveur modèle (port de la config) doit tourner ; sinon, --self-test
valide la mécanique des graders sans modèle.

Usage :
  uv run python -m evals.run_eval --self-test          # hors-ligne, valide les graders
  uv run python -m evals.run_eval --runs 3             # éval réelle (serveur up)
  uv run python -m evals.run_eval --runs 3 --no-judge  # graders code seuls (plus rapide)
  uv run python -m evals.run_eval --variant new        # une seule variante
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from loom.agent.conversation import Conversation
from loom.tools import AVAILABLE_TOOLS, build_registry

from evals.cases import CASES
from evals.harness import (
    _RT,
    git_show,
    load_eval_config,
    make_client,
    make_perm,
)

_OUT = _RT.parent / "evals" / "out"


# --- trajectoire -------------------------------------------------------------


@dataclass
class Trajectory:
    tool_calls: list = field(default_factory=list)  # [(name, args_dict)]
    tool_results: list = field(default_factory=list)  # [{name, ok, preview}]
    final_text: str = ""
    reasoning: str = ""
    error: str | None = None
    # Métriques de COÛT par run (le pass/fail seul cache un « ça passe mais en 11 tours
    # et 38 appels ») : tours MODÈLE (un event usage par appel modèle) distincts des
    # appels OUTILS, tokens réels, raison d'arrêt (event 'done' de la boucle), durée.
    model_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    stop_reason: str = ""
    duration_s: float = 0.0

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    # Rétro-compat des checks existants : n_turns a toujours compté les APPELS D'OUTILS.
    n_turns = n_tool_calls


# --- prompts variantes -------------------------------------------------------


def _git_show(rel: str) -> str:
    return git_show(rel).strip()


def load_variants(which: str) -> dict:
    """Renvoie {nom: (chat_prompt, subagent_prompt)} pour la/les variante(s) demandée(s)."""
    disk_chat = (_RT / "prompts" / "chat.system.md").read_text(encoding="utf-8").strip()
    disk_sub = (
        (_RT / "prompts" / "subagent.system.md").read_text(encoding="utf-8").strip()
    )
    old = (
        _git_show("loom/prompts/chat.system.md"),
        _git_show("loom/prompts/subagent.system.md"),
    )
    new = (disk_chat, disk_sub)
    allv = {"old": old, "new": new}
    if which == "both":
        return allv
    return {which: allv[which]}


# --- un run ------------------------------------------------------------------


def run_one(
    client, model, chat_prompt, sub_prompt, case, ws: Path, cfg, perm, max_iters
):
    """Exécute la boucle agentique sur un cas dans un workspace neuf ; renvoie la Trajectory."""
    # Override du prompt sous-agent (lu par build_registry au moment de l'appel).
    import loom.prompts as _p

    _p.SUBAGENT_SYSTEM = sub_prompt

    convo = Conversation(system_prompt=chat_prompt)
    registry = build_registry(
        workspace_dir=str(ws),
        max_bytes=cfg.chat.read_file_max_bytes,
        enabled=[t["name"] for t in AVAILABLE_TOOLS],
        web_cfg=cfg.chat.web_search,
        client=client,
        conversation=convo,
        model=model,
        sub_max_tokens=cfg.chat.max_tokens,
        permission=perm,
        active_model=model,
    )
    prompt = case.prompt.replace("{NOTES_PATH}", (ws / "docs" / "notes.md").as_posix())
    # Cas à HISTORIQUE pré-rempli (saturation de contexte) : les vieux tours synthétiques
    # précèdent la tâche, sans coûter leur génération live sur le modèle local.
    messages = [
        *(getattr(case, "history", None) or []),
        {"role": "user", "content": prompt},
    ]
    traj = Trajectory()
    t0 = time.monotonic()
    try:
        for kind, payload in client.stream_chat_tools(
            messages,
            chat_prompt,
            max_tokens=cfg.chat.max_tokens,
            model=model,
            registry=registry,
            thinking=False,
            max_iters=max_iters,
            permission=perm,
            # Par cas : un seuil bas force la compaction préventive (cas saturation) ;
            # None = comme avant (pas de compaction préventive sur le chemin d'éval).
            compact_after_tokens=getattr(case, "compact_tokens", None),
        ):
            if kind == "content":
                traj.final_text += payload
            elif kind == "reasoning":
                traj.reasoning += payload
            elif kind == "usage":
                # Un event usage par appel modèle (réel ou estimé) -> compteur de TOURS
                # MODÈLE + cumul des tokens. L'usage estimé (provider muet) compte pareil.
                traj.model_turns += 1
                traj.prompt_tokens += payload.get("prompt_tokens") or 0
                traj.completion_tokens += payload.get("completion_tokens") or 0
                traj.cached_tokens += payload.get("cached_tokens") or 0
            elif kind == "done":
                traj.stop_reason = payload.get("reason") or ""
            elif kind == "tool_result":
                # L'event `tool_call` ne porte que {id, name} ; les ARGUMENTS réels
                # (path lu/écrit, commande shell) ne sont exposés que dans `tool_result`
                # (clés `path` et `cmd`). On reconstruit les args d'ici.
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
    except Exception as e:  # un run qui plante = donnée, pas un crash du harnais
        traj.error = f"{type(e).__name__}: {e}"
        traj.stop_reason = traj.stop_reason or "crash"
    traj.duration_s = round(time.monotonic() - t0, 1)
    return traj


# --- juge LLM ----------------------------------------------------------------

_JUDGE_SYS = (
    "Tu es un évaluateur STRICT et impartial du travail d'un agent. On te donne une tâche, "
    "un critère de réussite, et la trace de ce que l'agent a fait. Tu juges UNIQUEMENT "
    "d'après la trace. Réponds en JSON sur une ligne : "
    '{"pass": true/false, "score": 1-5, "reason": "…"}. '
    "pass=true seulement si le critère est clairement rempli."
)


def judge(client, model, case, traj) -> dict:
    tools_seen = ", ".join(n for n, _ in traj.tool_calls) or "(aucun)"
    user = (
        f"TÂCHE : {case.prompt}\n\n"
        f"CRITÈRE DE RÉUSSITE : {case.rubric}\n\n"
        f"OUTILS APPELÉS (ordre) : {tools_seen}\n\n"
        f"RÉPONSE FINALE DE L'AGENT :\n{(traj.final_text or '(vide)')[:1500]}"
    )
    txt = ""
    try:
        for kind, chunk in client.stream_chat(
            [{"role": "user", "content": user}],
            _JUDGE_SYS,
            max_tokens=400,
            model=model,
            thinking=False,
        ):
            if kind == "content":
                txt += chunk
    except Exception as e:
        return {"pass": None, "score": None, "reason": f"juge indisponible : {e}"}
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {
            "pass": None,
            "score": None,
            "reason": f"JSON juge illisible : {txt[:120]}",
        }
    try:
        d = json.loads(m.group(0))
        return {
            "pass": bool(d.get("pass")),
            "score": d.get("score"),
            "reason": str(d.get("reason", ""))[:300],
        }
    except json.JSONDecodeError:
        return {
            "pass": None,
            "score": None,
            "reason": f"JSON juge invalide : {txt[:120]}",
        }


# --- agrégation & rapport ----------------------------------------------------


def _critical(checks: dict) -> dict:
    """Checks bloquants = ceux dont le nom ne commence pas par '_' (informatifs)."""
    return {k: v for k, v in checks.items() if not k.startswith("_")}


def case_passed(checks: dict) -> bool:
    crit = _critical(checks)
    return bool(crit) and all(crit.values())


def run_variant(
    client, model, name, prompts, cfg, perm, runs, max_iters, do_judge, only
):
    chat_p, sub_p = prompts
    results = {}  # case_id -> list[run dict]
    for case in CASES:
        if only and case.id not in only:
            continue
        runs_data = []
        for k in range(runs):
            with tempfile.TemporaryDirectory(prefix=f"loom_eval_{case.id}_") as tmp:
                ws = Path(tmp)
                case.setup(ws)
                traj = run_one(
                    client, model, chat_p, sub_p, case, ws, cfg, perm, max_iters
                )
                checks = case.check(traj, ws)
                jd = judge(client, model, case, traj) if do_judge else None
                runs_data.append(
                    {
                        "checks": checks,
                        "passed": case_passed(checks),
                        "n_model_turns": traj.model_turns,
                        "n_tool_calls": traj.n_tool_calls,
                        "prompt_tokens": traj.prompt_tokens,
                        "completion_tokens": traj.completion_tokens,
                        "cached_tokens": traj.cached_tokens,
                        "stop_reason": traj.stop_reason,
                        "duration_s": traj.duration_s,
                        "error": traj.error,
                        "tools": [n for n, _ in traj.tool_calls],
                        "final": (traj.final_text or "")[:800],
                        "judge": jd,
                    }
                )
                _save_transcript(name, case.id, k, traj, checks, jd)
                mark = "ok" if runs_data[-1]["passed"] else "XX"
                print(
                    f"  [{name}] {case.id} run{k + 1}/{runs} [{mark}] "
                    f"stop={traj.stop_reason or '?'} tours={traj.model_turns} "
                    f"outils={len(traj.tool_calls)} "
                    f"tok={traj.prompt_tokens}/{traj.completion_tokens} "
                    f"{traj.duration_s}s {runs_data[-1]['tools']}"
                    + (f" ERREUR={traj.error}" if traj.error else "")
                )
        results[case.id] = runs_data
    return results


def _save_transcript(variant, case_id, k, traj, checks, jd):
    d = _OUT / variant
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# {variant} / {case_id} / run {k + 1}", ""]
    lines.append("## Outils appelés")
    for n, a in traj.tool_calls:
        lines.append(f"- {n}({json.dumps(a, ensure_ascii=False)[:200]})")
    lines.append("\n## Réponse finale\n" + (traj.final_text or "(vide)"))
    lines.append(
        f"\n## Coût\nstop={traj.stop_reason or '?'} tours_modèle={traj.model_turns} "
        f"outils={len(traj.tool_calls)} tok_in={traj.prompt_tokens} "
        f"tok_out={traj.completion_tokens} durée={traj.duration_s}s"
    )
    lines.append("\n## Checks code")
    for c, v in checks.items():
        lines.append(f"- [{'x' if v else ' '}] {c}")
    if jd:
        lines.append(f"\n## Juge LLM\n{json.dumps(jd, ensure_ascii=False)}")
    if traj.error:
        lines.append(f"\n## ERREUR\n{traj.error}")
    (d / f"{case_id}_run{k + 1}.md").write_text("\n".join(lines), encoding="utf-8")


def report(all_results: dict, runs: int):
    """all_results : {variant: {case_id: [run...]}}. Imprime un tableau comparatif."""
    variants = list(all_results.keys())
    print("\n" + "=" * 70)
    print("RAPPORT D'ÉVAL — taux de réussite (runs réussis / total) par cas")
    print("=" * 70)
    head = "cas".ljust(16) + "".join(v.ljust(14) for v in variants)
    print(head)
    print("-" * len(head))
    summary = {
        v: {
            "pass": 0,
            "tot": 0,
            "jscore": [],
            "turns": [],
            "tools": [],
            "tok_in": [],
            "tok_out": [],
            "dur": [],
            "stops": {},
        }
        for v in variants
    }
    case_ids = [c.id for c in CASES if any(c.id in all_results[v] for v in variants)]
    for cid in case_ids:
        row = cid.ljust(16)
        for v in variants:
            rd = all_results[v].get(cid, [])
            p = sum(1 for r in rd if r["passed"])
            row += f"{p}/{len(rd)}".ljust(14)
            s = summary[v]
            s["pass"] += p
            s["tot"] += len(rd)
            s["turns"] += [r.get("n_model_turns", 0) for r in rd]
            s["tools"] += [r.get("n_tool_calls", 0) for r in rd]
            s["tok_in"] += [r.get("prompt_tokens", 0) for r in rd]
            s["tok_out"] += [r.get("completion_tokens", 0) for r in rd]
            s["dur"] += [r.get("duration_s", 0.0) for r in rd]
            for r in rd:
                sr = r.get("stop_reason") or "?"
                s["stops"][sr] = s["stops"].get(sr, 0) + 1
            s["jscore"] += [
                r["judge"]["score"]
                for r in rd
                if r.get("judge") and isinstance(r["judge"].get("score"), (int, float))
            ]
        print(row)
    print("-" * len(head))
    tot = "TOTAL".ljust(16)
    for v in variants:
        s = summary[v]
        tot += f"{s['pass']}/{s['tot']}".ljust(14)
    print(tot)

    def _avg(xs) -> str:
        return f"{sum(xs) / len(xs):.1f}" if xs else "n/a"

    for v in variants:
        s = summary[v]
        js = f"{sum(s['jscore']) / len(s['jscore']):.2f}" if s["jscore"] else "n/a"
        stops = " ".join(f"{k}={n}" for k, n in sorted(s["stops"].items()))
        print(
            f"  {v}: juge moyen={js}/5  tours modèle moy={_avg(s['turns'])}  "
            f"outils moy={_avg(s['tools'])}  tok in/out moy={_avg(s['tok_in'])}/"
            f"{_avg(s['tok_out'])}  durée moy={_avg(s['dur'])}s  stops: {stops}"
        )
    # COÛT PAR CAS : « ce cas passe, mais à quel prix » — le pass/fail seul le cache.
    print("\nCOÛT PAR CAS (moyennes par variante) :")
    for cid in case_ids:
        for v in variants:
            rd = all_results[v].get(cid, [])
            if not rd:
                continue
            stops = " ".join(
                f"{k}={n}"
                for k, n in sorted(
                    {
                        sr: sum(1 for r in rd if (r.get("stop_reason") or "?") == sr)
                        for sr in {r.get("stop_reason") or "?" for r in rd}
                    }.items()
                )
            )
            print(
                f"  {cid.ljust(16)} [{v}] tours={_avg([r.get('n_model_turns', 0) for r in rd])} "
                f"outils={_avg([r.get('n_tool_calls', 0) for r in rd])} "
                f"tok={_avg([r.get('prompt_tokens', 0) for r in rd])}/"
                f"{_avg([r.get('completion_tokens', 0) for r in rd])} "
                f"durée={_avg([r.get('duration_s', 0.0) for r in rd])}s  stops: {stops}"
            )
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "report.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDétail : {_OUT}\\report.json + transcripts par variante.")


# --- self-test (sans modèle) -------------------------------------------------


def self_test():
    """Valide que chaque grader s'exécute et renvoie un dict[str,bool], sans modèle."""
    print("SELF-TEST des graders (aucun modèle requis)\n")
    # Trajectoire synthétique « bon agent » minimale.
    traj = Trajectory(
        tool_calls=[
            ("read_file", {"path": "calc.py"}),
            ("edit_file", {"path": "calc.py"}),
            ("run_shell", {"command": "Get-ChildItem"}),
        ],
        tool_results=[
            {"name": "edit_file", "ok": True, "preview": ""},
            {"name": "run_shell", "ok": True, "preview": "OK"},
        ],
        final_text="C'est corrigé, le script tourne sans erreur.",
    )
    ok = True
    for case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            case.setup(ws)
            try:
                checks = case.check(traj, ws)
                assert isinstance(checks, dict) and all(
                    isinstance(v, bool) for v in checks.values()
                ), "check ne renvoie pas un dict[str,bool]"
                print(
                    f"  [ok] {case.id}: {len(checks)} checks -> "
                    f"{sum(_critical(checks).values())}/{len(_critical(checks))} critiques vrais"
                )
            except Exception as e:
                ok = False
                print(f"  [XX] {case.id}: {type(e).__name__}: {e}")
    print("\nSELF-TEST", "VERT" if ok else "ROUGE")
    return ok


# --- main --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--variant", choices=["old", "new", "both"], default="both")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--cases", default=None, help="ids séparés par des virgules")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    cfg = load_eval_config()
    model = args.model or cfg.default_model
    client, base_url = make_client(cfg, model)
    perm = make_perm(cfg)
    only = set(args.cases.split(",")) if args.cases else None

    print(
        f"Modèle : {model} @ {base_url}  | runs={args.runs}  variante={args.variant}  "
        f"juge={'non' if args.no_judge else 'oui'}\n"
    )
    variants = load_variants(args.variant)
    all_results = {}
    for name, prompts in variants.items():
        print(f"--- VARIANTE {name} ---")
        all_results[name] = run_variant(
            client,
            model,
            name,
            prompts,
            cfg,
            perm,
            args.runs,
            args.max_iters,
            not args.no_judge,
            only,
        )
    report(all_results, args.runs)


if __name__ == "__main__":
    main()
