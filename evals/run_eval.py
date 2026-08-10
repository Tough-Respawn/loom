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
from datetime import UTC
from pathlib import Path

from evals.cases import CASES
from evals.harness import (
    _RT,
    git_show,
    load_eval_config,
    make_client,
    make_perm,
)
from loom.agent.conversation import Conversation
from loom.tools import AVAILABLE_TOOLS, build_registry

_OUT = _RT.parent / "evals" / "out"


def new_campaign_dir(sha: str | None = None, stamp: str | None = None) -> Path:
    """Dossier UNIQUE d'une campagne : out/runs/<sha>_<horodatage>[-N].

    PREUVES CONSERVÉES : deux campagnes ne s'écrasent JAMAIS, même commit et même
    seconde comprises (suffixe -N par création exclusive). L'échec windows_shell
    2/3 du 2026-07-24 est indiagnosticable parce que out/<variante>/ était écrasé
    à chaque campagne — ce dossier par run corrige ça. Transcripts détaillés +
    report.json + campaign.json du run SEULEMENT ; les baselines compactes restent
    dans out/history/ (résumés versionnés, séparés des transcripts). out/runs/ est
    couvert par le .gitignore existant (evals/out/*) : rien de gros n'est
    versionné. Les transcripts contiennent les cas d'éval (publics) et les SORTIES
    DU MODÈLE : aucune garantie TECHNIQUE d'absence de secrets dans ces sorties —
    c'est le gitignore qui les garde hors dépôt, pas un filtrage ; ne pas les
    publier tels quels."""
    from datetime import datetime

    base = _OUT / "runs"
    base.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = f"{sha or 'nosha'}_{stamp}"
    n = 1
    while True:
        d = base / (name if n == 1 else f"{name}-{n}")
        try:
            d.mkdir(parents=False, exist_ok=False)
            return d
        except FileExistsError:
            n += 1


def write_campaign_meta(
    run_dir: Path,
    *,
    sha: str,
    dirty: bool,
    model: str,
    variants: list[str],
    runs: int,
    cases: set[str] | None,
    judge: bool,
) -> None:
    """campaign.json : les conditions EXACTES de la campagne, à côté de ses
    transcripts (sans le diff complet — `dirty` suffit à savoir si la variante
    « new » mesurait du code non commité)."""
    from datetime import datetime

    meta = {
        "sha": sha or "",
        "dirty": bool(dirty),
        "model": model,
        "variants": list(variants),
        "runs": runs,
        "cases": sorted(cases) if cases else None,
        "judge": bool(judge),
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (run_dir / "campaign.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@dataclass
class Trajectory:
    tool_calls: list = field(default_factory=list)  # [(name, args_dict)]
    tool_results: list = field(default_factory=list)  # [{name, ok, preview}]
    final_text: str = ""
    reasoning: str = ""
    error: str | None = None
    # Séparer tours modèle et appels d'outils révèle le coût derrière un simple succès.
    model_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    stop_reason: str = ""
    duration_s: float = 0.0
    # ST-01 : activité RÉELLE des sous-boucles (une entrée par tier appelé par un
    # dispatch_agent) — modèle effectif, tours, outils, tokens, stop, synthèse.
    sub_agents: list = field(default_factory=list)

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    # `n_turns` conserve son ancien sens d'appels d'outils.
    n_turns = n_tool_calls


def _record_subagents(client, model: str, sink: list):
    """Instrumentation du HARNAIS (jamais de la production) : remplace
    `client.stream_chat_tools` par un enregistreur. Le parent est appelé via la
    référence d'origine (renvoyée), donc TOUT appel passant par l'attribut vient
    d'un SubAgentRunner (`_run_tier`) : une entrée par tier réellement appelé,
    avec le modèle effectif, les tours, les outils, les tokens, la raison d'arrêt
    et la synthèse COMPLÈTE (bornée large, pas un aperçu).

    Renvoie (orig, restore) : itérer `orig` pour le parent, appeler `restore()`
    en finally."""
    orig = client.stream_chat_tools

    def recorded(messages, system_prompt, max_tokens, **kw):
        rec = {
            # _run_tier passe model=tier ; None = la route locale par défaut,
            # c'est-à-dire le modèle de la campagne.
            "model": kw.get("model") or model,
            "model_turns": 0,
            "tools": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "stop_reason": "",
            "synthesis": "",
        }
        sink.append(rec)
        for kind, payload in orig(messages, system_prompt, max_tokens, **kw):
            if kind == "usage":
                rec["model_turns"] += 1
                rec["prompt_tokens"] += payload.get("prompt_tokens") or 0
                rec["completion_tokens"] += payload.get("completion_tokens") or 0
            elif kind == "tool_result":
                rec["tools"].append(payload.get("name"))
            elif kind == "content" and isinstance(payload, str):
                rec["synthesis"] += payload
            elif kind == "done":
                rec["stop_reason"] = payload.get("reason") or ""
            yield (kind, payload)
        rec["synthesis"] = rec["synthesis"].strip()[:4000]

    client.stream_chat_tools = recorded

    def restore():
        client.stream_chat_tools = orig

    return orig, restore


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
    mcp_hub=None,
    deferred_tools=False,
):
    """Exécute la boucle agentique sur un cas dans un workspace neuf ; renvoie la Trajectory."""
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
        deferred_tools=deferred_tools,
        mcp_hub=mcp_hub,
    )
    prompt = case.prompt.replace("{NOTES_PATH}", (ws / "docs" / "notes.md").as_posix())
    # Préinjecter l'historique teste la saturation sans payer sa génération.
    messages = [
        *(getattr(case, "history", None) or []),
        {"role": "user", "content": prompt},
    ]
    traj = Trajectory()
    # L'enregistreur intercepte les sous-boucles (dispatch_agent) ; le parent
    # passe par `orig` et n'est donc jamais compté comme sous-agent.
    orig_stream, restore_stream = _record_subagents(client, model, traj.sub_agents)
    t0 = time.monotonic()
    try:
        for kind, payload in orig_stream(
            messages,
            chat_prompt,
            max_tokens=cfg.chat.max_tokens,
            model=model,
            registry=registry,
            thinking=False,
            max_iters=max_iters,
            permission=perm,
            # Un seuil par cas permet de forcer le chemin de compaction.
            compact_after_tokens=getattr(case, "compact_tokens", None),
        ):
            if kind == "content":
                traj.final_text += payload
            elif kind == "reasoning":
                traj.reasoning += payload
            elif kind == "usage":
                # Chaque événement d'usage représente un tour modèle, réel ou estimé.
                traj.model_turns += 1
                traj.prompt_tokens += payload.get("prompt_tokens") or 0
                traj.completion_tokens += payload.get("completion_tokens") or 0
                traj.cached_tokens += payload.get("cached_tokens") or 0
            elif kind == "done":
                traj.stop_reason = payload.get("reason") or ""
            elif kind == "tool_result":
                # Reconstruire les arguments depuis `tool_result`, seul événement qui les expose.
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
    finally:
        restore_stream()  # l'instrumentation ne survit jamais au run
    traj.duration_s = round(time.monotonic() - t0, 1)
    return traj


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


def _critical(checks: dict) -> dict:
    """Checks bloquants = ceux dont le nom ne commence pas par '_' (informatifs)."""
    return {k: v for k, v in checks.items() if not k.startswith("_")}


def case_passed(checks: dict) -> bool:
    crit = _critical(checks)
    return bool(crit) and all(crit.values())


def _run_record(traj, checks: dict, jd, model: str) -> dict:
    """Entrée d'UN run pour report.json : trajectoire du parent ET activité réelle
    des sous-agents (`sub_agents`, capturée par _record_subagents — modèle effectif,
    tours, outils, tokens, stop, synthèse). Factorisé pour être testable tel quel."""
    return {
        "checks": checks,
        "passed": case_passed(checks),
        "model": model,
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
        "sub_agents": traj.sub_agents,
        "judge": jd,
    }


def run_variant(
    client,
    model,
    name,
    prompts,
    cfg,
    perm,
    runs,
    max_iters,
    do_judge,
    only,
    mcp_hub=None,
    deferred_tools=False,
    run_dir: Path | None = None,
):
    chat_p, sub_p = prompts
    results = {}  # case_id -> list[run dict]
    for case in CASES:
        if only and case.id not in only:
            continue
        runs_data = []
        for k in range(runs):
            # Sous Windows, un handle tardif ne doit pas faire perdre tout le run d'évaluation.
            with tempfile.TemporaryDirectory(
                prefix=f"loom_eval_{case.id}_", ignore_cleanup_errors=True
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
                    max_iters,
                    mcp_hub,
                    deferred_tools,
                )
                checks = case.check(traj, ws)
                jd = judge(client, model, case, traj) if do_judge else None
                runs_data.append(_run_record(traj, checks, jd, model))
                _save_transcript(
                    run_dir, name, case.id, k, traj, checks, jd, model=model
                )
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


def _save_transcript(
    run_dir: Path | None, variant, case_id, k, traj, checks, jd, model=None
):
    """Transcript détaillé d'UN run de cas, sous le dossier de campagne (jamais
    écrasé d'une campagne à l'autre). `run_dir` None (appel hors campagne) ->
    repli sur l'ancien emplacement out/<variante>/. Chaque appel d'outil est
    suivi de l'aperçu borné de son RÉSULTAT : c'est ce qui rend diagnosticable
    une délégation (synthèse du sous-agent, marqueur « [relève : …] »)."""
    d = (run_dir / variant) if run_dir is not None else (_OUT / variant)
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# {variant} / {case_id} / run {k + 1}", ""]
    lines.append("## Outils appelés")
    for i, (n, a) in enumerate(traj.tool_calls):
        lines.append(f"- {n}({json.dumps(a, ensure_ascii=False)[:200]})")
        if i < len(traj.tool_results):
            r = traj.tool_results[i]
            mark = "ok" if r.get("ok") else "KO"
            preview = " ".join(str(r.get("preview", "")).split())[:160]
            lines.append(f"  -> [{mark}] {preview}")
    # L'activité RÉELLE des sous-boucles (pas un aperçu) : c'est la donnée qui
    # rend une délégation diagnosticable — modèle effectif, coût, synthèse entière.
    if traj.sub_agents:
        lines.append("\n## Sous-agents")
        for i, s in enumerate(traj.sub_agents, 1):
            lines.append(
                f"### sous-agent {i} — modèle={s.get('model', '?')} "
                f"tours={s.get('model_turns', 0)} outils={s.get('tools', [])} "
                f"tok={s.get('prompt_tokens', 0)}/{s.get('completion_tokens', 0)} "
                f"stop={s.get('stop_reason') or '?'}"
            )
            lines.append("Synthèse :\n" + (s.get("synthesis") or "(vide)"))
    lines.append("\n## Réponse finale\n" + (traj.final_text or "(vide)"))
    lines.append(
        f"\n## Coût\nmodèle={model or '?'} stop={traj.stop_reason or '?'} "
        f"tours_modèle={traj.model_turns} "
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


def report(all_results: dict, runs: int, run_dir: Path | None = None):
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
    # Afficher le coût par cas complète le verdict binaire.
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
            turns = [r.get("n_model_turns", 0) for r in rd]
            print(
                f"  {cid.ljust(16)} [{v}] tours={_avg(turns)} "
                f"(min {min(turns)}/max {max(turns)}) "
                f"outils={_avg([r.get('n_tool_calls', 0) for r in rd])} "
                f"tok={_avg([r.get('prompt_tokens', 0) for r in rd])}/"
                f"{_avg([r.get('completion_tokens', 0) for r in rd])} "
                f"durée={_avg([r.get('duration_s', 0.0) for r in rd])}s  stops: {stops}"
            )
    # Le rapport JSON vit avec les transcripts de SA campagne (preuves conservées) ;
    # out/report.json reste une copie « dernier run » pour les habitudes existantes.
    out_dir = run_dir if run_dir is not None else _OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(all_results, ensure_ascii=False, indent=2)
    (out_dir / "report.json").write_text(payload, encoding="utf-8")
    if run_dir is not None:
        _OUT.mkdir(parents=True, exist_ok=True)
        (_OUT / "report.json").write_text(payload, encoding="utf-8")
    print(f"\nDétail : {out_dir}\\report.json + transcripts par variante.")


def pin_baseline(all_results: dict, runs: int, model: str) -> None:
    """Épingle un résumé COMPACT du run sous out/history/<sha>.json : la baseline
    persistante par commit. L'A/B git HEAD vs disque mesure le delta du diff COURANT ;
    l'historique épinglé mesure la DÉRIVE sur des semaines (re-run même commit = remplacé).
    Résumé seul (pass + coûts moyens par cas), pas les transcripts : diff-able et léger."""
    from datetime import datetime

    from evals.harness import git_head_sha

    sha = git_head_sha()
    if not sha:
        print("(baseline non épinglée : git indisponible)")
        return

    def _mean(xs) -> float:
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    cases_summary: dict = {}
    for variant, cases in all_results.items():
        for cid, rd in cases.items():
            entry = cases_summary.setdefault(cid, {})
            turns = [r.get("n_model_turns", 0) for r in rd]
            toks = [r.get("prompt_tokens", 0) for r in rd]
            entry[variant] = {
                "pass": sum(1 for r in rd if r["passed"]),
                "runs": len(rd),
                "model_turns": _mean(turns),
                # Les extrêmes révèlent les runs pathologiques masqués par la moyenne.
                "model_turns_minmax": [min(turns), max(turns)] if turns else [0, 0],
                "tool_calls": _mean([r.get("n_tool_calls", 0) for r in rd]),
                "prompt_tokens": _mean(toks),
                "prompt_tokens_max": max(toks) if toks else 0,
                "completion_tokens": _mean([r.get("completion_tokens", 0) for r in rd]),
                "duration_s": _mean([r.get("duration_s", 0.0) for r in rd]),
                "stops": sorted({r.get("stop_reason") or "?" for r in rd}),
            }
    hist = _OUT / "history"
    hist.mkdir(parents=True, exist_ok=True)
    payload = {
        "sha": sha,
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "runs": runs,
        "cases": cases_summary,
    }
    path = hist / f"{sha}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Baseline épinglée : {path}")


def _injection_tests() -> bool:
    """Tests DÉTERMINISTES des garde-fous du harnais, par INJECTION de payloads cassés.

    Ces chemins (JSON d'appel malformé, appel émis en texte, dégénérescence en boucle,
    compaction) ne se testent PAS en E2E : on ne force pas un modèle stochastique à
    produire un appel cassé à la demande. On injecte donc directement les payloads
    dans les fonctions de garde (même patron que les trajectoires synthétiques)."""
    # Importer depuis les modules propriétaires évite de dépendre d'alias privés de façade.
    from loom.agent.compaction import _force_fit, _microcompact_tools
    from loom.agent.guards import _verify_streak_update
    from loom.agent.streaming import _salvage_tool_calls, _scan_repeat
    from loom.agent.toolrun import _safe_args

    checks: dict[str, bool] = {}

    # Un JSON tronqué ne doit jamais contaminer l'historique suivant.
    checks["JSON cassé -> args remis à {}"] = (
        _safe_args('{"path": "a.py", "old') == "{}"
    )
    checks["JSON valide -> conservé"] = (
        _safe_args('{"path": "a.py"}') == '{"path": "a.py"}'
    )

    # Récupérer les appels d'outils textuels dans les deux formats tolérés.
    hermes = 'bla <tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
    got = _salvage_tool_calls(hermes, "")
    checks["salvage Hermes/JSON"] = bool(got) and got[0]["name"] == "read_file"
    xmlish = (
        "<function=run_shell><parameter=command>Get-ChildItem</parameter></function>"
    )
    got = _salvage_tool_calls("", xmlish)
    checks["salvage XML-ish"] = bool(got) and got[0]["name"] == "run_shell"
    checks["texte sans appel -> rien"] = (
        _salvage_tool_calls("bonjour, voilà.", "") == []
    )

    # Couper les longues répétitions sans confondre la ponctuation répétée du code.
    counts: dict[str, int] = {}
    loop_line = "Je vais maintenant créer les fichiers du projet.\n"
    hit = None
    for _ in range(12):
        _, hit = _scan_repeat(loop_line, counts)
        if hit:
            break
    checks["boucle détectée au seuil"] = hit is not None
    counts2: dict[str, int] = {}
    _, hit2 = _scan_repeat("},\n" * 50, counts2)
    checks["lignes courtes de code ignorées"] = hit2 is None

    # La microcompaction préserve les résultats d'outils récents.
    convo = [
        {"role": "tool", "tool_call_id": str(i), "content": f"gros résultat {i}" * 50}
        for i in range(5)
    ]
    cleared = _microcompact_tools(convo, keep_recent_tools=2)
    checks["microcompact vide les vieux"] = cleared == 3 and "gros résultat 4" in str(
        convo[4]["content"]
    )

    # Préserver les petites preuves denses et vider seulement les gros dumps.
    convo_s = [
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": "erreur: exit=1 (module manquant)",
        },
        {"role": "tool", "tool_call_id": "b", "content": "gros dump de fichier " * 100},
        {"role": "tool", "tool_call_id": "c", "content": "modifié : calc.py"},
    ]
    cleared_s = _microcompact_tools(convo_s, keep_recent_tools=0)
    checks["microcompact sélectif : petites preuves gardées"] = (
        cleared_s == 1
        and "exit=1" in convo_s[0]["content"]
        and convo_s[2]["content"] == "modifié : calc.py"
    )

    # Le force-fit doit converger sans supprimer les deux derniers messages.
    convo2 = [{"role": "user", "content": "x" * 20000} for _ in range(10)]
    ok_fit = _force_fit(convo2, "system", 5000)
    checks["force-fit converge sous budget"] = ok_fit and len(convo2) >= 2

    # Préserver la tâche courante tant qu'un autre contenu reste réductible.
    task = "Lis le fichier facts.txt et donne le code d'accès."
    convo3 = [
        {"role": "user", "content": "ballast archivé " * 2000},
        {"role": "assistant", "content": "ballast archivé " * 2000},
        {"role": "user", "content": task},
        {"role": "user", "content": "[harnais : note de recentrage]"},
    ]
    _force_fit(convo3, "system", 4000)
    checks["force-fit préserve la tâche courante"] = any(
        m.get("content") == task for m in convo3
    )

    # Un budget impossible ne doit ni perdre la tâche ni laisser un message outil orphelin.
    convo5 = [
        {"role": "user", "content": "vieux tour " * 200},
        {"role": "assistant", "content": "vieille réponse " * 200},
        {"role": "user", "content": "la tâche courante"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "résultat d'outil " * 100},
    ]
    _force_fit(convo5, "S" * 9000, 4500)
    task_alive = any(m.get("content") == "la tâche courante" for m in convo5)
    orphan = False
    for i, m in enumerate(convo5):
        if m.get("role") == "tool":
            j = i - 1
            while j >= 0 and convo5[j].get("role") == "tool":
                j -= 1
            if not (
                j >= 0
                and convo5[j].get("role") == "assistant"
                and convo5[j].get("tool_calls")
            ):
                orphan = True
    checks["force-fit pop : tâche vivante, zéro tool orphelin"] = (
        task_alive and not orphan
    )

    # Garder tête et fin préserve aussi les conclusions et messages d'erreur.
    convo4 = [
        {"role": "assistant", "content": "DEBUT " + "x" * 10000 + " FIN"},
        {"role": "user", "content": "tâche"},
    ]
    _force_fit(convo4, "", 6000)
    c4 = str(convo4[0]["content"])
    checks["force-fit garde tête ET queue"] = c4.startswith("DEBUT") and c4.endswith(
        "FIN"
    )

    # Une mutation ou un échec réinitialise la série de vérifications vertes.
    s = 0
    for _ in range(4):
        s = _verify_streak_update("check_page", True, s)
    after_checks = s  # 4 checks verts d'affilée
    s = _verify_streak_update("read_file", True, s)  # lire ne change rien
    after_read = s
    s = _verify_streak_update("edit_file", True, s)  # modifier périme la preuve
    after_edit = s
    s2 = _verify_streak_update("check_page", False, 5)  # check raté = info nouvelle
    checks["streak sur-vérification : monte/reset correctement"] = (
        after_checks == 4 and after_read == 4 and after_edit == 0 and s2 == 0
    )

    ok = all(checks.values())
    print("INJECTION des garde-fous (payloads cassés, aucun modèle requis)\n")
    for name, v in checks.items():
        print(f"  [{'ok' if v else 'XX'}] {name}")
    print()
    return ok


def self_test():
    """Valide que chaque grader s'exécute et renvoie un dict[str,bool], sans modèle,
    et exécute les tests d'injection des garde-fous du harnais."""
    guards_ok = _injection_tests()
    print("SELF-TEST des graders (aucun modèle requis)\n")
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
    ok = ok and guards_ok
    print("\nSELF-TEST", "VERT" if ok else "ROUGE")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--variant", choices=["old", "new", "both"], default="both")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--cases", default=None, help="ids séparés par des virgules")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--mcp-fixture",
        action="store_true",
        help=(
            "branche le serveur stdio hermétique sur la variante new seulement, "
            "pour mesurer le coût agentique du catalogue MCP"
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    cfg = load_eval_config()
    model = args.model or cfg.default_model
    client, base_url = make_client(cfg, model)
    perm = make_perm(cfg)
    only = set(args.cases.split(",")) if args.cases else None

    # Empêcher la veille durant les longues évaluations sans activité utilisateur.
    from loom.runtime.stay_awake import StayAwake

    _awake = StayAwake()
    _awake.acquire()

    print(
        f"Modèle : {model} @ {base_url}  | runs={args.runs}  variante={args.variant}  "
        f"juge={'non' if args.no_judge else 'oui'}\n"
    )
    variants = load_variants(args.variant)
    mcp_hub = None
    if args.mcp_fixture:
        import sys

        from loom.config import _parse_mcp_server
        from loom.tools.mcp import McpHub

        fixture = _RT.parent / "tests" / "fake_mcp_server.py"
        mcp_hub = McpHub(
            [
                _parse_mcp_server(
                    {
                        "name": "eval-fixture",
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(fixture)],
                        "timeout_s": 3.0,
                        "danger_override": False,
                    }
                )
            ]
        )
    all_results = {}
    from evals.harness import git_dirty, git_head_sha

    sha = git_head_sha()
    run_dir = new_campaign_dir(sha or None)
    write_campaign_meta(
        run_dir,
        sha=sha,
        dirty=git_dirty(),
        model=model,
        variants=list(variants),
        runs=args.runs,
        cases=only,
        judge=not args.no_judge,
    )
    print(f"Campagne : {run_dir}\n")
    try:
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
                mcp_hub=mcp_hub if name == "new" else None,
                deferred_tools=args.mcp_fixture,
                run_dir=run_dir,
            )
    finally:
        if mcp_hub is not None:
            mcp_hub.close()
    report(all_results, args.runs, run_dir)
    # Ne jamais remplacer une baseline complète par un sous-ensemble filtré.
    if only:
        print("(baseline non épinglée : run partiel via --cases)")
    else:
        pin_baseline(all_results, args.runs, model)


if __name__ == "__main__":
    main()
