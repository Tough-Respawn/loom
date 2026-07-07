"""Éval du skill code-review : détection à vérité-terrain (rappel / faux positifs / verdict).

Injecte le CORPS du skill (ce que renvoie use_skill) dans le prompt système - variante
`old` = git HEAD, `new` = disque - fait relire chaque cas étiqueté, puis grade :
  - RAPPEL : pour chaque problème planté, la revue l'a-t-elle attrapé ? (juge LLM)
  - FAUX POSITIFS : combien de problèmes infondés la revue invente-t-elle ? (juge LLM)
  - VERDICT : la revue tranche-t-elle correctement (NON-prêt si bugs, prêt si propre) ?

Usage :
  uv run python -m evals.run_review_eval --runs 2 --variant new   # baseline skill actuel
  uv run python -m evals.run_review_eval --runs 2                  # A/B old(HEAD) vs new(disque)
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass, field

from loom.agent.conversation import Conversation
from loom.extend.skills import _parse_skill_md
from loom.tools import AVAILABLE_TOOLS, build_registry

from evals.harness import (
    _RT,
    git_show,
    load_eval_config,
    make_client,
    make_perm,
)
from evals.review_cases import CASES, extract_verdict, verdict_ok

_OUT = _RT.parent / "evals" / "out_review"
_SKILL_REL = "loom/skills/code-review/SKILL.md"


@dataclass
class Trajectory:
    tool_calls: list = field(default_factory=list)
    final_text: str = ""
    error: str | None = None


# --- variantes du skill ------------------------------------------------------


def _git_show(rel: str) -> str:
    return git_show(rel)


def _skill_body(text: str) -> str:
    _name, _desc, body, _meta = _parse_skill_md(text, "code-review")
    return body


def load_variants(which: str) -> dict:
    new_body = _skill_body(
        (_RT / "skills" / "code-review" / "SKILL.md").read_text("utf-8")
    )
    old_body = _skill_body(_git_show(_SKILL_REL))
    allv = {"old": old_body, "new": new_body}
    return allv if which == "both" else {which: allv[which]}


# --- un run de revue ---------------------------------------------------------

_CHAT = (_RT / "prompts" / "chat.system.md").read_text("utf-8").strip()

_TASK = (
    "Relis le changement suivant AVANT merge et rends ton verdict. Le code est fourni "
    "ci-dessous (inutile de lancer git diff) :\n\n```\n{code}\n```"
)


def run_one(client, model, skill_body, case, cfg, perm, max_iters):
    system = f"{_CHAT}\n\n# Skill actif - code-review\n{skill_body}"
    convo = Conversation(system_prompt=system)
    traj = Trajectory()
    # ignore_cleanup_errors : cf. run_eval — un processus enfant qui tient encore le
    # workspace au cleanup (Windows) ne doit pas avorter l'éval entière.
    with tempfile.TemporaryDirectory(
        prefix=f"loom_rev_{case.id}_", ignore_cleanup_errors=True
    ) as tmp:
        registry = build_registry(
            workspace_dir=tmp,
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
        try:
            for kind, payload in client.stream_chat_tools(
                [{"role": "user", "content": _TASK.format(code=case.code)}],
                system,
                max_tokens=cfg.chat.max_tokens,
                model=model,
                registry=registry,
                thinking=False,
                max_iters=max_iters,
                permission=perm,
            ):
                if kind == "content":
                    traj.final_text += payload
                elif kind == "tool_result":
                    traj.tool_calls.append(payload.get("name"))
        except Exception as e:
            traj.error = f"{type(e).__name__}: {e}"
    return traj


# --- juge : rappel + faux positifs -------------------------------------------

_JUDGE_SYS = (
    "Tu évalues une revue de code, en juge impartial. On te donne le CODE relu, la REVUE "
    "produite, et la liste des PROBLÈMES attendus. Pour chaque problème attendu, compte-le "
    "comme identifié (true) si la revue le mentionne de façon SUBSTANTIELLE - le mot exact "
    "n'est pas requis, une description claire du même problème suffit. Compte aussi les "
    "problèmes DISTINCTS que la revue affirme mais qui sont CLAIREMENT INFONDÉS (absents du "
    "code) ; en cas de doute, ne les compte pas. Réponds en JSON sur une ligne : "
    '{"found": {"<id>": true/false, ...}, "spurious": <entier>}.'
)


def judge(client, model, case, review_text) -> dict:
    issues = "\n".join(f"- {i['id']} : {i['desc']}" for i in case.issues) or "(aucun)"
    user = (
        f"CODE RELU :\n```\n{case.code}\n```\n\n"
        f"REVUE PRODUITE :\n{(review_text or '(vide)')[:2000]}\n\n"
        f"PROBLÈMES ATTENDUS :\n{issues}"
    )
    txt = ""
    try:
        for kind, chunk in client.stream_chat(
            [{"role": "user", "content": user}],
            _JUDGE_SYS,
            max_tokens=500,
            model=model,
            thinking=False,
        ):
            if kind == "content":
                txt += chunk
    except Exception as e:
        return {"found": {}, "spurious": None, "error": f"juge indisponible : {e}"}
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {
            "found": {},
            "spurious": None,
            "error": f"JSON juge illisible : {txt[:120]}",
        }
    try:
        d = json.loads(m.group(0))
        found = {k: bool(v) for k, v in (d.get("found") or {}).items()}
        sp = d.get("spurious")
        return {
            "found": found,
            "spurious": int(sp) if isinstance(sp, (int, float)) else None,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "found": {},
            "spurious": None,
            "error": f"JSON juge invalide : {txt[:120]}",
        }


# --- run d'une variante ------------------------------------------------------


def run_variant(client, model, name, skill_body, cfg, perm, runs, max_iters, only):
    results = {}
    for case in CASES:
        if only and case.id not in only:
            continue
        rd = []
        for k in range(runs):
            traj = run_one(client, model, skill_body, case, cfg, perm, max_iters)
            jd = judge(client, model, case, traj.final_text)
            planted = [i["id"] for i in case.issues]
            caught = [i for i in planted if jd.get("found", {}).get(i)]
            v_ok = verdict_ok(traj.final_text, case.expect_clean)
            rec = {
                "planted": planted,
                "caught": caught,
                "recall": (len(caught) / len(planted)) if planted else None,
                "spurious": jd.get("spurious"),
                "verdict": extract_verdict(traj.final_text),
                "verdict_ok": v_ok,
                "error": traj.error or jd.get("error"),
                # texte COMPLET : tronquer coupait la ligne « Verdict » et faussait le
                # re-grade hors-ligne (verdict lu comme « inconnu »).
                "review": traj.final_text or "",
            }
            rd.append(rec)
            _save(name, case.id, k, case, traj, rec)
            r = f"{len(caught)}/{len(planted)}" if planted else "propre"
            print(
                f"  [{name}] {case.id} run{k + 1}/{runs} rappel={r} "
                f"FP={rec['spurious']} verdict={rec['verdict']}[{'ok' if v_ok else 'XX'}]"
                + (f" ERREUR={rec['error']}" if rec["error"] else "")
            )
        results[case.id] = rd
    return results


def _save(variant, case_id, k, case, traj, rec):
    d = _OUT / variant
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {variant} / {case_id} / run {k + 1}",
        "",
        "## Code relu",
        "```",
        case.code,
        "```",
        "",
        "## Revue produite",
        traj.final_text or "(vide)",
        "",
        f"## Grader\nplantés={rec['planted']} attrapés={rec['caught']} "
        f"FP={rec['spurious']} verdict={rec['verdict']} (ok={rec['verdict_ok']})",
    ]
    if rec["error"]:
        lines.append(f"\n## ERREUR\n{rec['error']}")
    (d / f"{case_id}_run{k + 1}.md").write_text("\n".join(lines), encoding="utf-8")


# --- rapport -----------------------------------------------------------------


def report(all_results: dict):
    variants = list(all_results.keys())
    print("\n" + "=" * 74)
    print("RAPPORT CODE-REVIEW - rappel (problèmes attrapés) · FP · verdict")
    print("=" * 74)
    head = "cas".ljust(24) + "".join(v.ljust(25) for v in variants)
    print(head)
    print("-" * len(head))
    agg = {v: {"caught": 0, "planted": 0, "vok": 0, "n": 0, "sp": []} for v in variants}
    case_ids = [c.id for c in CASES if any(c.id in all_results[v] for v in variants)]
    for cid in case_ids:
        row = cid.ljust(24)
        for v in variants:
            rd = all_results[v].get(cid, [])
            planted = sum(len(r["planted"]) for r in rd)
            caught = sum(len(r["caught"]) for r in rd)
            vok = sum(1 for r in rd if r["verdict_ok"])
            cell = (
                f"{caught}/{planted} rec · {vok}/{len(rd)} verd"
                if planted
                else f"propre · {vok}/{len(rd)} verd"
            )
            row += cell.ljust(25)
            a = agg[v]
            a["caught"] += caught
            a["planted"] += planted
            a["vok"] += vok
            a["n"] += len(rd)
            a["sp"] += [r["spurious"] for r in rd if isinstance(r["spurious"], int)]
        print(row)
    print("-" * len(head))
    for v in variants:
        a = agg[v]
        rec = f"{100 * a['caught'] / a['planted']:.0f}%" if a["planted"] else "n/a"
        verd = f"{100 * a['vok'] / a['n']:.0f}%" if a["n"] else "n/a"
        sp = f"{sum(a['sp']) / len(a['sp']):.2f}" if a["sp"] else "n/a"
        print(f"  {v}: rappel global={rec}  verdict correct={verd}  FP moyen/run={sp}")
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "report.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), "utf-8"
    )
    print(f"\nDétail : {_OUT}\\report.json + transcripts par variante.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--variant", choices=["old", "new", "both"], default="both")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--cases", default=None)
    args = ap.parse_args()

    cfg = load_eval_config()
    model = args.model or cfg.default_model
    client, base_url = make_client(cfg, model)
    perm = make_perm(cfg)
    only = set(args.cases.split(",")) if args.cases else None

    print(
        f"Modèle : {model} @ {base_url}  | runs={args.runs}  variante={args.variant}\n"
    )
    all_results = {}
    for name, body in load_variants(args.variant).items():
        print(f"--- VARIANTE {name} (skill {len(body)} car.) ---")
        all_results[name] = run_variant(
            client, model, name, body, cfg, perm, args.runs, args.max_iters, only
        )
    report(all_results)


if __name__ == "__main__":
    main()
