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

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    # `n_turns` conserve son ancien sens d'appels d'outils.
    n_turns = n_tool_calls




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
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "report.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDétail : {_OUT}\\report.json + transcripts par variante.")


def pin_baseline(all_results: dict, runs: int, model: str) -> None:
    """Épingle un résumé COMPACT du run sous out/history/<sha>.json : la baseline
    persistante par commit. L'A/B git HEAD vs disque mesure le delta du diff COURANT ;
    l'historique épinglé mesure la DÉRIVE sur des semaines (re-run même commit = remplacé).
    Résumé seul (pass + coûts moyens par cas), pas les transcripts : diff-able et léger."""
    from datetime import datetime, timezone

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
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
            )
    finally:
        if mcp_hub is not None:
            mcp_hub.close()
    report(all_results, args.runs)
    # Ne jamais remplacer une baseline complète par un sous-ensemble filtré.
    if only:
        print("(baseline non épinglée : run partiel via --cases)")
    else:
        pin_baseline(all_results, args.runs, model)


if __name__ == "__main__":
    main()
