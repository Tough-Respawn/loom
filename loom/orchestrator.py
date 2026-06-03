# loom/orchestrator.py
"""Orchestrateur : pipeline d'agents séquentiel (1 GPU, pas de parallélisme).

`run_pipeline` est un GÉNÉRATEUR : il yield les events au fil de l'eau (streaming
live + confirmation interactive possible). Chaque agent peut avoir ses propres
outils (registre construit via `tool_factory`) et passe par la même garde de
permission/confirmation que le chat. Boucle review→fix bornée : si le dernier
agent (relecteur) rend un verdict bloquant, le développeur repasse une fois.
"""

from __future__ import annotations

from collections.abc import Iterator

from loom.agents import (
    Agent,
    AgentRun,
    RunStep,
    build_step_messages,
    compose_agent_system_prompt,
    is_blocking,
    is_reviewer,
)
from loom.verify import format_report


def run_pipeline(
    agents: list[Agent],
    task: str,
    client,
    skills_dir: str,
    *,
    max_tokens: int,
    tool_factory=None,
    permission=None,
    confirm=None,
    max_revisions: int = 1,
    verifier=None,
) -> Iterator[dict]:
    """Déroule les agents et yield les events (agent_start/reasoning/content/
    tool_call/tool_request/tool_result/agent_done), chacun portant 'agent'.

    Renseigne `run` (AgentRun) au fil de l'eau ; l'appelant peut le récupérer via
    le dernier yield {'type':'run_done', 'run': AgentRun}.
    """
    run = AgentRun(task=task)

    def run_agent(agent: Agent, extra_context: str = "") -> Iterator[dict]:
        yield {
            "type": "agent_start",
            "agent": agent.id,
            "role": agent.role,
            "model": agent.model,
        }
        system_prompt = compose_agent_system_prompt(agent, skills_dir)
        messages = build_step_messages(task, run.steps)
        if extra_context:
            messages.append(
                {
                    "role": "user",
                    "content": "Rapport du VÉRIFICATEUR (déterministe) — corrige TOUS "
                    f"ces défauts :\n{extra_context}",
                }
            )
        registry = tool_factory(agent.tools) if (tool_factory and agent.tools) else None
        reasoning = ""
        content = ""
        written: list[str] = []
        if registry is not None and len(registry):
            source = client.stream_chat_tools(
                messages,
                system_prompt,
                agent.max_tokens or max_tokens,
                model=agent.model,
                registry=registry,
                thinking=agent.thinking,
                permission=permission,
                confirm=confirm,
            )
        else:
            source = client.stream_chat(
                messages,
                system_prompt,
                agent.max_tokens or max_tokens,
                model=agent.model,
                thinking=agent.thinking,
            )
        for kind, payload in source:
            if kind == "reasoning":
                reasoning += payload
                yield {"type": "reasoning", "agent": agent.id, "text": payload}
            elif kind == "content":
                content += payload
                yield {"type": "content", "agent": agent.id, "text": payload}
            elif kind == "usage":
                yield {"type": "usage", "agent": agent.id, **payload}
            elif kind in ("tool_begin", "tool_call", "tool_request", "tool_result"):
                if (
                    kind == "tool_result"
                    and payload.get("ok")
                    and payload.get("path")
                    and payload.get("name") in ("write_file", "edit_file")
                ):
                    written.append(payload["path"])
                yield {"type": kind, "agent": agent.id, **payload}
        run.steps.append(
            RunStep(
                agent_id=agent.id,
                role=agent.role,
                model=agent.model,
                reasoning=reasoning,
                content=content,
                written=written,
            )
        )
        yield {"type": "agent_done", "agent": agent.id}

    def _verify_phase():
        """Phase de vérification, signalée comme une ÉTAPE : yield 'verify_start' (dès
        qu'il y a des artefacts à vérifier) puis l'événement 'verify' avec le rapport.
        Renvoie le VerifyReport (ou None si rien à vérifier). Ne lève jamais."""
        seen: list[str] = []
        for step in run.steps:
            for p in step.written:
                if p and p not in seen:
                    seen.append(p)
        if not verifier or not seen:
            return None
        yield {"type": "verify_start", "count": len(seen)}
        try:
            report = verifier(seen)
        except Exception:  # noqa: BLE001 - un échec de vérif ne casse pas le run
            return None
        yield {
            "type": "verify",
            "ok": report.ok,
            "defects": [
                {"location": d.location, "kind": d.kind, "evidence": d.evidence}
                for d in report.defects
            ],
        }
        return report

    for agent in agents:
        yield from run_agent(agent)

    # HARD-GATE déterministe : le Vérificateur (exécution réelle) prime sur le texte
    # du relecteur. On boucle review→fix tant que verify a des défauts OU que le
    # relecteur bloque, en INJECTANT le rapport de défauts au développeur (P0.4).
    report = yield from _verify_phase()
    revisions = 0
    while (
        revisions < max_revisions
        and len(agents) >= 2
        and run.steps
        and (
            (report is not None and not report.ok)
            or (is_reviewer(agents[-1].role) and is_blocking(run.steps[-1].content))
        )
    ):
        revisions += 1
        coder, reviewer = agents[-2], agents[-1]
        yield {"type": "revision", "n": revisions}
        diag = format_report(report) if (report is not None and not report.ok) else ""
        yield from run_agent(coder, extra_context=diag)
        yield from run_agent(reviewer)
        report = yield from _verify_phase()

    yield {"type": "run_done", "run": run}


def run_build(
    task: str,
    client,
    *,
    model: str | None,
    write,
    workspace: str = ".",
    verifier=None,
    max_tokens: int = 2048,
    context: int = 8192,
    n_parallel: int = 1,
    max_rounds: int = 3,
    semantic_review: bool = False,
) -> Iterator[dict]:
    """Pipeline FAN-OUT (mode /run par défaut) : PLAN détaillé → génération PARALLÈLE
    isolée par fichier (1 appel non-streamé/fichier → pas de tool-call géant tronqué, pas
    de contexte partagé qui déborde, GPU batché) → verify → fix (boucle fermée).

    Budget DÉRIVÉ (P1) : `(max_workers, gen_max_tokens, file_char_cap)` viennent de
    `compute_budget(context, n_parallel, n_files)` — plus jamais de constantes hand-tunées
    qui font déborder le pool KV partagé (cf. F3/F4/F5, docs/plan-harness-robustesse).

    Boucle ROBUSTE (P3) : un ÉTAT PERSISTANT `path -> {content, abspath}` garde la
    DERNIÈRE bonne version de chaque fichier ; une génération qui échoue ne l'écrase pas,
    les transitoires (connexion/timeout) sont ré-essayées, et le verify tourne sur
    l'UNION de tous les fichiers connus (pas seulement ceux du dernier round).

    Yield les MÊMES events que run_pipeline (agent_start/content/agent_done/tool_begin/
    tool_result/verify_start/verify/revision/run_done) → l'UI Preact les rend sans
    modification. `write(path, content) -> chemin absolu`. `verifier(abs_paths)->Report`.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from openai import APIConnectionError, APITimeoutError

    from loom.parallel import (
        best_of,
        cap_rewrites,
        compute_budget,
        derive_modes,
        edit_one,
        fix_one,
        generate_one,
        plan_files,
    )

    # Transitoires = ré-essayables (le serveur a brièvement lâché/saturé) ; un APIError
    # 500 (overflow contexte) ne l'est PAS (ré-essayer ne change rien → on garde l'ancien).
    transient = (APIConnectionError, APITimeoutError)

    run = AgentRun(task=task)
    # État PERSISTANT : dernière bonne version par fichier. Source de vérité du run.
    state: dict[str, dict] = {}

    def _gen_with_retry(gen_fn, spec, attempts: int = 2):
        """Appelle gen_fn(spec) ; ré-essaie UNIQUEMENT sur erreur transitoire (borné)."""
        last: Exception | None = None
        for _ in range(max(1, attempts)):
            try:
                return gen_fn(spec)
            except transient as exc:  # noqa: PERF203 - ré-essai borné, pas un hot-loop
                last = exc
        raise last  # type: ignore[misc]

    def _gen_phase(agent_id, role, gen_fn, specs, max_workers):
        """tool_begin par fichier → génère EN PARALLÈLE (retry transitoire) → écrit et met
        à jour l'ÉTAT au fil de l'eau. Un échec garde la version précédente (last-good)."""
        yield {"type": "agent_start", "agent": agent_id, "role": role, "model": model}
        ids = {s.path: f"{agent_id}:{s.path}" for s in specs}
        for s in specs:
            yield {
                "type": "tool_begin",
                "agent": agent_id,
                "id": ids[s.path],
                "name": "write_file",
            }
        workers = max(1, min(max_workers, len(specs)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_gen_with_retry, gen_fn, s): s for s in specs}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    path, content = fut.result()
                    abspath = write(path, content)
                except Exception as exc:  # noqa: BLE001 - un fichier raté ne casse pas le run
                    kept = " (version précédente conservée)" if s.path in state else ""
                    yield {
                        "type": "tool_result",
                        "agent": agent_id,
                        "id": ids[s.path],
                        "name": "write_file",
                        "ok": False,
                        "path": s.path,
                        "preview": f"erreur: {exc}{kept}",
                    }
                    continue  # last-good : on n'écrase JAMAIS l'état par une erreur
                state[path] = {"content": content, "abspath": abspath}
                yield {
                    "type": "tool_result",
                    "agent": agent_id,
                    "id": ids[path],
                    "name": "write_file",
                    "ok": True,
                    "path": path,
                    "preview": f"écrit ({len(content)} car.)",
                    "detail": content[:4000],
                }
        yield {"type": "agent_done", "agent": agent_id}

    def _verify_phase(paths):
        """Vérifie l'UNION des fichiers connus (état), pas seulement le dernier round."""
        written = [state[p]["abspath"] for p in paths if p in state]
        if not verifier or not written:
            return None
        yield {"type": "verify_start", "count": len(written)}
        try:
            report = verifier(written)
        except Exception:  # noqa: BLE001 - une vérif qui casse ne casse pas le run
            return None
        yield {
            "type": "verify",
            "ok": report.ok,
            "defects": [
                {"location": d.location, "kind": d.kind, "evidence": d.evidence}
                for d in report.defects
            ],
        }
        return report

    # 1) PLAN détaillé (contrat : state, signatures, snippets, boucle, clavier, sélecteurs)
    yield {
        "type": "agent_start",
        "agent": "plan",
        "role": "Planificateur",
        "model": model,
    }
    try:
        from loom.explore import explore

        ground = explore(task, workspace, context=context)
        design, specs = plan_files(
            client,
            task,
            model=model,
            max_tokens=max_tokens,
            explore_summary=ground.summary,
        )
    except Exception as exc:  # noqa: BLE001
        yield {
            "type": "content",
            "agent": "plan",
            "text": f"erreur de planification : {exc}",
        }
        yield {"type": "agent_done", "agent": "plan"}
        yield {"type": "run_done", "run": run}
        return
    yield {
        "type": "content",
        "agent": "plan",
        "text": design + "\n\n**Fichiers :** " + ", ".join(s.path for s in specs),
    }
    yield {"type": "agent_done", "agent": "plan"}
    if not specs:
        yield {"type": "run_done", "run": run}
        return

    all_paths = [s.path for s in specs]
    # P1 : budget DÉRIVÉ du contexte serveur partagé (jamais hand-tuné).
    max_workers, gen_max_tokens, file_char_cap = compute_budget(
        context, n_parallel, len(specs)
    )

    # Mode par fichier (déterministe) : absent->create, existant+verify KO->rewrite,
    # existant+verify OK->patch ; rewrite d'un gros fichier dégradé en patch.
    _verifier_for_modes = verifier or (lambda paths: None)
    planned = cap_rewrites(
        derive_modes(specs, workspace, _verifier_for_modes), workspace
    )
    mode_by_path = {pf.spec.path: pf.mode for pf in planned}

    def _incomplete() -> bool:
        # Un fichier planifié totalement absent de l'état (génération jamais réussie) =
        # app incomplète : on continue à le retenter même si le verify des présents passe.
        return any(p not in state for p in all_paths)

    # 2) GÉNÉRATION PARALLÈLE
    def _gen_dispatch(s):
        if mode_by_path.get(s.path) == "patch":
            return edit_one(
                client,
                design,
                s,
                workspace,
                model=model,
                max_tokens=gen_max_tokens,
                file_char_cap=file_char_cap,
            )
        return generate_one(
            client,
            design,
            s,
            all_paths,
            model=model,
            max_tokens=gen_max_tokens,
            file_char_cap=file_char_cap,
        )

    yield from _gen_phase(
        "build",
        "Développeur (parallèle)",
        _gen_dispatch,
        specs,
        max_workers,
    )
    report = yield from _verify_phase(all_paths)

    # 3) FIX (boucle fermée) : régénère en parallèle avec l'union des fichiers + les défauts.
    # On boucle tant qu'il reste des défauts OU qu'un fichier planifié manque (borné).
    rounds = 0
    prev_locations: set[str] | None = None
    while rounds < max_rounds and (
        (report is not None and not report.ok) or (verifier and _incomplete())
    ):
        rounds += 1
        yield {"type": "revision", "n": rounds}
        diag = (
            format_report(report) if report is not None else "Fichier(s) manquant(s)."
        )
        current = [(p, state[p]["content"]) for p in all_paths if p in state]

        def _fix_dispatch(s, _c=current, _d=diag):
            def _make():
                if mode_by_path.get(s.path) == "patch":
                    return edit_one(
                        client,
                        design,
                        s,
                        workspace,
                        model=model,
                        max_tokens=gen_max_tokens,
                        file_char_cap=file_char_cap,
                        defects=_d,
                    )
                return fix_one(
                    client,
                    design,
                    s,
                    _c,
                    _d,
                    model=model,
                    max_tokens=gen_max_tokens,
                    file_char_cap=file_char_cap,
                )

            return best_of(_make, 2)

        yield from _gen_phase(
            f"fix{rounds}",
            f"Correction {rounds} (parallèle)",
            _fix_dispatch,
            specs,
            max_workers,
        )
        report = yield from _verify_phase(all_paths)

        # Stop anti-divergence : si l'ensemble des défauts ne DÉCROÎT pas (et qu'aucun
        # fichier ne manque), inutile de boucler — un 4B oscille (corrige A, casse B).
        cur_locations = (
            {d.location for d in report.defects} if report is not None else set()
        )
        if (
            not _incomplete()
            and prev_locations is not None
            and not cur_locations < prev_locations  # pas un sous-ensemble STRICT
        ):
            break
        prev_locations = cur_locations

    if semantic_review and report is not None and report.ok:
        from loom.parallel import review_semantic

        current = [(p, state[p]["content"]) for p in all_paths if p in state]
        sem = review_semantic(client, design, current, model=model)
        yield {
            "type": "agent_start",
            "agent": "review",
            "role": "Relecture sémantique",
            "model": model,
        }
        if sem:
            body = (
                "Défauts SÉMANTIQUES possibles (à vérifier — non bloquant) :\n"
                + "\n".join(f"- {d.location} : {d.evidence}" for d in sem)
            )
        else:
            body = "Relecture sémantique : aucun défaut comportemental détecté."
        yield {"type": "content", "agent": "review", "text": body}
        yield {"type": "agent_done", "agent": "review"}

    yield {"type": "run_done", "run": run}
