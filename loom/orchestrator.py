# loom/orchestrator.py
"""Orchestrateur : pipeline d'agents séquentiel (1 GPU, pas de parallélisme).

`run_pipeline` est un GÉNÉRATEUR : il yield les events au fil de l'eau (streaming
live + confirmation interactive possible). Chaque agent peut avoir ses propres
outils (registre construit via `tool_factory`) et passe par la même garde de
permission/confirmation que le chat. Boucle review→fix bornée : si le dernier
agent (relecteur) rend un verdict bloquant, le développeur repasse une fois.
"""

from __future__ import annotations

import re
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

# Contrat avec verify_web.js : un défaut kind='asset' a pour evidence
# "référence introuvable: <chemin> (...)". On en extrait le fichier à matérialiser.
_ASSET_REF_RE = re.compile(r"référence introuvable:\s*([\w\-./]+)")
_SAFE_ASSET_RE = re.compile(
    r"^[\w\-./]+\.(?:css|js|mjs|json|svg|png|jpg|jpeg|gif|webp)$", re.IGNORECASE
)


def _missing_assets(report, known_paths) -> list[str]:
    """Assets LOCAUX référencés mais introuvables (défauts kind='asset') à générer.
    Filtre : extension sûre, chemin relatif sans remontée, pas déjà planifié, borné."""
    if report is None:
        return []
    known = set(known_paths)
    out: list[str] = []
    for d in report.defects:
        if d.kind != "asset":
            continue
        m = _ASSET_REF_RE.search(d.evidence)
        if not m:
            continue
        raw = m.group(1)
        if ".." in raw or raw.startswith("/"):  # remontée de dossier / chemin absolu
            continue
        rel = raw.lstrip("./")
        if not _SAFE_ASSET_RE.match(rel):
            continue
        if rel not in known and rel not in out:
            out.append(rel)
    return out[:6]  # garde-fou : jamais un déluge de fichiers matérialisés


def _asset_role(path: str) -> str:
    """Rôle injecté au générateur pour un asset que le plan a oublié de spécifier."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext == "css":
        return (
            "Feuille de style CSS partagée, référencée par les pages via <link>. Style "
            "sobre et cohérent : barre de navigation, mise en page du contenu, cartes, "
            "boutons, tableau de bord. Pas de @import ni d'URL externe."
        )
    if ext in ("js", "mjs"):
        return (
            "Script référencé par une page (fonctions globales, sans import/export ES)."
        )
    return f"Asset {ext or '?'} référencé par une page."


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
    max_rounds: int = 6,
    semantic_review: bool = False,
    lesson_store=None,
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
    import queue
    from concurrent.futures import ThreadPoolExecutor

    from openai import APIConnectionError, APITimeoutError

    from loom.parallel import (
        FileSpec,
        best_of,
        cap_rewrites,
        compute_budget,
        derive_modes,
        edit_one,
        fix_one,
        generate_one,
    )

    # Transitoires = ré-essayables (le serveur a brièvement lâché/saturé) ; un APIError
    # 500 (overflow contexte) ne l'est PAS (ré-essayer ne change rien → on garde l'ancien).
    transient = (APIConnectionError, APITimeoutError)

    run = AgentRun(task=task)
    # État PERSISTANT : dernière bonne version par fichier. Source de vérité du run.
    state: dict[str, dict] = {}

    def _gen_with_retry(gen_fn, spec, on_token, attempts: int = 2):
        """Appelle gen_fn(spec, on_token) ; ré-essaie sur erreur transitoire (borné)."""
        last: Exception | None = None
        for _ in range(max(1, attempts)):
            try:
                return gen_fn(spec, on_token)
            except transient as exc:  # noqa: PERF203 - ré-essai borné, pas un hot-loop
                last = exc
        raise last  # type: ignore[misc]

    def _gen_phase(agent_id, role, gen_fn, specs, max_workers):
        """tool_begin par fichier → génère EN PARALLÈLE et STREAME la pensée/sortie de
        chaque fichier au fil de l'eau (via une queue thread-safe, events taggés par `id`)
        → écrit et met à jour l'ÉTAT. Un échec garde la version précédente (last-good).

        Le streaming rend le codeur VISIBLE (comme le planificateur) : chaque worker pousse
        ses deltas dans la queue, le générateur (seul thread à muter l'état) les relaie."""
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
        q: queue.Queue = queue.Queue()

        def _worker(s):
            sid = ids[s.path]

            def on_token(kind, text):
                q.put(("token", sid, kind, text))

            try:
                q.put(("result", s, _gen_with_retry(gen_fn, s, on_token), None))
            except Exception as exc:  # noqa: BLE001 - relayé au thread principal
                q.put(("result", s, None, exc))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s in specs:
                ex.submit(_worker, s)
            remaining = len(specs)
            while remaining:
                item = q.get()
                if item[0] == "token":
                    _, sid, kind, text = item
                    if kind in ("reasoning", "content"):
                        yield {"type": kind, "agent": agent_id, "id": sid, "text": text}
                    continue
                # ("result", spec, res, exc) — l'écriture et la MAJ d'état se font ICI
                # (seul thread), jamais dans un worker.
                _, s, res, exc = item
                remaining -= 1
                sid = ids[s.path]
                try:
                    if exc is not None:
                        raise exc
                    path, content = res
                    # NB: pour un patch, edit_one a déjà écrit ; cette ré-écriture
                    # (atomique) normalise juste les fins de ligne en \n.
                    abspath = write(path, content)
                except Exception as exc2:  # noqa: BLE001 - un fichier raté ne casse pas le run
                    kept = " (version précédente conservée)" if s.path in state else ""
                    yield {
                        "type": "tool_result",
                        "agent": agent_id,
                        "id": sid,
                        "name": "write_file",
                        "ok": False,
                        "path": s.path,
                        "preview": f"erreur: {exc2}{kept}",
                    }
                    continue  # last-good : on n'écrase JAMAIS l'état par une erreur
                state[path] = {"content": content, "abspath": abspath}
                yield {
                    "type": "tool_result",
                    "agent": agent_id,
                    "id": sid,
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
        from loom.explore import explore, list_project_files
        from loom.parallel import _PLAN_SYS, _parse_plan, _plan_prompt
        from loom.planning import (
            critique_design,
            decompose_into_stories,
            write_plan_artifacts,
        )

        ground = explore(task, workspace, context=context)
        # Brownfield strict : si des fichiers existent déjà, le plan DOIT les réutiliser
        # (tue la divergence d'archi observée : 3 fichiers -> 4 inventés).
        existing = list_project_files(workspace)
        messages = [
            {
                "role": "user",
                "content": _plan_prompt(task, ground.summary, existing_files=existing),
            }
        ]
        raw = ""
        if hasattr(client, "stream_chat"):
            # Streame le plan EN DIRECT : l'UI voit la réflexion ET le contrat se générer,
            # au lieu d'un appel bloquant opaque. thinking=True pour montrer la pensée.
            # Fallback `complete` pour les clients minimalistes (tests) sans stream_chat.
            for kind, payload in client.stream_chat(
                messages, _PLAN_SYS, max_tokens, model=model, thinking=True
            ):
                if kind == "reasoning":
                    yield {"type": "reasoning", "agent": "plan", "text": payload}
                elif kind == "content":
                    raw += payload
                    yield {"type": "content", "agent": "plan", "text": payload}
                elif kind == "usage":
                    yield {"type": "usage", "agent": "plan", **payload}
        else:
            raw = client.complete(
                messages, _PLAN_SYS, max_tokens=max_tokens, model=model, thinking=False
            )
            yield {"type": "content", "agent": "plan", "text": raw}
        design, specs = _parse_plan(raw)
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
        "text": "\n\n**Fichiers :** " + ", ".join(s.path for s in specs),
    }
    yield {"type": "agent_done", "agent": "plan"}
    if not specs:
        yield {"type": "run_done", "run": run}
        return

    # 1b) PLANIFICATION PROFONDE : auto-critique du plan (comble ses trous) puis découpe
    # en USER STORIES (critères d'acceptation observables), externalisées en .md sous
    # `.loom/` via le writer du run -> dev/vérificateur s'y appuient, contexte libéré.
    try:
        yield {
            "type": "agent_start",
            "agent": "deep_plan",
            "role": "Planification profonde",
            "model": model,
        }
        design = critique_design(client, design, task, model=model)
        stories = decompose_into_stories(client, design, specs, task, model=model)
        write_plan_artifacts(workspace, design, stories, write=write)
        run.stories = stories
        yield {
            "type": "content",
            "agent": "deep_plan",
            "text": "**User stories :**\n"
            + "\n".join(f"- {s.id} — {s.title}" for s in stories),
        }
        yield {"type": "agent_done", "agent": "deep_plan"}
    except Exception:  # noqa: BLE001 - la planif profonde ne doit jamais casser le run
        stories = []

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

    # 2) GÉNÉRATION PARALLÈLE — chaque fichier reçoit les US qui le concernent (le dev
    # déroule des US concrètes) ET les LEÇONS apprises de runs passés (auto-amélioration),
    # sans casser le batching (option « US informent la gen »).
    from loom.planning import stories_for_file

    lessons_text = (
        "\n".join(f"- {x}" for x in lesson_store.recent()) if lesson_store else ""
    )

    def _gen_dispatch(s, on_token=None):
        if mode_by_path.get(s.path) == "patch":
            return edit_one(
                client,
                design,
                s,
                workspace,
                all_paths,
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
            stories_text=stories_for_file(stories, s.path),
            lessons_text=lessons_text,
            on_token=on_token,
        )

    yield from _gen_phase(
        "build",
        "Développeur (parallèle)",
        _gen_dispatch,
        specs,
        max_workers,
    )
    report = yield from _verify_phase(all_paths)
    # Défauts du PREMIER jet = ce que le dev a raté -> matière à apprentissage (task 9).
    initial_defects = (
        list(report.defects) if (report is not None and not report.ok) else []
    )

    # SELF-HEAL : un asset LOCAL référencé mais jamais planifié (typiquement la feuille de
    # style partagée que le plan a oubliée) ne peut pas être réparé par la boucle de fix —
    # il n'est PAS dans les specs. On l'AJOUTE et on le génère : le plan sous-spécifié se
    # répare au lieu de laisser un lien cassé. Une seule passe, bornée.
    heal = _missing_assets(report, all_paths)
    if heal:
        new_specs = [FileSpec(path=p, role=_asset_role(p)) for p in heal]
        specs.extend(new_specs)
        all_paths.extend(heal)
        for p in heal:
            mode_by_path[p] = "create"
        yield from _gen_phase(
            "assets",
            "Assets manquants (auto-réparation)",
            _gen_dispatch,
            new_specs,
            max_workers,
        )
        report = yield from _verify_phase(all_paths)

    # 3) FIX (boucle fermée) : régénère en parallèle avec l'union des fichiers + les défauts.
    # On boucle tant qu'il reste des défauts OU qu'un fichier planifié manque (borné).
    rounds = 0
    prev_count: int | None = None
    while rounds < max_rounds and (
        (report is not None and not report.ok) or (verifier and _incomplete())
    ):
        rounds += 1
        yield {"type": "revision", "n": rounds}
        diag = (
            format_report(report) if report is not None else "Fichier(s) manquant(s)."
        )
        current = [(p, state[p]["content"]) for p in all_paths if p in state]

        def _fix_dispatch(s, on_token=None, _c=current, _d=diag):
            def _make():
                if mode_by_path.get(s.path) == "patch":
                    return edit_one(
                        client,
                        design,
                        s,
                        workspace,
                        all_paths,
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

        # Stop sur ABSENCE DE PROGRÈS : on continue tant que le NOMBRE de défauts décroît
        # strictement (pas les seules locations : tous les défauts d'une page sont sur le
        # même fichier, donc le set de locations ne rétrécit jamais et coupait à tort dès
        # le 2e tour). Le 1er tour établit la baseline ; ensuite, si ça ne baisse plus (et
        # qu'aucun fichier ne manque), on arrête — un 4B oscille sinon indéfiniment.
        cur_count = len(report.defects) if report is not None else 0
        if (
            not _incomplete()
            and prev_count is not None
            and cur_count >= prev_count  # pas de rétrécissement strict
        ):
            break
        prev_count = cur_count

    # Si on s'arrête en laissant des défauts, on le DIT clairement (pas de coupe silencieuse).
    if report is not None and not report.ok:
        yield {"type": "agent_start", "agent": "bilan", "role": "Bilan", "model": model}
        yield {
            "type": "content",
            "agent": "bilan",
            "text": (
                f"Arrêt après {rounds} correction(s) : {len(report.defects)} défaut(s) "
                "restant(s) (le modèle n'a pas convergé). Voir les défauts ci-dessus."
            ),
        }
        yield {"type": "agent_done", "agent": "bilan"}

    # VÉRIFICATION ORIENTÉE INTENTION : le verify déterministe dit « ça tourne », pas
    # « ça fait ce qui était demandé ». On confronte le code aux CRITÈRES D'ACCEPTATION
    # des US. Activée par défaut dès qu'il y a des critères (ou si semantic_review forcé).
    # Si des critères existent et sont violés, on lance UNE passe de correction bornée
    # (un 4B ne doit pas boucler) ; sinon advisory non bloquant.
    from loom.planning import acceptance_text

    accept = acceptance_text(stories)
    if report is not None and report.ok and (semantic_review or accept):
        from loom.parallel import review_semantic

        yield {
            "type": "agent_start",
            "agent": "review",
            "role": "Vérification d'intention",
            "model": model,
        }
        current = [(p, state[p]["content"]) for p in all_paths if p in state]
        sem = review_semantic(client, design, current, model=model, acceptance=accept)
        if sem:
            body = "Défauts d'INTENTION (critères non satisfaits) :\n" + "\n".join(
                f"- {d.location} : {d.evidence}" for d in sem
            )
        else:
            body = "Vérification d'intention : tous les critères semblent satisfaits."
        yield {"type": "content", "agent": "review", "text": body}
        yield {"type": "agent_done", "agent": "review"}

        # Si des CRITÈRES existent et sont violés : UNE passe de correction ciblée.
        if sem and accept:
            diag = "\n".join(f"- {d.location} : {d.evidence}" for d in sem)
            cur = [(p, state[p]["content"]) for p in all_paths if p in state]

            def _intent_fix(s, on_token=None, _c=cur, _d=diag):
                return best_of(
                    lambda: fix_one(
                        client,
                        design,
                        s,
                        _c,
                        _d,
                        model=model,
                        max_tokens=gen_max_tokens,
                        file_char_cap=file_char_cap,
                    ),
                    2,
                )

            yield {"type": "revision", "n": rounds + 1}
            yield from _gen_phase(
                "fix_intent", "Correction (intention)", _intent_fix, specs, max_workers
            )
            report = yield from _verify_phase(all_paths)

    # AUTO-AMÉLIORATION (task 9) : distille les erreurs du premier jet en UNE leçon
    # générale et la persiste. Les runs suivants la reçoivent dans le prompt de génération
    # -> Loom s'augmente de ses erreurs au lieu qu'on code chaque cas. Ne casse jamais.
    if lesson_store is not None and initial_defects:
        from loom.lessons import distill_lesson

        defects_text = "\n".join(f"{d.location}: {d.evidence}" for d in initial_defects)
        lesson = distill_lesson(client, defects_text, task, model=model)
        if lesson:
            lesson_store.add(lesson)
            yield {
                "type": "agent_start",
                "agent": "learn",
                "role": "Apprentissage",
                "model": model,
            }
            yield {
                "type": "content",
                "agent": "learn",
                "text": "Leçon retenue pour les prochains runs : " + lesson,
            }
            yield {"type": "agent_done", "agent": "learn"}

    yield {"type": "run_done", "run": run}
