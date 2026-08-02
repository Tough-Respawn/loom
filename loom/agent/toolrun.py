from __future__ import annotations

from __future__ import annotations
import json
import threading
import time
from collections.abc import Iterator
from loom.agent.inline_image import (
    image_user_message,
    is_inline_image,
    parse_inline_image,
)

from loom.agent.debuglog import log_event
from loom.agent.guards import _VERIFY_STREAK_NOTE, _verify_streak_update
from loom.agent.toolsets import (
    _BROWSER_CHECKS,
    _BUG_SIGNAL_TOOLS,
    _SERIAL_WRITE,
    _WRITE_TOOLS,
)


def _safe_args(raw: str) -> str:
    """Renvoie des arguments JSON VALIDES pour l'historique. Si l'appel a été tronqué
    (réponse coupée par max_tokens -> JSON cassé), on remet `{}` : sans ça, le JSON
    invalide reste dans la conversation et CHAQUE requête suivante échoue (500 'parse
    error') -> cascade infinie. Le message d'erreur d'outil signale déjà la troncature."""
    raw = raw or "{}"
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return "{}"


def _sub_activity_line(kind: str, payload) -> str:
    """Rend un event de sous-agent en ligne lisible pour le flux live de sa pastille.
    Le détail (synthèse) reste dans le résultat final ; ici on veut juste VOIR l'ouvrier
    agir : quel outil il appelle, s'il réussit, et sa synthèse au fil de l'eau."""
    if kind == "tool_call":
        return f"\n→ {payload.get('name', 'outil')}"
    if kind == "tool_result":
        mark = "✓" if payload.get("ok") else "✕"
        loc = f" {payload['path']}" if payload.get("path") else ""
        head = (payload.get("preview") or "").split("\n")[0][:80]
        return f"  {mark}{loc} {head}\n"
    if kind == "content":
        return payload if isinstance(payload, str) else ""
    return ""


def _stream_tool_events(registry, tc_id: str, name: str, args: dict):
    """Exécute un outil STREAMANT (dispatch_agent) en relayant son activité live.

    Générateur partagé par les chemins parallèle et séquentiel : yield les events
    à relayer (tool_stream, sub_usage) puis, en dernier, ("__result__", synthèse).
    L'appelant route les events vers son canal (yield direct ou queue de thread)."""
    parts: list[str] = []
    for sub_kind, sub_payload in registry.run_stream(name, args):
        line = _sub_activity_line(sub_kind, sub_payload)
        if line:
            yield ("tool_stream", {"id": tc_id, "text": line})
        if sub_kind == "content" and isinstance(sub_payload, str):
            parts.append(sub_payload)
        elif sub_kind == "usage":  # conso du sous-agent -> totaux de session
            yield ("sub_usage", sub_payload)
    yield ("__result__", "".join(parts).strip() or "(le sous-agent n'a rien renvoyé)")


def _tool_result_payload(
    tc_id: str, name: str, ok: bool, tool_content, args: dict
) -> dict:
    """Payload de l'event tool_result (pastille UI), UNIQUE pour les chemins parallèle
    et séquentiel. detail/in_full sont SPÉCIALISÉS pour les outils d'écriture/shell —
    jamais parallel-safe, donc le chemin parallèle retombe toujours sur le cas générique.
    - preview (1 ligne) : état replié ; detail conservé pour rétro-compat ;
    - in_full = ce que l'outil a REÇU (commande, contenu écrit, diff, chemin/args) ;
    - out_full = ce qu'il a RENVOYÉ. Le tout borné."""
    if name == "write_file":
        detail = args.get("content") or ""
    elif name == "edit_file":
        detail = f"- {args.get('old_string', '')}\n+ {args.get('new_string', '')}"
    else:
        detail = tool_content
    if name == "run_shell":
        in_full = args.get("command") or ""
    elif name in ("write_file", "append_file"):
        in_full = f"{args.get('path', '')}\n{args.get('content', '')}"
    elif name == "edit_file":
        in_full = (
            f"{args.get('path', '')}\n- {args.get('old_string', '')}"
            f"\n+ {args.get('new_string', '')}"
        )
    else:
        in_full = args.get("path") or json.dumps(args, ensure_ascii=False)
    return {
        "id": tc_id,
        "name": name,
        "ok": ok,
        "preview": str(tool_content)[:300],
        "path": args.get("path"),
        # Commande réellement lancée par run_shell : pour la VOIR dans la pastille
        # (sinon on ne voit que le résultat, pas ce qui a tourné).
        "cmd": args.get("command"),
        "detail": detail[:4000] if detail else None,
        "in_full": str(in_full)[:8000],
        "out_full": str(tool_content)[:8000],
    }


def _run_tools_parallel(
    registry, tool_calls: list[dict], convo: list[dict], st: dict
) -> Iterator[tuple[str, object]]:
    """Exécute un tour d'outils PARALLEL-SAFE concurremment (1 thread par outil),
    relaie l'activité live au fil de l'eau et recolle les résultats DANS L'ORDRE
    dans `convo`. Met à jour st["executed"] / st["fail_count"]."""
    import queue as _queue

    _q: _queue.Queue = _queue.Queue()
    _res: dict = {}  # id -> (result, ok, args)
    # Signale à l'UI un GROUPE parallèle -> rendu en « arène » côté à côté (animation
    # des agents qui tournent en même temps), pas des pastilles empilées.
    yield (
        "parallel",
        {
            "ids": [tc["id"] for tc in tool_calls],
            "names": [tc["name"] for tc in tool_calls],
        },
    )
    for _tc in tool_calls:  # pastilles d'abord : elles se rempliront en parallèle
        yield ("tool_call", {"id": _tc["id"], "name": _tc["name"]})
        log_event(
            "tool.call",
            name=_tc["name"],
            args_len=len(_tc["arguments"] or ""),
        )

    def _pworker(tc):
        name = tc["name"]
        try:
            pargs = json.loads(tc["arguments"] or "{}")
        except json.JSONDecodeError:
            _res[tc["id"]] = (
                "erreur: arguments tronqués (réponse coupée).",
                False,
                {},
            )
            _q.put((tc["id"], "__done__", None))
            return
        try:
            if registry.is_streaming(
                name
            ):  # dispatch_agent : activité relayée live (helper partagé)
                r = "(le sous-agent n'a rien renvoyé)"
                for ek, ep in _stream_tool_events(registry, tc["id"], name, pargs):
                    if ek == "__result__":
                        r = ep
                    else:
                        _q.put((tc["id"], ek, ep))
            else:  # lecture/recherche : exécution directe
                r = registry.run(name, pargs)
            okp = not r.startswith("erreur")
        except Exception as e:  # noqa: BLE001 - un outil qui casse n'arrête pas les autres
            r, okp = f"erreur: {e}", False
        _res[tc["id"]] = (r, okp, pargs)
        _q.put((tc["id"], "__done__", None))

    _threads = [
        threading.Thread(target=_pworker, args=(tc,), daemon=True, name="loom-parallel")
        for tc in tool_calls
    ]
    for _t in _threads:
        _t.start()
    _left = len(_threads)
    while _left > 0:  # relaie l'activité live des N outils au fil de l'eau
        _tid, _kind, _payload = _q.get()
        if _kind == "__done__":
            _left -= 1
        else:
            yield (_kind, _payload)
    for _t in _threads:
        _t.join()
    for (
        tc
    ) in tool_calls:  # résultats DANS L'ORDRE (messages `tool` cohérents pour l'API)
        name = tc["name"]
        r, okp, pargs = _res.get(tc["id"], ("(vide)", False, {}))
        # Streak de troncature tenu ICI (mono-thread) et pas dans _pworker : st
        # n'est pas protégé contre les écritures concurrentes des workers.
        if str(r).startswith("erreur: arguments tronqués"):
            st["truncated_streak"] = st.get("truncated_streak", 0) + 1
        else:
            st["truncated_streak"] = 0
        convo.append({"role": "tool", "tool_call_id": tc["id"], "content": r})
        if name == "dispatch_agent" and not str(r).startswith("refusé"):
            st["executed"] = True
        if not okp and name in _BUG_SIGNAL_TOOLS:
            st["fail_count"] += 1
        yield (
            "tool_result",
            _tool_result_payload(tc["id"], name, okp, r, pargs),
        )


def _run_tools_sequential(
    tool_calls: list[dict],
    registry,
    permission,
    confirm,
    convo: list[dict],
    strong: bool,
    st: dict,
) -> Iterator[tuple[str, object]]:
    """Exécute les appels d'outils d'un tour, SÉQUENTIELLEMENT : permission/confirm,
    garde-fou P1.1 (1 write par tour), images inline différées, anti sur-vérification.
    Met à jour convo et st (executed, files_written, fail_count, verify_streak)."""
    wrote_this_turn = False  # P1.1 : un seul write_file/append_file par tour
    # Images inline (read_image) à faire VOIR au modèle : différées après TOUS
    # les résultats d'outils (les messages `tool` doivent rester contigus).
    image_followups: list[dict] = []
    for tc in tool_calls:
        name = tc["name"]
        yield ("tool_call", {"id": tc["id"], "name": name})
        log_event("tool.call", name=name, args_len=len(tc["arguments"] or ""))
        _t_tool = time.monotonic()
        try:
            args = json.loads(tc["arguments"] or "{}")
            st["truncated_streak"] = 0
        except json.JSONDecodeError:
            # Arguments tronqués (réponse coupée par max_tokens). NE PAS exécuter
            # avec des args vides (erreur trompeuse 'path manquant') : signaler la
            # troncature pour que le modèle réémette l'appel en plus court.
            # Le streak est lu par _preventive_compaction : à saturation de fenêtre
            # (completion étranglée), « plus court » ne suffit JAMAIS — 2 troncatures
            # de suite déclenchent une compaction forcée avant l'appel suivant.
            st["truncated_streak"] = st.get("truncated_streak", 0) + 1
            result = (
                "erreur: arguments tronqués (réponse coupée). "
                "Réémets cet appel d'outil, en plus court."
            )
            convo.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            yield (
                "tool_result",
                {"id": tc["id"], "name": name, "ok": False, "preview": result},
            )
            continue
        # Rail dur des outils différés : rendre l'erreur tool_search AVANT la
        # permission. Un appel aveugle à un outil dangereux ne doit pas ouvrir
        # une confirmation pour une commande qui ne sera de toute façon pas lancée.
        if registry and getattr(registry, "requires_schema", lambda _name: False)(name):
            result = registry.run(name, args)
            convo.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            yield (
                "tool_result",
                _tool_result_payload(tc["id"], name, False, result, args),
            )
            continue
        # P1.1 : sérialiser les écritures à gros contenu (1 par tour) -> évite le
        # batch de N gros write_file/append_file qui sature max_tokens et tronque.
        # Les éditions par bloc (edit/replace/insert) ne passent PAS par ici.
        if name in _SERIAL_WRITE and wrote_this_turn:
            result = (
                "différé : un seul write_file/append_file par tour. Réémets "
                "cet appel (seul) au prochain tour."
            )
            convo.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            yield (
                "tool_result",
                {
                    "id": tc["id"],
                    "name": name,
                    "ok": False,
                    "preview": result,
                    "path": args.get("path"),
                },
            )
            continue
        # MCP est tiers et donc dangereux par défaut (nom inconnu de la politique
        # -> ask). Un serveur explicitement déclaré de confiance porte danger=False
        # et contourne cette confirmation, sans affaiblir les autres outils.
        trusted_mcp = bool(
            name.startswith("mcp_")
            and registry
            and not getattr(registry, "is_dangerous", lambda _name: True)(name)
        )
        decision = permission(name, args) if permission else None
        # La confiance explicite supprime seulement la demande prudente due au
        # statut tiers. Elle ne contourne jamais un refus global (deny_all).
        if trusted_mcp and decision is not None and decision.action == "ask":
            decision = None
        action = decision.action if decision else "allow"

        if action == "deny":
            # Garde-fou non contournable : jamais exécuté.
            result = f"refusé par la politique de sécurité: {decision.reason}"
            ok = False
        elif action == "ask":
            # Confirmation interactive : on signale l'UI puis on ATTEND la
            # décision via `confirm` (bloquant). Refus par défaut sans confirm.
            summary = str(args.get("command") or args.get("path") or "")
            yield (
                "tool_request",
                {"id": tc["id"], "name": name, "summary": summary},
            )
            if confirm and confirm(tc["id"], name, args):
                result = (
                    registry.run(name, args) if registry else "erreur: pas d'outils"
                )
                ok = not result.startswith("erreur")
            else:
                result = "refusé par l'utilisateur"
                ok = False
        elif registry and registry.is_streaming(name):  # allow + streamant
            # Outil streamant (dispatch_agent) : activité relayée EN DIRECT dans
            # sa pastille, synthèse reconstruite (helper partagé avec le parallèle).
            result = "(le sous-agent n'a rien renvoyé)"
            for ek, ep in _stream_tool_events(registry, tc["id"], name, args):
                if ek == "__result__":
                    result = ep
                else:
                    yield (ek, ep)
            ok = not result.startswith("erreur")
        else:  # allow
            result = registry.run(name, args) if registry else "erreur: pas d'outils"
            ok = not result.startswith("erreur")

        # read_image renvoie une image inline encodée : on ne met qu'un accusé
        # TEXTE dans le message `tool` (pas de base64 géant), et on diffère le
        # message `user` multimodal qui fera réellement VOIR l'image au modèle.
        if is_inline_image(result):
            caption, data_url = parse_inline_image(result)
            tool_content = f"[image « {caption} » chargée — fournie ci-dessous]"
            image_followups.append(image_user_message(caption, data_url))
            ok = True
        else:
            tool_content = result
        # Anti SUR-VÉRIFICATION (informatif, jamais bloquant) : au-delà de
        # _VERIFY_STREAK_NOTE checks navigateur VERTS d'affilée sans changement
        # d'état entre-temps, le résultat le dit au modèle. On n'empêche RIEN
        # (les checks restent exclus du non-progrès : re-prouver est légitime) ;
        # on nomme la preuve déjà faite. Coupé pour un modèle fort (strong).
        st["verify_streak"] = _verify_streak_update(name, ok, st["verify_streak"])
        if (
            not strong
            and ok
            and name in _BROWSER_CHECKS
            and st["verify_streak"] >= _VERIFY_STREAK_NOTE
        ):
            tool_content += (
                f"\n[harnais : {st['verify_streak']} vérifications vertes d'affilée "
                "sans changement d'état entre-temps — la preuve est faite. "
                "Conclus MAINTENANT ; ne re-vérifie que si tu modifies "
                "quelque chose.]"
            )
        convo.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": tool_content}
        )
        log_event(
            "tool.result",
            name=name,
            ok=ok,
            ms=round((time.monotonic() - _t_tool) * 1000),
            preview=str(tool_content)[:90],
        )
        yield (
            "tool_result",
            _tool_result_payload(tc["id"], name, ok, tool_content, args),
        )
        if name in _SERIAL_WRITE:
            wrote_this_turn = True
        # Suivi pour l'audit de claim : une EXÉCUTION réelle (run_shell/dispatch,
        # même en échec mais hors refus de permission) et les FICHIERS écrits.
        if name in ("run_shell", "dispatch_agent") and not str(result).startswith(
            "refusé"
        ):
            st["executed"] = True
        if ok and name in _WRITE_TOOLS and args.get("path"):
            st["files_written"].add(args["path"])
        # Cascade de bugs : on compte les échecs des outils d'EXÉCUTION/VÉRIF (pas les
        # erreurs d'usage type ligne hors limite). Au 2e échec, on IMPOSE la méthode debug.
        if not ok and name in _BUG_SIGNAL_TOOLS:
            st["fail_count"] += 1
    convo.extend(image_followups)  # images vues au tour suivant
