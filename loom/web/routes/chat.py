from __future__ import annotations

import json
import re
import shutil
import threading
import time
import traceback
from functools import partial
from pathlib import Path

from flask import Response, request

from loom.agent import context
from loom.agent.client import _msg_chars, log_event, set_debug_log_path
from loom.prompts import IMAGE_REFINE_SYSTEM
from loom.runtime.comfy import ComfyError
from loom.runtime.models_profile import load_profile
from loom.web.app import (
    _action_trace_line,
    _build_user_content,
    _detect_workspace,
    _infer_title,
    _should_adopt,
    _sse,
)
from loom.web.routes.commands import _handle_goal_command, _handle_init_command
from loom.web.routes.helpers import (
    _cancel_for,
    _confirm,
    _engine_for,
    _ensure_local_server,
    _free_image_engines,
    _get_session,
    _local_busy_notice,
    _local_size_mb,
    _lock_for,
    _model_limits,
    _price_of,
    _session,
    _totals,
)
from loom.web.routes.maintenance import _post_turn_maintenance
from loom.web.routes.models import _handle_add_model_command
from loom.web.routes.system_prompt import _build_system_prompt

# Source de vérité de la palette; garder ce catalogue aligné avec les handlers.
CHAT_COMMANDS = [
    {
        "name": "/add-model",
        "usage": "/add-model · /add-model <recherche HF> · /add-model distant [url]",
        "description": "Ajouter un modèle : local (recherche Hugging Face, quant "
        "recommandé selon ta machine) ou distant (API OpenAI-compatible — la clé "
        "se donne à une étape dédiée, jamais dans la commande).",
    },
    {
        "name": "/remove-model",
        "usage": "/remove-model",
        "description": "Supprimer un modèle : local (dossier et GGUF effacés du "
        "disque) ou distant géré par l'UI — liste numérotée + confirmation.",
    },
    {
        "name": "/rebench",
        "usage": "/rebench · /rebench <id modèle local>",
        "description": "Recalibrer un modèle LOCAL texte tel que configuré "
        "(contexte par pente mesurée + vitesse en profondeur) — verdict comparé "
        "à l'actuel, application sur confirmation.",
    },
    {
        "name": "/goal",
        "usage": "/goal <condition> · /goal (statut) · /goal clear",
        "description": "Poser un objectif vérifiable pour la session, le consulter, "
        "ou l'effacer.",
    },
    {
        "name": "/init",
        "usage": "/init [dossier]",
        "description": "Analyser le workspace et générer sa fiche projet loom.md "
        "(injectée ensuite au contexte).",
    },
    {
        "name": "/cancel",
        "usage": "/cancel",
        "description": "Annuler le wizard en cours (ex. /add-model).",
    },
]


_HANDOFF_MAX_CHARS = 100_000


def _handoff_provenance(raw: str, source) -> list[dict[str, str]]:
    """Valide la chaîne reçue puis ajoute la source directe, vérifiée côté serveur.

    Aucun plafond de profondeur : un message peut faire autant d'allers-retours que
    nécessaire. Chaque champ est seulement borné pour empêcher une métadonnée UI de
    gonfler le prompt sans limite.
    """
    previous = []
    if raw:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("provenance de transfert invalide") from exc
        if not isinstance(decoded, list):
            raise ValueError("provenance de transfert invalide")
        for item in decoded:
            if not isinstance(item, dict):
                raise ValueError("provenance de transfert invalide")
            previous.append(
                {
                    "session_id": str(item.get("session_id", ""))[:80],
                    "title": str(item.get("title", ""))[:120],
                    "model": str(item.get("model", ""))[:120],
                }
            )
    previous.append(
        {
            "session_id": source.id,
            "title": (source.title or "session")[:120],
            "model": (source.conversation.model or "modèle par défaut")[:120],
        }
    )
    return previous


def _handoff_prompt(content: str, provenance: list[dict[str, str]]) -> str:
    direct = provenance[-1]
    route = " → ".join(f"{p['model']} (session « {p['title']} »)" for p in provenance)
    return (
        "[Message transferred from another Loom session]\n"
        f"Direct source: model {direct['model']}, session « {direct['title']} ».\n"
        f"Transfer route: {route}.\n"
        "Use the transferred content below as context and respond to it.\n\n"
        "--- transferred content ---\n"
        f"{content}"
    )




def _register_chat_routes(app, S):
    @app.post("/handoff")
    @app.post("/chat")
    def chat():
        is_handoff = request.path == "/handoff"
        display_message = (request.form.get("message") or "").strip()
        provenance: list[dict[str, str]] = []
        handoff_id = ""

        if not display_message or len(display_message) > (
            _HANDOFF_MAX_CHARS if is_handoff else 5000
        ):
            return Response("message invalide", status=400)

        if is_handoff:
            source_sid = (request.form.get("source_session_id") or "").strip()
            source = _get_session(S, source_sid)
            if source is None:
                return Response("session source introuvable", status=404)
            handoff_id = (request.form.get("handoff_id") or "").strip()[:80]
            try:
                provenance = _handoff_provenance(
                    request.form.get("provenance") or "", source
                )
            except ValueError as exc:
                return Response(str(exc), status=400)
            message = _handoff_prompt(display_message, provenance)
        else:
            message = display_message

        # Chaque session possède son verrou pour préserver le parallélisme entre onglets.
        req_sid = (request.form.get("session_id") or "").strip()
        # Ne jamais rabattre un identifiant explicite invalide sur la session active.
        sess = _get_session(S, req_sid) if req_sid else _session(S)
        if sess is None:
            return Response("session introuvable", status=404)
        if is_handoff and sess.id == source.id:
            return Response("transfert vers la session source refusé", status=400)
        if not is_handoff:
            S.cur["session"] = sess  # focus (défaut de l'index)
        sid = sess.id
        chat_lock = _lock_for(S, sid)
        cancel_event = _cancel_for(S, sid)

        queued_payload: str | dict = message
        if is_handoff:
            queued_payload = {
                "text": message,
                "display": display_message,
                "provenance": provenance,
                "handoff_id": handoff_id,
            }

        # Refuser un handoff sans génération active évite une note orpheline.
        queue_only = is_handoff and request.form.get("queue_only") == "1"
        if queue_only:
            if chat_lock.acquire(blocking=False):
                chat_lock.release()
                return Response("session cible disponible", status=409)
            if S.notes.push(sid, queued_payload) < 0:
                return Response(
                    "file d'attente pleine — attendre le prochain point d'arrêt ou Stop",
                    status=429,
                )
            return Response(
                "message transféré mis en file d'attente — il sera pris en compte "
                "au prochain point d'arrêt",
                status=202,
            )

        if not chat_lock.acquire(blocking=False):
            # Après STOP, attendre le verrou; sinon mettre en file pour la génération active.
            if not (
                cancel_event.is_set() and chat_lock.acquire(timeout=S.interrupt_wait)
            ):
                if S.notes.push(sid, queued_payload) < 0:
                    return Response(
                        "file d'attente pleine — attendre le prochain point d'arrêt ou Stop",
                        status=429,
                    )
                # La file transporte seulement du texte.
                queued_msg = (
                    "message mis en file d'attente (génération en cours) — il sera pris "
                    "en compte au prochain point d'arrêt"
                )
                if request.files.getlist("image"):
                    queued_msg += (
                        " (images ignorées : la file ne transporte que du texte)"
                    )
                return Response(queued_msg, status=202)

        # On tient le verrou : repartir d'un signal d'annulation propre.

        cancel_event.clear()

        # Revérifier après le verrou empêche de recréer une session supprimée entre-temps.
        if not S.session_store.session_dir(sid).exists():
            chat_lock.release()
            return Response("session supprimée", status=404)

        conv = sess.conversation

        def save():
            return S.session_store.save(sess)

        if not is_handoff:
            # Un wizard actif capture tout le texte jusqu'à sa fin.
            message, _wiz_resp = _handle_add_model_command(
                S, message, conv, sess, save, chat_lock
            )
            if _wiz_resp is not None:
                return _wiz_resp

            message, _goal_resp = _handle_goal_command(
                S, message, conv, save, chat_lock
            )
            if _goal_resp is not None:
                return _goal_resp

            message = _handle_init_command(S, message, sess)

        title_message = display_message if is_handoff else message

        # Ne pas injecter d'image inline dans un modèle sans projecteur vision.

        # Regrouper les traces agent et serveur dans le dossier de la session.

        _sdir = S.session_store.session_dir(sess.id)

        set_debug_log_path(_sdir / "debug.log")

        _serve_log = S.session_store.root.parent / "logs" / "serve.log"

        if _serve_log.exists():
            try:
                _sdir.mkdir(parents=True, exist_ok=True)

                shutil.copyfile(_serve_log, _sdir / "serve.log")

            except OSError:
                pass

        # Adopter un dossier explicite pour y résoudre les commandes et chemins relatifs.

        adopted_ws = None

        # Un transfert apporte du contexte, jamais un changement de workspace implicite.
        detected = None if is_handoff else _detect_workspace(message, S.workspace_dir)

        if detected:
            # Ne pas invalider le cache KV pour un chemin déjà interne au workspace.
            if detected != sess.workspace and _should_adopt(sess.workspace, detected):
                sess.workspace = detected

                S.session_store.save(sess)

                adopted_ws = detected

        try:
            content = _build_user_content(
                message,
                request.files.getlist("image"),
                is_vision=bool(conv.model and conv.model in S.vision_models),
                stash_dir=_sdir / "uploads",
            )

            conv.add("user", content)

            save()

            # La timeline doit contenir le message utilisateur pour rejouer toute la vue.
            user_event = {
                "content": display_message if is_handoff else message,
            }
            if is_handoff:
                user_event.update(
                    {
                        "provenance": provenance,
                        "handoff_id": handoff_id,
                    }
                )
            S.session_store.append_event(sess.id, "user", user_event)

            # La compaction reste dans le générateur afin d'être visible dans le flux.
            system_prompt, strong = _build_system_prompt(
                S, conv, workspace=sess.workspace
            )
        except ValueError as exc:
            chat_lock.release()

            return Response(str(exc), status=400)

        except Exception:  # noqa: BLE001
            chat_lock.release()
            traceback.print_exc()
            return Response("erreur interne", status=500)

        # Image et LLM partagent le GPU: sérialiser et libérer la VRAM avant diffusion.
        if conv.model in S.image_model_ids:
            _im = S.image_by_id[conv.model]

            def generate_image():
                S.stay_awake.acquire()
                _img_held = False
                _sess = sess

                def _finish(md_text: str):
                    conv.add("assistant", md_text)
                    save()
                    _data = {"text": md_text}
                    if provenance:
                        _data["provenance"] = provenance
                    S.session_store.append_event(_sess.id, "text", _data)
                    return _sse("text", **_data)

                try:
                    yield _sse("status", label="préparation du moteur image…")
                    S.local_gen_lock.acquire()
                    _img_held = True
                    S.local_busy["reason"] = "image"
                    # Retirer le chemin de l'image pour ne garder que l'instruction d'édition.
                    src_image, msg_text = None, message
                    for cand in re.findall(
                        r'"([^"]+\.(?:png|jpe?g|webp|bmp))"', message, re.IGNORECASE
                    ) + re.findall(
                        r"[A-Za-z]:[\\/][^\s\"']+\.(?:png|jpe?g|webp|bmp)",
                        message,
                        re.IGNORECASE,
                    ):
                        if Path(cand).is_file():
                            src_image = cand
                            msg_text = (
                                message.replace(f'"{cand}"', " ")
                                .replace(cand, " ")
                                .strip()
                            )
                            break
                    # Le refiner est best-effort et doit être déchargé avant la diffusion.
                    prompt, refined = msg_text, False
                    if _im.refiner and _im.refiner in S.models:
                        yield _sse(
                            "status", label=f"affinage du prompt ({_im.refiner})…"
                        )
                        try:
                            if (
                                _im.refiner in S.remote_model_ids
                                or _ensure_local_server(S, wait=90.0)
                            ):
                                # Pour une photo, demander une instruction plutôt qu'une description.
                                _refine_in = (
                                    "[An input photo is attached; write an EDIT "
                                    "instruction: what to change, what must stay "
                                    f"identical.] {msg_text}"
                                    if src_image
                                    else msg_text
                                )
                                # Ajouter la grammaire propre au générateur ciblé.
                                _refine_sys = IMAGE_REFINE_SYSTEM
                                if _im.refine_hints:
                                    _refine_sys += (
                                        "\n\nTARGET-MODEL RULES (override the above "
                                        "when they conflict):\n" + _im.refine_hints
                                    )
                                out = ""
                                for kind, chunk in S.client.stream_chat(
                                    [{"role": "user", "content": _refine_in}],
                                    _refine_sys,
                                    max_tokens=512,
                                    model=_im.refiner,
                                    thinking=False,
                                ):
                                    if kind == "content":
                                        out += chunk
                                out = " ".join(out.split()).strip().strip('"')
                                if out:
                                    prompt, refined = out, True
                        except Exception:  # noqa: BLE001 - affinage best-effort
                            traceback.print_exc()
                        if not refined:
                            yield _sse(
                                "notice",
                                text="affinage indisponible — prompt envoyé tel quel.",
                            )
                    # Convertir le tag de format du refiner en résolution du workflow.
                    gen_w, gen_h = _im.width, _im.height
                    # Retirer même un tag inconnu pour qu'il ne fuite pas dans le prompt.
                    _fmt = re.search(
                        r"\[\s*format\s*:\s*([a-zà-ÿ -]+?)\s*\]\s*$",
                        prompt,
                        re.IGNORECASE,
                    )
                    if _fmt:
                        prompt = prompt[: _fmt.start()].strip()
                        _f = _fmt.group(1).lower()
                        if any(
                            k in _f
                            for k in ("portrait", "full-body", "full body", "vertical")
                        ):
                            gen_w, gen_h = 832, 1216
                        elif any(
                            k in _f
                            for k in ("landscape", "paysage", "wide", "horizontal")
                        ):
                            gen_w, gen_h = 1216, 832
                        elif any(k in _f for k in ("square", "carr")):
                            gen_w, gen_h = 1024, 1024
                    # Titrer avant de décharger le refiner évite un rechargement.
                    if _sess.title == "Nouvelle session":
                        _title = _infer_title(S.client, _im.refiner or None, msg_text)
                        if _title:
                            _sess.title = _title
                            S.session_store.save(_sess)
                            yield _sse("session_title", id=_sess.id, title=_title)
                    S.client.unload_local()  # VRAM libre pour la diffusion
                    eng = _engine_for(S, _im)
                    eng.ensure_up()
                    yield _sse("status", label="génération de l'image…")
                    data, ext = eng.generate(
                        Path(_im.workflow_path).read_text(encoding="utf-8"),
                        prompt,
                        timeout=float(_im.timeout),
                        image_path=src_image,
                        width=gen_w,
                        height=gen_h,
                    )
                    # Une seule copie liée au cycle de vie de la session évite les orphelins.
                    name = f"loom_{int(time.time() * 1000)}{ext}"
                    media_dir = S.session_store.root / _sess.id / "generated"
                    media_dir.mkdir(parents=True, exist_ok=True)
                    (media_dir / name).write_bytes(data)
                    loc = str(media_dir / name)
                    if ext in (".png", ".jpg", ".jpeg", ".webp"):
                        md = (
                            f"![{(prompt or 'image')[:80]}](/genimg/{_sess.id}/{name})\n\n"
                            f"Image écrite : `{loc}`"
                        )
                    else:
                        # Les vidéos exigent un lien, contrairement aux images Markdown.
                        md = (
                            f"[vidéo générée — cliquer pour lire](/genimg/{_sess.id}/{name})\n\n"
                            f"Vidéo écrite : `{loc}`"
                        )
                    if refined:
                        # Rendre visible la transformation effectuée par le refiner.
                        md += f"\n\nPrompt affiné ({_im.refiner}) : `{prompt}`"
                    yield _finish(md)
                except ComfyError as exc:
                    yield _finish(f"[génération d'image interrompue : {exc}]")
                except Exception as exc:  # noqa: BLE001 - jamais de stacktrace dans le chat
                    traceback.print_exc()
                    yield _finish(
                        f"[génération d'image interrompue : erreur interne — {str(exc)[:160]}]"
                    )
                finally:
                    if _img_held:
                        S.local_busy["reason"] = ""
                        S.local_gen_lock.release()
                    chat_lock.release()
                    S.stay_awake.release()
                    yield _sse("status", label="")

            return Response(generate_image(), mimetype="text/event-stream")

        def generate():
            # Empêcher la veille système pendant le tour, sans bloquer l'extinction de l'écran.
            S.stay_awake.acquire()
            S.confirm_local.ev = cancel_event
            # Le verrou global concerne seulement le serveur local à slot unique.
            _local_held = False
            # Une note transférée remplace la provenance pour conserver toute la chaîne.
            response_provenance = list(provenance)

            # Appliquer le profil aux réponses comme aux écritures d'outils.

            _profile = load_profile(conv.model) if conv.model else None

            if adopted_ws:  # informe l'UI que le dossier de travail a été adopté
                yield _sse("workspace", path=adopted_ws)

            # Démarrer dans le générateur rend chaque phase visible dans le flux.
            if conv.model and conv.model not in S.remote_model_ids:
                _reachable, _running_txt = S.client.running_local(timeout=2.0)
                if not _reachable:
                    yield _sse(
                        "notice",
                        text="serveur modèle éteint — démarrage de la stack en cours…",
                    )
                    _t_srv = time.monotonic()
                    S.server_manager.start()
                    # Tant que la stack vit, attendre avec des notices plutôt qu'un délai fixe.
                    _deadline = time.monotonic() + 600.0  # garde-fou absolu
                    _last_notice = time.monotonic()
                    while time.monotonic() < _deadline and not cancel_event.is_set():
                        time.sleep(0.7)
                        _reachable, _running_txt = S.client.running_local(timeout=2.0)
                        if _reachable:
                            # Tracer chaque phase rend les temps d'attente explicables.
                            log_event(
                                "turn.step",
                                etape="demarrage_serveur",
                                s=round(time.monotonic() - _t_srv, 1),
                            )
                            yield _sse("notice", text="serveur modèle démarré.")
                            break
                        if not S.server_manager.owns_running():
                            break  # process mort : inutile d'attendre le garde-fou
                        if time.monotonic() - _last_notice >= 15.0:
                            _last_notice = time.monotonic()
                            yield _sse(
                                "notice",
                                text=(
                                    "chargement du modèle en cours… "
                                    f"({int(time.monotonic() - _t_srv)} s — un gros "
                                    "modèle peut prendre plusieurs minutes)"
                                ),
                            )
                    if not _reachable and not cancel_event.is_set():
                        if S.server_manager.owns_running():
                            _txt = (
                                "le serveur modèle charge encore après "
                                f"{int(time.monotonic() - _t_srv)} s — la génération "
                                "est tentée quand même ; si elle échoue, réessaie "
                                "dans un moment (détails : var/logs/serve.log)."
                            )
                        else:
                            _txt = (
                                "le serveur modèle s'est ARRÊTÉ pendant le démarrage "
                                "— la génération va échouer (cause : var/logs/serve.log)."
                            )
                        yield _sse("notice", text=_txt)
                if _reachable and conv.model not in _running_txt:
                    yield _sse(
                        "notice",
                        text="chargement du modèle en mémoire — la première réponse met "
                        "plus de temps à démarrer…",
                    )

            answer = ""

            actions: list[str] = []  # trace compacte des outils (anti-amnésie)

            saved = False
            # Sauvegarder aux étapes marquantes limite les pertes lors d'une interruption.
            _turn = {"idx": None, "last": 0.0}

            # La timeline persiste uniquement les événements nécessaires pour rejouer la vue.
            _TL = {
                "reasoning",
                "text",
                "tool_call",
                "tool_result",
                "phase",
                "notice",
                "parallel",
                "user",  # note en vol injectée : à sa vraie position au rechargement
                "harness",  # intervention du garde-fou Loom (3e voix) : rejouable
                "monitor_event",  # stdout asynchrone à sa vraie position
            }

            def _tl(event, **data):
                if event in _TL:
                    S.session_store.append_event(sess.id, event, data)
                return _sse(event, **data)

            def _persist(final=False):
                # Garder une trace compacte des outils sans injecter leurs sorties brutes au contexte.
                nonlocal saved

                body = answer

                if actions:
                    trace = "[Actions de ce tour : " + " · ".join(actions[:20]) + "]"

                    body = f"{body}\n\n{trace}" if body else trace

                if not body:  # rien à dire ET rien fait -> pas de bulle vide
                    return

                # session.json porte le contexte; la timeline porte l'affichage temps réel.
                if _turn["idx"] is None:
                    conv.add("assistant", body)
                    _turn["idx"] = len(conv.messages) - 1
                else:
                    conv.messages[_turn["idx"]]["content"] = body

                save()
                saved = True

            # Lier le registre au workspace courant empêche les outils d'écrire à côté.

            # Compacter dans le flux, après démarrage, et seulement pour les modèles locaux.
            _is_local_model = bool(conv.model) and conv.model not in S.remote_model_ids
            # Un seuil relatif évite que le long prompt système déclenche une compaction constante.
            _, _pre_threshold = _model_limits(S, conv.model)
            if (
                _is_local_model
                and context.needs_summary(
                    conv.system_prompt, conv.messages, _pre_threshold
                )
                and len(conv.messages) > S.settings["keep_recent"]
            ):
                yield _sse("status", label="compaction du contexte…")
                if context.summarize(
                    conv, S.client, _pre_threshold, S.settings["keep_recent"]
                ):
                    save()
                    # Estimer aussitôt la jauge; l'usage réel la corrigera au prochain appel.
                    conv.context_tokens = (
                        len(conv.system_prompt)
                        + sum(_msg_chars(m.get("content")) for m in conv.messages)
                    ) // 3
                    yield _tl(
                        "tool_result",
                        name="(compaction)",
                        ok=True,
                        preview="Contexte résumé pour libérer de la place. Je reprends.",
                    )
                    yield _sse("totals", **_totals(S, conv))
                yield _sse("status", label="")  # efface le label d'activité

            ws = sess.workspace

            registry = S.tool_factory(conv.active_tools, ws, conv)

            use_tools = registry is not None and len(registry)

            eff_max_tokens, eff_compact = _model_limits(S, conv.model)

            # Les modèles distants forts gardent seulement les garde-fous indispensables.

            # `/cancel` ferme le flux distant courant pour libérer rapidement le verrou de session.
            stream_holder: dict = {}
            S.active_streams[sess.id] = stream_holder

            def _make_source():
                # Reconstruire la source après STOP pour inclure une éventuelle note en file.
                if use_tools:
                    return S.client.stream_chat_tools(
                        conv.to_messages(),
                        system_prompt,
                        eff_max_tokens,
                        model=conv.model or None,
                        registry=registry,
                        thinking=conv.thinking,
                        permission=S.perm["fn"],
                        confirm=partial(_confirm, S),
                        compact_after_tokens=eff_compact,
                        strong=strong,
                        notes_provider=lambda: S.notes.drain(sess.id),
                        monitor_events_provider=(
                            (lambda: S.monitor_hub.drain(sess.id))
                            if S.monitor_hub is not None
                            else None
                        ),
                        refocus_note=not S.refocus_handled.get(sess.id, False),
                        stream_holder=stream_holder,
                    )
                return S.client.stream_chat(
                    conv.to_messages(),
                    system_prompt,
                    eff_max_tokens,
                    model=conv.model or None,
                    thinking=conv.thinking,
                    stream_holder=stream_holder,
                )

            interrupted = False

            saw_compaction = (
                False  # une troncature (force-fit/compaction) a eu lieu ce tour
            )
            stop_reason = ""  # raison du done de la boucle (natural, repeat_stop, …)

            recv_confirmed = 0  # reçus confirmés par l'usage (tool-calls inclus)

            cur_turn = 0  # reçus live du tour en cours (reset à chaque usage)

            sent_tokens = 0  # envoyés (prompt) cumulés via l'usage

            last_rate = 0.0  # dernier débit mesuré

            burst_start = None  # début de rafale (débit hors pauses outils)

            burst_tokens = 0

            last_tok = None

            # Titrer le distant en arrière-plan; attendre la fin pour ne pas concurrencer le local.
            _titled = {"value": None, "emitted": False}
            _title_ready = threading.Event()
            _immediate_title = (
                sess.title == "Nouvelle session" and conv.model in S.remote_model_ids
            )
            if _immediate_title:

                def _do_title(_msg=title_message, _model=conv.model):
                    _t = ""
                    try:
                        _t = _infer_title(S.client, _model or None, _msg)
                    except Exception:  # noqa: BLE001 - titre best-effort, jamais bloquant
                        _t = ""
                    _titled["value"] = _t or ""
                    if _t:
                        sess.title = _t
                    _title_ready.set()

                threading.Thread(
                    target=_do_title, daemon=True, name="loom-title"
                ).start()

            try:
                # Sérialiser le slot local unique; laisser les API distantes parallèles.
                if conv.model and conv.model not in S.remote_model_ids:
                    if not S.local_gen_lock.acquire(blocking=False):
                        yield _sse("notice", text=_local_busy_notice(S))
                        S.local_gen_lock.acquire()
                    _local_held = True
                    S.local_busy["reason"] = "génération"
                    # Libérer la VRAM image; ne garder son cache RAM que si le LLM tient avec.
                    if _free_image_engines(S, _local_size_mb(S, conv.model)) is False:
                        yield _sse(
                            "notice",
                            text=(
                                "moteur image déchargé de la RAM (insuffisante pour "
                                "garder LLM + cache image ensemble) : la prochaine "
                                "image repaiera le chargement disque."
                            ),
                        )

                # Après STOP, une note en file relance un nouveau tour.
                while True:
                    source = _make_source()
                    interrupted = False
                    saw_compaction = False
                    stop_reason = ""
                    for kind, payload in source:
                        # Publier le titre distant dès qu'il est disponible.
                        if (
                            _immediate_title
                            and _title_ready.is_set()
                            and not _titled["emitted"]
                        ):
                            _titled["emitted"] = True
                            if _titled["value"]:
                                S.session_store.save(sess)
                                yield _sse(
                                    "session_title", id=sess.id, title=_titled["value"]
                                )

                        if cancel_event.is_set():
                            # Conserver le contenu déjà reçu lors d'une nouvelle soumission.

                            interrupted = True

                            break

                        if kind == "note":
                            # Persister la note injectée à la position réellement vue par le modèle.
                            if isinstance(payload, dict):
                                note_text = str(payload.get("text", ""))
                                note_display = str(payload.get("display", note_text))
                                note_provenance = payload.get("provenance") or []
                                response_provenance = list(note_provenance)
                                note_data = {
                                    "content": note_display,
                                    "provenance": note_provenance,
                                    "handoff_id": str(payload.get("handoff_id", "")),
                                }
                            else:
                                note_text = payload
                                note_display = payload
                                note_data = {"content": payload}
                            conv.add("user", note_text)
                            save()
                            yield _tl("user", **note_data)
                            yield _sse(
                                "note",
                                text=note_display,
                                provenance=note_data.get("provenance"),
                                handoff_id=note_data.get("handoff_id", ""),
                            )
                            continue

                        if kind == "monitor_event":
                            # Garde exactement la structure vue par le modèle : un
                            # assistant.tool_call synthétique suivi du role=tool.
                            conv.messages.extend(
                                (payload["assistant_message"], payload["tool_message"])
                            )
                            save()
                            yield _tl(
                                "monitor_event",
                                monitor_id=payload["monitor_id"],
                                description=payload["description"],
                                text=payload["text"],
                                final=bool(payload.get("final", False)),
                            )
                            continue

                        if kind == "harness":
                            # Distinguer la voix du garde-fou de celles du modèle et de l'utilisateur.
                            yield _tl("harness", **payload)
                            continue

                        if kind == "status":
                            yield _sse("status", **payload)

                        elif kind == "context_estimate":
                            # Estimer la jauge après compaction; le prochain usage la corrigera.
                            conv.context_tokens = int(payload.get("tokens", 0) or 0)
                            yield _sse("totals", **_totals(S, conv))

                        elif kind == "reasoning":
                            yield _tl("reasoning", text=payload)

                        elif kind == "content":
                            if _profile is not None:
                                payload = _profile.apply_to_text(payload)
                            answer += payload

                            # Journaliser chaque fragment pour rendre la vue rejouable.
                            text_data = {"text": payload}
                            if response_provenance:
                                text_data["provenance"] = response_provenance
                            yield _tl("text", **text_data)

                        elif kind == "parallel":
                            yield _tl("parallel", **payload)

                        elif kind == "tool_call":
                            yield _tl("tool_call", **payload)

                        elif kind == "tool_request":
                            yield _sse("tool_request", **payload)

                        elif kind == "tool_begin":
                            yield _sse("tool_begin", **payload)

                        elif kind == "tool_args":
                            yield _sse("tool_args", **payload)

                        elif kind == "tool_stream":
                            yield _sse("tool_stream", **payload)

                        elif kind == "tool_result":
                            if str(payload.get("name", "")).startswith("(compaction"):
                                saw_compaction = True

                            line = _action_trace_line(payload)

                            if line and line not in actions:
                                actions.append(line)

                            yield _tl("tool_result", **payload)

                            _persist()  # checkpoint contexte (event-driven) : l'outil vient de finir

                        elif kind == "usage":
                            # L'usage serveur inclut les appels d'outils et réconcilie le compteur live.

                            _p = payload.get("prompt_tokens", 0) or 0
                            _c = payload.get("completion_tokens", 0) or 0
                            _cached = payload.get("cached_tokens", 0) or 0

                            sent_tokens += _p

                            recv_confirmed += _c

                            cur_turn = 0

                            # Additionner chaque appel reflète le coût réellement facturé du contexte.
                            _pin, _pout, _pcached = _price_of(S, conv.model)
                            conv.add_usage(_p, _c, _cached, _pin, _pout, _pcached)

                            yield _sse("usage", **payload)

                            yield _sse(
                                "metrics",
                                sent=sent_tokens,
                                recv=recv_confirmed,
                                tok_s=last_rate,
                            )

                            yield _sse("totals", **_totals(S, conv))

                        elif kind == "sub_usage":
                            # Compter le coût du sous-agent sans modifier la jauge du fil principal.
                            _sp = payload.get("prompt_tokens", 0) or 0
                            _sc = payload.get("completion_tokens", 0) or 0
                            _scached = payload.get("cached_tokens", 0) or 0
                            _pin, _pout, _pcached = _price_of(S, conv.model)
                            conv.add_usage(
                                _sp,
                                _sc,
                                _scached,
                                _pin,
                                _pout,
                                _pcached,
                                set_context=False,
                            )
                            yield _sse("totals", **_totals(S, conv))

                        elif kind == "phase":
                            yield _tl("phase", **payload)

                        elif kind == "done":
                            # La raison d'arrêt pilote le réarmement du recentrage.
                            stop_reason = str(payload.get("reason", "") or "")

                        # Mesurer le débit par rafale exclut les pauses d'exécution des outils.

                        if kind in ("reasoning", "content", "tool_args"):
                            now = time.monotonic()

                            if last_tok is None or now - last_tok > 1.0:
                                burst_start = now

                                burst_tokens = 0

                            burst_tokens += 1

                            cur_turn += 1

                            last_tok = now

                            span = now - burst_start

                            tok_s = round(burst_tokens / span, 1) if span > 0 else 0.0

                            last_rate = tok_s

                            yield _sse(
                                "metrics",
                                sent=sent_tokens,
                                recv=recv_confirmed + cur_turn,
                                tok_s=tok_s,
                            )

                    if not interrupted:
                        break

                    # Persister le travail partiel avant le marqueur d'interruption.
                    _persist(final=True)
                    conv.add(
                        "user",
                        "[Interrupted by the user here — the answer above is "
                        "incomplete. Wait for the next instruction.]",
                    )
                    save()

                    # Une note en file transforme le STOP en reprise; sinon l'arrêt est définitif.
                    _pending = S.notes.drain(sess.id)
                    if not _pending:
                        return
                    for _n in _pending:
                        if isinstance(_n, dict):
                            _note_text = str(_n.get("text", ""))
                            _note_display = str(_n.get("display", _note_text))
                            _note_provenance = _n.get("provenance") or []
                            response_provenance = list(_note_provenance)
                            _note_data = {
                                "content": _note_display,
                                "provenance": _note_provenance,
                                "handoff_id": str(_n.get("handoff_id", "")),
                            }
                        else:
                            _note_text = _n
                            _note_display = _n
                            _note_data = {"content": _n}
                        conv.add("user", _note_text)
                        S.session_store.append_event(sess.id, "user", _note_data)
                        yield _sse(
                            "note",
                            text=_note_display,
                            provenance=_note_data.get("provenance"),
                            handoff_id=_note_data.get("handoff_id", ""),
                        )
                    save()
                    cancel_event.clear()

                # Réarmer le recentrage seulement si la compaction finit encore en boucle.
                if saw_compaction:
                    if stop_reason == "natural":
                        S.refocus_handled[sess.id] = True
                    elif stop_reason in ("repeat_stop", "loop_degenerate"):
                        S.refocus_handled[sess.id] = False
                else:
                    S.refocus_handled[sess.id] = False

                if not answer.strip():
                    answer = "(le modèle a seulement réfléchi — augmente max_tokens)"

                    text_data = {"text": answer}
                    if response_provenance:
                        text_data["provenance"] = response_provenance
                    yield _tl("text", **text_data)

                _persist(final=True)  # fin de tour : écriture finale garantie

                # Sauvegarder le slot avant le titre et la maintenance qui peuvent l'écraser.
                _kv_saved = S.client.save_slot(
                    conv.model, "turnend.kv", session_id=sess.id
                )

                # Déporter la maintenance évite de retarder `done` et protège le cache du fil.
                _do_reflect = (
                    S.settings["reflect_enabled"]
                    and S.reflect_stores is not None
                    and saved
                    and len(actions) >= S.settings["reflect_min_actions"]
                )

                threading.Thread(
                    target=_post_turn_maintenance,
                    args=(
                        S,
                        sess,
                        conv.to_messages(),
                        list(actions),
                        answer,
                        conv.model,
                        _do_reflect,
                        _kv_saved,
                    ),
                    daemon=True,
                    name="loom-post-turn",
                ).start()

                # Titrer la session du flux, jamais la session actuellement focalisée.

                if _immediate_title:
                    # Attendre brièvement le titre distant s'il n'a pas encore été publié.
                    if not _titled["emitted"]:
                        _title_ready.wait(timeout=8)
                        _titled["emitted"] = True
                        if _titled["value"]:
                            sess.title = _titled["value"]
                            S.session_store.save(sess)
                            yield _sse(
                                "session_title", id=sess.id, title=_titled["value"]
                            )
                elif saved and sess.title == "Nouvelle session":
                    # Titrer le local seulement lorsque son slot est libre.
                    _title = _infer_title(S.client, conv.model or None, title_message)
                    if _title:
                        sess.title = _title
                        S.session_store.save(sess)
                        yield _sse("session_title", id=sess.id, title=_title)

                yield _sse("done")

            except GeneratorExit:
                # Marquer la troncature avant la persistance destinée au tour suivant.
                try:
                    conv.add(
                        "user",
                        "[Interrupted by a new user submission — the answer above "
                        "is incomplete.]",
                    )
                except Exception:  # noqa: BLE001 - marqueur best-effort
                    pass
                # Persister la réponse partielle puis relayer la fermeture du générateur.

                _persist(final=True)  # client parti : écriture finale garantie

                raise

            except Exception:  # noqa: BLE001 - on remonte l'erreur au client SSE
                traceback.print_exc()
                yield _sse("error", message="erreur interne")

            finally:
                S.last_activity[0] = time.time()  # marque l'activité pour le keep-warm

                S.active_streams.pop(
                    sess.id, None
                )  # plus de stream à fermer pour /cancel

                if _local_held:
                    S.local_busy["reason"] = ""
                    S.local_gen_lock.release()

                chat_lock.release()
                S.stay_awake.release()  # plus de veille bloquée si plus aucune génération

        return Response(generate(), mimetype="text/event-stream")
