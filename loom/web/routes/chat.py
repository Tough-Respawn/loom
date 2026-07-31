from __future__ import annotations

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

# ---- Commandes /goal et /init (préambule de /chat) ------------------------------------

# Catalogue des commandes slash du chat — SOURCE DE VÉRITÉ de la palette « / » du
# composer (GET /commands). À tenir en phase avec les handlers ci-dessous : une
# commande non listée ici est indécouvrable pour l'utilisateur.
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


# ---- Route : /chat (génération SSE) -----------------------------------------------------


def _register_chat_routes(app, S):
    @app.post("/chat")
    def chat():
        message = (request.form.get("message") or "").strip()

        if not message or len(message) > 5000:
            return Response("message invalide", status=400)

        # Session CIBLE : par `session_id` (onglet) sinon la session focus. Chaque session a
        # son verrou : une nouvelle soumission n'interrompt QUE la génération de SA session,
        # les autres onglets continuent en parallèle.
        req_sid = (request.form.get("session_id") or "").strip()
        # Un session_id EXPLICITE mais inconnu -> 404 (comme /fork, /compact) : sinon
        # l'onglet enverrait silencieusement son message dans la session focus.
        sess = _get_session(S, req_sid) if req_sid else _session(S)
        if sess is None:
            return Response("session introuvable", status=404)
        S.cur["session"] = sess  # focus (défaut de l'index)
        sid = sess.id
        chat_lock = _lock_for(S, sid)
        cancel_event = _cancel_for(S, sid)

        if not chat_lock.acquire(blocking=False):
            # Le verrou de CETTE session est tenu. Deux cas se distinguent par
            # cancel_event (posé par /cancel, effacé seulement à l'acquisition d'un
            # nouveau tour) :
            #  - STOP en cours (cancel_event posé) : la génération interrompue relâche
            #    son verrou pendant son teardown. Un message RENVOYÉ après un stop ne
            #    doit PAS partir en file (elle n'est drainée que par une génération
            #    active, qui n'existe plus après l'arrêt -> message jamais généré). On
            #    ATTEND la libération (bornée par interrupt_wait) puis on génère.
            #  - vraie génération concurrente (cancel_event absent) : on ne l'interrompt
            #    PAS, le message part en FILE D'ATTENTE (même mécanique que les notes en
            #    vol), injecté au prochain point d'arrêt. L'annulation, c'est le stop.
            if not (
                cancel_event.is_set() and chat_lock.acquire(timeout=S.interrupt_wait)
            ):
                if S.notes.push(sid, message) < 0:
                    return Response(
                        "file d'attente pleine — attendre le prochain point d'arrêt ou Stop",
                        status=429,
                    )
                # La file est TEXTE-ONLY : d'éventuelles images jointes ne peuvent pas suivre.
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

        # Anti-résurrection : entre la capture de `sess` (plus haut) et l'acquisition du
        # verrou, un /session/delete concurrent a pu supprimer cette session. Maintenant
        # qu'on tient le verrou, plus aucune suppression n'est possible -> si le dossier a
        # disparu, on abandonne AVANT tout save() (qui la recréerait, session zombie).
        if not S.session_store.session_dir(sid).exists():
            chat_lock.release()
            return Response("session supprimée", status=404)

        conv = sess.conversation

        def save():
            return S.session_store.save(sess)

        # Commande /add-model + wizard actif : le wizard déterministe capte TOUT
        # message de la session tant qu'il est actif (y compris avant /goal — un
        # « /goal » tapé en plein wizard est une réponse au wizard, pas une commande).
        message, _wiz_resp = _handle_add_model_command(
            S, message, conv, sess, save, chat_lock
        )
        if _wiz_resp is not None:
            return _wiz_resp

        # Commande /goal : pilote l'OBJECTIF de complétion de la session. La logique
        # (pose/statut/efface) est factorisée dans _handle_goal_command, qui renvoie
        # (message, response) : response non None = ack immédiat à retourner directement.
        message, _goal_resp = _handle_goal_command(S, message, conv, save, chat_lock)
        if _goal_resp is not None:
            return _goal_resp

        # Commande /init : génère une fiche projet `loom.md` À LA RACINE DU DOSSIER
        # de TRAVAIL de la session. Factorisé dans _handle_init_command (adopte un
        # dossier cible si fourni, réécrit le message en consigne de génération).
        message = _handle_init_command(S, message, sess)

        # Plus de garde bloquant : un modèle texte-only ne reçoit PAS l'image inline (qui

        # ferait planter un llama-server sans mmproj) — on la stocke sur disque et il l'inspecte

        # via read_image, routé vers un modèle vision (cf. _build_user_content plus bas).

        # Logs PAR SESSION (au même titre que session.json) : (1) trace des échanges modèle

        # routée vers sessions/<id>/debug.log ; (2) copie du log serveur modèle global

        # (var/logs/serve.log) dans la session — doublon assumé, pour tout avoir sous la main.

        _sdir = S.session_store.session_dir(sess.id)

        set_debug_log_path(_sdir / "debug.log")

        _serve_log = S.session_store.root.parent / "logs" / "serve.log"

        if _serve_log.exists():
            try:
                _sdir.mkdir(parents=True, exist_ok=True)

                shutil.copyfile(_serve_log, _sdir / "serve.log")

            except OSError:
                pass

        # Auto-adoption du dossier de travail : si le message désigne un dossier EXISTANT,

        # la session l'adopte avant le tour -> run_shell tourne dedans et les chemins

        # relatifs s'y résolvent, sans que l'utilisateur ait à pointer le dossier dans l'UI.

        adopted_ws = None

        detected = _detect_workspace(message, S.workspace_dir)

        if detected:
            # Un chemin INTERNE au projet courant n'est pas un changement de contexte :
            # adopter casserait le cache KV (re-prefill intégral) pour rien — cf.
            # _should_adopt (vécu 2026-07-19 : var/sessions/<id> cité comme simple info).
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

            # Journal d'affichage temps réel : on y consigne le message user (le journal est la
            # source de RÉ-AFFICHAGE au rechargement -> il doit être complet, user inclus).
            S.session_store.append_event(sess.id, "user", {"content": message})

            # Résumé auto pré-tour : DÉPLACÉ dans generate() (plus bas) pour être VISIBLE
            # dans le stream (label d'activité « compaction… ») au lieu d'un blocage muet
            # avant le 1er octet. Le prompt système ne dépend pas de l'historique -> on le
            # construit ici sans attendre le résumé.
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

        # --- Modèle IMAGE sélectionné : court-circuit de la boucle tool-use. Un message
        # user = un prompt d'image = une image dans la conversation. Même GPU que le LLM
        # local -> même sérialisation (_local_gen_lock) ; VRAM libérée (unload_local)
        # avant la diffusion ; erreurs TOUJOURS lisibles (patron « génération interrompue »).
        if conv.model in S.image_model_ids:
            _im = S.image_by_id[conv.model]

            def generate_image():
                S.stay_awake.acquire()
                _img_held = False
                _sess = sess

                def _finish(md_text: str):
                    conv.add("assistant", md_text)
                    save()
                    S.session_store.append_event(_sess.id, "text", {"text": md_text})
                    return _sse("text", text=md_text)

                try:
                    yield _sse("status", label="préparation du moteur image…")
                    S.local_gen_lock.acquire()
                    _img_held = True
                    S.local_busy["reason"] = "image"
                    # Photo d'ENTRÉE (modèles d'édition, ex. Kontext) : un chemin de
                    # fichier image dans le message est détecté, vérifié sur disque,
                    # retiré du texte (le prompt ne doit porter que l'instruction) et
                    # transmis au moteur ({IMAGE} du workflow). Chemins avec espaces :
                    # entre guillemets.
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
                    # Affinage du prompt (best-effort, JAMAIS bloquant) : le refiner
                    # déclaré par le modèle image (model.toml, id d'un modèle Loom)
                    # réécrit la demande — quelle que soit la langue — en prompt de
                    # diffusion anglais. Séquence VRAM sûre : le refiner est servi par
                    # llama-swap D'ABORD, puis déchargé (unload_local ci-dessous) —
                    # LLM et diffusion ne co-résident jamais.
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
                                # Édition d'une photo : le refiner doit produire une
                                # INSTRUCTION (quoi changer / quoi garder), pas une
                                # description de scène — on le lui dit dans le message.
                                _refine_in = (
                                    "[An input photo is attached; write an EDIT "
                                    "instruction: what to change, what must stay "
                                    f"identical.] {msg_text}"
                                    if src_image
                                    else msg_text
                                )
                                # Prompt système = règles générales + grammaire propre
                                # au générateur (refine_hints du model.toml) : chaque
                                # modèle a son style de prompt optimal.
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
                    # FORMAT dynamique : le refiner termine par un tag
                    # [format: portrait|landscape|square] dérivé de la demande —
                    # extrait ici, retiré du prompt, converti en résolution. Absent
                    # -> dimensions du model.toml (les workflows sans {WIDTH}/{HEIGHT}
                    # ignorent simplement ces valeurs).
                    gen_w, gen_h = _im.width, _im.height
                    # Tag TOUJOURS retiré quelle que soit sa valeur (un petit modèle
                    # invente parfois la sienne, ex. "full-body" : elle ne doit JAMAIS
                    # fuir dans le prompt de diffusion) ; synonymes mappés, inconnu ->
                    # dimensions par défaut.
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
                    # Titre de session : une session image/vidéo mérite un nom comme
                    # les autres. Inféré par le REFINER (encore résident — coût nul en
                    # rechargement) ; sans refiner, _infer_title retombe sur le début
                    # du message. Fait AVANT unload_local.
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
                    # UNIQUE copie : dans le dossier de LA session (comme sa timeline).
                    # Le média suit le cycle de vie de la session — la supprimer emporte
                    # ses médias, aucun orphelin (décision user 2026-07-09 : fini les
                    # duplications var/generated + workspace + output ComfyUI).
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
                        # Vidéo (webm/mp4) : le markdown image ne la lit pas — lien
                        # cliquable, le navigateur la joue dans un onglet.
                        md = (
                            f"[vidéo générée — cliquer pour lire](/genimg/{_sess.id}/{name})\n\n"
                            f"Vidéo écrite : `{loc}`"
                        )
                    if refined:
                        # Le prompt réellement envoyé au diffuseur, visible dans le fil :
                        # l'utilisateur voit ce que l'affinage a fait de sa demande.
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
            # Empêche la mise en veille du système tant que CE tour génère (release au
            # finally) : sans ça, une veille par inactivité gèle loom.web + llama.cpp et la
            # génération meurt (« connexion perdue »). L'écran peut s'éteindre, le travail
            # continue en arrière-plan.
            S.stay_awake.acquire()
            # Annulation de CETTE session, lue par _confirm (même thread de génération).
            S.confirm_local.ev = cancel_event
            # Verrou modèle LOCAL : pris dans le try ci-dessous (avant le 1er appel modèle),
            # libéré au finally. Distant -> jamais pris (vrai parallèle entre onglets).
            _local_held = False

            # Profil du modèle : correctifs déterministes (cadratins, guillemets

            # typographiques) appliqués au texte streamé du chat. Le profil existe

            # déjà pour les outils d'écriture (via tool_factory) ; on le recharge ici

            # pour l'appliquer AUSSI aux réponses du modèle, pas seulement aux fichiers.

            _profile = load_profile(conv.model) if conv.model else None

            if adopted_ws:  # informe l'UI que le dossier de travail a été adopté
                yield _sse("workspace", path=adopted_ws)

            # Démarrage AUTO du serveur modèle, RACONTÉ dans le fil : notices streamées
            # pendant le démarrage de la stack puis le chargement du modèle — l'utilisateur
            # voit que ça travaille au lieu de paniquer devant un silence. Fait DANS le
            # générateur (pas avant la Response) pour que ces étapes s'affichent en direct.
            if conv.model and conv.model not in S.remote_model_ids:
                _reachable, _running_txt = S.client.running_local(timeout=2.0)
                if not _reachable:
                    yield _sse(
                        "notice",
                        text="serveur modèle éteint — démarrage de la stack en cours…",
                    )
                    _t_srv = time.monotonic()
                    S.server_manager.start()
                    # Attente pilotée par l'ÉTAT DU PROCESS, pas par un mur de temps :
                    # un 35 Go à froid dépasse largement 90 s, et l'ancien message
                    # prédisait « la génération va échouer » à tort (vécu 2026-07-21).
                    # Stack vivante -> on attend en le disant (notice périodique) ;
                    # stack MORTE -> vrai échec, on arrête d'attendre tout de suite.
                    _deadline = time.monotonic() + 600.0  # garde-fou absolu
                    _last_notice = time.monotonic()
                    while time.monotonic() < _deadline and not cancel_event.is_set():
                        time.sleep(0.7)
                        _reachable, _running_txt = S.client.running_local(timeout=2.0)
                        if _reachable:
                            # Étape TRACÉE (console debug) : chaque phase du tour porte
                            # sa durée — plus jamais un « 1min52 » opaque (2026-07-19).
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
            # Persistance AU FIL DE L'EAU : au lieu de tout sauver une seule fois EN FIN de tour
            # (un long audit interrompu/rechargé/relancé perdait TOUT), on met à jour EN PLACE
            # l'unique message assistant du tour (réponse en cours + trace compacte des actions)
            # et on sauve à CHAQUE étape marquante (outil terminé, flux de texte) + à la fin.
            _turn = {"idx": None, "last": 0.0}

            # Journal d'affichage TEMPS RÉEL : chaque événement visible est écrit à l'instant
            # dans timeline.jsonl (append, zéro batch) -> rejouable au rechargement. On y met
            # les événements qui reconstruisent la vue (raisonnement, texte, cartes d'outils) ;
            # pas les compteurs (metrics/totals) ni les décorations live (tool_stream/args).
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
            }

            def _tl(event, **data):
                if event in _TL:
                    S.session_store.append_event(sess.id, event, data)
                return _sse(event, **data)

            def _persist(final=False):
                # On NE persiste pas les messages `tool` bruts (gonflerait le contexte + casserait
                # le résumeur) : seulement le texte + la trace des actions. Un même tour = UN seul
                # message assistant, mis à jour en place (pas de doublons).
                nonlocal saved

                body = answer

                if actions:
                    trace = "[Actions de ce tour : " + " · ".join(actions[:20]) + "]"

                    body = f"{body}\n\n{trace}" if body else trace

                if not body:  # rien à dire ET rien fait -> pas de bulle vide
                    return

                # Piloté par ÉVÉNEMENT (chaque outil terminé + fin), plus par un timer : le
                # temps réel de l'affichage vient du journal `timeline.jsonl`, pas d'ici. Ce
                # session.json ne porte que le contexte lean du modèle, inutile à chaque token.
                if _turn["idx"] is None:
                    conv.add("assistant", body)
                    _turn["idx"] = len(conv.messages) - 1
                else:
                    conv.messages[_turn["idx"]]["content"] = body

                save()
                saved = True

            # Registre construit selon les outils activés pour CETTE conversation

            # (toggles UI) ET le workspace de la session active : sans ça les outils

            # (write/edit/run_shell + sous-agent) retombent sur cfg.chat.workspace_dir

            # et écrivent à côté du dossier ciblé.

            # Résumé PRÉ-TOUR (proactif), DANS le stream pour être VISIBLE : si l'historique
            # dépasse le budget, on émet le label d'activité « compaction… », on résume, on
            # trace une carte, puis on efface le label. Gate `needs_summary` d'abord (sans
            # appel modèle) pour ne montrer le label QUE si un résumé va vraiment tourner.
            # Placé APRÈS le démarrage du serveur modèle (le résumé appelle le modèle).
            # LOCAL UNIQUEMENT : un modèle DISTANT a une grande fenêtre et gère lui-même son
            # contexte + son prefix-cache ; réécrire son historique casserait ce cache et
            # coûterait des tokens pour rien. On ne compacte donc que le local.
            _is_local_model = bool(conv.model) and conv.model not in S.remote_model_ids
            # SEUIL relatif à la FENÊTRE, pas le `context_budget` (3000) absolu : ce dernier
            # est comparé à system_prompt + messages, or le prompt système SEUL fait ~11k
            # tokens -> le seuil 3000 était TOUJOURS dépassé et la compaction partait à CHAQUE
            # message (même « poursuis »), en appelant le modèle (lent). On la déclenche
            # désormais seulement près de la saturation (même seuil que le microcompact).
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
                    # Jauge à jour TOUT DE SUITE (estimation ~3 car./token), sans attendre
                    # l'usage réel du 1er appel du tour.
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

            # Limites du modèle courant (distant = sa grande fenêtre ; local = global).

            eff_max_tokens, eff_compact = _model_limits(S, conv.model)

            # `strong` (tier distant=fort) est calculé plus haut, à la construction du prompt :

            # il coupe ici les gardes de comportement (act_nudge, claim_audit, coupe non-progrès).

            # On ne garde que outils + mémoire + sécurité. Un modèle local garde le harnais complet.

            # Holder du stream distant EN COURS : rempli par _stream_model_turn à la création
            # du stream, fermé par /cancel pour débloquer une itération figée (modèle distant
            # lent/bloqué) -> verrou de session libéré de façon bornée. Vidé au finally.
            stream_holder: dict = {}
            S.active_streams[sess.id] = stream_holder

            if use_tools:
                source = S.client.stream_chat_tools(
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
                    # Notes en vol : remarques poussées par /note PENDANT ce tour,
                    # injectées au prochain point d'arrêt sans interrompre.
                    notes_provider=lambda: S.notes.drain(sess.id),
                    # Note de recentrage : seulement si l'épisode de troncature
                    # précédent n'a pas déjà été géré proprement (cf. _refocus_handled).
                    refocus_note=not S.refocus_handled.get(sess.id, False),
                    stream_holder=stream_holder,
                )

            else:
                source = S.client.stream_chat(
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

            # Auto-titre DÈS L'ENVOI (le titre dérive du MESSAGE, pas de la réponse). Pour un
            # modèle DISTANT : on l'infère en tâche de fond tout de suite et on le pousse au
            # client dès qu'il est prêt (interleavé), sans attendre la fin du tour -> l'onglet
            # prend son vrai nom en ~1-2s même sur une génération longue. Pour un modèle LOCAL,
            # on NE le fait PAS ici (llama-swap = 1 slot ; un appel concurrent contendrait avec
            # la génération) : on garde le titrage en fin de tour, quand le slot est libre.
            _titled = {"value": None, "emitted": False}
            _title_ready = threading.Event()
            _immediate_title = (
                sess.title == "Nouvelle session" and conv.model in S.remote_model_ids
            )
            if _immediate_title:

                def _do_title(_msg=message, _model=conv.model):
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
                # Modèle LOCAL : llama-swap n'en sert qu'UN à la fois -> on sérialise via le
                # verrou global (limitation machine connue, signalée à l'UI). Modèle DISTANT :
                # pas de verrou -> cette session génère EN PARALLÈLE des autres onglets.
                if conv.model and conv.model not in S.remote_model_ids:
                    if not S.local_gen_lock.acquire(blocking=False):
                        yield _sse("notice", text=_local_busy_notice(S))
                        S.local_gen_lock.acquire()
                    _local_held = True
                    S.local_busy["reason"] = "génération"
                    # Un moteur image encore chargé tiendrait la VRAM que le LLM va
                    # réclamer : le vider d'abord (best-effort, rapide si rien à vider).
                    # Son cache RAM n'est gardé que si le LLM tient à côté (64 Go : oui ;
                    # machine étroite : non, et on le DIT — l'utilisateur comprend
                    # pourquoi la prochaine image rechargera depuis le disque).
                    if _free_image_engines(S, _local_size_mb(S, conv.model)) is False:
                        yield _sse(
                            "notice",
                            text=(
                                "moteur image déchargé de la RAM (insuffisante pour "
                                "garder LLM + cache image ensemble) : la prochaine "
                                "image repaiera le chargement disque."
                            ),
                        )

                for kind, payload in source:
                    # Titre distant prêt (thread de fond) -> on le pousse dès la 1re occasion.
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
                        # Une nouvelle soumission demande l'arrêt : on stoppe net

                        # et on persiste ce qui a déjà été généré.

                        interrupted = True

                        break

                    if kind == "note":
                        # Note en vol INJECTÉE par la boucle : on la PERSISTE telle
                        # quelle (même contenu que ce que le modèle a vu) et on
                        # l'affiche dans le fil à sa vraie position.
                        conv.add("user", payload)
                        save()
                        yield _tl("user", content=payload)
                        yield _sse("note", text=payload)
                        continue

                    if kind == "harness":
                        # 3e voix : intervention du garde-fou Loom (relance, audit,
                        # recentrage…). Ni toi ni le modèle -> bulle distincte,
                        # persistée pour être rejouée au rechargement.
                        yield _tl("harness", **payload)
                        continue

                    if kind == "status":
                        # Signal d'activité (ex. compaction en cours) : piloté vers le label
                        # animé au-dessus du composer, comme « le modèle tourne ».
                        yield _sse("status", **payload)

                    elif kind == "context_estimate":
                        # Compaction : la jauge de contexte est rafraîchie IMMÉDIATEMENT
                        # (estimation), sans attendre l'usage réel du prochain appel — sinon
                        # elle resterait au pic pendant tout l'appel suivant. L'usage réel du
                        # tour d'après la corrigera de toute façon.
                        conv.context_tokens = int(payload.get("tokens", 0) or 0)
                        yield _sse("totals", **_totals(S, conv))

                    elif kind == "reasoning":
                        yield _tl("reasoning", text=payload)

                    elif kind == "content":
                        if _profile is not None:
                            payload = _profile.apply_to_text(payload)
                        answer += payload

                        # Temps réel : le texte est journalisé à l'instant (rejouable). Le
                        # session.json (contexte) se met à jour aux frontières d'outils + fin.
                        yield _tl("text", text=payload)

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
                        # Fin d'un tour : llama-server donne le prompt réel et le completion

                        # EXACT (tool-calls inclus) -> on cumule envoyés/reçus à travers les

                        # tours ET les outils, et on réconcilie le tour courant.

                        _p = payload.get("prompt_tokens", 0) or 0
                        _c = payload.get("completion_tokens", 0) or 0
                        _cached = payload.get("cached_tokens", 0) or 0

                        sent_tokens += _p

                        recv_confirmed += _c

                        cur_turn = 0

                        # Cumul RÉEL de la session : chaque appel refacture tout le contexte en
                        # INPUT -> on somme input/output/cache/coût sur TOUS les appels
                        # (persisté), pas seulement le tour. C'est LA vraie somme facturée, et
                        # `cached` mesure si le prompt caching du provider mord.
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
                        # Conso d'un SOUS-AGENT (dispatch_agent) : ses tokens sont RÉELS et
                        # facturés -> on les ajoute aux totaux de session (coût, N×, in/out/
                        # cache). `set_context=False` : son prompt n'est PAS le contexte du fil
                        # principal, on ne touche donc pas la jauge de remplissage ni les
                        # métriques per-tour (sent/recv) qui décrivent le tour principal.
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
                        # Raison d'arrêt de la boucle (natural, repeat_stop,
                        # loop_degenerate…) : nourrit la boucle de feedback de la
                        # note de recentrage ci-dessous.
                        stop_reason = str(payload.get("reason", "") or "")

                    # Compteur live : chaque delta (texte OU arguments d'un tool_call) =

                    # 1 vrai token streamé par llama-server. On compte aussi tool_args

                    # pour que le compteur avance pendant la génération d'un appel (gros

                    # write_file inclus) au lieu de se figer. On affiche le cumul + un débit

                    # mesuré sur la rafale courante ; le timer se réinitialise après >1s sans

                    # token (pause d'exécution) pour que les tok/s reflètent la génération.

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

                if interrupted:
                    _persist(
                        final=True
                    )  # interrompu : on force la sauvegarde du travail fait

                    # Stop PROPRE : marqueur PERSISTÉ -> au tour suivant le modèle
                    # SAIT que sa réponse est tronquée volontairement (sans ça il
                    # reprenait une réponse incomplète comme si de rien n'était).
                    conv.add(
                        "user",
                        "[Interrupted by the user here — the answer above is "
                        "incomplete. Wait for the next instruction.]",
                    )
                    save()

                    return

                # Feedback de la note de recentrage : troncature + tour fini PROPREMENT
                # (stop naturel) = épisode GÉRÉ, on cesse de ré-injecter la note ;
                # dérapage (non-progrès/boucle) = ré-armée ; tour sans troncature =
                # remise à zéro (le prochain épisode aura sa note).
                if saw_compaction:
                    if stop_reason == "natural":
                        S.refocus_handled[sess.id] = True
                    elif stop_reason in ("repeat_stop", "loop_degenerate"):
                        S.refocus_handled[sess.id] = False
                else:
                    S.refocus_handled[sess.id] = False

                if not answer.strip():
                    answer = "(le modèle a seulement réfléchi — augmente max_tokens)"

                    yield _sse("text", text=answer)

                _persist(final=True)  # fin de tour : écriture finale garantie

                # Cache souverain : le slot local contient LA conversation à cet
                # instant précis -> on le sauve MAINTENANT (~ms), avant que le titre
                # (inline ci-dessous) et le reflect (maintenance) ne l'écrasent ; la
                # maintenance le RESTAURERA (~ms) au lieu de re-préfiller des minutes.
                # session_id : sidecar meta pour la reprise à CHAUD one-shot
                # (try_hot_resume ne restaure jamais le save d'une autre session).
                _kv_saved = S.client.save_slot(
                    conv.model, "turnend.kv", session_id=sess.id
                )

                # Apprentissage post-tour + restauration du cache : DÉPORTÉS dans un
                # thread (cf. _post_turn_maintenance). Avant, reflect tournait ICI,
                # avant le `done` -> l'UI restait sur « le modèle travaille » pendant
                # un appel modèle entier, ET le cache KV de la conversation était
                # écrasé -> re-prefill INTÉGRAL au message suivant (bug 2026-07-10).
                # Le thread attend le verrou local (libéré à la fermeture du flux).
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

                # Auto-titre : à la 1re vraie réponse, nommer la session (le modèle infère le
                # sujet). On titre LA session de CETTE génération (`sess`), pas la session
                # focus (_cur) — sinon, en multi-onglets concurrent, on titrerait la mauvaise.

                if _immediate_title:
                    # Distant : filet de secours si le thread de titre n'a pas fini avant la
                    # fin de la boucle (ou tour sans événement) -> on l'attend brièvement.
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
                    # Local : titrage en fin de tour (slot llama-swap libre, pas de contention).
                    _title = _infer_title(S.client, conv.model or None, message)
                    if _title:
                        sess.title = _title
                        S.session_store.save(sess)
                        yield _sse("session_title", id=sess.id, title=_title)

                yield _sse("done")

            except GeneratorExit:
                # Marqueur d'interruption AVANT la persistance finale : le tour
                # suivant (celui qui a remplacé ce flux) saura que cette réponse
                # est volontairement tronquée.
                try:
                    conv.add(
                        "user",
                        "[Interrupted by a new user submission — the answer above "
                        "is incomplete.]",
                    )
                except Exception:  # noqa: BLE001 - marqueur best-effort
                    pass
                # L'utilisateur a soumis un nouveau message : le client a fermé le

                # flux. On persiste la réponse PARTIELLE déjà reçue, puis on relaie

                # l'interruption (re-raise obligatoire pour le protocole générateur).

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
