# loom/web/routes/maintenance.py — helper de chat sorti de chat.py (comportement constant).
from __future__ import annotations
import time

from loom.agent.calltrace import call_purpose
from loom.agent.debuglog import log_event, set_debug_log_path
from loom.web.routes.priming import _prime_slot


def _maint_log(etape: str, t0: float, **fields) -> None:
    """Une ligne `maint.step` : étape, durée et résultat (journal de la session)."""
    log_event(
        "maint.step", etape=etape, duree_s=round(time.monotonic() - t0, 1), **fields
    )


def _post_turn_maintenance(
    S,
    sess,
    msgs,
    actions,
    answer,
    model,
    do_reflect,
    title_request=None,
):
    """Fin de tour déportée hors du flux SSE : sauvegarde du slot de la
    conversation, reflect (apprentissage) PUIS restauration de ce cache (repli =
    ré-amorçage par re-prefill si le save a échoué). Local : sérialisé
    derrière le verrou (attend la fermeture du flux ; si l'utilisateur a déjà
    relancé, on passe après son tour). Distant : reflect seul."""
    is_local = bool(model) and model not in S.remote_model_ids
    holder = getattr(S, "warm_holder", None)
    # Thread à part : sans ça, reflect, restauration et ré-amorçage finissaient dans
    # le journal GLOBAL, invisibles dans celui de la session (post-mortem 2026-09-30).
    try:
        set_debug_log_path(S.session_store.session_dir(sess.id) / "debug.log")
    except Exception:  # noqa: BLE001 - journal best-effort
        pass
    t_maint = time.monotonic()
    log_event(
        "maint.start",
        model=model or "",
        local=is_local,
        reflect=bool(do_reflect),
        titre=title_request is not None,
    )

    def _aborted() -> bool:
        # Un message attend le verrou : plus RIEN ne démarre (warm, titre, ping),
        # le signal reste posé jusqu'à la libération (revue croisée 2026-09-13).
        return bool(holder and holder.get("abort"))

    t0 = time.monotonic()
    if is_local and not S.local_gen_lock.acquire(timeout=600):
        _maint_log("verrou", t0, resultat="timeout")
        return
    if is_local:
        S.local_busy["reason"] = "maintenance"
        _maint_log("verrou", t0, resultat="obtenu")

    kv_saved = False
    try:
        # Sauvegarde EN PREMIER, sous le verrou : ni reflect ni le titre n'ont encore
        # touché le slot. Un message en attente la rend inutile (son tour sauvera).
        if is_local and not _aborted():
            t0 = time.monotonic()
            kv_saved = bool(S.client.save_slot(model, "turnend.kv", session_id=sess.id))
            _maint_log("sauvegarde", t0, resultat="ok" if kv_saved else "echec")
        if do_reflect:
            t0 = time.monotonic()
            try:
                from loom.agent.reflect import reflect as _reflect

                with call_purpose("reflect"):
                    _res = _reflect(
                        msgs,
                        actions,
                        answer,
                        client=S.client,
                        model=model or S.reflect_model,
                        provider=S.reflect_stores.provider,
                        paths=S.reflect_stores.paths,
                        learned_dir=S.reflect_stores.learned_dir,
                        stream_holder=getattr(S, "warm_holder", None),
                    )

                # Trace VISIBLE (console/serve.log) : sinon l'apprentissage est
                # une boîte noire — on ne sait pas s'il a tourné ni retenu quoi.
                if _res is None:
                    _maint_log("reflect", t0, resultat="rien_retenu")
                    print(
                        "[reflect] rien retenu (tour peu généralisable)",
                        flush=True,
                    )
                else:
                    # Une note identité RÉÉCRIT le system prompt (USER/MEMORY) : sur un
                    # modèle hybride, le prochain tour recalcule tout (2026-09-30).
                    _maint_log(
                        "reflect",
                        t0,
                        resultat="retenu",
                        skills=len(_res.new_skills) + len(_res.improved_skills),
                        episodes=len(_res.episodes),
                        notes_identite=len(_res.memory_updates)
                        + len(_res.user_updates)
                        + len(_res.soul_updates),
                    )
                    print(
                        f"[reflect] retenu : {len(_res.new_skills)} skill(s), "
                        f"{len(_res.improved_skills)} amélioré(s), "
                        f"{len(_res.episodes)} épisode(s), "
                        f"{len(_res.memory_updates) + len(_res.user_updates) + len(_res.soul_updates)} "
                        "note(s) identité",
                        flush=True,
                    )

            except Exception as _e:  # noqa: BLE001 - best-effort, jamais bloquant
                _maint_log("reflect", t0, resultat="erreur", erreur=str(_e))
                print(f"[reflect] erreur ignorée : {_e}", flush=True)

        if is_local and _aborted():
            _maint_log("restauration", t_maint, resultat="sautee_message_en_attente")
        if is_local and not _aborted():
            t0 = time.monotonic()
            _restored = bool(kv_saved) and S.client.restore_slot(model, "turnend.kv")
            _maint_log(
                "restauration",
                t0,
                resultat="ok"
                if _restored
                else ("refusee_ou_echec" if kv_saved else "pas_de_sauvegarde"),
            )
            if _restored:
                print(
                    "[slot] cache de la conversation RESTAURÉ après fin de tour "
                    "(~ms, save/restore du slot KV)",
                    flush=True,
                )
            else:
                t0 = time.monotonic()
                _ok = _prime_slot(S, sess)
                _maint_log(
                    "reamorcage",
                    t0,
                    resultat="ok"
                    if _ok
                    else ("annule" if _aborted() else "echec_ou_sans_objet"),
                )
                print(
                    f"[prime] repli ré-amorçage par re-prefill : "
                    f"{'ok' if _ok else 'échec/sans objet'}",
                    flush=True,
                )
            S.last_activity[0] = time.time()
            if title_request is not None and not _aborted():
                # Vrai titre APRÈS le warm, séquentiel (les deux slots partagent le
                # matériel : en parallèle, warm à 6,5 t/s et titre annulé au timeout,
                # vécu 2026-09-13) et INTERRUPTIBLE par un message via le porte-flux.
                from loom.web.routes.helpers import _title_in_background

                message, provisional = title_request
                t0 = time.monotonic()
                _title_in_background(
                    S,
                    sess,
                    model,
                    message,
                    provisional,
                    stream_holder=getattr(S, "warm_holder", None),
                )
                _maint_log("titre", t0)

    finally:
        _interrupted = _aborted()  # lu AVANT la remise à zéro du signal
        if is_local:
            if holder is not None:
                holder["abort"] = False  # le verrou se libère : signal remis à zéro
            S.local_busy["reason"] = ""
            S.local_gen_lock.release()
        log_event(
            "maint.end",
            duree_s=round(time.monotonic() - t_maint, 1),
            interrompue=_interrupted,
        )


# --- Keep-warm : empêche l'OS d'évincer le modèle inactif (cold start après pause). --

# Thread daemon qui ping le modèle de la session ACTIVE (1 token) quand : keep-warm

# activé, une vraie requête a déjà eu lieu (_last_activity > 0), et on est resté idle

# depuis >= keepwarm_interval. `_local_gen_lock` non bloquant => on ne ping JAMAIS pendant

# une génération LOCALE (--parallel 1). On ne ping QUE le modèle déjà chargé => pas de swap.


def _keepwarm_loop(S):
    while True:
        interval = float(S.settings["keepwarm_interval"])  # relu à chaud
        time.sleep(max(15.0, min(interval / 3.0, 60.0)))

        # Activable/désactivable à chaud : si coupé, on ne ping pas (thread reste en veille).
        if not S.settings["keepwarm_enabled"]:
            continue

        last = S.last_activity[0]

        if last <= 0 or (time.time() - last) < interval:
            continue

        _keepwarm_tick(S)


def _keepwarm_tick(S) -> None:
    """Une itération du keep-warm (extraite pour être testable) : verrou, prime du
    préfixe (ou ping de repli), signal d'annulation honoré et remis à zéro."""
    if not S.local_gen_lock.acquire(blocking=False):
        return  # génération locale en cours => déjà chaud

    holder = getattr(S, "warm_holder", None)
    S.local_busy["reason"] = "keepwarm"
    try:
        sess = S.cur["session"]

        model = sess.conversation.model if sess else None

        if not model:
            return

        try:  # journal de la session, pas le journal global
            set_debug_log_path(S.session_store.session_dir(sess.id) / "debug.log")
        except Exception:  # noqa: BLE001 - journal best-effort
            pass
        log_event("keepwarm.tick", model=model)

        # Keep-warm = garder chaud le modèle LOCAL (éviter le cold start). Un modèle
        # DISTANT n'a pas de cold start côté machine ET est PAYANT à l'appel : le
        # pinger en boucle brûlerait des crédits pour rien -> on saute.
        if model in S.remote_model_ids:
            return

        # Keep-warm v2 : on ré-amorce le PRÉFIXE DE LA CONVERSATION au lieu
        # d'un « ping » — l'ancien ping gardait le modèle chaud mais ÉCRASAIT
        # le cache KV du fil (slot unique) : chaque reprise re-préfillait
        # TOUT (bug 2026-07-10). Ici : modèle chaud ET cache chaud ; si le
        # cache est déjà bon, le prefill est ~nul -> quasi gratuit. Repli
        # ping pour une session encore vide (rien à amorcer, juste chauffer).
        with call_purpose("keepwarm"):
            primed = _prime_slot(S, sess)
        if not primed and not (holder and holder.get("abort")):
            # Un warm INTERROMPU n'est pas un échec : pas de ping pendant qu'un
            # message attend le verrou (revue croisée 2026-09-13).
            with call_purpose("keepwarm"):
                for _kind, _chunk in S.client.stream_chat(
                    [{"role": "user", "content": "ping"}],
                    "",
                    1,
                    model=model,
                    thinking=False,
                    stream_holder=holder,
                ):
                    pass

        S.last_activity[0] = time.time()  # gardé chaud => relance un intervalle

    except Exception:  # noqa: BLE001 - keep-warm best-effort, jamais bloquant
        pass

    finally:
        if holder is not None:
            holder["abort"] = False
        S.local_busy["reason"] = ""
        S.local_gen_lock.release()
