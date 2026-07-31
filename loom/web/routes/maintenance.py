# loom/web/routes/maintenance.py — helper de chat sorti de chat.py (comportement constant).
from __future__ import annotations
import time

from loom.web.routes.priming import _prime_slot




def _post_turn_maintenance(
    S, sess, msgs, actions, answer, model, do_reflect, kv_saved=False
):
    """Fin de tour déportée hors du flux SSE : reflect (apprentissage) PUIS
    restauration du cache de la conversation (save fait en fin de génération ;
    repli = ré-amorçage par re-prefill si le save a échoué). Local : sérialisé
    derrière le verrou (attend la fermeture du flux ; si l'utilisateur a déjà
    relancé, on passe après son tour). Distant : reflect seul."""
    is_local = bool(model) and model not in S.remote_model_ids

    if is_local and not S.local_gen_lock.acquire(timeout=600):
        return
    if is_local:
        S.local_busy["reason"] = "maintenance"

    try:
        if do_reflect:
            try:
                from loom.agent.reflect import reflect as _reflect

                _res = _reflect(
                    msgs,
                    actions,
                    answer,
                    client=S.client,
                    model=model or S.reflect_model,
                    provider=S.reflect_stores.provider,
                    paths=S.reflect_stores.paths,
                    learned_dir=S.reflect_stores.learned_dir,
                )

                # Trace VISIBLE (console/serve.log) : sinon l'apprentissage est
                # une boîte noire — on ne sait pas s'il a tourné ni retenu quoi.
                if _res is None:
                    print(
                        "[reflect] rien retenu (tour peu généralisable)",
                        flush=True,
                    )
                else:
                    print(
                        f"[reflect] retenu : {len(_res.new_skills)} skill(s), "
                        f"{len(_res.improved_skills)} amélioré(s), "
                        f"{len(_res.episodes)} épisode(s), "
                        f"{len(_res.memory_updates) + len(_res.user_updates) + len(_res.soul_updates)} "
                        "note(s) identité",
                        flush=True,
                    )

            except Exception as _e:  # noqa: BLE001 - best-effort, jamais bloquant
                print(f"[reflect] erreur ignorée : {_e}", flush=True)

        if is_local:
            if kv_saved and S.client.restore_slot(model, "turnend.kv"):
                print(
                    "[slot] cache de la conversation RESTAURÉ après fin de tour "
                    "(~ms, save/restore du slot KV)",
                    flush=True,
                )
            else:
                _ok = _prime_slot(S, sess)
                print(
                    f"[prime] repli ré-amorçage par re-prefill : "
                    f"{'ok' if _ok else 'échec/sans objet'}",
                    flush=True,
                )
            S.last_activity[0] = time.time()

    finally:
        if is_local:
            S.local_busy["reason"] = ""
            S.local_gen_lock.release()


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

        if not S.local_gen_lock.acquire(blocking=False):
            continue  # génération locale en cours => déjà chaud

        S.local_busy["reason"] = "keepwarm"
        try:
            sess = S.cur["session"]

            model = sess.conversation.model if sess else None

            if not model:
                continue

            # Keep-warm = garder chaud le modèle LOCAL (éviter le cold start). Un modèle
            # DISTANT n'a pas de cold start côté machine ET est PAYANT à l'appel : le
            # pinger en boucle brûlerait des crédits pour rien -> on saute.
            if model in S.remote_model_ids:
                continue

            # Keep-warm v2 : on ré-amorce le PRÉFIXE DE LA CONVERSATION au lieu
            # d'un « ping » — l'ancien ping gardait le modèle chaud mais ÉCRASAIT
            # le cache KV du fil (slot unique) : chaque reprise re-préfillait
            # TOUT (bug 2026-07-10). Ici : modèle chaud ET cache chaud ; si le
            # cache est déjà bon, le prefill est ~nul -> quasi gratuit. Repli
            # ping pour une session encore vide (rien à amorcer, juste chauffer).
            if not _prime_slot(S, sess):
                for _kind, _chunk in S.client.stream_chat(
                    [{"role": "user", "content": "ping"}],
                    "",
                    1,
                    model=model,
                    thinking=False,
                ):
                    pass

            S.last_activity[0] = time.time()  # gardé chaud => relance un intervalle

        except Exception:  # noqa: BLE001 - keep-warm best-effort, jamais bloquant
            pass

        finally:
            S.local_busy["reason"] = ""
            S.local_gen_lock.release()
