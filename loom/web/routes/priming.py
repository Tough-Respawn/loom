# loom/web/routes/priming.py — helper de chat sorti de chat.py (comportement constant).
from __future__ import annotations

import threading
import time

from loom.web.routes.helpers import (
    _ensure_local_server,
    _ensure_model,
)
from loom.web.routes.system_prompt import _build_system_prompt

# ---- Maintenance post-tour et keep-warm ------------------------------------------------


def _prime_slot(S, sess) -> bool:
    """Ré-amorce le cache KV du slot local avec le fil de `sess` : re-prefill
    silencieux du MÊME préfixe que le prochain tour (system prompt + messages +
    schémas d'outils — mêmes ingrédients que /chat, sinon zéro réutilisation).
    Le message suivant ne préfille alors que son delta. Fil VIDE accepté : sur une
    session neuve on amorce le préfixe statique (system prompt + schémas), c'est
    justement là que le premier message payait tout le prefill. False si rien à
    amorcer (modèle distant : cache provider + appel payant ; image/vidéo)."""
    try:
        conv = sess.conversation
        model = conv.model
        if (
            not model
            or model in S.remote_model_ids
            or model in S.image_model_ids
            or model in S.video_model_ids
        ):
            return False
        # Reprise à CHAUD one-shot AVANT le re-prefill : sur slot froid (serveur
        # (re)démarré, swap de modèle, boot loom.web), si un save de CETTE session
        # existe, un restore (~0,6 s) remplace le re-prefill intégral (~60-85 s
        # mesurés sur un 35B). Best-effort : le warm_context ci-dessous reste le
        # repli ET la validation (préfixe identique -> ne paie que le delta).
        # getattr : les fakes de test (PrimeSpy…) n'implémentent que warm_context.
        _thr = getattr(S.client, "try_hot_resume", None)
        if _thr is not None and _thr(model, sess.id):
            print(
                "[prime] reprise à CHAUD : slot restauré depuis le save de fin de "
                "tour (le re-prefill ci-dessous ne paie que le delta)",
                flush=True,
            )
        msgs = conv.to_messages()
        if not msgs:
            # Fil vide : certains templates (Qwen3-coder/Agents-A1) EXIGENT un message
            # user (« No user query found » -> 400). On amorce avec un placeholder
            # minimal : le préfixe commun (system prompt + schémas + en-tête user)
            # reste réutilisé tel quel, seul le contenu du placeholder diverge du
            # vrai premier message (quelques tokens re-préfillés, pas des milliers).
            msgs = [{"role": "user", "content": "."}]
        system_prompt, _strong = _build_system_prompt(S, conv, workspace=sess.workspace)
        registry = S.tool_factory(conv.active_tools, sess.workspace, conv)
        return S.client.warm_context(
            msgs,
            system_prompt,
            model=model,
            registry=registry if (registry is not None and len(registry)) else None,
            thinking=conv.thinking,
        )
    except Exception as e:  # noqa: BLE001 - amorçage best-effort, jamais bloquant
        print(f"[prime] erreur ignorée : {e}", flush=True)
        return False


def _prime_async(S, sess, *, wait_server: float = 0.0, require_running: bool = False):
    """Amorce le cache KV en FOND (thread daemon) dès qu'un modèle local devient la
    cible du prochain tour : le prefill du préfixe se paie pendant le temps mort
    (chargement/choix du modèle, bascule de session), plus au premier message.
    Remplace l'ancien ping warmup qui chargeait le modèle mais écrasait le slot
    avec un préfixe poubelle -> le 1er message re-préfillait TOUT.
    `wait_server` > 0 : démarre le serveur modèle s'il est éteint et attend (choix
    de modèle, bouton start = intention claire). `require_running` : n'amorce que
    si le serveur tourne déjà (bascule de session : changer de fil ne doit pas
    booter la machine). Best-effort : jamais bloquant, jamais d'erreur visible."""
    conv = sess.conversation
    model = conv.model
    if (
        S.client is None
        or not model
        or model in S.remote_model_ids
        or model in S.image_model_ids
        or model in S.video_model_ids
    ):
        return

    def _run():
        try:
            if wait_server > 0:
                if not _ensure_local_server(S, wait=wait_server):
                    print(
                        "[prime] serveur modèle indisponible — amorçage abandonné",
                        flush=True,
                    )
                    return
            elif require_running:
                reachable, _ = S.client.running_local(timeout=2.0)
                if not reachable:
                    print("[prime] serveur éteint — amorçage sauté", flush=True)
                    return
            # Une génération locale en cours = modèle déjà chaud et cache géré en
            # fin de tour (_post_turn_maintenance) : on ne s'y empile pas.
            if not S.local_gen_lock.acquire(blocking=False):
                print("[prime] génération en cours — amorçage sauté", flush=True)
                return
            S.local_busy["reason"] = "prime"
            try:
                ok = _prime_slot(S, sess)
                print(
                    f"[prime] amorçage au chargement : "
                    f"{'ok' if ok else 'échec/sans objet'}",
                    flush=True,
                )
                if ok:
                    # Préfixe chaud : le keep-warm prend le relais pour le garder.
                    S.last_activity[0] = time.time()
            finally:
                S.local_busy["reason"] = ""
                S.local_gen_lock.release()
        except Exception as e:  # noqa: BLE001 - amorçage best-effort
            print(f"[prime] erreur ignorée : {e}", flush=True)

    threading.Thread(target=_run, daemon=True, name="loom-prime").start()


def _boot_prime(S):
    """Amorce au DÉMARRAGE de loom.web : si une session active persistée existe et
    que le serveur modèle tourne déjà (instance externe ou restart de loom.web),
    son préfixe est pré-préfillé sans attendre le premier message. Ne crée jamais
    de session et ne démarre jamais le serveur : boot = zéro effet de bord."""
    try:
        if S.client is None:
            return
        sess = S.session_store.active()
        if sess is None:
            return
        with S.gen_guard:
            sess = S.sessions_cache.setdefault(sess.id, sess)
        _ensure_model(S, sess)
        _prime_async(S, sess, require_running=True)
    except Exception as e:  # noqa: BLE001 - best-effort, jamais bloquant au boot
        print(f"[prime] erreur ignorée (boot) : {e}", flush=True)
