from __future__ import annotations

from __future__ import annotations
import threading
import time
from loom.runtime.comfy import ComfyEngine
from loom.runtime.hardware import ram_available_mb



# Marge RAM (Mo) gardée libre AU-DELÀ du LLM à charger : OS, cache KV, pics
# transitoires. En dessous, on ne garde pas le cache image (jamais d'OOM pour
# une optimisation de confort).
_RAM_KEEP_MARGIN_MB = 4096


# ---- Modèles : limites, prix, jauges -------------------------------------------------


def _price_of(S, model_id):
    return S.model_prices.get(model_id, (0.0, 0.0, 0.0))


def _ctx_info(S, model_id):
    """(fenêtre de contexte, source) du modèle -> dénominateur de la jauge + provenance.

    Distant : on demande D'ABORD au PROVIDER (`client.remote_context`, mis en cache) —
    c'est le modèle lui-même qui fait autorité. S'il ne publie rien (Z.ai/OpenAI), repli
    sur la valeur déclarée en config. Local : la fenêtre est celle qu'on a ALLOUÉE au
    serveur (n_ctx) = notre limite volontaire, signalée comme telle. Sources possibles :
    `provider` (fait autorité), `config` (déclaré, non vérifiable), `local` (notre limite)."""
    declared = S.model_contexts.get(model_id) or S.context_window
    if model_id in S.remote_model_ids:
        provided = S.client.remote_context(model_id)
        if provided:
            return provided, "provider"
        return declared, "config"
    return declared, "local"


def _model_limits(S, model_id):
    """(plafond de sortie, seuil de microcompact) pour `model_id`.

    Le max_tokens global est une contrainte LOCALE (calibrée pour la VRAM de la machine).
    Un modèle DISTANT ne l'hérite PAS : sa machine est plus puissante. Non défini -> None
    (plafond OMIS dans la requête, le provider applique SA limite). La réserve de
    microcompact reste modeste côté distant (leur fenêtre est large, le seuil compte peu)."""
    win = S.model_contexts.get(model_id) or S.context_window
    explicit = S.model_max_tokens.get(model_id)
    if model_id in S.remote_model_ids:
        cap = explicit  # None possible -> pas de cap imposé
        reserve = explicit or 8192
    else:
        cap = explicit or S.settings["max_tokens"]  # local : plafond global
        reserve = cap
    return cap, max(1024, win - reserve - 1024)


def _totals(S, conv):
    """Compteurs de session + fenêtre du modèle (jauge de remplissage du contexte).
    La fenêtre dépend du modèle (que l'app connaît), pas de la Conversation -> jointe ici,
    avec sa source (provider/config/local) pour que l'UI signale si le chiffre fait autorité."""
    win, src = _ctx_info(S, conv.model)
    return {**conv.usage_totals(), "context_window": win, "context_source": src}


# ---- Moteurs image (ComfyUI) ----------------------------------------------------------


def _engine_for(S, im) -> ComfyEngine:
    key = (im.comfy_dir, im.comfy_port)
    with S.engines_lock:
        if key not in S.engines:
            S.engines[key] = ComfyEngine(im.comfy_dir, im.comfy_port)
        return S.engines[key]


def _free_image_engines(S, llm_size_mb: int = 0) -> bool | None:
    """Rend la VRAM tenue par un moteur image (best-effort, rapide) : appelé avant
    une génération LOCALE — 6 Go ne tiennent pas la diffusion ET le LLM.

    La RAM, elle, est arbitrée : si le LLM entrant tient À CÔTÉ du cache image
    (RAM disponible mesurée >= size_mb du LLM + marge), on garde le cache
    (keep_ram) — la prochaine image repart de la RAM, pas du disque. Machine
    étroite (ex. 32 Go) ou taille inconnue -> cache vidé, comportement historique.
    Renvoie True (cache gardé), False (cache vidé) ou None (aucun moteur actif)."""
    with S.engines_lock:
        engines = list(S.engines.values())
    up = [eng for eng in engines if eng.is_up(timeout=0.5)]
    if not up:
        return None
    keep = bool(llm_size_mb) and ram_available_mb() >= (
        llm_size_mb + _RAM_KEEP_MARGIN_MB
    )
    for eng in up:
        eng.free(keep_ram=keep)
    return keep


def _local_size_mb(S, mid) -> int:
    """size_mb (model.toml) d'un modèle local, 0 si inconnu."""
    spec = next((m for m in S.local_model_specs if m.get("id") == mid), None)
    return int(spec.get("size_mb") or 0) if spec else 0


def _client_mark_all_cold(S) -> None:
    """Marque tous les slots froids (reprise à chaud) — tolère un client absent ou
    un fake de test qui n'implémente pas la méthode (duck typing des tests web)."""
    fn = getattr(S.client, "mark_all_cold", None)
    if fn is not None:
        fn()


def _ensure_local_server(S, wait: float = 0.0) -> bool:
    """Serveur modèle joignable ? Sinon DÉMARRAGE AUTO, puis attente bornée à `wait` s.
    GGUF déjà présents -> llama-swap répond en ~1-2 s ; un premier téléchargement peut
    dépasser `wait` (pas grave : l'UI suit l'état via /machine_state)."""
    reachable, _ = S.client.running_local(timeout=2.0)
    if reachable:
        return True
    # Serveur injoignable -> tout slot présumé chaud ne l'est plus (reprise à
    # chaud : le prochain amorçage repassera par try_hot_resume).
    _client_mark_all_cold(S)
    S.server_manager.start()
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        time.sleep(0.7)
        reachable, _ = S.client.running_local(timeout=2.0)
        if reachable:
            return True
    return False


# ---- Sessions : verrous, cache, contexte ---------------------------------------------


def _local_busy_notice(S) -> str:
    """Message de mise en file quand le verrou local est tenu : dit la VRAIE raison.
    Un prefill d'amorçage (prime/keepwarm/maintenance) n'est PAS « une autre session
    qui génère » — la notice mentait dans ce cas (retour user 2026-07-19)."""
    reason = (getattr(S, "local_busy", None) or {}).get("reason") or ""
    if reason in ("prime", "keepwarm", "maintenance"):
        return (
            "modèle local en préparation de contexte (prefill d'amorçage) — "
            "ton message part dès que c'est prêt."
        )
    if reason == "image":
        return (
            "machine occupée par une génération d'image — mise en file derrière elle."
        )
    return (
        "modèle local occupé : une autre session génère déjà sur la machine — "
        "mise en file (le parallèle réel n'existe qu'avec un modèle distant)."
    )


def _lock_for(S, sid: str) -> threading.Lock:
    with S.gen_guard:
        return S.sess_locks.setdefault(sid, threading.Lock())


def _cancel_for(S, sid: str) -> threading.Event:
    with S.gen_guard:
        return S.sess_cancel.setdefault(sid, threading.Event())


def _ensure_model(S, sess):
    # Une session neuve peut naître sans modèle -> requête model="" -> llama-swap renvoie
    # 404. On garantit un modèle valide (le 1er = défaut) ; corrige aussi les vides.
    if sess is not None and not sess.conversation.model and S.models:
        sess.conversation.set_model(S.models[0])
        S.session_store.save(sess)
    return sess


def _get_session(S, sid: str):
    """Session par id, depuis le cache (une instance) ou chargée du disque. None si absente."""
    if not sid:
        return None
    with S.gen_guard:
        s = S.sessions_cache.get(sid)
    if s is None:
        s = S.session_store.load(sid)
        if s is not None:
            with S.gen_guard:
                s = S.sessions_cache.setdefault(sid, s)
    return _ensure_model(S, s)


def _session(S):
    cur = S.cur["session"]
    if cur is None:
        cur = S.session_store.active() or S.session_store.create(
            workspace=S.workspace_dir
        )
        with S.gen_guard:
            cur = S.sessions_cache.setdefault(cur.id, cur)
        S.cur["session"] = cur
    return _ensure_model(S, cur)


def _ctx(S):
    """Renvoie (conversation, save) : la conversation de la session active et sa

    persistance. Point de vérité unique pour tous les endpoints."""

    sess = _session(S)

    return sess.conversation, (lambda: S.session_store.save(sess))


def _confirm(S, tool_id: str, name: str, args: dict) -> bool:
    """Bloque jusqu'à la décision UI (OK/Refuser). Interruptible et borné.

    Renvoie False si refus, timeout, ou si une nouvelle soumission annule
    (cancel_event) — évite tout deadlock sur le verrou de chat.
    """

    ev = threading.Event()

    S.pending[tool_id] = {"event": ev, "approved": False}

    deadline = time.monotonic() + S.confirm_timeout

    # Annulation de LA session dont on exécute la génération (thread-local, posé par /chat).
    cancel_ev = getattr(S.confirm_local, "ev", None)

    try:
        while not ev.wait(0.2):
            if (cancel_ev is not None and cancel_ev.is_set()) or (
                time.monotonic() > deadline
            ):
                return False

        return bool(S.pending[tool_id]["approved"])

    finally:
        S.pending.pop(tool_id, None)
