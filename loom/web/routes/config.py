from __future__ import annotations

from __future__ import annotations
import threading
from flask import request

from loom.agent.debuglog import log_event
from loom.web.routes.helpers import _client_mark_all_cold


# ---- Config vivante ----------------------------------------------------------------


def _reload_app_config(S):
    """Relit defaults.toml + local.toml et met à jour le holder + la permission (à chaud).
    Best-effort : une config invalide ne casse pas l'app en cours (on garde l'ancienne)."""
    if not (S.config_defaults_path and S.config_local_path):
        return
    try:
        from loom.agent.context import effective_context_budget
        from loom.config import load_config
        from loom.permissions import evaluate

        c = load_config(S.config_defaults_path, S.config_local_path)
    except Exception as e:  # noqa: BLE001 - reload best-effort, jamais fatal
        print(f"[loom] reload config échoué: {e}", flush=True)
        return
    S.settings.update(
        max_tokens=c.chat.max_tokens,
        context_budget=effective_context_budget(
            c.chat.context_token_budget, c.context, c.chat.max_tokens
        ),
        keep_recent=c.chat.keep_recent_messages,
        identity_max_tokens=c.chat.identity_max_tokens,
        project_memory_max_tokens=c.chat.project_memory_max_tokens,
        reflect_enabled=c.chat.reflect_enabled,
        reflect_min_actions=c.chat.reflect_min_actions,
        keepwarm_enabled=c.chat.keepwarm_enabled,
        keepwarm_interval=c.chat.keepwarm_interval,
        permission_mode=c.permissions.mode,
    )
    S.perm["fn"] = lambda name, args: evaluate(name, args, c.permissions)


# Délai accordé à llama-swap pour charger une config écrite (rechargement mesuré
# ~1,4 s sur v241, surveillance toutes les 2 s) avant de refuser la requête.
SWAP_RELOAD_TIMEOUT_S = 20.0


def _probe_swap(S) -> tuple[str, set[str] | None, str | None]:
    """État du serveur local vu par la config llama-swap : (nature, modèles, racine).

    - 'absent'      : pas de client, rien à interrompre ;
    - 'injoignable' : /running en échec ou en délai -> modèles INCONNUS (None) : on
      n'en déduit jamais « rien ne tourne » (revue du 2026-10-14) ;
    - 'direct'      : llama-server sans llama-swap, personne ne surveille le yaml ;
    - 'swap'        : llama-swap, modèles en cours lus dans /running."""
    from loom.runtime.swap_sync import running_models

    client = getattr(S, "client", None)
    if client is None:
        return "absent", set(), None
    reachable, txt = client.running_local(timeout=2.0)
    if not reachable:
        return "injoignable", None, None
    running = running_models(txt)
    if running is None:
        return "direct", set(), client.local_server_root()
    return "swap", running, client.local_server_root()


def _running_local_models(S) -> set[str] | None:
    """Modèles servis par llama-swap ; None si inconnu (sonde en échec)."""
    return _probe_swap(S)[1]


def _swap_watch(S):
    """État « config écrite, rechargement pas encore confirmé » (un par app)."""
    from loom.runtime.swap_sync import ReloadWatch

    watch = getattr(S, "swap_watch", None)
    if watch is None:
        watch = S.swap_watch = ReloadWatch()
    return watch


def _read_text(path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _regen_swap_yaml(S):
    """Recalcule le llama-swap.yaml depuis la config SANS interrompre le modèle qui
    tourne : llama-swap (-watch-config) recharge GLOBALEMENT, donc une modification qui
    ne touche pas le modèle actif est mise en attente jusqu'à la prochaine transition
    (cf. loom.runtime.swap_sync ; incident du 2026-10-14). Une écriture immédiate ouvre
    l'attente du rechargement : la requête locale suivante attend que llama-swap l'ait
    chargée.

    Renvoie 'written', 'deferred' ou 'unchanged' (vrais), False en cas d'échec."""
    if not (S.config_defaults_path and S.config_local_path):
        return False
    from loom.runtime import serve, swap_sync

    try:
        kind, running, root = _probe_swap(S)
        y = serve.SWAP_YAML
        previous = _read_text(y)
        # Ancre AVANT l'écriture : seul un rechargement postérieur confirme la config.
        anchor = swap_sync.fetch_logs(root) if kind == "swap" else None
        status = serve.sync_swap_yaml(
            S.config_defaults_path,
            S.config_local_path,
            y,
            running=running,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[loom] regen swap yaml échoué: {e}", flush=True)
        return False
    if status is None:
        print("[loom] regen swap yaml échoué: config illisible", flush=True)
        return False
    log_event(
        "swap.config",
        statut=status,
        serveur=kind,
        actifs=",".join(sorted(running or ())) or "-",
    )
    if status == "written" and kind == "swap":
        _swap_watch(S).mark(anchor, previous, _read_text(y))
        if running:
            # llama-swap recharge tout : aucun slot ne peut plus être présumé chaud.
            _client_mark_all_cold(S)
    if getattr(S, "client", None) is not None:
        try:
            # Les slots effectifs suivent la config (cache_isolation posé après un bench).
            from loom.config import load_config
            from loom.runtime.server_args import compute_slot_counts

            cfg = load_config(S.config_defaults_path, S.config_local_path)
            S.client.slot_counts = compute_slot_counts(cfg.models, cfg.n_parallel)
        except Exception as e:  # noqa: BLE001 - la config yaml reste appliquée
            print(f"[loom] slots effectifs non recalculés: {e}", flush=True)
    return status


_SWAP_PREFLIGHT_LOCK = threading.Lock()


def _swap_preflight(S, model: str) -> None:
    """Avant toute requête vers le modèle local `model` :

    1. une config EN ATTENTE est posée si ce modèle ne tourne pas (le modèle actif
       s'arrêterait de toute façon : llama-swap permute) ;
    2. toute config écrite et pas encore confirmée chargée est ATTENDUE (sans quoi la
       requête part vers l'ancien catalogue : 404 sur un modèle ajouté, ou modèle
       démarré puis tué par le rechargement). Pas de confirmation dans le délai ->
       SwapConfigNotLoaded, explicite, l'état restant actif pour l'appel suivant.

    Rien en attente ni à confirmer : un test d'existence de fichier, aucun réseau."""
    from loom.runtime import serve, swap_sync

    y = serve.SWAP_YAML
    watch = _swap_watch(S)
    if not swap_sync.pending_path(y).exists() and not watch.active:
        return
    with _SWAP_PREFLIGHT_LOCK:
        kind, running, root = _probe_swap(S)
        if kind in ("absent", "injoignable"):
            # Rien ne répond : la requête échouera, ou lancera la pile (yaml frais).
            return
        if swap_sync.pending_path(y).exists():
            if kind == "direct":
                swap_sync.apply_pending(y)  # personne ne surveille ce fichier
            elif model not in running:
                anchor = swap_sync.fetch_logs(root)
                previous = _read_text(y)
                swap_sync.apply_pending(y)
                watch.mark(anchor, previous, _read_text(y))
                log_event("swap.config", statut="applied", cible=model)
                if running:
                    _client_mark_all_cold(S)
        if kind == "swap" and watch.active:
            if not watch.wait(root, timeout=SWAP_RELOAD_TIMEOUT_S):
                log_event(
                    "swap.config", level="WARN", statut="non_confirme", cible=model
                )
                raise swap_sync.SwapConfigNotLoaded(
                    "llama-swap n'a pas confirmé le chargement de la nouvelle "
                    f"configuration des modèles en {SWAP_RELOAD_TIMEOUT_S:.0f} s "
                    f"(détails : var/logs/serve.log) — réessaie dans un instant."
                )
            log_event("swap.config", statut="charge", cible=model)


def _install_swap_preflight(S) -> None:
    """Branche le préalable sur le client : il passe avant chaque requête locale."""
    if getattr(S, "client", None) is not None:
        S.client.local_preflight = lambda model: _swap_preflight(S, model)


def _apply_to_model_server(S, section):
    """Param SERVEUR/OVERRIDE (affecte le lancement de llama-server) : régénère le yaml et
    décharge les modèles locaux -> ils se relancent avec les nouveaux args au prochain usage
    (llama-swap -watch-config). Ne touche pas les autres process. Best-effort, en tâche de fond."""
    if section not in ("server", "override"):
        return
    if _regen_swap_yaml(S) == "written":
        threading.Thread(
            target=S.client.unload_local, daemon=True, name="loom-reload-models"
        ).start()


# ---- Routes : console de configuration -------------------------------------------------

# ---- Console de configuration : introspection + édition des vrais fichiers TOML (deux
# couches commun/système), commentaires préservés via tomlkit (loom.runtime.config_schema).


def _cfg_paths_ok(S):
    return bool(S.config_defaults_path and S.config_local_path)


def _register_config_routes(app, S):
    @app.get("/config")
    def config_describe():
        if not _cfg_paths_ok(S):
            return {"error": "chemins de config indisponibles"}, 500
        from loom.runtime import config_schema

        return config_schema.describe(S.config_defaults_path, S.config_local_path)

    @app.post("/config/set")
    def config_set():
        if not _cfg_paths_ok(S):
            return {"error": "chemins de config indisponibles"}, 500
        from loom.runtime import config_schema

        b = request.get_json(silent=True) or {}
        section = (b.get("section") or "").strip()
        key = (b.get("key") or "").strip()
        if not (section and key):
            return {"error": "section et key requis"}, 400
        try:
            with S.toml_lock:
                res = config_schema.set_value(
                    S.config_defaults_path,
                    S.config_local_path,
                    section,
                    key,
                    b.get("value"),
                )
        except (ValueError, OSError) as e:
            return {"ok": False, "error": str(e)[:160]}, 400
        if res.get("ok"):
            _reload_app_config(
                S
            )  # applique À CHAUD les params app (permissions, tokens…)
            _apply_to_model_server(
                S, section
            )  # régénère le yaml si param serveur/modèle
        code = 200 if res.get("ok") else 400
        return res, code

    @app.post("/config/reset")
    def config_reset():
        if not _cfg_paths_ok(S):
            return {"error": "chemins de config indisponibles"}, 500
        from loom.runtime import config_schema

        b = request.get_json(silent=True) or {}
        section = (b.get("section") or "").strip()
        key = (b.get("key") or "").strip()
        if not (section and key):
            return {"error": "section et key requis"}, 400
        with S.toml_lock:
            res = config_schema.reset_value(
                S.config_defaults_path, S.config_local_path, section, key
            )
        if res.get("ok"):
            _reload_app_config(S)
            _apply_to_model_server(S, section)
        return res, (200 if res.get("ok") else 400)

    @app.get("/config/effective")
    def config_effective():
        """Valeurs de config ACTUELLEMENT en vigueur dans l'app en cours (mémoire vive). Sert
        à vérifier qu'une édition s'applique à chaud, sans redémarrer loom.web."""
        return dict(S.settings)
