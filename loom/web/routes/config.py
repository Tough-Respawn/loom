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


def _running_local_models(S) -> set[str] | None:
    """Modèles servis par llama-swap : set vide si rien n'écoute (rien à interrompre),
    None si la réponse est illisible (prudence : on n'interrompt pas)."""
    from loom.runtime.swap_sync import running_models

    client = getattr(S, "client", None)
    if client is None:
        return set()
    reachable, txt = client.running_local(timeout=2.0)
    return running_models(txt) if reachable else set()


def _regen_swap_yaml(S):
    """Recalcule le llama-swap.yaml depuis la config SANS interrompre le modèle qui
    tourne : llama-swap (-watch-config) recharge GLOBALEMENT, donc une modification qui
    ne touche pas le modèle actif est mise en attente jusqu'à la prochaine transition
    (cf. loom.runtime.swap_sync ; incident du 2026-10-14).

    Renvoie 'written', 'deferred' ou 'unchanged' (vrais), False en cas d'échec."""
    if not (S.config_defaults_path and S.config_local_path):
        return False
    from loom.runtime import serve

    try:
        running = _running_local_models(S)
        status = serve.sync_swap_yaml(
            S.config_defaults_path,
            S.config_local_path,
            serve.SWAP_YAML,
            running=running,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[loom] regen swap yaml échoué: {e}", flush=True)
        return False
    if status is None:
        print("[loom] regen swap yaml échoué: config illisible", flush=True)
        return False
    log_event(
        "swap.config", statut=status, actifs=",".join(sorted(running or ())) or "-"
    )
    if status == "written" and running:
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
    """Avant toute requête vers le modèle local `model` : si une config llama-swap est
    EN ATTENTE et que ce modèle ne tourne pas, l'appliquer et attendre la fin du
    rechargement. Le modèle actif s'arrêterait de toute façon (llama-swap permute) ; et
    sans attente, la requête démarrerait `model` sous l'ancienne config, aussitôt tué
    par le rechargement. Sans attente en cours : un simple test d'existence de fichier."""
    from loom.runtime import serve, swap_sync

    y = serve.SWAP_YAML
    if not swap_sync.pending_path(y).exists():
        return
    with _SWAP_PREFLIGHT_LOCK:
        if not swap_sync.pending_path(y).exists():
            return
        reachable, txt = S.client.running_local(timeout=2.0)
        running = swap_sync.running_models(txt) if reachable else set()
        if running is not None and model in running:
            return  # il sert déjà avec une entrée inchangée : l'attente peut attendre
        root = S.client.local_server_root() if reachable else None
        result = swap_sync.apply_pending(y, root=root)
        log_event("swap.config", statut=result or "rien", cible=model)
        if reachable:
            _client_mark_all_cold(S)


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
