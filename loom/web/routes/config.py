from __future__ import annotations

from __future__ import annotations
import threading
from flask import request




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


def _regen_swap_yaml(S):
    """Régénère le llama-swap.yaml depuis la config (llama-swap -watch-config le recharge).
    Best-effort, silencieux si serve indispo. Renvoie True si écrit."""
    if not (S.config_defaults_path and S.config_local_path):
        return False
    try:
        from loom.runtime.serve import regenerate_swap_yaml

        return bool(regenerate_swap_yaml(S.config_defaults_path, S.config_local_path))
    except Exception as e:  # noqa: BLE001
        print(f"[loom] regen swap yaml échoué: {e}", flush=True)
        return False


def _apply_to_model_server(S, section):
    """Param SERVEUR/OVERRIDE (affecte le lancement de llama-server) : régénère le yaml et
    décharge les modèles locaux -> ils se relancent avec les nouveaux args au prochain usage
    (llama-swap -watch-config). Ne touche pas les autres process. Best-effort, en tâche de fond."""
    if section not in ("server", "override"):
        return
    if _regen_swap_yaml(S):
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
