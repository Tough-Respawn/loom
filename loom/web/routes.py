# loom/web/routes.py

"""Routes et helpers de loom.web, opérant sur l'objet d'état partagé `S` construit
par create_app (loom/web/app.py) : fonctions d'enregistrement module-level
`_register_*(app, S)` + helpers `_xxx(S, ...)` — aucun état module, tout vit sur S."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from functools import partial
from pathlib import Path

from flask import Response, render_template, request

from loom.agent import context
from loom.agent.client import _msg_chars, log_event, set_debug_log_path
from loom.extend.skills import (
    collect_skills,
    effective_skills,
    read_skill_source,
    render_catalog,
    write_skill_source,
)
from loom.prompts import CHAT_SYSTEM_STRONG, IMAGE_REFINE_SYSTEM
from loom.runtime import model_install, model_store
from loom.runtime.comfy import ComfyEngine, ComfyError
from loom.runtime.hardware import ram_available_mb
from loom.runtime.models_profile import load_profile
from loom.runtime.platform_info import detect as platform_detect
from loom.web import wizard as _wizard
from loom.web.app import (
    _GOAL_CLEAR_WORDS,
    _NOTE_MAX_CHARS,
    _action_trace_line,
    _build_user_content,
    _detect_workspace,
    _infer_title,
    _init_message,
    _should_adopt,
    _sse,
)

# Marge RAM (Mo) gardée libre AU-DELÀ du LLM à charger : OS, cache KV, pics
# transitoires. En dessous, on ne garde pas le cache image (jamais d'OOM pour
# une optimisation de confort).
_RAM_KEEP_MARGIN_MB = 4096


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


# ---- Skills ---------------------------------------------------------------------------


def _all_skills(S) -> list:
    return collect_skills(
        S.skills_dir,
        S.plugins_dir,
        learned_dir=S.learned_skills_dir,
        user_dir=S.user_skills_dir,
    )


def _skills_ctx(S, conv) -> dict:
    """Contexte du panneau Skills : la liste COMPLÈTE (pour les cases), l'ensemble des
    skills ACTIFS (non désactivés) et ceux qui ont un override de session (badge UI)."""
    skills = _all_skills(S)
    disabled = set(conv.disabled_skills)
    return {
        "skills": skills,
        "active_skills": [s.name for s in skills if s.name not in disabled],
        "overridden_skills": [s.name for s in skills if s.name in conv.skill_overrides],
    }


def _skill_deletable(S, skill) -> bool:
    """Un skill n'est supprimable depuis l'UI que s'il vient d'un dossier GÉRÉ par
    l'utilisateur (appris ou ajouté) ET que son dossier est bien confiné dedans —
    jamais un skill du package (versionné) ni d'un plugin (géré par le store)."""
    if getattr(skill, "origin", "") not in ("learned", "user"):
        return False
    if not skill.base_dir:
        return False
    roots = [
        d for d in (S.learned_skills_dir, S.user_skills_dir) if d
    ]  # dossiers gérés
    base = Path(skill.base_dir).resolve()
    for root in roots:
        r = Path(root).resolve()
        if base != r and r in base.parents:
            return True
    return False


def _index_context(S) -> dict:
    sess = _session(S)

    conv = sess.conversation

    ws = sess.workspace

    sessions = [
        {"id": m.id, "title": m.title, "workspace": m.workspace}
        for m in S.session_store.list()
    ]

    active_id = sess.id

    return {
        "messages": conv.messages,
        **_skills_ctx(S, conv),
        "usage_totals": _totals(S, conv),
        "models": S.models,
        "remote_model_ids": S.remote_model_ids,
        "image_model_ids": S.image_model_ids,
        "video_model_ids": S.video_model_ids,
        "model_descriptions": S.model_descriptions,
        "current_model": conv.model,
        "thinking": conv.thinking,
        "local_only": conv.local_only,
        "available_tools": S.available_tools,
        "active_tools": conv.active_tools,
        "workspace_dir": ws,
        "sessions": sessions,
        "active_session": active_id,
        "permission_mode": S.settings["permission_mode"],
        # État initial pour l'hydratation côté client (Preact). On échappe '<'
        # pour ne pas pouvoir fermer la balise <script> depuis le contenu.
        "init_json": json.dumps(
            {
                "messages": conv.messages,
                "thinking": conv.thinking,
                "local_only": conv.local_only,
                "usage_totals": _totals(S, conv),
                # Onglet initial : la session active (id/titre/modèle/workspace) + toutes
                # les sessions (pour la sidebar). Le multi-onglets s'hydrate là-dessus.
                "active_session": active_id,
                "title": sess.title,
                "model": conv.model,
                "workspace": ws,
                "sessions": sessions,
                # Racine des dossiers de session sur disque : le front en dérive
                # root/<sid> (session.json, timeline.jsonl, debug.log) pour le menu
                # contextuel d'onglet/panneau (chemin réel copiable).
                "sessions_root": str(S.session_store.root.resolve()),
            },
            ensure_ascii=False,
        ).replace("<", "\\u003c"),
    }


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


def _handle_goal_command(S, message, conv, save, chat_lock):
    """Traite la commande /goal : pose/statut/efface l'objectif de session.

    Retourne (message, response) : response non None = ack immédiat (return direct
    dans chat()) ; response None = continuer le flux normal (message éventuellement
    réécrit en consigne de démarrage)."""

    if message == "/goal" or message.startswith("/goal "):
        arg = message[len("/goal") :].strip()
        if arg and arg.lower() not in _GOAL_CLEAR_WORDS:
            # Pose l'objectif et AMORCE le travail : on remplace le message par une consigne
            # de démarrage et on laisse le flux normal tourner, objectif désormais actif.
            conv.set_goal(arg)
            save()
            message = (
                f"Objectif à atteindre : {arg}\n"
                "Commence MAINTENANT à agir pour l'atteindre, et PROUVE-le (exécute, montre "
                "la sortie réelle). Ne t'arrête pas tant qu'il n'est pas démontré atteint."
            )
            # (pas de return : on tombe dans la génération normale ci-dessous)
        else:
            if not arg:
                ack = (
                    f"Objectif courant : « {conv.goal} » (actif jusqu'à preuve d'atteinte, "
                    "/goal clear pour l'effacer)."
                    if conv.goal
                    else "Aucun objectif actif. Pose-en un : /goal <condition vérifiable>."
                )
            else:
                conv.set_goal("")
                save()
                ack = "Objectif effacé - retour au mode normal (arrêt au stop naturel)."
            chat_lock.release()

            def _goal_ack():
                yield _sse("text", text=ack)
                yield _sse("done")

            return message, Response(_goal_ack(), mimetype="text/event-stream")
    return message, None


def _handle_init_command(S, message):
    """Traite /init : adopte un dossier cible si fourni, et réécrit le message en consigne
    de génération de fiche projet. Retourne le message (éventuellement réécrit)."""
    if message == "/init" or message.startswith("/init "):
        arg = message[len("/init") :].strip()
        _sess = _session(S)
        target_dir = _sess.workspace
        if arg:
            cand = Path(arg).expanduser()
            if cand.is_dir():
                target_dir = str(cand.resolve())
                if target_dir != _sess.workspace:
                    _sess.workspace = target_dir
                    S.session_store.save(_sess)
        target_display = str(Path(target_dir)).replace("\\", "/")
        message = _init_message(target_display)
        # (pas de return : le flux normal ci-dessous exécute la consigne)
    return message


# ---- Commande /add-model : wizard déterministe d'ajout de modèle -----------------------


def _list_remote_models(base_url: str, api_key: str) -> list[str] | None:
    """Ids exposés par une API OpenAI-compatible (GET /models), triés — ou None si
    l'endpoint est injoignable/refuse : le wizard retombe sur la saisie manuelle.
    Évite de taper un nom de modèle qui n'existe pas chez le provider."""
    import httpx

    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json().get("data") or []
        ids = sorted({str(m["id"]) for m in data if m.get("id")})
        return ids or None
    except Exception:  # noqa: BLE001 - best-effort, la saisie manuelle reste possible
        return None


def _removable_models(S) -> list[dict]:
    """Modèles supprimables via /remove-model : TOUT ce que le sélecteur affiche.
    kind ∈ {local, remote (dossier remote/<id>/), image, video} — le wizard adapte
    son message de confirmation au kind (_step_d_pick)."""
    import tomllib

    items = []
    for m in S.local_model_specs:
        size = (m.get("size_mb") or 0) / 1024
        items.append(
            {
                "id": m["id"],
                "kind": "local",
                "label": f"{m['id']} — local, {size:.1f} Go sur disque",
            }
        )
    # Distants : un dossier remote/<id>/ par modèle — si c'est le default_model,
    # la confirmation avertit (repli boot sur le 1er modèle).
    default_model = ""
    if S.config_local_path and Path(S.config_local_path).exists():
        try:
            cfg = tomllib.loads(Path(S.config_local_path).read_text(encoding="utf-8"))
            default_model = str(cfg.get("chat", {}).get("default_model") or "")
        except (OSError, tomllib.TOMLDecodeError):
            pass
    for mid in sorted(S.remote_model_ids):
        items.append(
            {
                "id": mid,
                "kind": "remote",
                "is_default": mid == default_model,
                "label": f"{mid} — distant "
                f"({S.remote_model_names.get(mid, '?')}, remote/{mid}/)",
            }
        )
    # Image/vidéo (ComfyUI) : seule la DÉFINITION Loom (model.toml + workflow.json)
    # est supprimable — les poids vivent côté ComfyUI, partagés entre modèles.
    for im in sorted(S.image_by_id.values(), key=lambda m: m.id):
        kind = "video" if im.id in S.video_model_ids else "image"
        items.append(
            {
                "id": im.id,
                "kind": kind,
                "label": f"{im.id} — {kind} (ComfyUI), définition seule",
            }
        )
    return items


def _forget_remote(S, mid: str) -> None:
    """Retire un modèle distant de TOUS les registres montés (route client comprise).
    Partagé entre la route DELETE du panneau engrenage et /remove-model."""
    if getattr(S, "client", None) is not None:
        S.client.remove_remote_route(mid)
    S.remote_model_ids.discard(mid)
    S.remote_model_names.pop(mid, None)
    S.model_contexts.pop(mid, None)
    S.model_max_tokens.pop(mid, None)
    S.model_prices.pop(mid, None)
    S.vision_models.discard(mid)
    if mid in S.models:
        S.models.remove(mid)


def _models_roots(S) -> list[str]:
    """Racines des modèles, dans l'ordre de priorité (racine[0] = cible des écritures :
    nouveaux distants remote/<id>/, installs locaux). Repli : dérivée de models_dir
    (<racine>/local/text) quand create_app n'a pas reçu models_roots (vieux appels)."""
    roots = getattr(S, "models_roots", None)
    if roots:
        return [str(r) for r in roots]
    # Repli sûr : uniquement si models_dir suit la convention <racine>/local/text —
    # sinon remonter de deux crans pointerait n'importe où.
    if S.models_dir:
        p = Path(S.models_dir)
        if p.name == "text" and p.parent.name == "local":
            return [str(p.parent.parent)]
    return []


def _install_roots(S) -> list[dict]:
    """Racines candidates à l'installation d'un LOCAL, avec l'espace libre (Go) pour que
    le wizard affiche un choix éclairé quand il y a plusieurs disques. racine[0] = défaut
    (la plus rapide par convention, cf. [storage] models_root)."""
    import shutil

    out = []
    for r in _models_roots(S):
        try:
            free_gb = shutil.disk_usage(r).free // (1024**3)
        except OSError:
            free_gb = None
        out.append({"path": r, "free_gb": free_gb})
    return out


def _wizard_deps(S):
    """Dépendances du wizard (INJECTÉES : la machine à états reste pure et testable).
    Point de patch des tests — garder la construction ici, jamais dans wizard.py."""
    from types import SimpleNamespace

    from loom.runtime import hardware, hf_catalog

    hw = hardware.detect_hardware()
    # Budget de CAPACITÉ : RAM TOTALE (le modèle courant sera déchargé par
    # llama-swap avant le nouveau), et VRAM ajoutée SEULEMENT si discrète (sur
    # mémoire unifiée elle EST la RAM -> 0, pas de double comptage). Corrige le
    # « ne tiendra pas » erroné du 2026-07-23 (budget mesuré sur la dispo).
    ram_total = hardware.ram_total_mb()
    vram_budget = hw.vram_total_mb if hw.vram_is_discrete else 0
    return SimpleNamespace(
        search_models=hf_catalog.search_models,
        list_gguf_files=hf_catalog.list_gguf_files,
        recommend=lambda files: model_install.recommend_quant(
            files, vram_budget, ram_total
        ),
        derive_id=model_install.derive_model_id,
        existing_ids=set(S.models),
        list_remote_models=_list_remote_models,
        removable_models=lambda: _removable_models(S),
        image_dir_state=lambda ikind, mid: _image_dir_state(S, ikind, mid),
        check_workflow=_check_workflow,
        rebenchable_models=lambda: _rebenchable_models(S),
        model_kind=lambda mid: _model_kind(S, mid),
        install_roots=lambda: _install_roots(S),
    )


def _mount_local(S, mid, mdir, size_mb, vision=False):
    """Monte À CHAUD un modèle local fraîchement installé : registres partagés +
    régénération du llama-swap.yaml (llama-swap -watch-config le recharge). Le
    sélecteur voit le modèle sans redémarrer loom.web (spec §3.4)."""
    if not any(m.get("id") == mid for m in S.local_model_specs):
        S.local_model_specs.append(
            {"id": mid, "dir": str(mdir), "size_mb": int(size_mb)}
        )
    if mid not in S.local_model_ids:
        S.local_model_ids.append(mid)
    if mid not in S.models:
        S.models.append(mid)
    if vision:
        S.vision_models.add(mid)
    _regen_swap_yaml(S)


def _image_base_dir(S, ikind: str) -> Path:
    """Dossier local/{image,video} où vivent les modèles de ce type : la racine qui
    en héberge déjà, sinon celle d'un modèle image existant, sinon à côté de
    models_dir (<root>/local/text -> <root>/local/<ikind>)."""
    for im in S.image_by_id.values():
        d = Path(im.dir).parent
        if d.name == ikind:
            return d
    if S.image_by_id:
        any_dir = Path(next(iter(S.image_by_id.values())).dir)
        return any_dir.parent.parent / ikind
    return Path(S.models_dir).parent / ikind


def _image_dir_state(S, ikind: str, mid: str) -> str | None:
    """État du dossier d'un modèle image/vidéo : None (absent), "partial" (scaffold
    sans recette) ou "complete" (montable). Sert au wizard pour la reprise."""
    d = _image_base_dir(S, ikind) / mid
    if not d.is_dir():
        return None
    return (
        "complete"
        if (d / "model.toml").is_file() and (d / "workflow.json").is_file()
        else "partial"
    )


def _check_workflow(path: str) -> dict:
    """Validation légère d'un export ComfyUI « format API » : JSON parsable +
    placeholder {PROMPT} (warning si absent, jamais bloquant — recette exotique)."""
    import json as _json

    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": f"fichier introuvable ({path})", "warnings": []}
    try:
        raw = p.read_text(encoding="utf-8")
        _json.loads(raw)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"JSON invalide : {exc}", "warnings": []}
    warnings = []
    if "{PROMPT}" not in raw:
        warnings.append(
            "placeholder {PROMPT} absent — le prompt du chat ne sera pas injecté"
        )
    return {"ok": True, "error": None, "warnings": warnings}


def _mount_image(S, ikind: str, mid: str):
    """(Re)découvre <root>/local/{image,video}/<mid> via le parseur officiel et le
    monte À CHAUD dans tous les registres du sélecteur. None si absent/incomplet."""
    from loom.runtime.image_models import discover_image_models

    root = _image_base_dir(S, ikind).parent.parent
    im = next((m for m in discover_image_models([root]) if m.id == mid), None)
    if im is None:
        return None
    S.image_by_id[mid] = im
    S.image_model_ids.add(mid)
    if im.kind == "video":
        S.video_model_ids.add(mid)
    if mid not in S.models:
        S.models.append(mid)
    if im.description:
        S.model_descriptions[mid] = im.description
    return im


def _forget_image(S, mid: str) -> None:
    """Démonte un modèle image/vidéo de tous les registres du sélecteur."""
    S.image_by_id.pop(mid, None)
    S.image_model_ids.discard(mid)
    S.video_model_ids.discard(mid)
    S.model_descriptions.pop(mid, None)
    if mid in S.models:
        S.models.remove(mid)


# ---- /rebench : recalibration topologique d'un LOCAL TEXTE (loom.setup réutilisé) ----

# Un seul rebench à la fois : la mesure sature CPU/GPU et exige la VRAM libre.
_REBENCH = {"job": None}


def _model_kind(S, mid: str) -> str | None:
    if mid in S.remote_model_ids:
        return "remote"
    if mid in S.video_model_ids:
        return "video"
    if mid in S.image_model_ids:
        return "image"
    if any(m.get("id") == mid for m in S.local_model_specs):
        return "local"
    return None


def _rebenchable_models(S) -> list[dict]:
    """Modèles calibrables par /rebench : les LOCAUX TEXTE uniquement."""
    return [
        {
            "id": m["id"],
            "label": f"{m['id']} — contexte actuel {m.get('context', '?')}, "
            f"{(m.get('size_mb') or 0) / 1024:.1f} Go",
        }
        for m in S.local_model_specs
    ]


def _run_calibration(S, spec, progress):
    """Cœur de mesure (préconditions + topologie + calibrate), avec les flags EXACTS
    du modèle. Lève RuntimeError actionnable si la machine n'est pas prête.
    Isolé pour être stubbable dans les tests (aucun subprocess en CI)."""
    import os
    import tomllib

    import psutil

    from loom.runtime.gguf_meta import read_gguf_meta
    from loom.setup import bench as bench_mod
    from loom.setup import topology as topo_mod
    from loom.setup.steps import read_raw_config, resolve_bin, server_bin_status
    from loom.web.__main__ import CONFIG_PATH, PERSONAL_CONFIG_PATH

    raw = read_raw_config(CONFIG_PATH, PERSONAL_CONFIG_PATH)
    _, bin_name = server_bin_status(raw)
    server_bin = resolve_bin(bin_name)
    if server_bin is None:
        raise RuntimeError("binaire llama-server introuvable")
    mdir = Path(spec["dir"])
    mt = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
    gguf = mdir / mt["filename"]
    if not gguf.is_file():
        raise RuntimeError(f"GGUF introuvable ({gguf})")
    meta = read_gguf_meta(gguf)
    is_moe = bool(meta.get("expert_count"))
    vram = topo_mod.gpu_vram_total_mb()
    topo = topo_mod.discover_topology(meta, bench_mod.has_gpu_backend(server_bin), vram)
    headroom = int((raw.get("server") or {}).get("gpu_kv_headroom_mb", 640) or 640)
    ram = int(psutil.virtual_memory().total // (1024 * 1024))
    budget = topo_mod.memory_budget_mb(topo, vram, ram, headroom)
    over = raw.get("override") or {}
    # Threads : même résolution que l'exécutant (serve.py) — override machine,
    # sinon cœurs physiques (≈ logiques/2) en GPU, tous les threads en CPU pur.
    logical = os.cpu_count() or 4
    threads = int(
        over.get("threads")
        or (logical if topo == topo_mod.TOPO_RAM else max(1, logical // 2))
    )
    # ngl : la borne PAR MODÈLE (model.toml n_gpu_layers) PRIME — c'est elle qui
    # évite le spill (ex. gemma4 à 36/42 couches). Sinon doctrine MoE (99, experts
    # en RAM), sinon l'override machine.
    if mt.get("n_gpu_layers") is not None:
        ngl = int(mt["n_gpu_layers"])
    elif is_moe and topo != topo_mod.TOPO_RAM:
        ngl = 99
    else:
        ngl = int(over.get("n_gpu_layers", 99 if topo != topo_mod.TOPO_RAM else 0))
    mmproj = mt.get("mmproj_filename")
    probe = topo_mod.ServerProbe(
        server_bin=str(server_bin),
        model_path=str(gguf),
        threads=threads,
        ngl=ngl,
        topology=topo,
        mmproj_path=str(mdir / mmproj) if mmproj else None,
        cpu_moe=bool(mt.get("cpu_moe", is_moe)),
        n_cpu_moe=mt.get("n_cpu_moe"),
    )
    # Sonde d'isolation AVANT la calibration : si le modèle exige un 2e slot,
    # la calibration doit mesurer avec le KV réellement doublé (même séquence
    # que loom-setup step_bench — le conseilleur simule l'exécutant).
    progress("sonde d'isolation du cache (A -> pollution -> A)…")
    isolation = None
    iso_detail = ""
    try:
        first, back = probe.probe_isolation()
        isolation = topo_mod.isolation_needed(first, back)
        iso_detail = f"retour {back}/{first} tokens retraités"
        if isolation:
            probe.n_parallel = 2
    except Exception:  # noqa: BLE001 - sonde best-effort : la calibration vaut sans verdict
        pass
    progress(f"topologie {topo}, budget {budget} Mo")
    calib = topo_mod.calibrate(
        probe, meta, topology=topo, budget_mb=budget, progress=progress
    )
    calib["isolation"] = isolation
    calib["isolation_detail"] = iso_detail
    calib["isolation_avant"] = bool(mt.get("cache_isolation", False))
    return calib, gguf


def _rebench_worker(S, sess, chat_lock, mid, job):
    """Thread du job : mesure, verdict comparé, message PERSISTÉ + état b_apply si
    une application a du sens. `job.done` posé EN DERNIER (le stream lit final)."""
    spec = next((m for m in S.local_model_specs if m.get("id") == mid), None)
    try:
        calib, _gguf = _run_calibration(
            S, spec, lambda m: setattr(job, "label", f"calibration : {m}")
        )
        current = int(spec.get("context") or S.context_window or 0)
        new = calib["context"]
        iso = calib.get("isolation")
        iso_change = iso is not None and iso != calib.get("isolation_avant", False)
        if iso is None:
            iso_line = "sonde d'isolation : illisible (réglage inchangé)."
        elif iso:
            iso_line = (
                f"sonde d'isolation : cache PERDU après pollution du slot "
                f"({calib['isolation_detail']}) -> 2 slots pour ce modèle."
            )
        else:
            iso_line = (
                f"sonde d'isolation : cache survit à la pollution "
                f"({calib['isolation_detail']}) -> 1 slot suffit."
            )
        if new == current and not iso_change:
            msg = (
                f"✅ « {mid} » est déjà au top : contexte actuel {current} = "
                f"mesuré {new} ({calib['mecanisme']}).\n{iso_line}\n"
                "Rien à changer."
            )
            wiz = None
        else:
            changes = []
            if new != current:
                sens = (
                    "amélioration"
                    if new > current
                    else "RÉDUCTION (l'actuel déborde le budget mesuré)"
                )
                changes.append(f"contexte {current} → {new} ({sens})")
            if iso_change:
                changes.append(
                    "cache_isolation → "
                    + ("true (2 slots)" if iso else "false (1 slot)")
                )
            msg = (
                f"Verdict pour « {mid} » : " + " · ".join(changes) + "\n"
                f"(pente {calib['slope_kb_tok']} Ko/token, vitesse validée "
                f"jusqu'à {calib['valide_jusqua']} tokens)\n"
                f"mécanisme : {calib['mecanisme']}\n{iso_line}\n"
                "Tape « oui » pour appliquer — toute autre réponse laisse tout "
                "en l'état."
            )
            wiz = {
                "step": "b_apply",
                "id": mid,
                "context": new,
                "mecanisme": calib["mecanisme"],
                # Verdict d'isolation appliqué EN MÊME TEMPS que le contexte : le
                # contexte a été mesuré avec ce nombre de slots-là — appliquer l'un
                # sans l'autre recréerait un couple (fenêtre, KV) jamais mesuré.
                "isolation": iso if iso_change else None,
                "isolation_detail": calib.get("isolation_detail", ""),
            }
    except (RuntimeError, ValueError) as exc:
        msg = f"❌ Recalibration de « {mid} » échouée : {exc} — config inchangée."
        wiz = None
    got = chat_lock.acquire(timeout=2)
    try:
        conv = sess.conversation
        conv.add("assistant", msg)
        if wiz is not None:
            conv.set_wizard(wiz)
        S.session_store.append_event(sess.id, "text", {"text": msg})
        S.session_store.save(sess)
    finally:
        if got:
            chat_lock.release()
    job.final = msg
    # Boutons du verdict (l'état b_apply attend oui/annuler) — lus par le stream.
    job.choices = ["oui", "annuler"] if wiz is not None else None
    job.done = True


def _persist_wizard_exchange(S, sess, conv, save, message, reply):
    """Chaque étape du wizard est un VRAI échange du fil : persistée dans la
    conversation ET le journal (ré-affichage au rechargement) — exigence spec."""
    conv.add("user", message)
    S.session_store.append_event(sess.id, "user", {"content": message})
    conv.add("assistant", reply)
    S.session_store.append_event(sess.id, "text", {"text": reply})
    save()


def _finish_install(S, sess, chat_lock, mid, mdir, job):
    """Fin de download (appelé DANS le thread du job, succès ou échec) : finalise le
    toml (métadonnées GGUF), monte le modèle, pousse le message de fin dans la
    conversation + le journal — visible même si l'onglet a été fermé entre-temps."""
    if job.error:
        msg = (
            f"Échec du téléchargement de « {mid} » :\n{job.error}\n"
            "Relance /add-model, ou pose le fichier à la main — la reprise est "
            "automatique au premier lancement du modèle."
        )
    else:
        meta = model_install.finalize_model_toml(mdir, Path(mdir) / job.filenames[0])
        _mount_local(
            S,
            mid,
            mdir,
            job.total_mb,
            vision=any("mmproj" in f.lower() for f in job.filenames),
        )
        extras = []
        if meta.get("n_layers"):
            extras.append(f"{meta['n_layers']} couches")
        if meta.get("expert_count"):
            extras.append("MoE détecté -> cpu_moe = true")
        det = f" ({', '.join(extras)})" if extras else ""
        msg = f"Modèle « {mid} » installé{det} — disponible dans le sélecteur."
    job.final_message = msg
    # Écriture de la conversation sous le verrou de session pour ne pas courir contre
    # une génération. Acquire à timeout COURT avec repli : on écrit de toute façon
    # (le journal est append-only, le risque réel est borné) — et dans le cas
    # synchrone (download instantané) le verrou est encore tenu par la requête du
    # wizard, inutile d'attendre longtemps.
    got = chat_lock.acquire(timeout=2)
    try:
        conv = sess.conversation
        conv.add("assistant", msg)
        S.session_store.append_event(sess.id, "text", {"text": msg})
        S.session_store.save(sess)
    finally:
        if got:
            chat_lock.release()


def _handle_add_model_command(S, message, conv, sess, save, chat_lock):
    """Intercepte /add-model ET tout message d'une session au wizard actif.
    Même contrat que _handle_goal_command : (message, response) — response non None
    = ack SSE immédiat, le flux LLM n'est jamais sollicité (wizard déterministe)."""
    active = bool(conv.wizard)
    is_cmd = message == "/add-model" or message.startswith("/add-model ")
    is_rm = message == "/remove-model" or message.startswith("/remove-model ")
    is_rb = message == "/rebench" or message.startswith("/rebench ")
    if not (active or is_cmd or is_rm or is_rb):
        return message, None

    # Étape AVANT transition : si c'était la saisie de la clé (r_key), le message
    # EST la clé — on la MASQUE dans tout ce qui persiste (conversation, journal).
    prev_step = (conv.wizard or {}).get("step") if active and not is_cmd else None

    try:
        deps = _wizard_deps(S)
        if is_rb:
            res = _wizard.start_rebench(message[len("/rebench") :].strip(), deps)
        elif is_rm:
            res = _wizard.start_remove(deps)
        elif is_cmd:
            res = _wizard.start(message[len("/add-model") :].strip(), deps)
        else:
            res = _wizard.step(conv.wizard, message, deps)
    except Exception as exc:  # noqa: BLE001 - HfCatalogError & co : actionnable, jamais de stacktrace
        res = _wizard.WizardResult(
            conv.wizard,
            f"{exc}\n(l'assistant reste à cette étape — réessaie, ou /cancel)",
        )

    conv.set_wizard(res.state)
    shown = message
    if prev_step == "r_key" and message.strip().lower() not in ("aucune", "none", "-"):
        shown = "•••••••• (clé masquée)"

    job = None
    rb_job = None
    extra_reply = ""
    models_changed = False
    if res.action and res.action["kind"] == "rebench":
        cur = _REBENCH["job"]
        if cur is not None and not cur.done:
            extra_reply = (
                "\n⏳ Une calibration est déjà en cours — attends sa fin "
                "(le verdict s'affichera dans son fil)."
            )
        else:
            from types import SimpleNamespace

            # Le banc charge le modèle lui-même : il lui faut la VRAM -> serveur off.
            S.server_manager.stop()
            _client_mark_all_cold(S)
            rb_job = SimpleNamespace(done=False, label="", final=None)
            _REBENCH["job"] = rb_job
            threading.Thread(
                target=_rebench_worker,
                args=(S, sess, chat_lock, res.action["id"], rb_job),
                daemon=True,
                name="loom-rebench",
            ).start()
    elif res.action and res.action["kind"] == "rebench_apply":
        a = res.action
        spec = next((m for m in S.local_model_specs if m.get("id") == a["id"]), None)
        if spec is None:
            extra_reply = f"\n❌ Modèle « {a['id']} » introuvable — rien d'appliqué."
        else:
            import tomllib

            from loom.setup.cli import _set_model_cache_isolation, _set_model_context

            mdir = Path(spec["dir"])
            mt = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
            gguf = mdir / mt["filename"]
            _set_model_context(gguf, a["context"], a["mecanisme"])
            applied = f"contexte {a['context']}"
            if a.get("isolation") is not None:
                _set_model_cache_isolation(
                    gguf, a["isolation"], a.get("isolation_detail", "")
                )
                applied += f" + cache_isolation={'true' if a['isolation'] else 'false'}"
            spec["context"] = a["context"]
            S.model_contexts[a["id"]] = a["context"]
            _regen_swap_yaml(S)
            extra_reply = (
                f"\n✅ {applied} écrit dans le model.toml de "
                f"« {a['id']} » — effet au prochain chargement du modèle."
            )
    elif res.action and res.action["kind"] == "upsert_remote":
        rec = {k: v for k, v in res.action["record"].items() if v is not None}
        roots = _models_roots(S)
        if roots:
            with S.toml_lock:
                model_store.write_remote_dir(roots[0], rec)
            if getattr(S, "client", None) is not None:
                _mount_remote(S, rec)
            models_changed = True
        else:
            extra_reply = "\n(racine des modèles indisponible : ajout NON persisté)"
    elif res.action and res.action["kind"] == "remove":
        a = res.action
        if a["model_kind"] in ("remote", "remote_config"):
            # Un distant = son dossier remote/<id> (purgé sur TOUTES les racines) ;
            # les emplacements hérités (local.toml, store JSON) sont nettoyés en
            # filet, sans jamais recréer un fichier vide.
            with S.toml_lock:
                model_store.delete_remote_dir(_models_roots(S), a["id"])
                if S.config_local_path:
                    model_store.delete_remote_in_toml(S.config_local_path, a["id"])
                if S.remote_store_path and Path(S.remote_store_path).exists():
                    model_store.delete(S.remote_store_path, a["id"])
            _forget_remote(S, a["id"])
            extra_reply = f"\n✅ « {a['id']} » retiré (dossier remote/ + sélecteur)."
            models_changed = True
        elif a["model_kind"] in ("image", "video"):
            import shutil

            im = S.image_by_id.get(a["id"])
            try:
                if im is not None:
                    shutil.rmtree(im.dir)
                _forget_image(S, a["id"])
                extra_reply = (
                    f"\n✅ Définition de « {a['id']} » supprimée (sélecteur compris) ; "
                    "les poids ComfyUI partagés ne sont PAS touchés."
                )
                models_changed = True
            except OSError as exc:
                extra_reply = f"\n❌ Suppression impossible : {exc}"
        else:
            import shutil

            spec = next(
                (m for m in S.local_model_specs if m.get("id") == a["id"]), None
            )
            try:
                if spec and spec.get("dir"):
                    shutil.rmtree(spec["dir"])
                S.local_model_specs[:] = [
                    m for m in S.local_model_specs if m.get("id") != a["id"]
                ]
                if a["id"] in S.local_model_ids:
                    S.local_model_ids.remove(a["id"])
                if a["id"] in S.models:
                    S.models.remove(a["id"])
                S.vision_models.discard(a["id"])
                _regen_swap_yaml(S)
                extra_reply = f"\n✅ « {a['id']} » supprimé du disque et du sélecteur."
                models_changed = True
            except PermissionError:
                # Windows verrouille un GGUF chargé (mmap llama-server) : on ne
                # touche à RIEN et on guide — pas de suppression partielle.
                extra_reply = (
                    f"\n❌ Fichiers de « {a['id']} » verrouillés — le modèle est "
                    "probablement CHARGÉ. Éteins le serveur modèle (ou charge un "
                    "autre modèle), puis relance /remove-model."
                )
            except OSError as exc:
                extra_reply = f"\n❌ Suppression impossible : {exc}"
    elif res.action and res.action["kind"] == "mount_image":
        a = res.action
        im = _mount_image(S, a["model_kind"], a["id"])
        extra_reply = (
            f"\n✅ « {a['id']} » monté — disponible dans le sélecteur."
            if im is not None
            else f"\n❌ Dossier de « {a['id']} » introuvable ou incomplet — "
            "relance /add-model."
        )
        models_changed = im is not None
    elif res.action and res.action["kind"] == "install_image":
        a = res.action
        base = _image_base_dir(S, a["model_kind"])
        mdir = base / a["model_id"]
        try:
            mdir.mkdir(parents=True, exist_ok=True)
            if not (mdir / "model.toml").is_file():  # reprise : ne pas écraser
                desc = a["description"].replace('"', "'")
                tmpl = next(iter(S.image_by_id.values()), None)
                from loom.runtime.image_models import default_comfy_dir

                comfy_dir = tmpl.comfy_dir if tmpl else default_comfy_dir()
                comfy_port = tmpl.comfy_port if tmpl else 8188
                refiner = tmpl.refiner if tmpl else ""
                timeout = 3600 if a["model_kind"] == "video" else 600
                (mdir / "model.toml").write_text(
                    f'label = "{a["model_id"]}"\n'
                    f"width = {a['width']}\nheight = {a['height']}\n"
                    f'comfy_dir = "{comfy_dir}"\ncomfy_port = {comfy_port}\n'
                    f'refiner = "{refiner}"\ntimeout = {timeout}\n'
                    f'description = "{desc}"\n',
                    encoding="utf-8",
                )
            if a["workflow_path"]:
                import shutil

                shutil.copyfile(a["workflow_path"], mdir / "workflow.json")
                im = _mount_image(S, a["model_kind"], a["model_id"])
                extra_reply = (
                    f"\n✅ « {a['model_id']} » créé et monté — disponible dans le "
                    "sélecteur. (Les poids ComfyUI ne sont pas gérés par Loom : la "
                    "recette doit référencer des checkpoints déjà présents côté "
                    "ComfyUI.)"
                    if im is not None
                    else f"\n❌ Recette copiée mais montage impossible — vérifie {mdir}."
                )
                models_changed = im is not None
            else:
                extra_reply = (
                    f"\n📁 Dossier préparé : {mdir}\nDépose ton export ComfyUI "
                    "(format API) sous le nom workflow.json, puis relance "
                    f"/add-model {a['model_kind']} avec le même id "
                    f"« {a['model_id']} » pour le monter (sinon il sera découvert "
                    "au prochain démarrage)."
                )
        except OSError as exc:
            extra_reply = f"\n❌ Création impossible : {exc}"
    elif res.action and res.action["kind"] == "install":
        a = res.action
        if not S.models_dir:
            extra_reply = (
                "\n(models_dir non configuré : installation locale indisponible)"
            )
        else:
            # `root` : racine choisie à l'étape disque du wizard (multi-racines) ;
            # sans elle, racine prioritaire = S.models_dir (<racine[0]>/local/text).
            base = (
                Path(a["root"]) / "local" / "text"
                if a.get("root")
                else Path(S.models_dir)
            )
            mdir = base / a["model_id"]
            files = list(a["files"])
            if a.get("mmproj_filename"):
                files.append(a["mmproj_filename"])
            model_install.write_model_toml(
                mdir,
                a["repo"],
                a["filename"],
                a["size_mb"],
                mmproj_filename=a.get("mmproj_filename"),
            )
            job = model_install.start_download(
                a["repo"],
                files,
                mdir,
                a["size_mb"],
                on_done=lambda j: _finish_install(
                    S, sess, chat_lock, a["model_id"], mdir, j
                ),
            )

    # Persistance APRÈS les actions : le résultat (✅/❌ d'extra_reply) fait partie
    # de l'échange — sinon il n'existait qu'en SSE et disparaissait au rechargement
    # du fil (vécu : « Suppression de … » sans verdict après F5).
    _persist_wizard_exchange(S, sess, conv, save, shown, res.reply + extra_reply)

    chat_lock.release()

    def _stream():
        yield _sse("text", text=res.reply + extra_reply)
        # Boutons de réponse (confort : purs raccourcis de frappe, cf. wizard.choices).
        if res.choices:
            yield _sse("choices", options=res.choices)
        # Un modèle vient d'être monté/retiré à chaud -> le front recharge le
        # sélecteur (vécu : « disponible dans le sélecteur »… qui ne l'affichait pas).
        if models_changed:
            yield _sse("models")
        if rb_job is not None:
            # Calibration en fond : progression live, verdict déjà PERSISTÉ par le
            # worker (visible même si ce flux est coupé/onglet fermé).
            while not rb_job.done:
                yield _sse("status", label=rb_job.label or "calibration en cours…")
                time.sleep(2)
            yield _sse("status", label="")
            if rb_job.final:
                yield _sse("text", text="\n" + rb_job.final)
            if getattr(rb_job, "choices", None):
                yield _sse("choices", options=rb_job.choices)
        if job is not None:
            while not job.done:
                yield _sse(
                    "status",
                    label=f"téléchargement… {job.progress_mb()}/{job.total_mb} Mo",
                )
                time.sleep(2)
            yield _sse("status", label="")
            if job.final_message:
                yield _sse("text", text="\n" + job.final_message)
            yield _sse("models")
        yield _sse("done")

    return message, Response(_stream(), mimetype="text/event-stream")


# ---- System prompt --------------------------------------------------------------------


def _build_system_prompt(S, conv):
    """Construit le system prompt complet : identité always-on + base (strong/local) +
    catalogue des skills + déclaration du moteur + conventions OS + dossier de travail +
    objectif de session. Retourne (system_prompt, strong)."""

    skills = effective_skills(
        _all_skills(S),
        overrides=conv.skill_overrides,
        disabled=conv.disabled_skills,
    )

    catalog = render_catalog(skills)

    # Identité always-on (SOUL/USER/MEMORY) EN TÊTE : c'est la définition qui FAIT FOI
    # de qui est Loom (rôle, persona, style). Le mode d'emploi opérationnel (outils,
    # règles) de chat.system.md vient APRÈS et s'y conforme - on ne plante plus un
    # cadrage générique d'abord pour le corriger 12k caractères plus loin. Always-on =>
    # survit toujours à la microcompaction/summarization (qui ne touchent que
    # l'historique). Bornée par identity_max_tokens. Cf. design §5.6.
    _idblk = ""

    if S.identity_paths:
        from loom.memory.identity import identity_block

        _idblk = identity_block(
            S.identity_paths["soul_path"],
            S.identity_paths["user_path"],
            S.identity_paths["memory_md_path"],
            max_tokens=S.settings["identity_max_tokens"],
        )

    # TIER du harnais : un modèle DISTANT (API, non quantifié) se pilote seul -> prompt
    # ALLÉGÉ (identité + outils + mémoire + sécurité), sans le scaffolding de comportement
    # de chat.system.md qui ne sert qu'à un petit modèle local. Le flag `strong` sert
    # aussi (plus bas) à couper les gardes de comportement dans la boucle d'outils.
    strong = bool(
        conv.model
        and conv.model in S.remote_model_ids
        and conv.model not in S.remote_weak_ids
    )

    base_prompt = CHAT_SYSTEM_STRONG if strong else conv.system_prompt

    system_prompt = f"{_idblk}\n\n{base_prompt}" if _idblk else base_prompt

    # Distant (strong) : la machine du provider encaisse le parallélisme -> on incite à
    # GROUPER les sous-agents indépendants dans un même tour (ils tournent en parallèle).
    if strong:
        system_prompt += (
            "\n\nParallélisme : quand plusieurs sous-tâches sont INDÉPENDANTES (auditer/"
            "explorer des pans distincts), émets PLUSIEURS dispatch_agent dans le MÊME "
            "tour - ils s'exécutent EN PARALLÈLE, bien plus vite qu'un par tour. Un pan = "
            "un agent, lance-les ensemble."
        )

    if catalog:
        system_prompt += f"\n\n{catalog}"

    # Le modèle ignore par défaut sous quel backend il tourne (le prompt dit
    # "Tu es Loom") -> il baratine quand on lui demande "quel modèle ?". On lui
    # injecte son modèle courant pour qu'il réponde honnêtement. DISTANT vs LOCAL :
    # sans ça un modèle servi par une API répétait « je tourne en local/offline sur
    # llama.cpp » (la persona de Loom est « agent local ») -> confabulation d'infra.
    if conv.model:
        if conv.model in S.remote_model_ids:
            _pm = S.remote_model_names.get(conv.model)

            _label = (
                f"« {_pm} » (route « {conv.model} »)" if _pm else f"« {conv.model} »"
            )

            system_prompt += (
                f"\n\n# Ton moteur\nTon raisonnement est servi par le modèle DISTANT "
                f"{_label}, via une API externe - PAS en local. Tes OUTILS, eux, "
                "s'exécutent bien sur la machine de l'utilisateur, mais toi (le cerveau) "
                "non. Ne prétends donc JAMAIS être offline, ni tourner sur llama.cpp / "
                "llama-swap / une carte graphique locale : ce serait faux. Si on te "
                "demande quel modèle/moteur tu utilises, donne ce nom honnêtement, sans "
                "inventer de détails d'infrastructure."
            )

        else:
            system_prompt += (
                f"\n\n# Ton moteur\nTu tournes sur le modèle local « {conv.model} ». "
                "Si on te demande quel modèle/moteur tu utilises, réponds-le "
                "honnêtement et directement (ce nom), sans esquiver."
            )

    # Système : Loom détecte SEUL l'OS et injecte ses conventions (shell, commandes,
    # chemins) -> le modèle produit du PowerShell sous Windows, du bash/unix sous
    # macOS/Linux, sans qu'on code l'OS en dur dans le prompt. Source unique partagée
    # avec run_shell (loom.runtime.platform_info) : jamais de divergence.
    system_prompt += "\n\n" + platform_detect().prompt_block()

    # Dossier de travail courant : le modèle l'IGNORE sinon et le devine en sondant
    # (git rev-parse à l'aveugle, list_dir…) -> tours gaspillés. On le lui dit, avec
    # le réflexe anti-tâtonnement quand ce dossier n'est pas un repo git. Reste EN BAS
    # (contexte volatil, près de l'action).
    _ws = _session(S).workspace

    system_prompt += (
        f"\n\n# Dossier de travail courant\nTes commandes (run_shell) tournent dans "
        f"`{_ws}` et les chemins relatifs s'y résolvent - n'y répète pas le nom de ce "
        "dossier dans tes chemins. Si une commande git échoue par « not a git "
        "repository », c'est que CE dossier n'est pas un repo : fais UN list_dir pour "
        "repérer le bon sous-dossier (puis `git -C <sous-dossier>`), ne relance pas la "
        "même commande à l'identique."
    )

    # Fiche projet auto-injectée : si `<workspace>/loom.md` existe (générée par /init),
    # le modèle la reçoit d'office au lieu de re-sonder le projet à chaque session.
    # Cache mtime (read_md) -> préfixe stable en session = prompt caching préservé ;
    # suit le workspace courant (adoption/changement en cours de session). Les DEUX
    # tiers la reçoivent (c'est de la mémoire, pas du scaffolding de comportement).
    # L'en-tête la cadre en CONTEXTE (pas instructions, possiblement périmée) : une
    # fiche est écrite en lisant le projet, un repo piégé ne doit pas pouvoir élever
    # ses consignes au rang de system prompt.
    from loom.memory.identity import project_block

    _pm_blk = project_block(_ws, max_tokens=S.settings["project_memory_max_tokens"])

    if _pm_blk:
        system_prompt += f"\n\n{_pm_blk}"

    # Objectif de session (/goal), en DIRECTIVE DOUCE : pas de juge externe qui te
    # contredit (retiré - il recalait des preuves correctes). Tu restes seul maître de
    # ta propre vérification : ne te déclare pas fini tant que l'objectif n'est pas
    # ATTEINT ET PROUVÉ par tes exécutions (montre la sortie réelle) ; une fois prouvé,
    # dis-le et arrête-toi. L'utilisateur l'efface avec « /goal clear ».
    if conv.goal:
        system_prompt += (
            f"\n\n# Objectif de session\nTant qu'il est actif, oriente ton travail vers "
            f"cet objectif et ne le déclare atteint qu'une fois PROUVÉ par tes propres "
            f"exécutions (sortie réelle affichée) :\n{conv.goal}"
        )

    return system_prompt, strong


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
        system_prompt, _strong = _build_system_prompt(S, conv)
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


# ---- Routes : socle (index, statiques, toggles) ---------------------------------------


def _register_misc_routes(app, S):
    @app.get("/")
    def index() -> str:
        return render_template("index.html", **_index_context(S))

    @app.get("/genimg/<sid>/<name>")
    def genimg_session(sid: str, name: str):
        # Sert les médias générés depuis le dossier de LA session (unique copie).
        # `sid` compose le CHEMIN de base : validé strictement (12 hex, format
        # uuid4.hex[:12] de SessionStore.create) sinon 404 — un sid forgé (`..`,
        # antislash Windows) ferait pointer la base hors de var/sessions.
        # send_from_directory protège `name` ; mimetype déduit du nom (png/webm).
        from flask import send_from_directory

        if not re.fullmatch(r"[0-9a-f]{12}", sid):
            return Response("session invalide", status=404)
        return send_from_directory(S.session_store.root / sid / "generated", name)

    @app.get("/genimg/<name>")
    def genimg(name: str):
        # LEGACY : messages d'avant 2026-07-09, servis depuis var/generated.
        from flask import send_from_directory

        return send_from_directory(S.generated_dir, name)

    @app.get("/favicon.ico")
    def favicon():
        # Requête par défaut du navigateur (silence le 404) : on sert le SVG de la trame.
        # Le <link rel="icon" type="image/svg+xml"> reste la source primaire de l'onglet.
        from flask import send_from_directory

        return send_from_directory(
            app.static_folder, "favicon.svg", mimetype="image/svg+xml"
        )

    @app.post("/tool_decision")
    def tool_decision():
        pend = S.pending.get(request.form.get("id", ""))

        if pend is not None:
            pend["approved"] = request.form.get("approve") == "1"

            pend["event"].set()

        return Response("", status=204)

    @app.post("/tools")
    def tools_update():
        conv, save = _ctx(S)

        conv.set_tools(request.form.getlist("tool"))

        save()

        return render_template(
            "_tools.html",
            available_tools=S.available_tools,
            active_tools=conv.active_tools,
        )

    @app.post("/thinking")
    def thinking_update():
        conv, save = _ctx(S)

        conv.set_thinking(request.form.get("thinking") == "1")

        save()

        return Response(str(int(conv.thinking)), mimetype="text/plain")

    @app.post("/local_only")
    def local_only_update():
        # Session PRIVÉE : coupe tout routage distant des sous-agents (chaîne
        # dispatch_models ignorée). Décision humaine par session, persistée.
        conv, save = _ctx(S)

        conv.set_local_only(request.form.get("local_only") == "1")

        save()

        return Response(str(int(conv.local_only)), mimetype="text/plain")

    @app.route("/pick-folder", methods=["POST"])
    def pick_folder():
        # Sous-processus : évite les soucis tkinter hors du thread principal de Flask.

        script = (
            "import tkinter, tkinter.filedialog as fd;"
            "r=tkinter.Tk(); r.withdraw(); r.attributes('-topmost', True);"
            "p=fd.askdirectory(); print(p if p else '')"
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
            )

        except Exception as exc:  # noqa: BLE001 - tkinter absent / timeout
            return {"path": "", "error": str(exc)[:200]}

        path = (proc.stdout or "").strip()

        if proc.returncode != 0 and not path:
            return {
                "path": "",
                "error": (proc.stderr or "sélecteur indisponible")[:200],
            }

        return {"path": path}

    @app.get("/sysmon")
    def sysmon_metrics():
        # Métriques système LIVE (CPU/RAM/GPU) pour le moniteur affiché avec un modèle LOCAL.
        # nvidia-smi + psutil ; champs à None si une source manque (le front s'adapte).
        from loom.runtime.sysmon import read_metrics

        return read_metrics()


# ---- Routes : sessions (fil, notes, fork, compaction) ---------------------------------


def _register_session_routes(app, S):
    @app.post("/reset")
    def reset() -> str:
        conv, save = _ctx(S)

        conv.reset()

        save()

        # Le fil repart à neuf -> on efface aussi le journal d'affichage temps réel.
        S.session_store.clear_timeline(_session(S).id)

        return render_template("index.html", **_index_context(S))

    @app.post("/fork")
    def fork():
        """Repart d'un message utilisateur : tronque l'historique APRES ce message (exclus),

        renvoie son texte pour pre-remplir l'input. user_index = N-ieme message user (0-based)
        COMPTÉ SUR L'AFFICHAGE — or la compaction fusionne les vieux tours côté serveur, donc
        l'index peut avoir glissé : on retrouve alors le message par son CONTENU (`text`,
        envoyé par l'UI), et sinon on répond une erreur EXPLICITE (plus d'échec muet)."""

        user_index = int(request.form.get("user_index", "-1"))
        ui_text = (request.form.get("text") or "").strip()

        # Cible la session de l'ONGLET appelant (session_id), pas la session focus
        # globale : avec la split view, le bouton « repartir » existe dans des panneaux
        # non-focus, et la resynchro du focus (/session/activate) COURT contre ce POST —
        # sans session_id, /fork pouvait tronquer une AUTRE session que celle affichée.
        # Un session_id fourni mais INCONNU est une erreur franche : surtout pas de
        # repli silencieux sur la session focus (ce serait re-tronquer la mauvaise).
        req_sid = (request.form.get("session_id") or "").strip()
        if req_sid:
            sess = _get_session(S, req_sid)
            if sess is None:
                return Response("session inconnue", status=404)
        else:
            sess = _session(S)
        conv, save = sess.conversation, (lambda: S.session_store.save(sess))

        msgs = conv.messages

        def _as_text(content) -> str:
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                ).strip()
            return str(content).strip()

        # Trouve le N-ieme message user dans l'historique persiste

        user_msgs = [i for i, m in enumerate(msgs) if m.get("role") == "user"]

        target_idx = None
        if 0 <= user_index < len(user_msgs):
            cand = user_msgs[user_index]
            # L'index ne vaut que si le contenu correspond encore (pas de glissement).
            if not ui_text or _as_text(msgs[cand].get("content", "")) == ui_text:
                target_idx = cand
        if target_idx is None and ui_text:
            # Index périmé (compaction) : dernier message user au MÊME contenu.
            for i in reversed(user_msgs):
                if _as_text(msgs[i].get("content", "")) == ui_text:
                    target_idx = i
                    break
        if target_idx is None:
            return Response(
                "ce message a été résumé par la compaction : « repartir » n'est "
                "possible que depuis un message encore présent dans le contexte "
                "(les plus récents).",
                status=400,
            )

        content = msgs[target_idx].get("content", "")

        if isinstance(content, list):
            text = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )

        else:
            text = str(content)

        # Tronque : garde jusqu'au message user (inclus), efface la suite

        conv.messages = msgs[: target_idx + 1]

        save()

        return {"text": text}

    @app.post("/compact")
    def compact():
        """Compaction MANUELLE (bouton près de la jauge de contexte) : résume les vieux
        tours de la session ciblée en un bloc dense et remplace l'historique. session_id =
        onglet ciblé, sinon la session focus. Refuse si une génération tourne (le verrou de
        session est pris) — on ne mute pas l'historique sous une boucle active. Renvoie les
        compteurs d'usage à jour (jauge de contexte ré-estimée) + `collapsed`."""
        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(S, req_sid) if req_sid else _session(S)
        if sess is None:
            return Response("session introuvable", status=404)
        lock = _lock_for(S, sess.id)
        if not lock.acquire(blocking=False):
            return Response("occupé : cette session génère déjà", status=429)
        try:
            conv = sess.conversation
            # DÉTERMINISTE et INSTANTANÉ (aucun appel modèle -> pas de blocage de plusieurs
            # minutes). Cible = prompt système (INCOMPRESSIBLE) + ~4000 car. (~1,3k tokens)
            # de conversation : on clippe le reste de l'historique.
            target_chars = len(conv.system_prompt) + 4000
            new_msgs, freed = S.client.compact_conversation(
                conv.messages,
                system_prompt=conv.system_prompt,
                target_chars=target_chars,
            )
            if freed:
                conv.messages = new_msgs
                # `freed` = tokens réellement libérés (delta de la conversation, le prompt
                # système s'annule). On le RETRANCHE du ctx réel du dernier appel -> jauge à
                # jour immédiatement ET exacte (pas une ré-estimation qui oublierait les
                # schémas d'outils). L'usage réel du prochain tour confirmera au token près.
                conv.context_tokens = max(0, conv.context_tokens - freed)
                S.session_store.save(sess)
        finally:
            lock.release()
        return {**_totals(S, conv), "collapsed": freed}

    @app.post("/cancel")
    def cancel():
        # Bouton Stop : pose le signal d'annulation de LA session ciblée (par session_id, sinon
        # la session focus) -> SA boucle /chat s'arrête net et libère son verrou. Les AUTRES
        # sessions (onglets) ne sont PAS touchées. Sans effet si rien ne tourne pour elle.

        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(S, req_sid) if req_sid else S.cur["session"]
        if sess is not None:
            _cancel_for(S, sess.id).set()

        return Response("", status=204)

    @app.post("/note")
    def note():
        # Note en vol (« btw » natif) : remarque envoyée PENDANT une génération.
        # Mise en file par session ; la boucle tool-use l'injecte au prochain point
        # d'arrêt (avant l'appel modèle suivant) SANS interrompre le tour. Si le
        # tour se termine avant l'injection, la note reste en file et part au début
        # du tour suivant — jamais perdue.
        text = (request.form.get("text") or "").strip()
        if not text:
            return {"error": "note vide"}, 400
        if len(text) > _NOTE_MAX_CHARS:
            return {
                "error": f"note trop longue (max {_NOTE_MAX_CHARS} caractères)"
            }, 413
        req_sid = (request.form.get("session_id") or "").strip()
        sess = _get_session(S, req_sid) if req_sid else S.cur["session"]
        if sess is None:
            return {"error": "session inconnue"}, 404
        queued = S.notes.push(sess.id, text)
        if queued < 0:
            return {
                "error": "file de notes pleine — attendre le prochain point d'arrêt"
            }, 429
        return {"ok": True, "queued": queued}

    # --- Sessions : liste / nouvelle / bascule / suppression ---

    @app.get("/sessions")
    def sessions_list():
        active = _session(S).id

        return {
            "sessions": [
                {"id": m.id, "title": m.title, "workspace": m.workspace}
                for m in S.session_store.list()
            ],
            "active": active,
        }

    @app.get("/session_state")
    def session_state():
        # État CLIENT d'une session, pour OUVRIR un onglet sans recharger la page : messages,
        # modèle, thinking, workspace, outils actifs, compteur. Le multi-onglets s'appuie
        # dessus (chaque onglet hydrate sa session à l'ouverture).
        sid = (request.args.get("id") or "").strip()
        sess = _get_session(S, sid)
        if sess is None:
            return Response("session inconnue", status=404)
        conv = sess.conversation
        return {
            "id": sess.id,
            "title": sess.title,
            "workspace": sess.workspace,
            "messages": conv.messages,
            "thinking": conv.thinking,
            "local_only": conv.local_only,
            "model": conv.model,
            "active_tools": conv.active_tools,
            "usage_totals": _totals(S, conv),
            # A-t-on un journal d'affichage à rejouer ? (sinon l'UI retombe sur `messages`).
            "has_timeline": bool(S.session_store.read_timeline(sess.id)),
        }

    @app.get("/session/<sid>/timeline")
    def session_timeline(sid):
        """Journal d'affichage temps réel d'une session, pour REJOUER l'UI au rechargement
        (raisonnement, texte, cartes d'outils exactement comme en direct). Les chunks 'text'/
        'reasoning' consécutifs sont recollés pour un rejeu léger."""
        out: list[dict] = []
        for e in S.session_store.read_timeline(sid):
            ev = e.get("event")
            d = e.get("data") or {}
            if ev in ("text", "reasoning") and out and out[-1].get("event") == ev:
                out[-1]["data"]["text"] = (out[-1]["data"].get("text") or "") + (
                    d.get("text") or ""
                )
            else:
                out.append({"event": ev, "data": dict(d)})
        return {"events": out}

    @app.post("/session/new")
    def session_new():
        ws = (request.form.get("workspace") or "").strip() or S.workspace_dir

        title = (request.form.get("title") or "").strip()

        # Sessions FANTÔMES : créées puis jamais utilisées (« Nouvelle session » et
        # zéro message) — balayées quand on en crée une nouvelle : elles ne font que
        # polluer la sidebar. On épargne toute session dont le verrou de génération
        # est tenu (un tour vient peut-être de démarrer).
        for meta in S.session_store.list():
            if meta.title != "Nouvelle session":
                continue
            _lk = S.sess_locks.get(meta.id)
            if _lk is not None and _lk.locked():
                continue
            ghost = _get_session(S, meta.id)
            if ghost is not None and not ghost.conversation.messages:
                S.session_store.delete(meta.id)
                with S.gen_guard:
                    S.sessions_cache.pop(meta.id, None)
                if S.cur["session"] is not None and S.cur["session"].id == meta.id:
                    S.cur["session"] = None

        sess = S.session_store.create(workspace=ws, title=title)

        with S.gen_guard:
            S.sessions_cache[sess.id] = sess

        S.cur["session"] = sess

        # Amorce du cache KV en fond (si le serveur modèle tourne déjà) : le préfixe
        # statique de la session neuve se préfille pendant que l'utilisateur tape.
        _prime_async(S, _ensure_model(S, sess), require_running=True)

        return {"id": sess.id, "title": sess.title, "workspace": sess.workspace}

    # ---- Export / import de session (.zip clair, cf. SessionStore.export_zip) ----

    @app.get("/session/<sid>/export")
    def session_export(sid):
        data = S.session_store.export_zip(sid)
        if data is None:
            return {"error": "session inconnue"}, 404
        meta = S.session_store.load(sid)
        slug = (
            re.sub(r"[^A-Za-z0-9._-]+", "-", meta.title if meta else "session").strip(
                "-."
            )[:40]
            or "session"
        )
        return Response(
            data,
            mimetype="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="loom-session-{slug}-{sid}.zip"'
                )
            },
        )

    @app.post("/session/import")
    def session_import():
        f = request.files.get("file")
        if f is None:
            return {"error": "fichier manquant (champ multipart « file »)"}, 400
        data = f.read(S.session_store.MAX_IMPORT_BYTES + 1)
        if len(data) > S.session_store.MAX_IMPORT_BYTES:
            return {"error": "archive trop grosse — import refusé"}, 413
        try:
            sess = S.session_store.import_zip(data)
        except ValueError as exc:
            return {"error": str(exc)}, 400
        # Même prise en charge qu'une session neuve : cache, focus, prime en fond.
        with S.gen_guard:
            S.sessions_cache[sess.id] = sess
        S.cur["session"] = sess
        _prime_async(S, _ensure_model(S, sess), require_running=True)
        return {
            "id": sess.id,
            "title": sess.title,
            "workspace": sess.workspace,
            "model": sess.conversation.model,
        }

    @app.post("/session/activate")
    def session_activate():
        sid = (request.form.get("id") or "").strip()

        loaded = _get_session(S, sid)

        if loaded is None:
            return Response("session inconnue", status=404)

        S.session_store.set_active(sid)

        S.cur["session"] = loaded

        # Bascule = intention de reprendre CE fil : on ré-amorce son préfixe (fil
        # complet) en fond si le serveur tourne, sans jamais le démarrer pour ça.
        _prime_async(S, loaded, require_running=True)

        return {"id": loaded.id}

    @app.post("/session/workspace")
    def session_workspace():
        # Réaffecte le dossier de travail de la SESSION ACTIVE (appelé par le sélecteur

        # de dossier). Sans ça, choisir un dossier ne s'appliquerait qu'à la création

        # d'une nouvelle session -> les outils continueraient de cibler l'ancien.

        ws = (request.form.get("workspace") or "").strip()

        if not ws:
            return Response("workspace manquant", status=400)

        # Cible la session de l'ONGLET appelant (session_id), pas la session focus
        # globale : avec le multi-onglets, _session() peut désigner un autre fil ->
        # le dossier choisi s'écrivait ailleurs et le tour partait sur l'ancien
        # workspace (bug constaté le 2026-07-10).
        req_sid = (request.form.get("session_id") or "").strip()
        sess = (_get_session(S, req_sid) if req_sid else None) or _session(S)

        sess.workspace = ws

        S.session_store.save(sess)

        return {"workspace": sess.workspace}

    @app.post("/session/delete")
    def session_delete():
        sid = (request.form.get("id") or "").strip()

        S.session_store.delete(sid)

        # Si on supprime la session courante, on recharge l'active (ou on en crée une).

        if S.cur["session"] is not None and S.cur["session"].id == sid:
            S.cur["session"] = None

            _session(S)

        return {"ok": True}


# ---- Routes : skills -------------------------------------------------------------------


def _register_skill_routes(app, S):
    @app.post("/skills")
    def skills_update():
        # Toggle des skills (façon /tools) : le formulaire porte les skills COCHÉS. Les
        # décochés (tous les autres) deviennent `disabled_skills` de la session -> retirés
        # du catalogue et de use_skill. Re-render le panneau (case maître incluse).
        conv, save = _ctx(S)
        enabled = set(request.form.getlist("skill"))
        all_names = [s.name for s in _all_skills(S)]
        conv.set_disabled_skills([n for n in all_names if n not in enabled])
        save()
        return render_template("_skills.html", **_skills_ctx(S, conv))

    @app.get("/skill")
    def skill_get():
        # Source d'un skill pour l'éditeur : texte brut du SKILL.md, ou l'override de session
        # s'il existe (ce que le modèle voit réellement pour cette session).
        conv, _ = _ctx(S)
        name = request.args.get("name", "")
        skill = next((s for s in _all_skills(S) if s.name == name), None)
        if skill is None:
            return {"error": f"skill inconnu : {name}"}, 404
        override = conv.skill_overrides.get(name)
        return {
            "name": skill.name,
            "description": skill.description,
            "source": override if override is not None else read_skill_source(skill),
            "has_override": override is not None,
            "learned": bool(getattr(skill, "learned", False)),
            "editable_on_disk": skill.base_dir != "",
            "origin": getattr(skill, "origin", "loom"),
            "deletable": _skill_deletable(S, skill),
        }

    @app.post("/skill/save")
    def skill_save():
        # Enregistre l'édition d'un skill. scope=session -> override de session (n'écrit
        # PAS le disque) ; scope=global -> écrit le SKILL.md pour TOUTES les sessions et
        # lève l'override de session (le fichier fait désormais foi).
        conv, save = _ctx(S)
        name = request.form.get("name", "")
        body = request.form.get("body", "")
        scope = request.form.get("scope", "session")
        skill = next((s for s in _all_skills(S) if s.name == name), None)
        if skill is None:
            return {"error": f"skill inconnu : {name}"}, 404
        if scope == "global":
            try:
                write_skill_source(skill, body)
            except OSError as exc:
                return {"error": f"écriture impossible : {exc}"}, 400
            conv.set_skill_override(name, None)
            save()
            return {"ok": True, "scope": "global"}
        conv.set_skill_override(name, body)
        save()
        return {"ok": True, "scope": "session"}

    @app.post("/skill/create")
    def skill_create():
        # « + nouveau » : crée un squelette SKILL.md dans le dossier des skills USER
        # (hors package : loom/skills reste l'officiel versionné) puis l'éditeur s'ouvre
        # dessus. Le nom sert de slug de dossier -> alphanumérique/tirets uniquement.
        raw = (request.form.get("name") or "").strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
        if not slug or len(slug) < 3:
            return {
                "error": "nom invalide (3+ caractères, lettres/chiffres/tirets)"
            }, 400
        if any(s.name == slug for s in _all_skills(S)):
            return {"error": f"un skill « {slug} » existe déjà"}, 409
        desc = (request.form.get("description") or "").strip()
        body = (request.form.get("body") or "").strip()
        # Corps fourni par le drawer (écrit à la main ou généré par le modèle) : on le
        # prend s'il porte déjà son frontmatter, sinon on l'enveloppe. Le nom du
        # frontmatter est FORCÉ au slug (identité = dossier, cf. effective_skills) —
        # sinon un skill créé dans `mon-skill/` pourrait se déclarer autrement et
        # entrer en collision avec un skill existant.
        fm_end = body.find("\n---", 3) if body.startswith("---") else -1
        if fm_end != -1:
            front_lines = [
                ln
                for ln in body[3:fm_end].splitlines()
                if ln.strip() and not ln.strip().lower().startswith("name:")
            ]
            content = (
                "---\nname: " + slug + "\n" + "\n".join(front_lines) + body[fm_end:]
            )
            if not content.endswith("\n"):
                content += "\n"
        elif body.startswith("---"):
            # Frontmatter jamais fermé : le chargeur retombera sur le nom du dossier.
            content = body if body.endswith("\n") else body + "\n"
        else:
            content = (
                f"---\nname: {slug}\ndescription: {desc}\n---\n\n"
                + (
                    body
                    or "Décris ici la méthode : quand ce skill s'applique, les "
                    "étapes à suivre, les pièges à éviter."
                )
                + "\n"
            )
        target = Path(S.user_skills_dir) / slug
        try:
            target.mkdir(parents=True, exist_ok=False)
            (target / "SKILL.md").write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"error": f"création impossible : {exc}"}, 400
        return {"ok": True, "name": slug}

    @app.post("/skill/generate")
    def skill_generate():
        # « Générer » du drawer de création : le modèle de la session rédige le
        # SKILL.md complet depuis la description. Modèle LOCAL : verrou non bloquant
        # (une génération en cours a priorité) + save/restore du slot KV pour ne PAS
        # sacrifier le cache de la conversation (cache souverain, cf. 2026-07-10).
        conv, _ = _ctx(S)
        desc = (request.form.get("description") or "").strip()
        if not desc:
            return {"error": "décris d'abord le skill"}, 400
        name = (request.form.get("name") or "nouveau-skill").strip()
        model = conv.model
        is_local = bool(model) and model not in S.remote_model_ids
        if is_local and not S.local_gen_lock.acquire(blocking=False):
            return {
                "error": "modèle local occupé — réessaie après la génération en cours"
            }, 409
        if is_local:
            S.local_busy["reason"] = "skill"
        try:
            # Serveur modèle éteint (bouton non cliqué, session neuve…) : on le démarre
            # comme le fait le chat, au lieu de planter en Connection refused (vécu).
            if is_local and not _ensure_local_server(S, wait=45.0):
                return {
                    "error": "serveur modèle indisponible (démarrage trop long) — "
                    "réessaie dans quelques secondes"
                }, 503
            saved = S.client.save_slot(model, "uigen.kv") if is_local else False
            prompt = (
                "Write a Loom SKILL.md file for the skill idea below. Output ONLY the "
                "file content, nothing else: a YAML frontmatter (---\\nname: "
                f"{name}\\ndescription: <one factual trigger line>\\n---) followed by "
                "a concise, actionable markdown body (when it applies, the steps, the "
                "pitfalls). Write the body in the SAME language as the idea.\n\n"
                f"Skill idea: {desc}"
            )
            chunks: list[str] = []
            try:
                for kind, chunk in S.client.stream_chat(
                    [{"role": "user", "content": prompt}],
                    "",
                    900,
                    model=model or None,
                    thinking=False,
                ):
                    if kind == "content":
                        chunks.append(chunk)
            except Exception as exc:  # noqa: BLE001 - erreur PROPRE côté UI, pas un 500
                return {"error": f"modèle injoignable : {str(exc)[:140]}"}, 502
            finally:
                if saved:
                    S.client.restore_slot(model, "uigen.kv")
            text = "".join(chunks).strip()
            # Dé-clôture un éventuel bloc ```...``` autour du fichier.
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.rstrip().endswith("```"):
                    text = text.rstrip()[:-3].rstrip() + "\n"
            if not text.strip():
                return {"error": "le modèle n'a rien produit — réessaie"}, 502
            return {"ok": True, "source": text}
        finally:
            if is_local:
                S.local_busy["reason"] = ""
                S.local_gen_lock.release()

    @app.post("/skill/delete")
    def skill_delete():
        # Suppression (skills appris + ajoutés user UNIQUEMENT) : retire le dossier du
        # skill, l'override et l'entrée disabled de la session courante. Les skills du
        # package/plugins ne passent jamais ici (_skill_deletable) — pour eux : décocher.
        conv, save = _ctx(S)
        name = request.form.get("name", "")
        skill = next((s for s in _all_skills(S) if s.name == name), None)
        if skill is None:
            return {"error": f"skill inconnu : {name}"}, 404
        if not _skill_deletable(S, skill):
            return {
                "error": "seuls les skills appris ou ajoutés par l'utilisateur sont "
                "supprimables (pour un skill du package ou d'un plugin : décoche-le)"
            }, 403
        try:
            shutil.rmtree(skill.base_dir)
        except OSError as exc:
            return {"error": f"suppression impossible : {exc}"}, 400
        conv.set_skill_override(name, None)
        conv.set_disabled_skills([n for n in conv.disabled_skills if n != name])
        save()
        return {"ok": True}


# ---- Routes : modèles (sélection, gestionnaire, machine) -------------------------------

# ---- Modèles LOCAUX : liste + édition du tuning MACHINE (offload GPU) dans model.toml.
# La définition (repo/filename/n_layers) est commune au modèle -> lecture seule ici ; le
# tuning (context/n_gpu_layers/cpu_moe/n_cpu_moe) est propre à cette machine -> éditable.
_LOCAL_EDITABLE = {
    "context": "int",
    "n_gpu_layers": "int",
    "cpu_moe": "bool",
    "n_cpu_moe": "int",
    # Microbatch/batch de prefill (banc 2026-07-19 : levier x2,9 sur MoE offloadé).
    "ubatch": "int",
    "batch": "int",
}


def _models_payload(S):
    """Liste ordonnée pour reconstruire le <select> côté client (id + local/distant)."""
    return [
        {
            "id": m,
            "remote": m in S.remote_model_ids,
            "image": m in S.image_model_ids,
            "video": m in S.video_model_ids,
            "desc": S.model_descriptions.get(m, ""),
        }
        for m in S.models
    ]


def _remote_list(S):
    """Modèles distants montés, pour le panneau de config. Jamais la clé en clair :
    seulement sa présence. Tous vivent dans config/local.toml (source unique) donc
    tous éditables/supprimables — `managed` reste dans le payload pour le front."""
    out = []
    for mid in S.remote_model_ids:
        info = S.client.remote_route_info(mid)
        key = S.client.remote_api_key(mid)
        out.append(
            {
                "id": mid,
                "base_url": info["base_url"],
                "model": info["model"],
                "context": S.model_contexts.get(mid),
                "max_tokens": S.model_max_tokens.get(mid),
                "vision": mid in S.vision_models,
                "has_key": info["has_key"],
                # Indice masqué (4 derniers car.) : l'utilisateur voit sa propre clé de
                # façon partielle, jamais la clé entière renvoyée au client.
                "key_hint": ("…" + key[-4:]) if key else "",
                "managed": True,
            }
        )
    return sorted(out, key=lambda x: x["id"])


def _mount_remote(S, rec):
    """Monte à chaud un modèle distant `rec` (dict) dans TOUS les registres partagés."""
    mid = rec["id"]
    S.client.add_remote_route(
        mid,
        {
            "base_url": rec["base_url"],
            "api_key": rec.get("api_key", ""),
            "model": rec["model"],
            "enable_thinking_param": bool(rec.get("enable_thinking_param", False)),
        },
    )
    S.remote_model_ids.add(mid)
    S.remote_model_names[mid] = rec["model"]
    if rec.get("context"):
        S.model_contexts[mid] = int(rec["context"])
    if rec.get("max_tokens"):
        S.model_max_tokens[mid] = int(rec["max_tokens"])
    S.model_prices[mid] = (
        float(rec.get("price_in", 0.0) or 0.0),
        float(rec.get("price_out", 0.0) or 0.0),
        float(rec.get("price_cached", 0.0) or 0.0),
    )
    if rec.get("vision"):
        S.vision_models.add(mid)
    else:
        S.vision_models.discard(mid)
    if mid not in S.models:
        S.models.append(mid)


def _register_model_routes(app, S):
    @app.post("/model")
    def model_update():
        conv, save = _ctx(S)

        model = request.form.get("model", "")

        conv.set_model(model)

        save()

        # Mémorise ce choix : il devient le défaut des prochaines sessions / lancements.

        S.session_store.set_default_model(model)

        # Cycle de vie du modèle SUR LA MACHINE — invariant multi-onglets (2026-07-19) :
        # au plus UN local chargé (llama-swap le garantit), les distants n'imposent
        # AUCUNE limite. Sélectionner un DISTANT ne décharge donc PLUS le local : une
        # autre session l'utilise peut-être (voire génère dessus — l'unload la tuait).
        # Libérer la VRAM reste possible via les boutons « décharger / éteindre ».
        if model in S.remote_model_ids:
            pass
        elif model in S.image_model_ids:
            # Modèle IMAGE : libérer la VRAM du LLM et préchauffer ComfyUI en fond
            # (équivalent du warmup local : la 1re image n'attend pas le démarrage).
            # SAUF si une génération LOCALE tourne (autre session) : on ne lui vole ni
            # la VRAM ni le modèle — generate_image sérialisera au moment de générer.
            def _prep_image(m=model):
                if not S.local_gen_lock.acquire(blocking=False):
                    print(
                        "[loom] préchauffage image sauté : génération locale en cours",
                        flush=True,
                    )
                    return
                try:
                    S.client.unload_local()
                    _engine_for(S, S.image_by_id[m]).ensure_up()
                except ComfyError as exc:
                    print(f"[loom] préchauffage ComfyUI : {exc}", flush=True)
                finally:
                    S.local_gen_lock.release()

            threading.Thread(
                target=_prep_image, daemon=True, name="loom-image-warmup"
            ).start()
        elif model:
            # Modèle LOCAL : démarre le serveur s'il est éteint (démarrage auto), puis
            # AMORCE le préfixe de la session (l'amorce charge le modèle ET remplit le
            # slot KV avec le vrai préfixe — l'ancien ping warmup l'écrasait avec un
            # préfixe poubelle). En fond : la réponse UI reste instantanée, le chip suit.
            _prime_async(S, _session(S), wait_server=90.0)

        return render_template(
            "_models.html",
            models=S.models,
            current_model=conv.model,
            remote_model_ids=S.remote_model_ids,
            image_model_ids=S.image_model_ids,
            video_model_ids=S.video_model_ids,
            model_descriptions=S.model_descriptions,
        )

    # ---- Gestionnaire de modèles (UI) : ajouter/tester/supprimer un modèle DISTANT à chaud,
    # sans redémarrer. Un distant = URL + clé (rien en VRAM) -> l'ajout monte une route et met
    # à jour les registres partagés en place. Persisté dans config/local.toml (source unique).
    @app.get("/commands")
    def commands():
        """Catalogue des commandes slash — consommé par la palette « / » du composer."""
        return {"commands": CHAT_COMMANDS}

    @app.get("/models/config")
    def models_config():
        return {"remotes": _remote_list(S), "models": _models_payload(S)}

    @app.post("/models/remote/test")
    def models_remote_test():
        b = request.get_json(silent=True) or {}
        base_url = (b.get("base_url") or "").strip().rstrip("/")
        model = (b.get("model") or "").strip()
        mid = (b.get("id") or "").strip()
        key = (b.get("api_key") or "").strip()
        if not key and mid:  # édition sans re-saisir la clé -> celle de la route montée
            key = S.client.remote_api_key(mid)
        if not (base_url and model):
            return {"ok": False, "message": "base_url et model requis"}, 400
        ok, msg = S.client.ping_remote(base_url, key, model)
        return {"ok": ok, "message": msg}

    @app.post("/models/remote")
    def models_remote_upsert():
        roots = _models_roots(S)
        if not roots:
            return {"error": "racine des modèles indisponible"}, 500
        b = request.get_json(silent=True) or {}
        mid = (b.get("id") or "").strip()
        base_url = (b.get("base_url") or "").strip().rstrip("/")
        model = (b.get("model") or "").strip()
        if not (mid and base_url and model):
            return {"error": "id, base_url et model sont requis"}, 400
        if mid in S.models and mid not in S.remote_model_ids:
            return {"error": f"'{mid}' est déjà un modèle local"}, 400
        # Clé : si vide, on garde celle de la route montée (édition sans re-saisir).
        key = (b.get("api_key") or "").strip() or S.client.remote_api_key(mid)
        rec = {
            "id": mid,
            "base_url": base_url,
            "model": model,
            "api_key": key,
            "context": int(b["context"]) if b.get("context") else None,
            "max_tokens": int(b["max_tokens"]) if b.get("max_tokens") else None,
            "vision": bool(b.get("vision")),
        }
        # Un distant = un dossier remote/<id>/model.toml sur la racine prioritaire.
        # Édition en place (tomlkit) : si le dossier vit sur une AUTRE racine, on
        # l'édite là-bas plutôt que de créer un doublon masqué par la priorité.
        with S.toml_lock:
            dest = next(
                (r for r in roots if model_store.remote_dir(r, mid).is_dir()),
                roots[0],
            )
            model_store.write_remote_dir(dest, rec)
        _mount_remote(S, rec)
        return {"ok": True, "models": _models_payload(S), "remotes": _remote_list(S)}

    @app.delete("/models/remote/<mid>")
    def models_remote_delete(mid):
        roots = _models_roots(S)
        if not roots:
            return {"error": "racine des modèles indisponible"}, 500
        if mid not in S.remote_model_ids:
            return {"error": f"modèle distant '{mid}' inconnu"}, 404
        with S.toml_lock:
            model_store.delete_remote_dir(roots, mid)
            # Filets : emplacements hérités, sans jamais recréer un fichier vide.
            if S.config_local_path:
                model_store.delete_remote_in_toml(S.config_local_path, mid)
            if S.remote_store_path and Path(S.remote_store_path).exists():
                model_store.delete(S.remote_store_path, mid)
        _forget_remote(S, mid)
        return {"ok": True, "models": _models_payload(S), "remotes": _remote_list(S)}

    @app.get("/models/local")
    def models_local():
        import tomllib

        out = []
        for m in S.local_model_specs:
            cur = {k: v for k, v in m.items() if k != "dir"}
            d = m.get("dir")
            if d:
                tp = Path(d) / "model.toml"
                if tp.exists():
                    try:
                        raw = tomllib.loads(tp.read_text(encoding="utf-8"))
                        for k in _LOCAL_EDITABLE:
                            if k in raw:
                                cur[k] = raw[k]
                    except (OSError, ValueError):
                        pass
            out.append(cur)
        return {"models": out}

    @app.post("/models/local/set")
    def models_local_set():
        import tomlkit

        b = request.get_json(silent=True) or {}
        mid = (b.get("id") or "").strip()
        key = (b.get("key") or "").strip()
        if key not in _LOCAL_EDITABLE:
            return {"error": "champ non éditable"}, 400
        spec = next((m for m in S.local_model_specs if m.get("id") == mid), None)
        if not spec or not spec.get("dir"):
            return {"error": "modèle local inconnu"}, 404
        tp = Path(spec["dir"]) / "model.toml"
        if not tp.exists():
            return {"error": "model.toml introuvable"}, 404
        raw = b.get("value")
        t = _LOCAL_EDITABLE[key]
        empty = raw is None or (
            isinstance(raw, str) and raw.strip() == "" and t == "int"
        )
        truthy = ("1", "true", "on", "yes")
        try:
            with S.toml_lock:  # sérialise le read-modify-write (Flask threaded)
                doc = tomlkit.parse(tp.read_text(encoding="utf-8"))
                if empty:
                    if key in doc:
                        del doc[key]
                elif t == "int":
                    doc[key] = int(raw)
                else:  # bool
                    doc[key] = (
                        raw if isinstance(raw, bool) else str(raw).lower() in truthy
                    )
                tp.write_text(tomlkit.dumps(doc), encoding="utf-8")
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)[:120]}, 400
        # Applique À CHAUD côté serveur modèle : régénère le yaml (llama-swap -watch-config le
        # recharge) + décharge CE modèle -> il se relance avec le nouveau tuning au prochain
        # usage, sans toucher au TOML à la main ni tout redémarrer.
        applied = _regen_swap_yaml(S)
        if applied:
            threading.Thread(
                target=lambda: S.client.unload_local(mid),
                daemon=True,
                name="loom-reload-model",
            ).start()
        return {"ok": True, "applies": "model-reload" if applied else "restart"}

    @app.get("/machine_state")
    def machine_state():
        # État du modèle SUR LA MACHINE, pour l'indicateur UI. Vérité = llama-swap /running
        # (best-effort ; le modèle peut aussi s'être déchargé seul via son TTL). On teste par
        # sous-chaîne quel modèle est chargé, sans coupler au schéma JSON de llama-swap.
        conv, _ = _ctx(S)
        model = conv.model
        remote = model in S.remote_model_ids
        reachable, running_txt = S.client.running_local()
        # /running est parsé quand c'est possible : llama-swap distingue « starting »
        # (chargement en cours) de « ready » (servable). Sans ça, le chip disait
        # « chargé » dès le début du chargement, et un unload pendant « starting » est
        # ignoré par llama-swap -> le bouton « décharger » doit se cacher à ce moment.
        # Repli sous-chaîne si le JSON change (on reste découplé du schéma).
        states: dict[str, str] = {}
        try:
            for entry in json.loads(running_txt).get("running", []):
                states[str(entry.get("model", ""))] = str(entry.get("state", ""))
        except (ValueError, AttributeError):
            pass
        if states or reachable:
            model_loaded = bool(model and states.get(model) == "ready")
            model_loading = bool(model and states.get(model) == "starting")
            any_loaded = bool(
                any(
                    states.get(mid) in ("ready", "starting")
                    for mid in S.local_model_ids
                )
            )
        else:
            model_loaded = bool(reachable and model and model in running_txt)
            model_loading = False
            any_loaded = bool(
                reachable and any(mid in running_txt for mid in S.local_model_ids)
            )
        if reachable:
            S.server_manager.confirm_started()  # démarrage confirmé -> fin de l'état « démarrage »
        return {
            "mode": "remote" if remote else "home",
            "model": model,
            "reachable": reachable,
            "model_loaded": model_loaded,
            "loading": model_loading,
            "any_loaded": any_loaded,
            # Serveur GÉRÉ (lancé par loom.web) : conditionne le bouton « éteindre » —
            # on ne propose jamais de tuer une stack lancée à la main hors Loom.
            "managed": S.server_manager.owns_running(),
            "starting": S.server_manager.starting,
        }

    @app.post("/machine/unload")
    def machine_unload():
        # Déchargement À LA DEMANDE (bouton UI sous le chip machine) : libère la VRAM sans
        # changer de modèle sélectionné. Synchrone : la réponse reflète le résultat réel
        # (llama-swap tue le llama-server en ~1-2 s). Rechargé à la prochaine requête.
        _client_mark_all_cold(S)
        return {"ok": S.client.unload_local()}

    @app.post("/machine/server/start")
    def machine_server_start():
        # Trigger MANUEL (bouton « démarrer le serveur ») : lance sans bloquer la requête ;
        # l'UI suit la progression via /machine_state (état « démarrage… »). Puis AMORCE
        # du préfixe de la session active en fond : « démarrer » = « rendre prêt à
        # répondre », premier message compris (il ne préfille plus que son delta).
        ok = S.server_manager.start()
        _ctx(S)  # garantit une session active avec un modèle valide
        _prime_async(S, _session(S), wait_server=90.0)
        return {"ok": ok}

    @app.post("/machine/server/stop")
    def machine_server_stop():
        # Éteint l'arbre complet (serve.py + llama-swap + llama-server) et libère RAM/VRAM.
        # Ne concerne QUE l'instance gérée par loom.web (cf. managed dans /machine_state).
        return {"ok": S.server_manager.stop()}


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
        sess = _get_session(S, req_sid) or _session(S)
        S.cur["session"] = sess  # focus (défaut de l'index)
        sid = sess.id
        chat_lock = _lock_for(S, sid)
        cancel_event = _cancel_for(S, sid)

        if not chat_lock.acquire(blocking=False):
            # Une génération de CETTE session tourne déjà : on ne l'interrompt PAS.
            # Le message part en FILE D'ATTENTE (même mécanique que les notes en vol) :
            # injecté role=user au prochain point d'arrêt de la boucle tool-use, ou au
            # début du tour suivant s'il arrive trop tard — jamais perdu. L'annulation,
            # c'est UNIQUEMENT le bouton stop (/cancel).
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
                queued_msg += " (images ignorées : la file ne transporte que du texte)"
            return Response(queued_msg, status=202)

        # On tient le verrou : repartir d'un signal d'annulation propre.

        cancel_event.clear()

        conv = sess.conversation
        save = lambda: S.session_store.save(sess)

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
        message = _handle_init_command(S, message)

        # Plus de garde bloquant : un modèle texte-only ne reçoit PAS l'image inline (qui

        # ferait planter un llama-server sans mmproj) — on la stocke sur disque et il l'inspecte

        # via read_image, routé vers un modèle vision (cf. _build_user_content plus bas).

        # Logs PAR SESSION (au même titre que session.json) : (1) trace des échanges modèle

        # routée vers sessions/<id>/debug.log ; (2) copie du log serveur modèle global

        # (var/logs/serve.log) dans la session — doublon assumé, pour tout avoir sous la main.

        _sdir = S.session_store.session_dir(_session(S).id)

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
            sess = _session(S)

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
            system_prompt, strong = _build_system_prompt(S, conv)
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
                _sess = _session(S)

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

            ws = _session(S).workspace

            registry = S.tool_factory(conv.active_tools, ws, conv)

            use_tools = registry is not None and len(registry)

            # Limites du modèle courant (distant = sa grande fenêtre ; local = global).

            eff_max_tokens, eff_compact = _model_limits(S, conv.model)

            # `strong` (tier distant=fort) est calculé plus haut, à la construction du prompt :

            # il coupe ici les gardes de comportement (act_nudge, claim_audit, coupe non-progrès).

            # On ne garde que outils + mémoire + sécurité. Un modèle local garde le harnais complet.

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
                )

            else:
                source = S.client.stream_chat(
                    conv.to_messages(),
                    system_prompt,
                    eff_max_tokens,
                    model=conv.model or None,
                    thinking=conv.thinking,
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

                if _local_held:
                    S.local_busy["reason"] = ""
                    S.local_gen_lock.release()

                chat_lock.release()
                S.stay_awake.release()  # plus de veille bloquée si plus aucune génération

        return Response(generate(), mimetype="text/event-stream")


def _register_soul_routes(app, S):
    """L'Âme : export/import chiffré de l'état portable (sessions, mémoire,
    identité, skills). AUCUN secret ne part (remote_models.json exclu par
    construction dans soul.build_archive). Spec : specs/2026-07-21-ame-*."""
    from loom.web import soul as soul_mod

    def _soul_paths():
        idp = S.identity_paths or {}
        identity_dir = (
            Path(idp["soul_path"]).parent
            if idp.get("soul_path")
            else Path("var/identity")
        )
        return soul_mod.SoulPaths(
            sessions_root=Path(S.session_store.root),
            memory_db=Path(S.memory_db_path or "var/memory/memory.db"),
            identity_dir=identity_dir,
            learned_skills_dir=Path(S.learned_skills_dir or "var/skills_learned"),
            user_skills_dir=Path(S.user_skills_dir),
        )

    @app.get("/soul/sessions")
    def soul_sessions():
        return {
            "sessions": [
                {"id": m.id, "title": m.title, "updated_at": m.updated_at}
                for m in S.session_store.list()
            ]
        }

    @app.post("/soul/passphrase/check")
    def soul_passphrase_check():
        return soul_mod.check_passphrase(request.form.get("passphrase", ""))

    @app.post("/soul/passphrase/generate")
    def soul_passphrase_generate():
        return {"passphrase": soul_mod.generate_passphrase()}

    @app.post("/soul/export")
    def soul_export():
        passphrase = request.form.get("passphrase", "")
        if not soul_mod.check_passphrase(passphrase)["ok"]:
            return {"error": "passphrase trop faible (score zxcvbn < 3)"}, 400
        ids = [s for s in request.form.get("session_ids", "").split(",") if s.strip()]
        try:
            recap = soul_mod.export_soul(
                _soul_paths(), ids, request.form.get("dest_dir", ""), passphrase
            )
        except soul_mod.SoulError as e:
            return {"error": str(e)}, 400
        except OSError as e:
            # USB pleine/retirée, dossier en lecture seule : vraie cause au user,
            # pas un 500 que l'UI traduirait en faux « échec réseau ».
            return {"error": f"écriture impossible : {e}"}, 400
        log_event("soul.export", sessions=recap["sessions"], path=recap["path"])
        return recap

    @app.post("/soul/import")
    def soul_import():
        # Tout l'import sous gen_guard : (1) refus net si une génération est en vol
        # (l'objet session vivant sauverait son état par-dessus les fichiers importés),
        # (2) aucune génération ne peut DÉMARRER pendant la fusion (elle chargerait
        # l'objet périmé du cache), (3) purge du cache + rechargement de la session
        # focus AVANT de relâcher — sinon le prochain save de l'objet mémoire écrase
        # silencieusement les données importées (« rien n'est jamais détruit »).
        with S.gen_guard:
            if any(lk.locked() for lk in S.sess_locks.values()):
                return {
                    "error": "génération en cours — réessaie quand les sessions "
                    "sont au repos"
                }, 409
            try:
                report = soul_mod.import_soul(
                    _soul_paths(),
                    request.form.get("file", ""),
                    request.form.get("passphrase", ""),
                )
            except soul_mod.SoulError as e:
                return {"error": str(e)}, 400
            S.sessions_cache.clear()
            cur = S.cur.get("session")
            if cur is not None:
                fresh = S.session_store.load(cur.id)
                if fresh is not None:
                    S.sessions_cache[cur.id] = fresh
                    S.cur["session"] = fresh
        log_event(
            "soul.import",
            ajoutees=report["sessions"]["ajoutees"],
            remplacees=report["sessions"]["remplacees"],
        )
        return {"report": report}
