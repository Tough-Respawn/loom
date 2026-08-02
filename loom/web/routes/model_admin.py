from __future__ import annotations
import shutil
from pathlib import Path
from loom.runtime import model_install
from loom.web.routes.config import _regen_swap_yaml





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
    # Avertir avant de supprimer le modèle distant par défaut.
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
    # Supprimer seulement la définition Loom; les poids ComfyUI peuvent être partagés.
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
    # Ne remonter depuis models_dir que s'il suit la convention attendue.
    if S.models_dir:
        p = Path(S.models_dir)
        if p.name == "text" and p.parent.name == "local":
            return [str(p.parent.parent)]
    return []


def _install_roots(S) -> list[dict]:
    """Racines candidates à l'installation d'un LOCAL, avec l'espace libre (Go) pour que
    le wizard affiche un choix éclairé quand il y a plusieurs disques. racine[0] = défaut
    (la plus rapide par convention, cf. [storage] models_root)."""

    out = []
    for r in _models_roots(S):
        try:
            free_gb = shutil.disk_usage(r).free // (1024**3)
        except OSError:
            free_gb = None
        out.append({"path": r, "free_gb": free_gb})
    return out


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
    # Attendre brièvement le verrou; le journal append-only borne le risque de repli.
    got = chat_lock.acquire(timeout=2)
    try:
        conv = sess.conversation
        conv.add("assistant", msg)
        S.session_store.append_event(sess.id, "text", {"text": msg})
        S.session_store.save(sess)
    finally:
        if got:
            chat_lock.release()


# La définition du modèle reste partagée; seuls les réglages machine sont éditables ici.
_LOCAL_EDITABLE = {
    "context": "int",
    "n_gpu_layers": "int",
    "cpu_moe": "bool",
    "n_cpu_moe": "int",
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
                # Montrer seulement les quatre derniers caractères de la clé.
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
