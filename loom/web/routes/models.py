from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from flask import Response, render_template, request

from loom.runtime import model_install, model_store
from loom.runtime.comfy import ComfyError
from loom.web import wizard as _wizard
from loom.web.app import (
    _sse,
)
from loom.web.routes.config import _regen_swap_yaml
from loom.web.routes.helpers import _client_mark_all_cold, _ctx, _engine_for, _session

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
    # Import TARDIF (rompt le cycle models<->chat au chargement : chat importe models
    # au top, models n'a besoin de chat qu'ici, une fois, au montage des routes).
    from loom.web.routes.chat import CHAT_COMMANDS, _prime_async

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
