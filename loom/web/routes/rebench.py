# loom/web/routes/rebench.py — sorti de models.py (comportement constant).
from __future__ import annotations
from pathlib import Path




# ---- /rebench : recalibration topologique d'un LOCAL TEXTE (loom.setup réutilisé) ----

# Un seul rebench à la fois : la mesure sature CPU/GPU et exige la VRAM libre.
_REBENCH = {"job": None}


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
