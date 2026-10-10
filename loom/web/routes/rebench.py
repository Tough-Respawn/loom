# loom/web/routes/rebench.py — sorti de models.py (comportement constant).
from __future__ import annotations
from pathlib import Path


# ---- /rebench : recalibration topologique d'un LOCAL TEXTE (loom.setup réutilisé) ----

# Un seul rebench à la fois : la mesure sature CPU/GPU et exige la VRAM libre.
_REBENCH = {"job": None}


def _measure_placement(
    probe,
    meta: dict,
    *,
    model_size_mb: int,
    hw,
    ram_total_mb: int,
    headroom_mb: int,
    gpu_backend: bool,
    progress,
    useful_ctx: int | None = None,
    mt: dict | None = None,
    raw: dict | None = None,
):
    """Sonde de placement (loom.setup.placement) sur la sonde serveur `probe` : renvoie
    (verdict sérialisable | None, sonde alignée sur l'élu). None quand rien n'est
    mesurable ou que la validation du seul candidat échoue : la calibration vaut alors
    avec les flags actuels du modèle. La faisabilité s'estime au contexte UTILE
    (`useful_ctx`) avec le type de cache de l'exécutant, via le profil GGUF ; la
    configuration ACTUELLE (`mt`) est la ligne de base ; `raw` porte les contraintes
    de prefill optionnelles ([placement])."""
    from dataclasses import replace as _dc_replace

    from loom.runtime.model_profile import ModelProfile
    from loom.setup import placement as place_mod

    profile = ModelProfile.from_meta(meta, model_size_mb=int(model_size_mb or 0))
    kv_mb = place_mod.kv_estimate_mb(
        profile,
        int(useful_ctx or place_mod.PLACEMENT_PROBE_CTX),
        gpu_tuning=bool(getattr(hw, "has_gpu", False)),
        slots=1,
    )
    plan = place_mod.plan_placements(
        moe=bool(meta.get("expert_count")),
        n_layers=meta.get("n_layers"),
        model_size_mb=int(model_size_mb or 0),
        kv_mb=kv_mb,
        gpu_backend=bool(gpu_backend),
        vram_total_mb=int(getattr(hw, "vram_total_mb", 0) or 0),
        ram_total_mb=int(ram_total_mb),
        uma=not getattr(hw, "vram_is_discrete", True),
        headroom_mb=headroom_mb,
        current=place_mod.placement_from_config(
            mt or {}, n_layers=meta.get("n_layers")
        ),
        profile=profile,
    )
    prefill_c, pp_floor = place_mod.constraints_from_config(raw or {})
    try:
        res = place_mod.probe_placement(
            lambda pl: _dc_replace(
                probe, ngl=pl.ngl, cpu_moe=pl.cpu_moe, n_cpu_moe=pl.n_cpu_moe
            ),
            plan.candidates,
            progress=progress,
            useful_ctx=useful_ctx,
            non_explores=plan.non_explores,
            prefill=prefill_c,
            pp_floor_ratio=pp_floor,
        )
    except Exception:  # noqa: BLE001 - sonde best-effort : la calibration vaut sans
        res = None
    if not res or res.get("placement") is None or not res["mesures"]:
        return None, probe
    pl = res["placement"]
    probe = _dc_replace(probe, ngl=pl.ngl, cpu_moe=pl.cpu_moe, n_cpu_moe=pl.n_cpu_moe)
    verdict = {
        "label": pl.label,
        "key": pl.key,
        "ngl": pl.ngl,
        "cpu_moe": pl.cpu_moe,
        "n_cpu_moe": pl.n_cpu_moe,
        "tg_ts": res["tg_ts"],
        "pp_ts": res["pp_ts"],
        "gain_pct": res["gain_pct"],
        "baseline": res["baseline"],
        "mecanisme": res["mecanisme"],
        "mesures": res["mesures"],
        "preselection": res.get("preselection"),
        "finalistes": res.get("finalistes"),
        "ctx_final": res.get("ctx_final"),
        "depth_final": res.get("depth_final"),
        "non_explores": res.get("non_explores"),
        "compare": bool(res.get("compare")),
    }
    return verdict, probe


def _placement_implied_ngl(pl: dict):
    """n_gpu_layers que _set_model_placement écrira pour ce verdict (None = retiré) :
    999 tout-GPU, 0 CPU seul, le -ngl exact d'un partiel dense."""
    label = pl.get("label") if isinstance(pl, dict) else pl
    if label == "gpu_partiel" and isinstance(pl, dict):
        return int(pl.get("ngl") or 0)
    return {"gpu_total": 999, "cpu": 0}.get(label)


def _probe_settings(
    meta: dict, mt: dict, over: dict, hw, *, gpu_backend: bool, vram_fallback_mb: int
) -> tuple[str, int, int, int]:
    """(topologie, VRAM totale, threads, ngl) de la sonde, avec la dérivation de
    l'EXÉCUTANT. La VRAM vient du profil `--list-devices` (Vulkan compris) et
    nvidia-smi n'est qu'un repli : sans ça la 860M passait en topologie « ram » et
    la sonde mesurait sans profil GPU. Threads = effective.launch_flags (override
    machine, sinon cœurs physiques en GPU, tous en CPU). ngl : la borne PAR MODÈLE
    (model.toml n_gpu_layers) PRIME — c'est elle qui évite le spill (gemma4 à
    36/42) —, sinon doctrine MoE (99, experts en RAM), sinon l'override machine."""
    from loom.runtime.effective import launch_flags
    from loom.setup import topology as topo_mod

    vram = int(getattr(hw, "vram_total_mb", 0) or vram_fallback_mb or 0)
    topo = topo_mod.discover_topology(meta, bool(gpu_backend), vram)
    threads = launch_flags(hw, over.get("threads")).threads
    gpu = topo != topo_mod.TOPO_RAM
    if mt.get("n_gpu_layers") is not None:
        ngl = int(mt["n_gpu_layers"])
    elif meta.get("expert_count") and gpu:
        ngl = 99
    else:
        ngl = int(over.get("n_gpu_layers", 99 if gpu else 0))
    return topo, vram, threads, ngl


def _run_calibration(S, spec, progress):
    """Cœur de mesure (préconditions + topologie + calibrate), avec les flags EXACTS
    du modèle. Lève RuntimeError actionnable si la machine n'est pas prête.
    Isolé pour être stubbable dans les tests (aucun subprocess en CI)."""
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
    # Le profil matériel de l'EXÉCUTANT (`--list-devices`, Vulkan compris) fixe la
    # topologie, les flags machine de la sonde et le mode de mesure mémoire ;
    # nvidia-smi n'est qu'un repli de VRAM.
    from loom.runtime.hardware import detect_hardware

    hw = detect_hardware(str(server_bin))
    gpu_backend = bool(bench_mod.has_gpu_backend(server_bin) and hw.has_gpu)
    over = raw.get("override") or {}
    topo, vram, threads, ngl = _probe_settings(
        meta,
        mt,
        over,
        hw,
        gpu_backend=gpu_backend,
        vram_fallback_mb=topo_mod.gpu_vram_total_mb(),
    )
    server_cfg = raw.get("server") or {}
    headroom = int(server_cfg.get("gpu_kv_headroom_mb", 640) or 640)
    ram = int(psutil.virtual_memory().total // (1024 * 1024))
    # Mémoire unifiée : le device est la RAM, comptée une fois (= ce que la sonde mesure).
    uma = bool(hw.has_gpu and not hw.vram_is_discrete)
    budget = topo_mod.memory_budget_mb(topo, vram, ram, headroom, uma=uma)
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
        # Checkpoints des hybrides : mesurer la mémoire que l'exécutant prendra.
        checkpoint_min_step=(
            mt.get("checkpoint_min_step") or server_cfg.get("checkpoint_min_step")
        ),
        ctx_checkpoints=mt.get("ctx_checkpoints"),
        profile=hw,
    )
    # Placement MESURÉ avant isolation et calibration (même séquence que loom-setup),
    # faisabilité estimée au contexte UTILE du modèle.
    from loom.setup.placement import useful_context

    ctx_utile = useful_context(
        mt.get("context"), server_cfg.get("context"), meta.get("context_length")
    )
    progress("sonde de placement (où vivent les poids)…")
    pl_verdict, probe = _measure_placement(
        probe,
        meta,
        model_size_mb=int(spec.get("size_mb") or mt.get("size_mb") or 0),
        hw=hw,
        ram_total_mb=ram,
        headroom_mb=headroom,
        gpu_backend=gpu_backend,
        progress=progress,
        useful_ctx=ctx_utile,
        mt=mt,
        raw=raw,
    )
    # Sonde d'isolation AVANT la calibration : si le modèle exige un 2e slot,
    # la calibration doit mesurer avec le KV réellement doublé (même séquence
    # que loom-setup step_bench — le conseilleur simule l'exécutant).
    progress("sonde d'isolation du cache (A -> pollution -> A)…")
    isolation = None
    iso_detail = ""
    try:
        first, back = probe.probe_isolation()
        isolation = topo_mod.isolation_needed(first, back, meta.get("recurrent"))
        iso_detail = f"retour {back}/{first} tokens retraités"
        if meta.get("recurrent"):
            iso_detail += ", mémoire récurrente"
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
    # Sonde d'ubatch sur la MÊME sonde serveur (flags exacts, n_parallel inclus) :
    # un modèle installé par /add-model n'a jamais eu la sienne — c'est ici qu'il
    # la rattrape, sans réinstaller.
    try:
        from dataclasses import replace as _dc_replace

        calib["ubatch_probe"] = bench_mod.probe_ubatch(
            lambda ub, b: _dc_replace(probe, ubatch=ub, batch=b),
            progress=progress,
        )
    except Exception:  # noqa: BLE001 - sonde best-effort : la calibration vaut sans
        calib["ubatch_probe"] = None
    calib["ubatch_avant"] = mt.get("ubatch")
    calib["batch_avant"] = mt.get("batch")
    # Vérifier le cache avec la configuration FINALE (placement élu, slots décidés,
    # batchs mesurés) : la séquence réelle de Loom doit réutiliser le cache.
    try:
        from dataclasses import replace as _dc_replace

        ub = calib.get("ubatch_probe")
        probe_final = (
            _dc_replace(probe, ubatch=ub["ubatch"], batch=ub["batch"]) if ub else probe
        )
        progress("vérification du cache avec la configuration finale…")
        calib["cache_verifie"] = probe_final.verify_cache()
    except Exception:  # noqa: BLE001 - vérification best-effort : le verdict le dira
        calib["cache_verifie"] = None
    # Le moteur avec lequel tout a été mesuré, pour le commentaire du model.toml.
    try:
        from loom.setup.llama_release import verify_binary

        calib["build"] = verify_binary(str(server_bin)) or "build ?"
    except Exception:  # noqa: BLE001 - best-effort
        calib["build"] = "build ?"
    calib["ctx_utile"] = ctx_utile
    calib["placement"] = pl_verdict
    calib["placement_avant"] = {
        "cpu_moe": bool(mt.get("cpu_moe", is_moe)),
        "n_cpu_moe": mt.get("n_cpu_moe"),
        "n_gpu_layers": mt.get("n_gpu_layers"),
    }
    return calib, gguf


def _rebench_worker(S, sess, chat_lock, mid, job):
    """Thread du job : mesure, verdict comparé, message PERSISTÉ + état b_apply si
    une application a du sens. `job.done` posé EN DERNIER (le stream lit final)."""
    from loom.setup import bench as bench_mod
    from loom.setup import topology as topo_mod

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
        ub = calib.get("ubatch_probe")
        ub_change = bool(ub) and (
            calib.get("ubatch_avant") != ub["ubatch"]
            or calib.get("batch_avant") != ub["batch"]
        )
        if ub is None:
            ub_line = "sonde ubatch : illisible (réglage inchangé)."
        else:
            gain = (
                f" (+{ub['gain_pct']:.0f} % vs ubatch 512)"
                if ub.get("gain_pct")
                else ""
            )
            ub_line = (
                f"sonde ubatch : prefill optimal à ub={ub['ubatch']} / b={ub['batch']} "
                f"({ub['pp_ts']:.0f} t/s{gain})."
            )
        # Placement des poids : changement si les flags que l'on écrirait diffèrent
        # de ceux du model.toml (cpu_moe, n_cpu_moe, n_gpu_layers implicite).
        pl = calib.get("placement")
        pl_avant = calib.get("placement_avant") or {}
        pl_change = bool(pl) and (
            bool(pl["cpu_moe"]) != bool(pl_avant.get("cpu_moe"))
            or pl.get("n_cpu_moe") != pl_avant.get("n_cpu_moe")
            or _placement_implied_ngl(pl) != pl_avant.get("n_gpu_layers")
        )
        if pl is None:
            pl_line = "sonde de placement : non comparée (un seul candidat faisable, ou illisible)."
        else:
            pl_line = f"sonde de placement : {pl['mecanisme']}."
        # Vérification du cache avec la configuration finale (séquence réelle, slots finaux).
        cv = calib.get("cache_verifie")
        if not cv or cv.get("reused") is None:
            cache_line = (
                "vérification du cache (configuration finale) : non faite ou illisible."
            )
        elif cv["reused"]:
            cache_line = (
                "vérification du cache (configuration finale) : cache réutilisé après "
                f"routage des appels annexes ({topo_mod.cache_check_text(cv)})."
            )
        else:
            cache_line = (
                "vérification du cache (configuration finale) : cache NON réutilisé "
                f"({topo_mod.cache_check_text(cv)})."
            )
        # Un plancher n'est pas une mesure : le verdict le dit.
        valide = bool(calib.get("valide", True))
        vitesse_txt = (
            f"vitesse validée jusqu'à {calib['valide_jusqua']} tokens"
            if valide
            else f"contexte {new} = repli NON validé, aucun barreau de vitesse mesuré"
        )
        if new == current and not iso_change and not ub_change and not pl_change:
            # « Déjà au top » exige des PREUVES complètes : contexte validé en vitesse,
            # placement COMPARÉ, cache vérifié. Sinon le verdict dit ce qui manque.
            manques = []
            if not valide:
                manques.append("contexte non validé en vitesse (repli)")
            if pl is None or not pl.get("compare", True):
                manques.append("placement non comparé")
            if not cv or cv.get("reused") is None:
                manques.append("cache non vérifié")
            elif not cv["reused"]:
                manques.append("cache NON réutilisé")
            if not manques:
                msg = (
                    f"✅ « {mid} » est déjà au top : contexte actuel {current} = "
                    f"mesuré {new} ({calib['mecanisme']}).\n{iso_line}\n{ub_line}\n"
                    f"{pl_line}\n{cache_line}\nRien à changer."
                )
            else:
                msg = (
                    f"« {mid} » : rien à changer d'après les mesures disponibles — "
                    f"{', '.join(manques)}. Contexte actuel {current} = mesuré {new} "
                    f"({calib['mecanisme']}).\n{iso_line}\n{ub_line}\n{pl_line}\n{cache_line}"
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
            if ub_change:
                av = calib.get("ubatch_avant") or "défaut"
                changes.append(f"ubatch {av} → {ub['ubatch']} (b={ub['batch']})")
            if pl_change:
                gain = (
                    f" ({pl['gain_pct']:+.0f} % de génération)"
                    if pl.get("gain_pct") is not None
                    else ""
                )
                changes.append(f"placement → {pl['label']}{gain}")
            msg = (
                f"Verdict pour « {mid} » : " + " · ".join(changes) + "\n"
                f"(pente {calib['slope_kb_tok']} Ko/token, {vitesse_txt})\n"
                f"mécanisme : {calib['mecanisme']}\n{iso_line}\n{ub_line}\n{pl_line}\n"
                f"{cache_line}\n"
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
                "ubatch": ub["ubatch"] if ub_change else None,
                "batch": ub["batch"] if ub_change else None,
                "ubatch_detail": (
                    f"{ub['pp_ts']} t/s sur {bench_mod.UBATCH_PROBE_PROMPT} tokens"
                    if ub_change
                    else ""
                ),
                # Placement mesuré AVEC ce contexte et ces slots : appliqué d'un bloc,
                # avec ses mesures détaillées (échantillons) et le build du moteur.
                "placement": (
                    dict(pl, build=calib.get("build")) if pl_change else None
                ),
            }
    except (RuntimeError, ValueError) as exc:
        msg = f"❌ Recalibration de « {mid} » échouée : {exc} — config inchangée."
        wiz = None
    got = chat_lock.acquire(timeout=2)
    try:
        conv = sess.conversation
        # Journal seulement : un compte rendu de commande n'est pas un tour que le
        # modèle doit relire (cf. model_admin._persist_wizard_exchange).
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
