# loom/web/routes/rebench.py — sorti de models.py (comportement constant).
from __future__ import annotations

import os
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
    override_ngl: int | None = None,
    slots: int = 1,
    trace: dict | None = None,
    logical: int | None = None,
    physical: int | None = None,
):
    """Sonde de placement (loom.setup.placement) sur la sonde serveur `probe` : renvoie
    (verdict sérialisable | None, sonde alignée sur l'élu). None quand rien n'est
    mesurable ou que la validation du seul candidat échoue : la calibration vaut alors
    avec les flags actuels du modèle. La faisabilité s'estime au contexte UTILE
    (`useful_ctx`) avec le type de cache de l'exécutant, via le profil GGUF ; la
    configuration ACTUELLE (`mt`) est la ligne de base ; `raw` porte les contraintes
    de prefill optionnelles ([placement]). Avec `logical` (cœurs), chaque finaliste à
    calcul CPU est réglé en threads avant la finale (candidats du parc)."""
    from dataclasses import replace as _dc_replace

    from loom.runtime.model_profile import ModelProfile
    from loom.setup import placement as place_mod

    th_options = None
    if logical and getattr(probe, "threads", None):
        th_options = place_mod.thread_options(
            int(probe.threads), int(logical), physical
        )

    profile = ModelProfile.from_meta(meta, model_size_mb=int(model_size_mb or 0))
    # Mémoire par contexte au-delà des poids : KV au contexte utile + état récurrent
    # (état vivant + checkpoints par slot) — la faisabilité des hybrides ne dépend plus
    # de la seule pente mesurée.
    estimation = place_mod.memory_estimate_mb(
        profile,
        int(useful_ctx or place_mod.PLACEMENT_PROBE_CTX),
        gpu_tuning=bool(getattr(hw, "has_gpu", False)),
        slots=max(1, int(slots or 1)),
        checkpoints=getattr(probe, "ctx_checkpoints", None),
    )
    # Ventilée : KV + état vivant côté DEVICE, checkpoints côté HÔTE (RAM).
    kv_mb = estimation["device_mb"]
    host_mb = estimation["host_mb"]
    if trace is not None:
        trace["memoire_estimee"] = estimation
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
        # Référence = configuration ACTUELLE résolue comme l'exécutant (resolve_ngl).
        current=place_mod.current_placement(
            mt or {},
            n_layers=meta.get("n_layers"),
            size_mb=int(model_size_mb or 0),
            profile=hw,
            override_ngl=override_ngl,
            headroom=headroom_mb,
        ),
        profile=profile,
        host_extra_mb=host_mb,
    )
    if plan.aucun_faisable:
        # Résultat EXPLICITE (revue #14) : rien ne tient d'après l'estimation, CPU seul
        # et configuration actuelle compris — on ne mesure ni n'applique rien.
        if trace is not None:
            trace["plan"] = plan
            trace["profil"] = profile.describe()
            trace["kv_estime_mb"] = kv_mb
            trace["hote_estime_mb"] = host_mb
        raise place_mod.AucunPlacementFaisable(plan.raison)
    prefill_c, pp_floor = place_mod.constraints_from_config(raw or {})
    # Finalistes x deux couples (ubatch, batch) : celui de l'exécutant (la base) et
    # l'alternative du parc — la sonde ubatch séparée disparaît.
    couples = place_mod.batch_couples(
        (getattr(probe, "ubatch", None), getattr(probe, "batch", None))
    )
    if trace is not None:
        # Compte rendu commun : de quoi reproduire la mesure.
        trace["profil"] = profile.describe()
        trace["kv_estime_mb"] = kv_mb
        trace["hote_estime_mb"] = host_mb
        trace["plan"] = plan
        trace["couples"] = couples
        trace["contrainte_prefill"] = prefill_c
        trace["prefill_floor_ratio"] = pp_floor
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
            batch_couples=couples,
            thread_options=th_options,
        )
    except Exception:  # noqa: BLE001 - sonde best-effort : la calibration vaut sans
        res = None
    if not res or res.get("placement") is None or not res["mesures"]:
        return None, probe
    pl = res["placement"]
    extra = (
        {"ubatch": pl.ubatch or probe.ubatch, "batch": pl.batch or probe.batch}
        if hasattr(probe, "ubatch")
        else {}
    )
    if pl.threads and hasattr(probe, "threads"):
        extra["threads"] = int(pl.threads)
    probe = _dc_replace(
        probe, ngl=pl.ngl, cpu_moe=pl.cpu_moe, n_cpu_moe=pl.n_cpu_moe, **extra
    )
    verdict = {
        "label": pl.label,
        "key": pl.key,
        "ngl": pl.ngl,
        "cpu_moe": pl.cpu_moe,
        "n_cpu_moe": pl.n_cpu_moe,
        "ubatch": pl.ubatch,
        "batch": pl.batch,
        "threads": pl.threads,
        "threads_finalistes": res.get("threads_finalistes") or {},
        "couples": res.get("couples"),
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


def _measure_threads(
    probe,
    pl_verdict: dict | None,
    *,
    logical: int,
    physical: int | None,
    ctx: int,
    depth: int,
    progress,
    n_layers: int | None = None,
    prefill=None,
    pp_floor_ratio: float | None = None,
):
    """Sonde de threads sur le placement ÉLU (option par modèle) : renvoie (verdict |
    None, sonde alignée). Tout GPU : {"non_explore": …} sans mesure. Élu déjà réglé
    avant la finale (`threads_finalistes`) : ce verdict est repris, pas remesuré. Sans
    comparaison de placement exploitable, la sonde porte la configuration actuelle :
    on la juge par ses flags."""
    from dataclasses import replace as _dc_replace

    from loom.setup import placement as place_mod

    if pl_verdict is not None:
        pl_key = str(pl_verdict.get("key") or pl_verdict.get("label") or "").split("@")[
            0
        ]
        cpu = pl_verdict.get("label") != "gpu_total"
        deja = (pl_verdict.get("threads_finalistes") or {}).get(pl_key)
        if cpu and deja and deja.get("threads") is not None:
            res = dict(deja)
            res["placement"] = pl_key
            res["mecanisme"] = f"réglés avant la finale — {res.get('mecanisme', '')}"
            if hasattr(probe, "threads") and res["threads"] != getattr(
                probe, "threads", None
            ):
                probe = _dc_replace(probe, threads=int(res["threads"]))
            return res, probe
    else:
        pl_obj = place_mod.Placement.from_flags(
            int(getattr(probe, "ngl", 999) or 0),
            bool(getattr(probe, "cpu_moe", False)),
            getattr(probe, "n_cpu_moe", None),
            n_layers,
        )
        pl_key, cpu = pl_obj.key, place_mod.needs_cpu_compute(pl_obj)
    current = int(getattr(probe, "threads", 0) or 0)
    if not cpu:
        return {
            "non_explore": (
                f"non exploré : {pl_key} sans calcul CPU attendu (threads {current} "
                "conservés)"
            ),
            "placement": pl_key,
        }, probe
    options = place_mod.thread_options(current, logical, physical)
    try:
        res = place_mod.probe_threads(
            lambda o: _dc_replace(probe, threads=o.threads),
            options,
            ctx=ctx,
            depth=depth,
            progress=progress,
            prefill=prefill,
            pp_floor_ratio=pp_floor_ratio,
        )
    except Exception:  # noqa: BLE001 - sonde best-effort : la calibration vaut sans
        res = None
    if not res:
        return None, probe
    res["placement"] = pl_key
    if res["threads"] != current:
        probe = _dc_replace(probe, threads=res["threads"])
    return res, probe


def _placement_implied_ngl(pl: dict):
    """n_gpu_layers que _set_model_placement écrira pour ce verdict (None = retiré) :
    999 tout-GPU, 0 CPU seul, le -ngl exact d'un partiel dense."""
    label = pl.get("label") if isinstance(pl, dict) else pl
    if label == "gpu_partiel" and isinstance(pl, dict):
        return int(pl.get("ngl") or 0)
    return {"gpu_total": 999, "cpu": 0}.get(label)


def _probe_settings(
    meta: dict,
    mt: dict,
    over: dict,
    hw,
    *,
    gpu_backend: bool,
    vram_fallback_mb: int,
    size_mb: int = 0,
    headroom: int = 1024,
) -> tuple[str, int, int, int]:
    """(topologie, VRAM totale, threads, ngl) de la sonde, avec la dérivation de
    l'EXÉCUTANT. La VRAM vient du profil `--list-devices` (Vulkan compris) et
    nvidia-smi n'est qu'un repli : sans ça la 860M passait en topologie « ram » et
    la sonde mesurait sans profil GPU. Threads = effective.launch_flags (override
    machine, sinon cœurs physiques en GPU, tous en CPU). ngl = la configuration
    ACTUELLE résolue par le même résolveur que serve.py / swap.py
    (placement.current_placement -> resolve_ngl : model.toml, override, VRAM libre)."""
    from loom.runtime.effective import launch_flags
    from loom.setup import topology as topo_mod
    from loom.setup.placement import current_placement

    vram = int(getattr(hw, "vram_total_mb", 0) or vram_fallback_mb or 0)
    topo = topo_mod.discover_topology(meta, bool(gpu_backend), vram)
    threads = launch_flags(hw, over.get("threads")).threads
    if topo == topo_mod.TOPO_RAM:
        return topo, vram, threads, 0
    cur = current_placement(
        mt,
        n_layers=meta.get("n_layers"),
        size_mb=size_mb,
        profile=hw,
        override_ngl=over.get("n_gpu_layers"),
        headroom=headroom,
    )
    return topo, vram, threads, int(cur.ngl if cur is not None else 999)


def _run_calibration(S, spec, progress, trace_out: dict | None = None):
    """Cœur de mesure (préconditions + topologie + calibrate), avec les flags EXACTS
    du modèle. Lève RuntimeError actionnable si la machine n'est pas prête.
    Isolé pour être stubbable dans les tests (aucun subprocess en CI).

    `trace_out` : compte rendu PROGRESSIF (schéma archive.BENCH_SCHEMA) rempli étape
    par étape — l'archive le conserve même si une étape lève."""
    trace = trace_out if trace_out is not None else {}
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

    # Le binaire que l'EXÉCUTANT lancera pour CE modèle (model.toml server_bin, sinon le
    # global) : détection matérielle, sonde et build tracé portent sur lui.
    probe_bin = topo_mod.model_server_bin(mt, str(server_bin))
    hw = detect_hardware(probe_bin)
    trace.update(etape="préparation", gguf=str(gguf), server_bin=probe_bin, materiel=hw)
    # Le binaire fait foi (`--list-devices`) : un build statique n'a aucune DLL à côté.
    gpu_backend = bench_mod.gpu_backend_available(hw, probe_bin)
    over = raw.get("override") or {}
    server_cfg = raw.get("server") or {}
    headroom = int(server_cfg.get("gpu_kv_headroom_mb", 640) or 640)
    size_mb = int(spec.get("size_mb") or mt.get("size_mb") or 0)
    topo, vram, threads, ngl = _probe_settings(
        meta,
        mt,
        over,
        hw,
        gpu_backend=gpu_backend,
        vram_fallback_mb=topo_mod.gpu_vram_total_mb(),
        size_mb=size_mb,
        headroom=headroom,
    )
    ram = int(psutil.virtual_memory().total // (1024 * 1024))
    # Mémoire unifiée : le device est la RAM, comptée une fois (= ce que la sonde mesure).
    uma = bool(hw.has_gpu and not hw.vram_is_discrete)
    budget = topo_mod.memory_budget_mb(topo, vram, ram, headroom, uma=uma)
    mmproj = mt.get("mmproj_filename")
    probe = topo_mod.ServerProbe(
        server_bin=probe_bin,
        model_path=str(gguf),
        # Slots de l'exécutant : [server] n_parallel global monté par l'isolation
        # ACTUELLE (model.toml) — un nouveau verdict seul la remplacera. Sans ça, une
        # sonde d'isolation en échec faisait mesurer à 1 slot un modèle qui tourne à 2.
        n_parallel=topo_mod.probe_slots(server_cfg, bool(mt.get("cache_isolation"))),
        threads=threads,
        ngl=ngl,
        topology=topo,
        mmproj_path=str(mdir / mmproj) if mmproj else None,
        cpu_moe=bool(mt.get("cpu_moe", is_moe)),
        n_cpu_moe=mt.get("n_cpu_moe"),
        # Batchs de l'exécutant : modèle, sinon repli machine [server] ubatch/batch.
        ubatch=topo_mod.probe_batches(mt, server_cfg)[0],
        batch=topo_mod.probe_batches(mt, server_cfg)[1],
        # Checkpoints des hybrides : mesurer la mémoire que l'exécutant prendra.
        checkpoint_min_step=(
            mt.get("checkpoint_min_step") or server_cfg.get("checkpoint_min_step")
        ),
        ctx_checkpoints=mt.get("ctx_checkpoints"),
        profile=hw,
    )
    # Isolation D'ABORD (sur la configuration actuelle) : le placement se compare
    # ensuite avec les slots FINAUX, le KV doublé compte dans la faisabilité et dans la
    # mesure (même séquence que loom-setup step_bench — le conseilleur simule l'exécutant).
    progress("sonde d'isolation du cache (A -> pollution -> A)…")
    trace["etape"] = "isolation"
    trace["flags"] = {
        "threads": threads,
        "gpu_tuning": bool(hw.has_gpu),
        "unified_memory": bool(not hw.vram_is_discrete),
        "ubatch": probe.ubatch,
        "batch": probe.batch,
        "slots": probe.n_parallel,
        "checkpoint_min_step": probe.checkpoint_min_step,
    }
    isolation = None
    iso_detail = ""
    iso_first, iso_back = 0, 0
    try:
        first, back = probe.probe_isolation()
        iso_first, iso_back = int(first), int(back)
        isolation = topo_mod.isolation_needed(first, back, meta.get("recurrent"))
        iso_detail = f"retour {back}/{first} tokens retraités"
        if meta.get("recurrent"):
            iso_detail += ", mémoire récurrente"
        # Nouveau verdict : il remplace l'isolation actuelle (dans les deux sens).
        probe.n_parallel = topo_mod.probe_slots(server_cfg, isolation)
    except Exception as exc:  # noqa: BLE001 - sonde best-effort : l'isolation actuelle reste
        iso_detail = f"sonde illisible ({exc}) — isolation actuelle conservée"
    trace["isolation"] = {
        "necessaire": isolation,
        "first": iso_first,
        "back": iso_back,
        "detail": iso_detail,
        "avant": bool(mt.get("cache_isolation", False)),
        "slots_mesure": probe.n_parallel,
    }
    trace["flags"]["slots"] = probe.n_parallel
    # Placement MESURÉ x couples de batchs, avant la calibration, faisabilité estimée au
    # contexte UTILE du modèle avec les slots finaux.
    from loom.setup.placement import final_depth as place_mod_final_depth
    from loom.setup.placement import useful_context

    ctx_utile = useful_context(
        mt.get("context"), server_cfg.get("context"), meta.get("context_length")
    )
    progress("sonde de placement (où vivent les poids, x batchs)…")
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
        override_ngl=over.get("n_gpu_layers"),
        slots=int(getattr(probe, "n_parallel", 1) or 1),
        trace=trace,
        logical=os.cpu_count() or 4,
        physical=psutil.cpu_count(logical=False),
    )
    trace["placement"] = pl_verdict
    trace["placement_avant"] = {
        "cpu_moe": bool(mt.get("cpu_moe", is_moe)),
        "n_cpu_moe": mt.get("n_cpu_moe"),
        "n_gpu_layers": mt.get("n_gpu_layers"),
    }
    # Threads sur le placement ÉLU (option par modèle) : réglés avant la finale quand il
    # y a eu finale (verdict repris), sinon mesurés maintenant au contexte et à la
    # profondeur de la finale.
    from loom.setup.placement import constraints_from_config

    prefill_c, pp_floor = constraints_from_config(raw)
    trace["etape"] = "threads"
    progress("sonde de threads sur le placement élu…")
    th_verdict, probe = _measure_threads(
        probe,
        pl_verdict,
        logical=os.cpu_count() or 4,
        physical=psutil.cpu_count(logical=False),
        ctx=int((pl_verdict or {}).get("ctx_final") or ctx_utile),
        depth=int(
            (pl_verdict or {}).get("depth_final") or place_mod_final_depth(ctx_utile)
        ),
        progress=progress,
        n_layers=meta.get("n_layers"),
        prefill=prefill_c,
        pp_floor_ratio=pp_floor,
    )
    trace["threads"] = th_verdict
    trace["etape"] = "calibration"
    progress(f"topologie {topo}, budget {budget} Mo")
    calib = topo_mod.calibrate(
        probe, meta, topology=topo, budget_mb=budget, progress=progress
    )
    trace["calibration"] = dict(calib)
    trace["etape"] = "batchs"
    calib["isolation"] = isolation
    calib["isolation_detail"] = iso_detail
    calib["isolation_first"] = iso_first
    calib["isolation_back"] = iso_back
    calib["isolation_avant"] = bool(mt.get("cache_isolation", False))
    if pl_verdict and pl_verdict.get("ubatch"):
        # Les batchs viennent du 2x2 des finalistes (même contexte, même profondeur,
        # mêmes slots que le placement) : la sonde ubatch séparée est obsolète.
        calib["ubatch_probe"] = {
            "ubatch": int(pl_verdict["ubatch"]),
            "batch": int(pl_verdict.get("batch") or pl_verdict["ubatch"]),
            "pp_ts": float(pl_verdict.get("pp_ts") or 0.0),
            "gain_pct": None,
            "mesures": {
                k: v.get("pp_ts")
                for k, v in (pl_verdict.get("mesures") or {}).items()
                if "pp_ts" in v
            },
            "origine": "finalistes x batchs",
        }
    else:
        # Repli : aucune comparaison de placement exploitable, sonde ubatch classique.
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
    calib["threads_probe"] = th_verdict
    calib["threads_avant"] = mt.get("threads")
    trace["ubatch"] = calib.get("ubatch_probe")
    trace["ubatch_avant"] = [mt.get("ubatch"), mt.get("batch")]
    trace["etape"] = "réglage final"
    # Sonde FINALE = placement élu + slots décidés + batchs mesurés : valider ce réglage
    # complet au contexte CALIBRÉ, à la profondeur de la comparaison, puis vérifier le
    # cache avec ce contexte alloué. Le verdict n'assemble plus des mesures prises avec
    # des paramètres différents.
    from dataclasses import replace as _dc_replace

    from loom.setup import placement as place_mod

    ub = calib.get("ubatch_probe")
    probe_final = probe
    if ub:
        try:
            probe_final = _dc_replace(probe, ubatch=ub["ubatch"], batch=ub["batch"])
        except Exception:  # noqa: BLE001 - sonde non clonable : on garde l'originale
            probe_final = probe
    try:
        calib["final"] = place_mod.validate_final(
            probe_final,
            ctx=int(calib["context"]),
            depth=place_mod.final_depth(ctx_utile),
            n_layers=meta.get("n_layers"),
            reference_tg=(pl_verdict or {}).get("tg_ts"),
            prefill=prefill_c,
            # Insatisfiable = aucun candidat comparé ne la tenait : avertir, pas bloquer.
            prefill_insatisfiable=place_mod.prefill_satisfiable(
                (pl_verdict or {}).get("mesures") or {}, prefill_c
            )
            is False,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - validation best-effort, nommée
        calib["final"] = {
            "echec": f"{type(exc).__name__}: {exc}",
            "ctx": calib["context"],
        }
    trace["final"] = calib.get("final")
    trace["etape"] = "cache"
    try:
        progress("vérification du cache avec la configuration finale…")
        calib["cache_verifie"] = probe_final.verify_cache(ctx=int(calib["context"]))
    except Exception:  # noqa: BLE001 - vérification best-effort : le verdict le dira
        calib["cache_verifie"] = None
    # Le moteur avec lequel tout a été mesuré, pour le commentaire du model.toml.
    try:
        from loom.setup.llama_release import verify_binary

        calib["build"] = verify_binary(probe_bin) or "build ?"
    except Exception:  # noqa: BLE001 - best-effort
        calib["build"] = "build ?"
    trace["cache"] = calib.get("cache_verifie")
    trace["build"] = calib.get("build")
    trace["contexte_utile"] = ctx_utile
    trace["etape"] = "fin"
    calib["ctx_utile"] = ctx_utile
    calib["placement"] = pl_verdict
    calib["placement_avant"] = {
        "cpu_moe": bool(mt.get("cpu_moe", is_moe)),
        "n_cpu_moe": mt.get("n_cpu_moe"),
        "n_gpu_layers": mt.get("n_gpu_layers"),
    }
    return calib, gguf


def _sections_from_calib(calib: dict, gguf) -> dict:
    """Sections du compte rendu reconstituées depuis le résultat de calibration (repli
    quand la trace progressive n'a pas tout : stubs, anciens appelants)."""
    return {
        "gguf": str(gguf),
        "build": calib.get("build"),
        "contexte_utile": calib.get("ctx_utile"),
        "calibration": {
            k: calib.get(k)
            for k in (
                "context",
                "valide",
                "mode",
                "mecanisme",
                "slope_kb_tok",
                "base_mb",
                "budget_mb",
                "capacity_ctx",
                "rungs",
                "vitesses",
                "valide_jusqua",
                "duree_s",
            )
        },
        "isolation": {
            "necessaire": calib.get("isolation"),
            "first": calib.get("isolation_first"),
            "back": calib.get("isolation_back"),
            "detail": calib.get("isolation_detail"),
            "avant": calib.get("isolation_avant"),
        },
        "placement": calib.get("placement"),
        "placement_avant": calib.get("placement_avant"),
        "ubatch": calib.get("ubatch_probe"),
        "ubatch_avant": [calib.get("ubatch_avant"), calib.get("batch_avant")],
        "threads": calib.get("threads_probe"),
        "final": calib.get("final"),
        "cache": calib.get("cache_verifie"),
    }


def _rebench_worker(S, sess, chat_lock, mid, job):
    """Thread du job : mesure, verdict comparé, message PERSISTÉ + état b_apply si
    une application a du sens. `job.done` posé EN DERNIER (le stream lit final)."""
    from loom.setup import bench as bench_mod
    from loom.setup import topology as topo_mod
    from loom.setup.placement import AucunPlacementFaisable

    spec = next((m for m in S.local_model_specs if m.get("id") == mid), None)
    calib = None
    _gguf = None
    # Compte rendu PROGRESSIF (archive.BENCH_SCHEMA) : rempli étape par étape par
    # _run_calibration, archivé même si une étape lève.
    trace: dict = {"source": "/rebench"}
    erreur: str | None = None
    try:
        calib, _gguf = _run_calibration(
            S,
            spec,
            lambda m: setattr(job, "label", f"calibration : {m}"),
            trace_out=trace,
        )
        current = int(spec.get("context") or S.context_window or 0)
        new = calib["context"]
        iso = calib.get("isolation")
        iso_change = iso is not None and iso != calib.get("isolation_avant", False)
        if "isolation_first" in calib:
            # Libellé honnête : ce que la mesure a montré, et pourquoi on isole.
            iso_line = "sonde d'isolation : " + topo_mod.isolation_text(
                iso, calib["isolation_first"], calib["isolation_back"]
            )
        elif iso is None:
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
        # Threads mesurés sur le placement élu (par modèle) : changement si la valeur
        # mesurée diffère du `threads` du model.toml.
        th = calib.get("threads_probe")
        th_avant = calib.get("threads_avant")
        th_change = (
            bool(th)
            and th.get("threads") is not None
            and (th.get("compare") and th["threads"] != th_avant)
        )
        if not th:
            th_line = "sonde de threads : illisible (réglage inchangé)."
        elif th.get("non_explore"):
            th_line = f"sonde de threads : {th['non_explore']}."
        else:
            th_line = f"sonde de threads sur {th.get('placement')} : {th['mecanisme']}."
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
        # Validation du RÉGLAGE FINAL complet au contexte calibré.
        fin = calib.get("final")
        if not fin or "echec" in fin:
            final_line = "réglage final : non validé" + (
                f" ({fin['echec']})." if fin and fin.get("echec") else "."
            )
        else:
            from loom.setup.placement import checkpoints_text, final_checks_text

            # Génération et contrainte de prefill dites SÉPARÉMENT (revue #14).
            final_line = (
                f"réglage final {fin['placement']} (ctx {fin['ctx']}, {fin['slots']} slots, "
                f"ub {fin['ubatch']}/b {fin['batch']}) : génération {fin['tg_ts']} t/s, "
                f"prefill {fin['pp_ts']} t/s à profondeur {fin['depth']}"
                f"{final_checks_text(fin)}{checkpoints_text(fin)}."
            )
        # Un plancher n'est pas une mesure : le verdict le dit.
        valide = bool(calib.get("valide", True))
        vitesse_txt = (
            f"vitesse validée jusqu'à {calib['valide_jusqua']} tokens"
            if valide
            else f"contexte {new} = repli NON validé, aucun barreau de vitesse mesuré"
        )
        if (
            new == current
            and not iso_change
            and not ub_change
            and not pl_change
            and not th_change
        ):
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
            if not fin or "echec" in fin:
                manques.append("réglage final non validé")
            else:
                if fin.get("coherent") is False:
                    manques.append(
                        "génération du réglage final incohérente avec la mesure de "
                        "placement"
                    )
                pc_fin = fin.get("prefill_contrainte") or {}
                if pc_fin and not pc_fin.get("respectee"):
                    manques.append(
                        "contrainte prefill non respectée par le réglage final"
                    )
            if not manques:
                msg = (
                    f"✅ « {mid} » est déjà au top : contexte actuel {current} = "
                    f"mesuré {new} ({calib['mecanisme']}).\n{iso_line}\n{ub_line}\n"
                    f"{pl_line}\n{th_line}\n{final_line}\n{cache_line}\nRien à changer."
                )
            else:
                msg = (
                    f"« {mid} » : rien à changer d'après les mesures disponibles — "
                    f"{', '.join(manques)}. Contexte actuel {current} = mesuré {new} "
                    f"({calib['mecanisme']}).\n{iso_line}\n{ub_line}\n{pl_line}\n"
                    f"{th_line}\n{final_line}\n{cache_line}"
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
            if th_change:
                gain_th = (
                    f" ({th['gain_pct']:+.0f} % de génération)"
                    if th.get("gain_pct") is not None
                    else ""
                )
                changes.append(
                    f"threads {th_avant or 'machine'} → {th['threads']}{gain_th}"
                )
            entete = (
                f"Verdict pour « {mid} » : " + " · ".join(changes) + "\n"
                f"(pente {calib['slope_kb_tok']} Ko/token, {vitesse_txt})\n"
                f"mécanisme : {calib['mecanisme']}\n{iso_line}\n{ub_line}\n{pl_line}\n"
                f"{th_line}\n{final_line}\n{cache_line}\n"
            )
            if fin and "echec" in fin:
                # Une baisse de vitesse avertit ; un échec de FONCTIONNEMENT empêche :
                # rien à appliquer, la configuration actuelle est conservée.
                msg = entete + (
                    "⛔ non applicable : la configuration finale complète n'a pas "
                    "fonctionné — réglages actuels conservés."
                )
                wiz = None
            elif fin and fin.get("bloquant"):
                # La contrainte EXPLICITE de prefill, tenue par un candidat, n'est pas
                # tenue par le réglage final : ce qui a été demandé n'est pas livré.
                msg = entete + (
                    f"⛔ non applicable : {fin['bloquant']} — réglages actuels conservés."
                )
                wiz = None
            else:
                msg = entete + (
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
                    # Validation du réglage final : conservée avec le verdict.
                    "final": calib.get("final"),
                    # Threads mesurés sur le placement élu (par modèle).
                    "threads": th["threads"] if th_change else None,
                    "threads_detail": (
                        f"{th.get('mecanisme', '')} (sur {th.get('placement')} à ctx "
                        f"{th.get('ctx')}, profondeur {th.get('depth')})"
                        if th_change
                        else ""
                    ),
                }
    except AucunPlacementFaisable as exc:
        # Pas un plantage : un résultat de l'estimation, dit tel quel (revue #14).
        msg = (
            f"⛔ « {mid} » : {exc}. Rien n'a été mesuré ni appliqué — configuration "
            "actuelle conservée."
        )
        wiz = None
        erreur = str(exc)
        trace["etape"] = "placement"
    except Exception as exc:  # noqa: BLE001 - erreurs opérationnelles comprises (OSError…)
        # Une FileNotFoundError ou PermissionError pendant la calibration laissait le job
        # sans fin et sans archive : tout échec devient un verdict d'échec archivé.
        msg = (
            f"❌ Recalibration de « {mid} » échouée : {type(exc).__name__}: {exc} — "
            "config inchangée."
        )
        wiz = None
        erreur = f"{type(exc).__name__}: {exc}"
    # Archive DURABLE du bench (var/bench/<modèle>/<horodatage>.json) : le compte rendu
    # PROGRESSIF (schéma commun), en échec comme en succès — l'état de session est
    # consommé par « oui » ou effacé par « annuler », pas l'archive. Un échec
    # d'écriture est DIT dans le verdict, jamais silencieux.
    from loom.setup import archive as archive_mod

    sections = _sections_from_calib(calib, _gguf) if calib is not None else {}
    sections.update({k: v for k, v in trace.items() if v is not None})
    if erreur is not None:
        sections["echec"] = {"etape": trace.get("etape"), "erreur": erreur}
    try:
        arch = archive_mod.archive_bench(
            mid, archive_mod.bench_payload(**sections, verdict_texte=msg, verdict=wiz)
        )
        if wiz is not None:
            wiz["archive"] = str(arch)
    except Exception as exc:  # noqa: BLE001 - l'archive n'empêche jamais le verdict
        msg += f"\n⚠ archive non écrite : {exc}"
    try:
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
    finally:
        # TOUJOURS finaliser : le flux et le verrou anti-double attendent ce drapeau.
        if getattr(job, "final", None) is None:
            job.final = msg
        job.done = True
