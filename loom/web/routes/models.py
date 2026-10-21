from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from flask import Response, render_template, request

from loom.runtime import model_install, model_store
from loom.runtime.comfy import ComfyError
from loom.utils import atomic_write_text
from loom.web import wizard as _wizard
from loom.web.routes.config import _regen_swap_yaml
from loom.web.routes.helpers import _client_mark_all_cold, _ctx, _engine_for, _session
from loom.web.routes.model_admin import (
    _LOCAL_EDITABLE,
    _check_workflow,
    _finish_install,
    _forget_image,
    _forget_remote,
    _image_base_dir,
    _image_dir_state,
    _install_roots,
    _list_remote_models,
    _model_kind,
    _models_payload,
    _models_roots,
    _mount_image,
    _mount_remote,
    _rebenchable_models,
    _remote_list,
    _removable_models,
)
from loom.web.routes.rebench import _REBENCH, _rebench_worker


@dataclass
class _ModelAddFlow:
    """UI state only: never stored in a conversation or its event journal."""

    wizard: dict | None = None
    reply: str = ""
    choices: list = field(default_factory=list)
    job: object = None
    lock: object = field(default_factory=threading.Lock)

    def set_wizard(self, state):
        self.wizard = state

    def busy_now(self):
        """Travail de fond en cours (hors verrou de la requête qui répond)."""
        return bool(self.job and not self.job.done)

    def snapshot(self):
        job = self.job
        busy = self.lock.locked() or self.busy_now()
        return {
            "reply": self.reply
            + (
                "\n" + job.final_message
                if job and job.done and job.final_message
                else ""
            ),
            "choices": self.choices,
            "active": bool(self.wizard),
            "secret": (self.wizard or {}).get("step") == "r_key",
            "busy": busy,
            "status": f"Téléchargement… {job.progress_mb()}/{job.total_mb} Mo"
            if job and not job.done
            else "",
        }


@dataclass
class _RebenchFlow(_ModelAddFlow):
    """Panneau « Recalibrer un modèle » : UN seul pour toute la machine (une calibration
    à la fois, cf. _REBENCH), hors conversation comme l'ajout. Le job est le job
    global ; son journal de progression est relu par le panneau."""

    def busy_now(self):
        job = _REBENCH["job"]
        return bool(job and not job.done)

    def snapshot(self):
        job = _REBENCH["job"]
        running = self.busy_now()
        return {
            "reply": self.reply,
            "choices": self.choices,
            "active": bool(self.wizard),
            "secret": False,
            "busy": self.lock.locked() or running,
            "status": (job.label or "calibration en cours…") if running else "",
            "log": list(getattr(job, "log", None) or ()) if job else [],
        }


def _cancel_interface_command(S, sess):
    """Cancel a dialog, never generate or enqueue a chat message."""
    flow = S.model_add_flows.get(sess.id)
    if flow is not None:
        if not flow.lock.acquire(blocking=False):
            return {"error": "Une étape est en cours. Réessaie à sa fin."}, 409
        try:
            if flow.job and not flow.job.done:
                return {
                    "error": "Le téléchargement continue en arrière-plan ; tu peux fermer le panneau."
                }, 409
            flow.set_wizard(None)
            flow.reply, flow.choices = "Ajout annulé.", []
            flow.job = None
        finally:
            flow.lock.release()
    rb = getattr(S, "rebench_flow", None)
    if rb is not None and rb.lock.acquire(blocking=False):
        try:
            job = _REBENCH["job"]
            if job is not None and not job.done:
                return {
                    "error": "La calibration continue ; son verdict s'affichera dans le panneau."
                }, 409
            if rb.wizard:
                rb.set_wizard(None)
                rb.reply, rb.choices = (
                    "Recalibration annulée — rien n'a été touché.",
                    [],
                )
        finally:
            rb.lock.release()
    # Clear a legacy dialog only while its conversation is idle.
    from loom.web.routes.helpers import _lock_for

    lock = _lock_for(S, sess.id)
    if lock.acquire(blocking=False):
        try:
            if sess.conversation.wizard:
                sess.conversation.set_wizard(None)
                S.session_store.save(sess)
        finally:
            lock.release()
    return {"ui_action": "cancel_interface", "message": "Commande d’interface annulée."}


def _repo_note(analysis: dict, vram_mb: int) -> str:
    """Une ligne sur ce qui gouverne la vitesse du modèle sur CETTE machine."""
    if analysis["moe"]:
        kind = (
            f"MoE : {analysis['expert_used']} experts actifs sur "
            f"{analysis['expert_count']}, seuls les actifs sont relus à chaque token"
        )
    else:
        kind = "modèle dense : tous les poids sont relus à chaque token"
    vram = f"{vram_mb / 1024:.1f} Go de VRAM" if vram_mb else "mémoire unifiée"
    return f"{kind} ({vram})"


def _wizard_deps(S):
    """Dépendances du wizard (INJECTÉES : la machine à états reste pure et testable).
    Point de patch des tests — garder la construction ici, jamais dans wizard.py."""
    from types import SimpleNamespace

    from loom.runtime import gguf_remote, hardware, hf_catalog
    from loom.web.routes.model_admin import _configured_server_bin

    # Sans le binaire, la détection ne voit pas la VRAM totale (vécu 2026-10-17 : 0 Mo
    # au lieu de 6 143 sur une RTX 2060, donc aucun quant jugé « en VRAM »).
    server_bin = _configured_server_bin()
    hw = hardware.detect_hardware(server_bin)
    # Compter la VRAM seulement si elle est discrète pour ne pas doubler la mémoire unifiée.
    ram_total = hardware.ram_total_mb()
    vram_budget = hw.vram_total_mb if hw.vram_is_discrete else 0
    supported = gguf_remote.supported_formats(str(server_bin) if server_bin else None)

    import tomllib

    from loom.runtime.recommendation import recommend_repository

    server_settings = {}
    for path in (S.config_defaults_path, S.config_local_path):
        if path and Path(path).exists():
            server_settings.update(
                tomllib.loads(Path(path).read_text(encoding="utf-8")).get("server", {})
            )

    def evaluate(repo, files):
        # Match the configured working context; never advertise architectural max context.
        context = max(16384, int(server_settings.get("context", 16384)))
        return recommend_repository(
            repo,
            files,
            vram_mb=vram_budget,
            ram_mb=ram_total,
            server_bin=server_bin,
            supported=supported,
            context=context,
            slots=int(server_settings.get("n_parallel", 1)),
            gpu_tuning=hw.has_gpu,
            max_tokens=int(S.settings.get("max_tokens", 8192)),
        )

    def alternatives(repo, analysis):
        # Stay within the selected family, then its declared base model.
        hits = []
        try:
            stem = model_install.family_stem(repo)
            hits = hf_catalog.search_family(stem) if stem else []
            base = next((h.get("base") for h in hits if h["repo_id"] == repo), None)
            base_stem = model_install.family_stem(base)
            if base_stem and base_stem != stem:
                hits += hf_catalog.search_family(base_stem)
        except hf_catalog.HfCatalogError:
            pass
        found = model_install.suggest_alternatives(
            repo,
            analysis,
            hits,
            hf_catalog.list_gguf_files,
            vram_budget,
            ram_total,
            supported,
            recommender=lambda candidate, files: evaluate(candidate, files)[0],
        )
        # Filiation lue sur Hugging Face, en parallèle : une variante tierce est signalée
        # (vécu : un « Qwen3.8-35B-A3B » d'Empero pris pour une sortie officielle).
        from concurrent.futures import ThreadPoolExecutor
        from functools import lru_cache

        parent_of = lru_cache(maxsize=64)(hf_catalog.base_model_of)

        def origin(alt):
            try:
                return model_install.third_party_origin(alt["repo_id"], parent_of)
            except hf_catalog.HfCatalogError:
                return None  # filiation illisible : rien n'est affirmé

        with ThreadPoolExecutor(max_workers=4) as pool:
            for alt, source in zip(found, pool.map(origin, found)):
                alt["derived_from"] = source
        return found

    def recommend(files, repo=None):
        out, analysis = evaluate(repo, files)
        if out:
            if analysis:
                out[0]["repo_note"] = _repo_note(analysis, vram_budget)
            if repo and not any(f["recommended"] for f in out):
                out[0]["alternatives"] = alternatives(repo, analysis)
        return out

    return SimpleNamespace(
        search_models=hf_catalog.search_models,
        list_gguf_files=hf_catalog.list_gguf_files,
        recommend=recommend,
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


def _checked_model_dir(S, directory):
    target = Path(directory).resolve()
    for root in _models_roots(S):
        root = Path(root).resolve()
        if target.is_relative_to(root):
            parts = target.relative_to(root).parts
            if (
                len(parts) == 3
                and parts[0] == "local"
                and parts[1] in {"text", "image", "video"}
            ):
                return target
    raise OSError("dossier hors des racines de modèles configurées")


def _remove_model(S, a):
    """Removal shared by the management panel and legacy wizard; never calls the LLM."""
    extra_reply, models_changed = "", False
    if a["model_kind"] in ("remote", "remote_config"):
        # Purger toutes les racines et les emplacements hérités sans les recréer.
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
                shutil.rmtree(_checked_model_dir(S, im.dir))
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

        spec = next((m for m in S.local_model_specs if m.get("id") == a["id"]), None)
        # Un modèle non monté (refusé, interrompu) n'a pas de spec : dossier de l'item.
        mdir = (spec or {}).get("dir") or a.get("dir")
        try:
            if mdir:
                shutil.rmtree(_checked_model_dir(S, mdir))
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
            # Sous Windows, refuser toute suppression partielle d'un GGUF encore mappé.
            extra_reply = (
                f"\n❌ Fichiers de « {a['id']} » verrouillés — le modèle est "
                "probablement CHARGÉ. Éteins le serveur modèle (ou charge un "
                "autre modèle), puis relance /remove-model."
            )
        except OSError as exc:
            extra_reply = f"\n❌ Suppression impossible : {exc}"
    return extra_reply.strip(), models_changed


def _handle_add_model_command(S, message, conv):
    """Une étape d'un panneau (ajout `_ModelAddFlow`, recalibration `_RebenchFlow`) :
    machine à états pure (wizard.py) + effets ici, jamais le modèle. Renvoie le
    snapshot du panneau (dict JSON)."""
    is_cmd = message == "/add-model" or message.startswith("/add-model ")
    is_rm = message == "/remove-model" or message.startswith("/remove-model ")
    is_rb = message == "/rebench" or message.startswith("/rebench ")

    try:
        if message.strip().lower() in _wizard.CANCEL_WORDS:
            res = _wizard.WizardResult(None, "Commande annulée.")
        elif is_rb:
            deps = _wizard_deps(S)
            res = _wizard.start_rebench(message[len("/rebench") :].strip(), deps)
        elif is_rm:
            deps = _wizard_deps(S)
            res = _wizard.start_remove(deps)
        elif is_cmd:
            deps = _wizard_deps(S)
            res = _wizard.start(message[len("/add-model") :].strip(), deps)
        else:
            deps = _wizard_deps(S)
            res = _wizard.step(conv.wizard, message, deps)
    except Exception as exc:  # noqa: BLE001 - HfCatalogError & co : actionnable, jamais de stacktrace
        res = _wizard.WizardResult(
            conv.wizard,
            (
                "Le serveur Loom utilise une ancienne version du code. Redémarre Loom pour charger les changements."
                if isinstance(exc, ImportError) and "GPU_KV_BYTES" in str(exc)
                else f"{exc}\nRéessaie ou annule cette opération."
            ),
        )

    conv.set_wizard(res.state)

    job = None
    extra_reply = ""
    models_changed = False
    if res.action and res.action["kind"] == "rebench":
        cur = _REBENCH["job"]
        if cur is not None and not cur.done:
            extra_reply = (
                "\n⏳ Une calibration est déjà en cours — attends sa fin "
                "(le verdict s'affichera dans ce panneau)."
            )
        else:
            from types import SimpleNamespace

            # Le banc doit récupérer la VRAM occupée par le serveur.
            S.server_manager.stop()
            _client_mark_all_cold(S)
            rb_job = SimpleNamespace(done=False, label="", final=None, log=[])
            _REBENCH["job"] = rb_job
            threading.Thread(
                target=_rebench_worker,
                args=(S, conv, res.action["id"], rb_job),
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

            from loom.setup.cli import (
                _set_model_cache_isolation,
                _set_model_context,
                _set_model_placement,
                _set_model_threads,
                _set_model_ubatch,
            )

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
            if a.get("ubatch"):
                _set_model_ubatch(
                    gguf, a["ubatch"], a["batch"], a.get("ubatch_detail", "")
                )
                applied += f" + ubatch={a['ubatch']}/batch={a['batch']}"
            if a.get("placement"):
                import datetime as _dt

                from loom.setup.placement import Placement

                pl = a["placement"]
                _set_model_placement(
                    gguf,
                    Placement(
                        pl["label"],
                        int(pl["ngl"]),
                        cpu_moe=bool(pl.get("cpu_moe")),
                        n_cpu_moe=pl.get("n_cpu_moe"),
                    ),
                    f"{_dt.date.today().isoformat()} (/rebench, "
                    f"{pl.get('build') or 'build ?'}) — {pl.get('mecanisme', '')}",
                )
                applied += f" + placement={pl['label']}"
            if a.get("threads"):
                _set_model_threads(gguf, int(a["threads"]), a.get("threads_detail", ""))
                applied += f" + threads={a['threads']}"
            note_ok = True
            if a.get("archive"):
                # Archive durable du bench : noter ce qui vient d'être appliqué. Un
                # échec d'annotation est DIT : appliqué, mais sans trace.
                from loom.setup.archive import note_application

                pl_a = a.get("placement") or {}
                note_ok = note_application(
                    a["archive"],
                    {
                        "context": a["context"],
                        "isolation": a.get("isolation"),
                        "ubatch": a.get("ubatch"),
                        "batch": a.get("batch"),
                        "placement": pl_a.get("key") or pl_a.get("label"),
                    },
                )
            spec["context"] = a["context"]
            S.model_contexts[a["id"]] = a["context"]
            _regen_swap_yaml(S)
            extra_reply = (
                f"\n✅ {applied} écrit dans le model.toml de "
                f"« {a['id']} » — effet au prochain chargement du modèle."
            )
            if not note_ok:
                extra_reply += (
                    "\n⚠ configuration appliquée, annotation de l'archive échouée "
                    f"({a.get('archive')})."
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
        extra_reply, models_changed = _remove_model(S, res.action)
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
            # Respecter le disque choisi par le wizard multi-racines.
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
                    S, None, conv.lock, a["model_id"], mdir, j
                ),
            )

    conv.reply, conv.choices, conv.job = res.reply + extra_reply, res.choices or [], job
    # `busy` hors verrou de CETTE requête : seul un travail de fond (téléchargement,
    # calibration) fait sonder le panneau.
    return dict(conv.snapshot(), busy=conv.busy_now(), models_changed=models_changed)


def _register_model_routes(app, S):
    S.model_add_flows = {}
    S.rebench_flow = _RebenchFlow()
    flows_lock = threading.Lock()

    @app.route("/models/add", methods=["GET", "POST"])
    def models_add():
        from loom.web.routes.helpers import _get_session

        sid = request.values.get("session_id", "").strip()
        sess = _get_session(S, sid) if sid else _session(S)
        if sess is None:
            return {"error": "session introuvable"}, 404
        with flows_lock:
            flow = S.model_add_flows.setdefault(sess.id, _ModelAddFlow())
        if request.method == "GET":
            return flow.snapshot()
        message = request.form.get("message", "").strip()
        if message == "/cancel":
            return _cancel_interface_command(S, sess)
        if not message or message.split()[0] in {"/rebench", "/remove-model"}:
            return {"error": "Réponse d’ajout invalide."}, 400
        if message.split()[0] == "/add-model":
            message = " ".join(message.split(maxsplit=1))
        if not flow.lock.acquire(blocking=False):
            return {"error": "Une étape est déjà en cours."}, 409
        try:
            if flow.job and not flow.job.done:
                return {"error": "Un téléchargement est déjà en cours."}, 409
            if not flow.wizard and message.split()[0] != "/add-model":
                return {"error": "Ouvre d’abord l’assistant d’ajout."}, 409
            # Existing installer and recommendations, with isolated UI state and lock.
            return _handle_add_model_command(S, message, flow)
        finally:
            flow.lock.release()

    @app.route("/models/rebench", methods=["GET", "POST"])
    def models_rebench():
        # Panneau machine-wide : pas de session (le verdict ne va dans aucun journal).
        flow = S.rebench_flow
        if request.method == "GET":
            return flow.snapshot()
        message = request.form.get("message", "").strip()
        if message == "/cancel":
            return _cancel_interface_command(S, _session(S))
        if not message or message.split()[0] in {"/add-model", "/remove-model"}:
            return {"error": "Réponse de recalibration invalide."}, 400
        if message.split()[0] == "/rebench":
            message = " ".join(message.split(maxsplit=1))
        if not flow.lock.acquire(blocking=False):
            return {"error": "Une étape est déjà en cours."}, 409
        try:
            if not flow.wizard and message.split()[0] != "/rebench":
                return {"error": "Ouvre d’abord le panneau de recalibration."}, 409
            return _handle_add_model_command(S, message, flow)
        finally:
            flow.lock.release()

    @app.get("/models/removable")
    def models_removable():
        busy = {
            h.get("model")
            for h in list(S.active_streams.values())
            if isinstance(h, dict)
        }
        busy.update(
            sess.conversation.model
            for sid, sess in list(S.sessions_cache.items())
            if S.sess_locks.get(sid) and S.sess_locks[sid].locked()
        )
        return {"models": [dict(m, busy=m["id"] in busy) for m in _removable_models(S)]}

    @app.delete("/models/installed/<mid>")
    def models_installed_delete(mid):
        from loom.web.routes.helpers import _lock_for

        held = []
        S.model_admin_lock.acquire()
        try:
            item = next((m for m in _removable_models(S) if m["id"] == mid), None)
            if item is None:
                return {"error": "modèle introuvable"}, 404
            # Protect active and about-to-start turns without stopping other models.
            for sess in list(S.sessions_cache.values()):
                if sess.conversation.model != mid:
                    continue
                lock = _lock_for(S, sess.id)
                if not lock.acquire(blocking=False):
                    return {
                        "error": "Ce modèle travaille encore. Attends la fin de sa génération."
                    }, 409
                held.append(lock)
            if any(
                isinstance(h, dict) and h.get("model") == mid
                for h in list(S.active_streams.values())
            ):
                return {"error": "Ce modèle travaille encore."}, 409
            message, ok = _remove_model(S, dict(item, model_kind=item["kind"]))
            if ok:
                fallback = S.models[0] if S.models else ""
                if S.session_store.default_model == mid:
                    S.session_store.set_default_model(fallback)
                for sess in list(S.sessions_cache.values()):
                    if sess.conversation.model == mid:
                        sess.conversation.set_model(fallback)
                        S.session_store.save(sess)
            return {
                "ok": ok,
                "message": message,
                "models": _models_payload(S),
            }, 200 if ok else 409
        except OSError as exc:
            return {"error": f"Suppression impossible : {exc}"}, 409
        finally:
            for lock in reversed(held):
                lock.release()
            S.model_admin_lock.release()

    # Import tardif requis pour rompre le cycle avec les routes de chat.
    from loom.web.routes.chat import CHAT_COMMANDS
    from loom.web.routes.priming import _prime_async

    @app.post("/model")
    def model_update():
        conv, save = _ctx(S)

        model = request.form.get("model", "")

        if getattr(conv, "local_only", False) and model in S.remote_model_ids:
            # Session privée : son historique ne part jamais vers une API distante. On
            # renvoie le sélecteur inchangé (HTMX le remet en place) et on explique.
            resp = Response(
                render_template(
                    "_models.html",
                    models=S.models,
                    current_model=conv.model,
                    remote_model_ids=S.remote_model_ids,
                    image_model_ids=S.image_model_ids,
                    video_model_ids=S.video_model_ids,
                    model_descriptions=S.model_descriptions,
                )
            )
            resp.headers["HX-Trigger"] = json.dumps(
                {
                    "loomToast": f"Session privée : « {model} » est distant, il ne "
                    "peut pas servir cette session. Ouvre une nouvelle session pour "
                    "l'utiliser."
                }
            )
            return resp

        conv.set_model(model)

        save()

        # Le choix courant devient le défaut des prochaines sessions.

        S.session_store.set_default_model(model)

        # Sélectionner un distant ne doit pas décharger un local utilisé par un autre onglet.
        if model in S.remote_model_ids:
            pass
        elif model in S.image_model_ids:
            # Préchauffer ComfyUI seulement si aucune génération locale ne possède la VRAM.
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
            # Amorcer le vrai préfixe en fond après le démarrage du serveur local.
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

    # Les modèles distants se montent à chaud depuis leur model.toml.
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
        b = request.get_json(silent=True) or {}
        mid = (b.get("id") or "").strip()
        if mid and not model_store.valid_model_id(mid):
            return {
                "error": f"id invalide « {mid} » (lettres, chiffres, -_. ; "
                "pas de point initial)"
            }, 400
        roots = _models_roots(S)
        if not roots:
            return {"error": "racine des modèles indisponible"}, 500
        base_url = (b.get("base_url") or "").strip().rstrip("/")
        model = (b.get("model") or "").strip()
        if not (mid and base_url and model):
            return {"error": "id, base_url et model sont requis"}, 400
        if mid in S.models and mid not in S.remote_model_ids:
            return {"error": f"'{mid}' est déjà un modèle local"}, 400
        # Une édition sans clé conserve celle déjà montée.
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
        # Modifier la racine existante évite un doublon masqué par la priorité.
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
        if mid not in S.remote_model_ids:
            return {"error": "modèle distant inconnu"}, 404
        payload, status = models_installed_delete(mid)
        if status == 200:
            payload["remotes"] = _remote_list(S)
        return payload, status

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
                atomic_write_text(tp, tomlkit.dumps(doc))
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)[:120]}, 400
        # Régénérer puis décharger applique le nouveau tuning au prochain usage. Seule
        # une écriture réelle le justifie : « différé » vise un modèle qui ne tourne pas
        # (appliqué à son prochain chargement), « inchangé » n'a rien à recharger.
        applied = _regen_swap_yaml(S)
        if applied == "written":
            threading.Thread(
                target=lambda: S.client.unload_local(mid),
                daemon=True,
                name="loom-reload-model",
            ).start()
        return {"ok": True, "applies": "model-reload" if applied else "restart"}

    @app.get("/machine_state")
    def machine_state():
        # L'état machine vient de `/running`, sans dépendre strictement de son schéma JSON.
        conv, _ = _ctx(S)
        model = conv.model
        remote = model in S.remote_model_ids
        reachable, running_txt = S.client.running_local()
        # Distinguer `starting` de `ready`; garder un repli texte si le JSON change.
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
            # Ne proposer l'arrêt que pour une stack lancée par Loom.
            "managed": S.server_manager.owns_running(),
            "starting": S.server_manager.starting,
        }

    @app.post("/machine/unload")
    def machine_unload():
        # Décharger libère la VRAM sans changer la sélection de la session.
        _client_mark_all_cold(S)
        return {"ok": S.client.unload_local()}

    @app.post("/machine/server/start")
    def machine_server_start():
        # Démarrer en fond puis amorcer le préfixe rend aussi le premier message rapide.
        ok = S.server_manager.start()
        _ctx(S)  # garantit une session active avec un modèle valide
        _prime_async(S, _session(S), wait_server=90.0)
        return {"ok": ok}

    @app.post("/machine/server/stop")
    def machine_server_stop():
        # Éteindre uniquement l'arbre géré par Loom.
        return {"ok": S.server_manager.stop()}
