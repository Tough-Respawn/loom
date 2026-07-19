# /add-model + /remove-model tous types — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tout modèle du sélecteur (local texte, distant API — UI ou config —, image, vidéo) devient ajoutable et supprimable via `/add-model` (filtré par type) et `/remove-model`.

**Architecture:** Étendre la machine à états pure `loom/web/wizard.py` (étape `kind` à 4 choix + nouveau flux `i_*` image/vidéo), les effets dans `loom/web/routes.py` (`install_image`, `mount_image`, `remove` × 4 kinds, `_mount_image`/`_forget_image`), et `loom/runtime/model_store.py` (`delete_remote_in_toml`). Wizard 100 % pur : tout I/O passe par `deps` injectées (`image_dir_state`, `check_workflow`) ou par `action`.

**Tech Stack:** Python (Flask), tomlkit, pytest. Spec : `docs/superpowers/specs/2026-07-19-add-remove-model-tous-types-design.md`.

## Global Constraints

- wizard.py reste SANS I/O (ni disque ni réseau) — tout par `deps`/`action`.
- Les poids ComfyUI (`E:/comfyui-models`) ne sont jamais touchés.
- tomlkit pour local.toml (commentaires préservés) ; jamais d'écriture directe.
- Messages wizard en français, préfixe d'étape `[add-model — image X/4]`.
- Commits courts Conventional Commits (<10 lignes), pas de Co-Authored-By.
- Suite existante : `uv run pytest tests/ -q` doit rester verte à chaque tâche.

---

### Task 1: `model_store.delete_remote_in_toml`

**Files:**
- Modify: `loom/runtime/model_store.py` (après `upsert_remote_in_toml`, l.136)
- Test: `tests/test_model_store.py` (existant ou créer)

**Interfaces:**
- Produces: `delete_remote_in_toml(local_path: str | Path, model_id: str) -> bool` — True si une entrée a été retirée, False si fichier/entrée absents (no-op).

- [ ] **Step 1: Test failing**

```python
def test_delete_remote_in_toml_retire_par_id_et_preserve_le_reste(tmp_path):
    p = tmp_path / "local.toml"
    p.write_text(
        "# commentaire preserve\n"
        'default_model = "glm-zai"\n\n'
        "[[remote_models]]\n"
        'id = "glm-zai"\n'
        'base_url = "https://api.z.ai/api/paas/v4"\n'
        'model = "glm-5.2"\n\n'
        "[[remote_models]]\n"
        'id = "glm-flash"\n'
        'base_url = "https://api.z.ai/api/paas/v4"\n'
        'model = "glm-5-flash"\n',
        encoding="utf-8",
    )
    assert model_store.delete_remote_in_toml(p, "glm-flash") is True
    out = p.read_text(encoding="utf-8")
    assert "glm-flash" not in out
    assert "# commentaire preserve" in out and 'id = "glm-zai"' in out

def test_delete_remote_in_toml_absent_est_noop(tmp_path):
    p = tmp_path / "local.toml"
    p.write_text('[[remote_models]]\nid = "a"\nbase_url = "https://x"\nmodel = "m"\n', encoding="utf-8")
    assert model_store.delete_remote_in_toml(p, "inconnu") is False
    assert model_store.delete_remote_in_toml(tmp_path / "absent.toml", "a") is False
```

- [ ] **Step 2: Run** `uv run pytest tests/test_model_store.py -q` → FAIL (AttributeError)

- [ ] **Step 3: Implémentation** (fin de `model_store.py`)

```python
def delete_remote_in_toml(local_path: str | Path, model_id: str) -> bool:
    """Retire un [[remote_models]] de local.toml par id (tomlkit : commentaires et
    structure PRÉSERVÉS). Pendant de upsert_remote_in_toml pour /remove-model sur un
    distant déclaré en config. Renvoie False si fichier ou entrée absents (no-op)."""
    import tomlkit

    p = Path(local_path)
    if not p.exists():
        return False
    doc = tomlkit.parse(p.read_text(encoding="utf-8"))
    arr = doc.get("remote_models")
    if arr is None:
        return False
    idx = next((i for i, t in enumerate(arr) if t.get("id") == model_id), None)
    if idx is None:
        return False
    del arr[idx]
    p.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return True
```

- [ ] **Step 4: Run** même commande → PASS
- [ ] **Step 5: Commit** `feat(model-store): delete_remote_in_toml (suppression distant config)`

---

### Task 2: wizard — étape `kind` à 4 choix + raccourcis `image`/`video`

**Files:**
- Modify: `loom/web/wizard.py` (`start` l.31, `_step_kind` l.94, `_STEPS` l.387)
- Test: `tests/test_wizard_add_model.py`

**Interfaces:**
- Consumes: rien de neuf.
- Produces: état `{"step": "i_id", "ikind": "image"|"video"}` (le flux i_* de Task 3 le consomme).

- [ ] **Step 1: Tests failing**

```python
def test_start_sans_arg_menu_a_4_types():
    r = wizard.start("", deps())
    assert r.state == {"step": "kind"}
    for w in ("local", "distant", "image", "vidéo"):
        assert w in r.reply

def test_kind_3_et_4_routent_vers_le_flux_image():
    r = wizard.step({"step": "kind"}, "3", deps())
    assert r.state == {"step": "i_id", "ikind": "image"}
    r = wizard.step({"step": "kind"}, "4", deps())
    assert r.state == {"step": "i_id", "ikind": "video"}

def test_start_raccourcis_image_video_local():
    assert wizard.start("image", deps()).state == {"step": "i_id", "ikind": "image"}
    assert wizard.start("video", deps()).state == {"step": "i_id", "ikind": "video"}
    # « local <recherche> » = recherche HF directe ; « local » seul = l_query
    assert wizard.start("local", deps()).state == {"step": "l_query"}
    r = wizard.start("local qwen3", deps(hits=[{"repo_id": "a/b", "downloads": 1, "likes": 1}]))
    assert r.state["step"] == "l_repo"

def test_backcompat_recherche_libre_reste_hf():
    r = wizard.start("qwen3 0.6b", deps(hits=[{"repo_id": "a/b", "downloads": 1, "likes": 1}]))
    assert r.state["step"] == "l_repo"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_wizard_add_model.py -q` → FAIL

- [ ] **Step 3: Implémentation.** Dans `start()` : menu sans-arg remplacé par 4 lignes (1. local — GGUF Hugging Face, 2. distant — API OpenAI-compatible, 3. image — générateur ComfyUI, 4. vidéo — générateur ComfyUI ; « réponds 1-4 »). Avant le test `distant`/URL, ajouter :

```python
    if low in ("image", "video", "vidéo"):
        return _start_image("video" if low.startswith("vid") else low, deps)
    if low == "local":
        return WizardResult(
            {"step": "l_query"},
            "[add-model 1/4] Que cherches-tu sur Hugging Face ? (ex. « qwen3 30b »)",
        )
    if low.startswith("local "):
        return _search(a.split(None, 1)[1], deps)
```

`_start_image(ikind, deps)` (défini en Task 3 ; pour cette tâche, version minimale qui renvoie `{"step": "i_id", "ikind": ikind}` avec le prompt id). `_step_kind` : réponses `3`/`image` et `4`/`video`/`vidéo` → `_start_image` ; message d'erreur « Réponds 1 (local), 2 (distant), 3 (image) ou 4 (vidéo), ou /cancel. »

- [ ] **Step 4: Run** → PASS (les tests Task 2 + toute la suite)
- [ ] **Step 5: Commit** `feat(wizard): étape type à 4 choix + raccourcis image/video`

---

### Task 3: wizard — flux image/vidéo `i_*`

**Files:**
- Modify: `loom/web/wizard.py` (nouvelles étapes + `_STEPS`)
- Test: `tests/test_wizard_add_model.py`

**Interfaces:**
- Consumes: `deps.image_dir_state(ikind, mid) -> "complete" | "partial" | None` ; `deps.check_workflow(path) -> {"ok": bool, "error": str | None, "warnings": list[str]}` ; `deps.existing_ids`.
- Produces: actions `{"kind": "install_image", "model_id", "model_kind", "width", "height", "description", "workflow_path": str | None}` et `{"kind": "mount_image", "id", "model_kind"}` (consommées en Task 5).

- [ ] **Step 1: Tests failing** (stub deps enrichi : `image_dir_state=lambda k, m: None`, `check_workflow=lambda p: {"ok": True, "error": None, "warnings": []}` — paramétrables)

```python
def test_flux_image_complet_avec_chemin():
    d = deps()
    r = wizard.step({"step": "i_id", "ikind": "image"}, "mon-modele", d)
    assert r.state["step"] == "i_dims" and "1024x1024" in r.reply
    r = wizard.step(r.state, "ok", d)
    assert r.state["step"] == "i_desc"
    r = wizard.step(r.state, "mon générateur", d)
    assert r.state["step"] == "i_workflow"
    r = wizard.step(r.state, "C:/tmp/wf_api.json", d)
    assert r.state is None
    assert r.action == {
        "kind": "install_image", "model_id": "mon-modele", "model_kind": "image",
        "width": 1024, "height": 1024, "description": "mon générateur",
        "workflow_path": "C:/tmp/wf_api.json",
    }

def test_flux_video_defauts_et_plus_tard():
    d = deps()
    r = wizard.step({"step": "i_id", "ikind": "video"}, "mon-clip", d)
    assert "832x480" in r.reply
    r = wizard.step(r.state, "640x360", d)
    r = wizard.step(r.state, "non", d)          # description vide
    r = wizard.step(r.state, "plus tard", d)
    assert r.state is None and r.action["workflow_path"] is None
    assert r.action["width"] == 640 and r.action["description"] == ""

def test_i_id_refuse_doublon_et_invalide():
    d = deps(existing={"pris"})
    assert wizard.step({"step": "i_id", "ikind": "image"}, "pris", d).state["step"] == "i_id"
    assert wizard.step({"step": "i_id", "ikind": "image"}, "a b", d).state["step"] == "i_id"

def test_i_id_dossier_complet_propose_le_montage():
    d = deps()
    d.image_dir_state = lambda k, m: "complete"
    r = wizard.step({"step": "i_id", "ikind": "image"}, "deja-pret", d)
    assert r.state is None
    assert r.action == {"kind": "mount_image", "id": "deja-pret", "model_kind": "image"}

def test_i_id_dossier_partiel_saute_a_la_recette():
    d = deps()
    d.image_dir_state = lambda k, m: "partial"
    r = wizard.step({"step": "i_id", "ikind": "image"}, "en-cours", d)
    assert r.state["step"] == "i_workflow" and r.state.get("resume") is True

def test_i_workflow_chemin_invalide_redemande():
    d = deps()
    d.check_workflow = lambda p: {"ok": False, "error": "introuvable", "warnings": []}
    st = {"step": "i_workflow", "ikind": "image", "id": "x", "width": 1024,
          "height": 1024, "description": ""}
    r = wizard.step(st, "C:/nexiste/pas.json", d)
    assert r.state["step"] == "i_workflow" and "introuvable" in r.reply

def test_i_workflow_warning_placeholder_transmis():
    d = deps()
    d.check_workflow = lambda p: {"ok": True, "error": None, "warnings": ["{PROMPT} absent"]}
    st = {"step": "i_workflow", "ikind": "image", "id": "x", "width": 1024,
          "height": 1024, "description": ""}
    r = wizard.step(st, "C:/tmp/wf.json", d)
    assert r.action["kind"] == "install_image" and "{PROMPT} absent" in r.reply
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implémentation** (section `# ---------- flux image/vidéo ----------`)

```python
_IMG_DEFAULT_DIMS = {"image": (1024, 1024), "video": (832, 480)}

def _start_image(ikind, deps):
    lab = "image" if ikind == "image" else "vidéo"
    return WizardResult(
        {"step": "i_id", "ikind": ikind},
        f"[add-model — {lab} 1/4] Id du modèle (nom du dossier + sélecteur UI, "
        "ex. « z-image-turbo ») :",
    )

def _step_i_id(state, t, deps):
    ikind = state["ikind"]
    if not _valid_id(t):
        return WizardResult(state, f"Id invalide « {t} » (lettres/chiffres/-_.). Réessaie :")
    if t in deps.existing_ids:
        return WizardResult(state, f"« {t} » existe déjà. Choisis un autre id :")
    found = deps.image_dir_state(ikind, t)
    if found == "complete":  # dossier « plus tard » complété -> montage direct
        return WizardResult(
            None,
            f"Le dossier de « {t} » existe déjà avec sa recette — montage direct.",
            {"kind": "mount_image", "id": t, "model_kind": ikind},
        )
    if found == "partial":  # dossier scaffoldé sans recette -> il ne manque qu'elle
        return WizardResult(
            {"step": "i_workflow", "ikind": ikind, "id": t, "resume": True},
            f"Le dossier de « {t} » existe mais il manque workflow.json. "
            "Colle le chemin de ton export ComfyUI (format API), ou « plus tard » :",
        )
    w, h = _IMG_DEFAULT_DIMS[ikind]
    return WizardResult(
        {"step": "i_dims", "ikind": ikind, "id": t},
        f"[add-model — 2/4] Dimensions de génération — « ok » pour {w}x{h}, "
        "ou tape LxH (ex. 1280x720) :",
    )

def _step_i_dims(state, t, deps):
    w, h = _IMG_DEFAULT_DIMS[state["ikind"]]
    if t.lower() not in ("ok", "oui"):
        parts = t.lower().replace("×", "x").split("x")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            return WizardResult(state, "Format attendu : LxH (ex. 1024x1024), ou « ok ». Réessaie :")
        w, h = int(parts[0]), int(parts[1])
    return WizardResult(
        dict(state, step="i_desc", width=w, height=h),
        "[add-model — 3/4] Description en une ligne (infobulle du sélecteur) — ou « non » :",
    )

def _step_i_desc(state, t, deps):
    desc = "" if t.lower() in ("non", "no", "aucune", "-") else t
    return WizardResult(
        dict(state, step="i_workflow", description=desc),
        "[add-model — 4/4] La recette ComfyUI : colle le chemin de ton export "
        "« format API » (ex. C:\\Users\\toi\\Downloads\\workflow_api.json), "
        "ou « plus tard » pour préparer le dossier :",
    )

def _step_i_workflow(state, t, deps):
    ikind = state["ikind"]
    later = t.lower() in ("plus tard", "later", "non")
    path = None if later else t.strip().strip('"').strip("'")
    warn = ""
    if path:
        chk = deps.check_workflow(path)
        if not chk["ok"]:
            return WizardResult(state, f"Recette illisible : {chk['error']}. "
                                "Colle un autre chemin, ou « plus tard » :")
        if chk["warnings"]:
            warn = "\n⚠️ " + " ; ".join(chk["warnings"])
    action = {
        "kind": "install_image",
        "model_id": state["id"],
        "model_kind": ikind,
        "width": state.get("width", _IMG_DEFAULT_DIMS[ikind][0]),
        "height": state.get("height", _IMG_DEFAULT_DIMS[ikind][1]),
        "description": state.get("description", ""),
        "workflow_path": path,
    }
    return WizardResult(None, f"Création de « {state['id']} »…" + warn, action)
```

`_STEPS` : ajouter `"i_id": _step_i_id, "i_dims": _step_i_dims, "i_desc": _step_i_desc, "i_workflow": _step_i_workflow`. Stub `deps()` du fichier de test : ajouter `image_dir_state=lambda k, m: None` et `check_workflow=lambda p: {"ok": True, "error": None, "warnings": []}`.

- [ ] **Step 4: Run** suite complète → PASS
- [ ] **Step 5: Commit** `feat(wizard): flux image/vidéo (id, dims, description, recette ComfyUI)`

---

### Task 4: wizard — confirmations `/remove-model` par kind

**Files:**
- Modify: `loom/web/wizard.py` (`start_remove` l.225, `_step_d_pick` l.244)
- Test: `tests/test_wizard_add_model.py`

**Interfaces:**
- Consumes: items `deps.removable_models()` enrichis : `kind` ∈ {local, remote, remote_config, image, video}, flag optionnel `is_default` (remote_config).
- Produces: action `{"kind": "remove", "id", "model_kind"}` inchangée (model_kind porte le kind étendu).

- [ ] **Step 1: Tests failing**

```python
def test_remove_liste_avec_rappel_et_confirmations_par_kind():
    items = [
        {"id": "loc", "kind": "local", "label": "loc — local, 1.0 Go sur disque"},
        {"id": "cfg", "kind": "remote_config", "label": "cfg — distant (m, config/local.toml)", "is_default": True},
        {"id": "img", "kind": "image", "label": "img — image (ComfyUI), définition seule"},
    ]
    r = wizard.start_remove(deps(removable=items))
    assert "config/local.toml" in r.reply and "poids ComfyUI" in r.reply
    # local : message disque inchangé
    c = wizard.step(r.state, "1", deps(removable=items))
    assert "SUPPRIMÉS du disque" in c.reply
    # remote_config : retrait du fichier + avertissement default_model
    c = wizard.step(r.state, "2", deps(removable=items))
    assert "config/local.toml" in c.reply and "défaut" in c.reply
    # image : définition seule, poids non touchés
    c = wizard.step(r.state, "3", deps(removable=items))
    assert "workflow.json" in c.reply and "PAS touchés" in c.reply
    ok = wizard.step(c.state, "oui", deps(removable=items))
    assert ok.action == {"kind": "remove", "id": "img", "model_kind": "image"}
```

- [ ] **Step 2: Run** → FAIL

- [ ] **Step 3: Implémentation.** `start_remove` : après la liste, ajouter la ligne « (réponds par un numéro — /cancel pour annuler ; un distant de config/local.toml sera retiré du fichier ; image/vidéo : définition seule, les poids ComfyUI partagés ne sont pas touchés) ». `_step_d_pick` : table des avertissements :

```python
    warns = {
        "local": "son DOSSIER et ses fichiers (GGUF compris) seront SUPPRIMÉS du disque",
        "remote": "il sera retiré du store (sa clé avec) et démonté",
        "remote_config": "il sera retiré de config/local.toml (sa clé avec) et démonté",
        "image": "sa définition Loom (model.toml + workflow.json) sera supprimée ; "
                 "les poids ComfyUI partagés ne sont PAS touchés",
        "video": "sa définition Loom (model.toml + workflow.json) sera supprimée ; "
                 "les poids ComfyUI partagés ne sont PAS touchés",
    }
    warn = warns[it["kind"]]
    if it.get("is_default"):
        warn += (" ⚠️ c'est le modèle par défaut de local.toml : au prochain boot, "
                 "repli sur le premier modèle installé")
```

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `feat(wizard): confirmations /remove-model par type (config, image, vidéo)`

---

### Task 5: routes — liste complète, effets image, suppression config, persistance ✅

**Files:**
- Modify: `loom/web/routes.py` (`_removable_models` l.526, `_wizard_deps` l.566, `_handle_add_model_command` l.658, nouveaux `_mount_image`/`_forget_image`/`_image_bases` près de `_mount_local` l.588)
- Test: `tests/test_add_model_routes.py`

**Interfaces:**
- Consumes: `delete_remote_in_toml` (Task 1), actions Task 3/4, `discover_image_models` (loom/runtime/image_models.py), registres S (`image_by_id`, `image_model_ids`, `video_model_ids`, `models`, `model_descriptions`, `remote_model_names`, `config_local_path`, `models_dir`).
- Produces: rien de nouveau pour d'autres tâches.

- [ ] **Step 1: Tests failing** (pattern stub S de `test_add_model_routes.py`) — cas à couvrir :

```python
def test_removable_models_liste_les_4_familles(...):
    # S avec : 1 local spec, 1 remote store, 1 remote_config (remote_model_ids sans store,
    # default_model dans un local.toml tmp), 1 image + 1 video dans image_by_id
    items = routes._removable_models(S)
    kinds = {i["id"]: i["kind"] for i in items}
    assert kinds == {"loc": "local", "ui": "remote", "cfg": "remote_config",
                     "img": "image", "vid": "video"}
    assert next(i for i in items if i["id"] == "cfg")["is_default"] is True

def test_remove_remote_config_edite_local_toml_et_demonte(...):
    # action remove kind remote_config -> delete_remote_in_toml appelé sur
    # S.config_local_path (tmp), id retiré de S.models / remote_model_ids
def test_remove_image_rmtree_dossier_et_demonte(...):
    # dossier tmp local/image/img avec model.toml+workflow.json ; action remove ->
    # dossier absent, id hors de S.models / image_by_id / image_model_ids
def test_install_image_scaffold_et_montage(...):
    # action install_image avec workflow_path tmp -> dossier créé sous
    # <models_dir parent>/image/<id>/, model.toml + workflow.json présents,
    # id monté dans S.models + image_by_id
def test_install_image_plus_tard_scaffold_sans_montage(...):
    # workflow_path None -> model.toml présent, PAS de workflow.json, id PAS monté
def test_extra_reply_persistee_dans_le_journal(...):
    # après un remove, le DERNIER event "text" du journal contient le ✅
```

- [ ] **Step 2: Run** `uv run pytest tests/test_add_model_routes.py -q` → FAIL

- [ ] **Step 3: Implémentation.**

`_removable_models` — remplacer le corps :

```python
def _removable_models(S) -> list[dict]:
    """Modèles supprimables via /remove-model : TOUT ce que le sélecteur affiche.
    kind ∈ {local, remote (store UI), remote_config (local.toml), image, video}."""
    import tomllib

    items = [
        {"id": m["id"], "kind": "local",
         "label": f"{m['id']} — local, {(m.get('size_mb') or 0) / 1024:.1f} Go sur disque"}
        for m in S.local_model_specs
    ]
    managed = set()
    if S.remote_store_path:
        for r in model_store.load(S.remote_store_path):
            managed.add(r["id"])
            items.append({"id": r["id"], "kind": "remote",
                          "label": f"{r['id']} — distant ({r.get('model', '?')})"})
    default_model = ""
    if S.config_local_path and Path(S.config_local_path).exists():
        try:
            cfg = tomllib.loads(Path(S.config_local_path).read_text(encoding="utf-8"))
            default_model = str(cfg.get("chat", {}).get("default_model") or cfg.get("default_model") or "")
        except (OSError, tomllib.TOMLDecodeError):
            pass
    for mid in sorted(S.remote_model_ids - managed):
        items.append({"id": mid, "kind": "remote_config", "is_default": mid == default_model,
                      "label": f"{mid} — distant ({S.remote_model_names.get(mid, '?')}, config/local.toml)"})
    for im in sorted(S.image_by_id.values(), key=lambda m: m.id):
        kind = "video" if im.id in S.video_model_ids else "image"
        items.append({"id": im.id, "kind": kind,
                      "label": f"{im.id} — {kind} (ComfyUI), définition seule"})
    return items
```

NB : vérifier où `default_model` vit réellement dans local.toml (`[chat] default_model` — cf. le fichier réel) et ne garder QUE la bonne clé.

Helpers image (sous `_mount_local`) :

```python
def _image_base_dir(S, ikind: str) -> Path:
    """Dossier local/{image,video} où créer un nouveau modèle : la racine qui héberge
    déjà ce type, sinon celle d'un modèle image existant, sinon à côté de models_dir
    (<root>/local/text -> <root>/local/<ikind>)."""
    for im in S.image_by_id.values():
        d = Path(im.dir).parent
        if d.name == ikind:
            return d
    if S.image_by_id:
        any_dir = Path(next(iter(S.image_by_id.values())).dir)
        return any_dir.parent.parent / ikind
    return Path(S.models_dir).parent / ikind

def _mount_image(S, ikind: str, mid: str) -> object | None:
    """(Re)découvre <root>/local/{image,video}/<mid> via le parseur officiel et monte
    le modèle à chaud dans tous les registres du sélecteur. None si introuvable/incomplet."""
    from loom.runtime.image_models import discover_image_models

    root = _image_base_dir(S, ikind).parent.parent  # .../local/<ikind> -> racine
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
```

`_wizard_deps` — ajouter :

```python
        image_dir_state=lambda ikind, mid: _image_dir_state(S, ikind, mid),
        check_workflow=_check_workflow,
```

avec :

```python
def _image_dir_state(S, ikind: str, mid: str) -> str | None:
    d = _image_base_dir(S, ikind) / mid
    if not d.is_dir():
        return None
    return "complete" if (d / "model.toml").is_file() and (d / "workflow.json").is_file() else "partial"

def _check_workflow(path: str) -> dict:
    """Validation légère d'un export ComfyUI format API (JSON + placeholders)."""
    import json

    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": f"fichier introuvable ({path})", "warnings": []}
    try:
        raw = p.read_text(encoding="utf-8")
        json.loads(raw)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"JSON invalide : {exc}", "warnings": []}
    warnings = []
    if "{PROMPT}" not in raw:
        warnings.append("placeholder {PROMPT} absent — le prompt du chat ne sera pas injecté")
    return {"ok": True, "error": None, "warnings": warnings}
```

`_handle_add_model_command` — nouveaux bras d'action (après le bras `remove` existant) :

```python
    elif res.action and res.action["kind"] == "mount_image":
        a = res.action
        im = _mount_image(S, a["model_kind"], a["id"])
        extra_reply = (
            f"\n✅ « {a['id']} » monté — disponible dans le sélecteur."
            if im is not None
            else f"\n❌ Dossier de « {a['id']} » introuvable ou incomplet — relance /add-model."
        )
        models_changed = im is not None
    elif res.action and res.action["kind"] == "install_image":
        a = res.action
        base = _image_base_dir(S, a["model_kind"])
        mdir = base / a["model_id"]
        try:
            mdir.mkdir(parents=True, exist_ok=True)
            if not (mdir / "model.toml").is_file():  # reprise : ne pas écraser l'existant
                desc = a["description"].replace('"', "'")
                tmpl = next(iter(S.image_by_id.values()), None)
                comfy_dir = tmpl.comfy_dir if tmpl else "C:/tools/ComfyUI"
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
                    f"\n✅ « {a['model_id']} » créé et monté — disponible dans le sélecteur. "
                    "(Les poids ComfyUI ne sont pas gérés par Loom : la recette doit "
                    "référencer des checkpoints déjà présents côté ComfyUI.)"
                    if im is not None
                    else f"\n❌ Recette copiée mais montage impossible — vérifie {mdir}."
                )
                models_changed = im is not None
            else:
                extra_reply = (
                    f"\n📁 Dossier préparé : {mdir}\nDépose ton export ComfyUI (format API) "
                    f"sous le nom workflow.json, puis relance /add-model {a['model_kind']} "
                    f"avec le même id « {a['model_id']} » pour le monter (sinon il sera "
                    "découvert au prochain démarrage)."
                )
        except OSError as exc:
            extra_reply = f"\n❌ Création impossible : {exc}"
```

Bras `remove` — étendre le branchement :

```python
        if a["model_kind"] in ("remote", "remote_config"):
            if a["model_kind"] == "remote" and S.remote_store_path:
                with S.toml_lock:
                    model_store.delete(S.remote_store_path, a["id"])
                where = "store"
            else:
                with S.toml_lock:
                    model_store.delete_remote_in_toml(S.config_local_path, a["id"])
                where = "config/local.toml"
            _forget_remote(S, a["id"])
            extra_reply = f"\n✅ « {a['id']} » retiré ({where} + sélecteur)."
            models_changed = True
        elif a["model_kind"] in ("image", "video"):
            import shutil

            im = S.image_by_id.get(a["id"])
            try:
                if im is not None:
                    shutil.rmtree(im.dir)
                _forget_image(S, a["id"])
                extra_reply = (f"\n✅ Définition de « {a['id']} » supprimée (sélecteur "
                               "compris) ; les poids ComfyUI partagés ne sont PAS touchés.")
                models_changed = True
            except OSError as exc:
                extra_reply = f"\n❌ Suppression impossible : {exc}"
        else:
            ... (bras local existant inchangé)
```

Persistance du résultat — déplacer l'appel `_persist_wizard_exchange(S, sess, conv, save, shown, res.reply)` (l.690) APRÈS le bloc d'actions, en persistant `res.reply + extra_reply` (une seule écriture, toujours sous le verrou). Le `conv.set_wizard(res.state)` reste où il est.

- [ ] **Step 4: Run** `uv run pytest tests/ -q` → PASS (suite complète)
- [ ] **Step 5: Commit** `feat(web): /add-model image/vidéo + /remove-model complet (config, image) + ✅ persisté`

---

### Task 6: E2E réel (Playwright sur loom.web) + rapport

**Files:** aucun (validation runtime).

- [ ] **Step 1:** Redémarrer loom.web (changements backend — gotcha connu), recharger l'onglet.
- [ ] **Step 2:** Dans une session de test : `/add-model image` → id `zz-e2e-image` → `ok` → description → chemin d'un `workflow_api.json` factice (JSON minimal avec `{PROMPT}`, créé dans le scratchpad) → vérifier : dossier créé sous `E:/loom-models/local/image/zz-e2e-image/`, entrée `image · zz-e2e-image` dans le sélecteur SANS reload.
- [ ] **Step 3:** `/remove-model` → vérifier que la liste montre locaux + glm-* (config) + image/vidéo → supprimer `zz-e2e-image` → sélecteur à jour sans reload, dossier disparu du disque.
- [ ] **Step 4:** Recharger la page → le fil montre bien le `✅` persisté.
- [ ] **Step 5:** Suppression d'un distant config : sur un local.toml de TEST uniquement (couvert par les tests unit Task 1/5) — ne PAS supprimer un vrai glm-*. En E2E, vérifier seulement que la liste les propose avec le bon libellé et que `/cancel` sort proprement.
- [ ] **Step 6:** Nettoyage (session de test supprimée), rapport final avec ce qui est prouvé vs non prouvé.
