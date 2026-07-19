# Machine à états PURE de /add-model : aucun réseau, deps entièrement fakés.
from types import SimpleNamespace

from loom.web import wizard


def deps(existing=(), hits=None, files=None, remote_models=None, removable=None):
    return SimpleNamespace(
        existing_ids=set(existing),
        search_models=lambda q: list(hits or []),
        list_gguf_files=lambda repo: list(files or []),
        recommend=lambda fs: [
            dict(f, fits=True, recommended=(i == len(fs) - 1)) for i, f in enumerate(fs)
        ],
        derive_id=lambda repo: "id-propose",
        # None = GET /models du provider injoignable -> saisie manuelle
        list_remote_models=lambda base_url, key: remote_models,
        removable_models=lambda: list(removable or []),
        # Flux image/vidéo : dossier inexistant + recette valide par défaut
        image_dir_state=lambda ikind, mid: None,
        check_workflow=lambda p: {"ok": True, "error": None, "warnings": []},
        # Flux /rebench : rien de calibrable par défaut
        rebenchable_models=lambda: [],
        model_kind=lambda mid: None,
    )


# ---------- tronc commun ----------


def test_start_sans_arg_demande_le_type():
    r = wizard.start("", deps())
    assert r.state == {"step": "kind"}
    assert "1" in r.reply and "local" in r.reply and "distant" in r.reply


def test_start_sans_arg_menu_a_4_types():
    r = wizard.start("", deps())
    for w in ("local", "distant", "image", "vidéo"):
        assert w in r.reply


def test_kind_3_et_4_routent_vers_le_flux_image():
    r = wizard.step({"step": "kind"}, "3", deps())
    assert r.state == {"step": "i_id", "ikind": "image"}
    r = wizard.step({"step": "kind"}, "4", deps())
    assert r.state == {"step": "i_id", "ikind": "video"}
    r = wizard.step({"step": "kind"}, "vidéo", deps())
    assert r.state == {"step": "i_id", "ikind": "video"}


def test_start_raccourcis_image_video_local():
    assert wizard.start("image", deps()).state == {"step": "i_id", "ikind": "image"}
    assert wizard.start("video", deps()).state == {"step": "i_id", "ikind": "video"}
    # « local <recherche> » = recherche HF directe ; « local » seul = l_query
    assert wizard.start("local", deps()).state == {"step": "l_query"}
    r = wizard.start(
        "local qwen3", deps(hits=[{"repo_id": "a/b", "downloads": 1, "likes": 1}])
    )
    assert r.state["step"] == "l_repo"


def test_backcompat_recherche_libre_reste_hf():
    r = wizard.start(
        "qwen3 0.6b", deps(hits=[{"repo_id": "a/b", "downloads": 1, "likes": 1}])
    )
    assert r.state["step"] == "l_repo"


def test_cancel_a_toute_etape():
    r = wizard.step(
        {"step": "r_model", "id": "x", "base_url": "https://a"}, "/cancel", deps()
    )
    assert r.state is None and r.action is None
    assert "annulé" in r.reply.lower()


def test_etat_inconnu_annule_proprement():
    r = wizard.step({"step": "n-existe-pas"}, "x", deps())
    assert r.state is None


# ---------- boutons de confirmation (choices) ----------


def test_choices_sur_menu_type_et_confirmations():
    # menu de type /add-model
    r = wizard.start("", deps())
    assert r.choices == ["local", "distant", "image", "vidéo"]
    # confirmation /remove-model
    items = [{"id": "x", "kind": "local", "label": "x — local, 1.0 Go sur disque"}]
    r = wizard.step({"step": "d_pick", "items": items}, "1", deps(removable=items))
    assert r.choices == ["oui", "annuler"]
    # confirmation /rebench
    d = deps()
    d.rebenchable_models = lambda: [{"id": "x", "label": "x — contexte actuel 4096"}]
    r = wizard.start_rebench("x", d)
    assert r.choices == ["oui", "annuler"]
    # étape de saisie libre : pas de boutons
    r = wizard.step({"step": "i_id", "ikind": "image"}, "mon-modele", deps())
    assert r.choices is None


# ---------- /rebench ----------


def test_rebench_liste_et_confirmation():
    items = [{"id": "orn", "label": "orn — contexte actuel 49152, 34.4 Go"}]
    d = deps()
    d.rebenchable_models = lambda: items
    r = wizard.start_rebench("", d)
    assert "orn" in r.reply and r.state["step"] == "b_pick"
    r = wizard.step(r.state, "1", d)
    assert r.state["step"] == "b_confirm" and "éteint" in r.reply
    r = wizard.step(r.state, "oui", d)
    assert r.state is None and r.action == {"kind": "rebench", "id": "orn"}


def test_rebench_id_direct_et_refus_par_type():
    items = [{"id": "orn", "label": "orn — contexte actuel 49152, 34.4 Go"}]
    d = deps()
    d.rebenchable_models = lambda: items
    d.model_kind = lambda m: {"glm": "remote", "chroma": "image"}.get(m)
    r = wizard.start_rebench("orn", d)
    assert r.state["step"] == "b_confirm"
    r = wizard.start_rebench("glm", d)
    assert r.state is None and "texte" in r.reply
    r = wizard.start_rebench("chroma", d)
    assert r.state is None and "texte" in r.reply
    r = wizard.start_rebench("inconnu", d)
    assert r.state is None and "inconnu" in r.reply


def test_rebench_confirm_non_annule():
    r = wizard.step({"step": "b_confirm", "id": "orn"}, "non", deps())
    assert r.state is None and r.action is None


def test_rebench_apply_oui_et_annulation():
    st = {"step": "b_apply", "id": "orn", "context": 65536, "mecanisme": "pente x"}
    r = wizard.step(st, "oui", deps())
    assert r.action == {
        "kind": "rebench_apply",
        "id": "orn",
        "context": 65536,
        "mecanisme": "pente x",
    }
    r = wizard.step(st, "non", deps())
    assert r.state is None and r.action is None and "inchangé" in r.reply


# ---------- /remove-model : confirmations par kind ----------


def test_remove_liste_avec_rappel_et_confirmations_par_kind():
    items = [
        {"id": "loc", "kind": "local", "label": "loc — local, 1.0 Go sur disque"},
        {
            "id": "cfg",
            "kind": "remote_config",
            "label": "cfg — distant (m, config/local.toml)",
            "is_default": True,
        },
        {
            "id": "img",
            "kind": "image",
            "label": "img — image (ComfyUI), définition seule",
        },
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


# ---------- flux image/vidéo ----------


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
        "kind": "install_image",
        "model_id": "mon-modele",
        "model_kind": "image",
        "width": 1024,
        "height": 1024,
        "description": "mon générateur",
        "workflow_path": "C:/tmp/wf_api.json",
    }


def test_flux_video_defauts_et_plus_tard():
    d = deps()
    r = wizard.step({"step": "i_id", "ikind": "video"}, "mon-clip", d)
    assert "832x480" in r.reply
    r = wizard.step(r.state, "640x360", d)
    r = wizard.step(r.state, "non", d)  # description vide
    r = wizard.step(r.state, "plus tard", d)
    assert r.state is None and r.action["workflow_path"] is None
    assert r.action["width"] == 640 and r.action["description"] == ""
    assert r.action["model_kind"] == "video"


def test_i_id_refuse_doublon_et_invalide():
    d = deps(existing={"pris"})
    r = wizard.step({"step": "i_id", "ikind": "image"}, "pris", d)
    assert r.state["step"] == "i_id"
    r = wizard.step({"step": "i_id", "ikind": "image"}, "a b", d)
    assert r.state["step"] == "i_id"


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
    st = {
        "step": "i_workflow",
        "ikind": "image",
        "id": "x",
        "width": 1024,
        "height": 1024,
        "description": "",
    }
    r = wizard.step(st, "C:/nexiste/pas.json", d)
    assert r.state["step"] == "i_workflow" and "introuvable" in r.reply


def test_i_workflow_warning_placeholder_transmis():
    d = deps()
    d.check_workflow = lambda p: {
        "ok": True,
        "error": None,
        "warnings": ["{PROMPT} absent"],
    }
    st = {
        "step": "i_workflow",
        "ikind": "image",
        "id": "x",
        "width": 1024,
        "height": 1024,
        "description": "",
    }
    r = wizard.step(st, "C:/tmp/wf.json", d)
    assert r.action["kind"] == "install_image" and "{PROMPT} absent" in r.reply


# ---------- flux distant ----------


def test_start_distant_et_url_routent_vers_le_flux_distant():
    # « /add-model distant » -> flux distant direct, sans passer par le menu
    r = wizard.start("distant", deps())
    assert r.state == {"step": "r_id"}
    # une URL brute n'est JAMAIS une recherche HF -> distant, base_url pré-remplie
    r = wizard.start("https://api.z.ai/api/paas/v4/", deps())
    assert r.state == {"step": "r_id", "base_url": "https://api.z.ai/api/paas/v4"}
    assert "base_url notée" in r.reply
    # l'id fourni ensuite SAUTE l'étape base_url et passe à la clé
    r2 = wizard.step(r.state, "glm-5", deps())
    assert r2.state["step"] == "r_key"
    assert r2.state["base_url"] == "https://api.z.ai/api/paas/v4"


def test_start_url_plus_cle_ignore_la_cle_et_avertit():
    # une clé collée dans la commande est IGNORÉE (jamais interprétée) + alerte
    r = wizard.start("distant https://api.z.ai/api/paas/v4/ sk-tres-secret", deps())
    assert r.state == {"step": "r_id", "base_url": "https://api.z.ai/api/paas/v4"}
    assert "sk-tres-secret" not in r.reply
    assert "IGNORÉ" in r.reply and "régénérer" in r.reply


def test_distant_parcours_complet():
    # provider qui expose GET /models -> la clé (étape 3) débloque la LISTE (étape 4)
    d = deps(existing={"deja-la"}, remote_models=["glm-5", "glm-5-flash"])
    r = wizard.step({"step": "kind"}, "2", d)
    assert r.state == {"step": "r_id"}

    r = wizard.step(r.state, "deja-la", d)  # id déjà pris -> re-demande
    assert r.state == {"step": "r_id"}

    r = wizard.step(r.state, "glm-5", d)
    assert r.state["step"] == "r_base_url"

    r = wizard.step(r.state, "pas-une-url", d)  # invalide -> re-demande
    assert r.state["step"] == "r_base_url"

    r = wizard.step(r.state, "https://api.z.ai/api/paas/v4/", d)
    assert r.state["step"] == "r_key"
    assert r.state["base_url"] == "https://api.z.ai/api/paas/v4"  # slash retiré
    assert "MASQUÉE" in r.reply  # la clé est annoncée comme masquée

    r = wizard.step(r.state, "sk-secret", d)
    assert r.state["step"] == "r_model"
    assert r.state["choices"] == ["glm-5", "glm-5-flash"]
    assert "glm-5-flash" in r.reply  # liste numérotée affichée

    r = wizard.step(r.state, "2", d)  # choix PAR NUMÉRO dans la liste
    assert r.state["step"] == "r_adv"

    r = wizard.step(r.state, "contexte=200000 vision=oui", d)
    assert r.state is None
    assert r.action == {
        "kind": "upsert_remote",
        "record": {
            "id": "glm-5",
            "base_url": "https://api.z.ai/api/paas/v4",
            "model": "glm-5-flash",
            "api_key": "sk-secret",
            "context": 200000,
            "max_tokens": None,
            "vision": True,
        },
    }


def test_distant_sans_liste_ni_avance_ni_cle():
    # provider muet sur GET /models -> saisie manuelle du nom, comme avant
    d = deps()
    st = {"step": "r_key", "id": "m", "base_url": "https://a"}
    r = wizard.step(st, "aucune", d)
    assert r.state["step"] == "r_model" and "choices" not in r.state
    assert "tape le nom" in r.reply
    r = wizard.step(r.state, "mm", d)  # texte libre accepté
    r = wizard.step(r.state, "non", d)
    assert r.action["record"]["model"] == "mm"
    assert r.action["record"]["api_key"] == ""
    assert r.action["record"]["context"] is None
    assert r.action["record"]["vision"] is False


# ---------- flux suppression (/remove-model) ----------

REMOVABLE = [
    {"id": "qwen-local", "kind": "local", "label": "qwen-local — local, 5.4 Go"},
    {"id": "glm-flash", "kind": "remote", "label": "glm-flash — distant (glm-4.7)"},
]


def test_remove_liste_puis_confirme():
    d = deps(removable=REMOVABLE)
    r = wizard.start_remove(d)
    assert r.state["step"] == "d_pick"
    assert "qwen-local" in r.reply and "glm-flash" in r.reply

    r = wizard.step(r.state, "9", d)  # hors liste -> re-demande
    assert r.state["step"] == "d_pick"

    r = wizard.step(r.state, "1", d)
    assert r.state["step"] == "d_confirm"
    assert "SUPPRIMÉS du disque" in r.reply  # local = destruction annoncée

    r = wizard.step(r.state, "oui", d)
    assert r.state is None
    assert r.action == {"kind": "remove", "id": "qwen-local", "model_kind": "local"}


def test_remove_tout_sauf_oui_annule():
    d = deps(removable=REMOVABLE)
    r = wizard.start_remove(d)
    r = wizard.step(r.state, "2", d)
    r = wizard.step(r.state, "vas-y", d)  # ni « oui » ni variantes -> annulation
    assert r.state is None and r.action is None
    assert "annulée" in r.reply


def test_remove_sans_rien_a_supprimer():
    r = wizard.start_remove(deps())
    assert r.state is None and r.action is None
    assert "Aucun modèle" in r.reply


# ---------- flux local ----------

HITS = [
    {"repo_id": "unsloth/Qwen3-30B-GGUF", "downloads": 9000, "likes": 42},
    {"repo_id": "org/autre-GGUF", "downloads": 100, "likes": 1},
]
FILES = [
    {
        "filename": "m.Q4_K_M.gguf",
        "part_files": ["m.Q4_K_M.gguf"],
        "size_mb": 10_000,
        "is_mmproj": False,
    },
    {
        "filename": "m.Q8_0.gguf",
        "part_files": ["m.Q8_0.gguf"],
        "size_mb": 20_000,
        "is_mmproj": False,
    },
    {
        "filename": "mmproj-F16.gguf",
        "part_files": ["mmproj-F16.gguf"],
        "size_mb": 800,
        "is_mmproj": True,
    },
]


def test_les_fichiers_auxiliaires_ne_sont_pas_des_quants():
    # Un mtp-*.gguf (is_aux) ne doit JAMAIS apparaître dans les quants proposés
    # (vécu : recommandé comme seul quant « qui tient » -> coquille vide installée).
    files = [
        {
            "filename": "mtp-ggml-model-bf16.gguf",
            "part_files": ["mtp-ggml-model-bf16.gguf"],
            "size_mb": 200,
            "is_mmproj": False,
            "is_aux": True,
        },
        {
            "filename": "m.Q4_K.gguf",
            "part_files": ["m.Q4_K.gguf"],
            "size_mb": 4900,
            "is_mmproj": False,
            "is_aux": False,
        },
    ]
    d = deps(hits=[{"repo_id": "org/r", "downloads": 1, "likes": 0}], files=files)
    r = wizard.step({"step": "l_repo", "hits": d.search_models("x")}, "1", d)
    assert r.state["step"] == "l_quant"
    assert "mtp-ggml" not in r.reply
    assert "m.Q4_K.gguf" in r.reply


def test_local_parcours_complet():
    d = deps(hits=HITS, files=FILES)
    r = wizard.start("qwen3 30b", d)  # /add-model <recherche> : direct à la shortlist
    assert r.state["step"] == "l_repo"
    assert "unsloth/Qwen3-30B-GGUF" in r.reply and "9000" in r.reply

    r = wizard.step(r.state, "1", d)  # choix du repo -> quants annotés
    assert r.state["step"] == "l_quant"
    assert "recommandé" in r.reply
    assert "mmproj-F16.gguf" in r.reply  # vision signalée
    # seuls les POIDS sont proposés (le mmproj n'est pas un choix de quant)
    assert len(r.state["files"]) == 2

    r = wizard.step(r.state, "2", d)  # choix du quant
    assert r.state["step"] == "l_id"
    assert "id-propose" in r.reply

    r = wizard.step(r.state, "ok", d)  # id proposé accepté
    assert r.state is None
    assert r.action == {
        "kind": "install",
        "model_id": "id-propose",
        "repo": "unsloth/Qwen3-30B-GGUF",
        "filename": "m.Q8_0.gguf",
        "files": ["m.Q8_0.gguf"],
        "size_mb": 20_000,
        "mmproj_filename": "mmproj-F16.gguf",
    }


def test_local_recherche_vide_redemande():
    d = deps(hits=[])
    r = wizard.start("nexistepas", d)
    assert r.state == {"step": "l_query"}
    assert "Aucun" in r.reply


def test_local_texte_libre_sur_shortlist_relance_la_recherche():
    d = deps(hits=HITS, files=FILES)
    r = wizard.start("qwen", d)
    r = wizard.step(r.state, "autre recherche", d)
    assert r.state["step"] == "l_repo"  # nouvelle shortlist, pas d'erreur


def test_local_id_deja_pris_redemande():
    d = deps(existing={"id-propose"}, hits=HITS, files=FILES)
    r = wizard.start("qwen", d)
    r = wizard.step(r.state, "1", d)
    r = wizard.step(r.state, "1", d)
    r = wizard.step(r.state, "ok", d)  # id proposé mais déjà pris
    assert r.state["step"] == "l_id"
    r = wizard.step(r.state, "mon-id", d)
    assert r.action["model_id"] == "mon-id"


def test_local_kind_1_va_a_la_recherche():
    r = wizard.step({"step": "kind"}, "1", deps())
    assert r.state == {"step": "l_query"}
