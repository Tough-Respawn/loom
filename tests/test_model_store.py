# Store des distants en DOSSIERS (<racine>/remote/<id>/model.toml) : découverte
# multi-racines, écriture/suppression, et migrations des emplacements hérités
# ([[remote_models]] de local.toml, store JSON var/remote_models.json).
import json

from loom.runtime import model_store


def _mk_remote(root, mid, base_url="https://x", model="m", extra=""):
    d = root / "remote" / mid
    d.mkdir(parents=True)
    d.joinpath("model.toml").write_text(
        f'base_url = "{base_url}"\nmodel = "{model}"\n{extra}', encoding="utf-8"
    )
    return d


def test_discover_remote_multi_racines_premiere_gagnante(tmp_path):
    r1, r2 = tmp_path / "c", tmp_path / "e"
    _mk_remote(r1, "kimi", model="m-c")
    _mk_remote(r2, "kimi", model="m-e")  # doublon d'id : la 1re racine gagne
    _mk_remote(r2, "glm", extra="context = 200000\n")
    (r2 / "remote" / "_TEMPLATE").mkdir()  # préfixe '_' : ignoré
    (r2 / "remote" / "sans-toml").mkdir()  # pas de model.toml : ignoré
    _mk_remote(r2, "bancal", base_url="")  # sans base_url : jamais monté
    out = model_store.discover_remote([r1, r2])
    assert [(m["id"], m["model"]) for m in out] == [("glm", "m"), ("kimi", "m-c")]
    assert out[0]["context"] == 200000
    # l'id vient du DOSSIER, même si le toml en portait un
    _mk_remote(r1, "dossier-roi", extra='id = "autre-nom"\n')
    assert any(m["id"] == "dossier-roi" for m in model_store.discover_remote([r1]))


def test_write_remote_dir_cree_puis_edite_en_preservant(tmp_path):
    p = model_store.write_remote_dir(
        tmp_path,
        {
            "id": "kimi",
            "base_url": "https://api.moonshot.ai/v1",
            "model": "k3",
            "api_key": "sk-x",
            "context": None,
        },
    )
    assert p == tmp_path / "remote" / "kimi" / "model.toml"
    txt = p.read_text(encoding="utf-8")
    assert 'base_url = "https://api.moonshot.ai/v1"' in txt
    assert "context" not in txt  # champ None : jamais posé
    # édition partielle : un commentaire ajouté à la main survit, la clé reste
    p.write_text("# note perso\n" + txt, encoding="utf-8")
    model_store.write_remote_dir(tmp_path, {"id": "kimi", "context": 131072})
    txt = p.read_text(encoding="utf-8")
    assert "# note perso" in txt and "context = 131072" in txt
    assert 'api_key = "sk-x"' in txt


def test_delete_remote_dir_purge_toutes_les_racines(tmp_path):
    r1, r2 = tmp_path / "c", tmp_path / "e"
    _mk_remote(r1, "kimi")
    _mk_remote(r2, "kimi")  # reliquat sur une 2e racine : purgé AUSSI
    assert model_store.delete_remote_dir([r1, r2], "kimi") is True
    assert not (r1 / "remote" / "kimi").exists()
    assert not (r2 / "remote" / "kimi").exists()
    assert model_store.delete_remote_dir([r1, r2], "absent") is False


def test_migrate_into_dirs_replie_toml_et_json(tmp_path):
    local = tmp_path / "local.toml"
    local.write_text(
        "# commentaire machine preserve\n[server]\ncontext = 24576\n\n"
        "[[remote_models]]\n"
        'id = "glm"\nbase_url = "https://z"\nmodel = "glm-5"\napi_key = "sk-g"\n',
        encoding="utf-8",
    )
    store = tmp_path / "remote_models.json"
    store.write_text(
        json.dumps([{"id": "kimi", "base_url": "https://k", "model": "k3"}]),
        encoding="utf-8",
    )
    root = tmp_path / "models"
    leftovers = model_store.migrate_into_dirs(local, store, root)
    assert leftovers == []
    ids = [m["id"] for m in model_store.discover_remote([root])]
    assert ids == ["glm", "kimi"]
    # sources vidées : plus de [[remote_models]], JSON supprimé, machine intacte
    txt = local.read_text(encoding="utf-8")
    assert "remote_models" not in txt and "# commentaire machine preserve" in txt
    assert "context = 24576" in txt
    assert not store.exists()
    # idempotent : re-run = no-op
    assert model_store.migrate_into_dirs(local, store, root) == []


def test_migrate_into_dirs_dossier_existant_fait_foi(tmp_path):
    root = tmp_path / "models"
    _mk_remote(root, "glm", model="version-dossier")
    local = tmp_path / "local.toml"
    local.write_text(
        '[[remote_models]]\nid = "glm"\nbase_url = "https://z"\nmodel = "version-toml"\n',
        encoding="utf-8",
    )
    model_store.migrate_into_dirs(local, None, root)
    out = model_store.discover_remote([root])
    assert out[0]["model"] == "version-dossier"  # jamais écrasé par l'entrée héritée
    assert "remote_models" not in local.read_text(encoding="utf-8")


def test_migrate_into_dirs_sans_racine_renvoie_les_entrees(tmp_path):
    local = tmp_path / "local.toml"
    local.write_text(
        '[[remote_models]]\nid = "a"\nbase_url = "https://x"\nmodel = "m"\n',
        encoding="utf-8",
    )
    leftovers = model_store.migrate_into_dirs(local, None, None)
    assert [m["id"] for m in leftovers] == ["a"]
    # rien d'écrit nulle part -> la source n'est PAS vidée (zéro perte)
    assert "remote_models" in local.read_text(encoding="utf-8")


def test_delete_remote_in_toml_retire_par_id_et_preserve_le_reste(tmp_path):
    p = tmp_path / "local.toml"
    p.write_text(
        "# commentaire preserve\n"
        '[chat]\ndefault_model = "glm-zai"\n\n'
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
    p.write_text(
        '[[remote_models]]\nid = "a"\nbase_url = "https://x"\nmodel = "m"\n',
        encoding="utf-8",
    )
    assert model_store.delete_remote_in_toml(p, "inconnu") is False
    assert model_store.delete_remote_in_toml(tmp_path / "absent.toml", "a") is False
    assert 'id = "a"' in p.read_text(encoding="utf-8")
