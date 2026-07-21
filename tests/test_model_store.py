# delete_remote_in_toml : suppression d'un distant déclaré en config, via tomlkit
# (commentaires et structure préservés) — pendant de upsert_remote_in_toml.
# migrate_to_toml : repli du store JSON hérité dans config/local.toml (source unique).
import json

from loom.runtime import model_store


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


def test_migrate_to_toml_replie_le_store_et_le_supprime(tmp_path):
    store = tmp_path / "remote_models.json"
    store.write_text(
        json.dumps(
            [{"id": "kimi", "base_url": "https://api.moonshot.ai/v1", "model": "k3"}]
        ),
        encoding="utf-8",
    )
    local = tmp_path / "local.toml"
    local.write_text(
        "# commentaire preserve\n[chat]\nmax_tokens = 1\n", encoding="utf-8"
    )
    migrated = model_store.migrate_to_toml(store, local)
    assert [m["id"] for m in migrated] == ["kimi"]
    out = local.read_text(encoding="utf-8")
    assert 'id = "kimi"' in out and "# commentaire preserve" in out
    assert not store.exists()  # le JSON hérité disparaît : source unique = TOML
    # Idempotent : plus de store -> no-op
    assert model_store.migrate_to_toml(store, local) == []


def test_migrate_to_toml_sans_local_path_lit_sans_supprimer(tmp_path):
    store = tmp_path / "remote_models.json"
    store.write_text(
        json.dumps([{"id": "a", "base_url": "https://x", "model": "m"}]),
        encoding="utf-8",
    )
    migrated = model_store.migrate_to_toml(store, None)
    assert [m["id"] for m in migrated] == ["a"]
    assert store.exists()  # lecture seule : rien à écrire, rien à détruire
