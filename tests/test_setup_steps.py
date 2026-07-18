# Installeur loom-setup : logique pure (config brute, constats, écriture
# local.toml) — SANS réseau, tout sur tmp_path.
import tomllib

from loom.setup.steps import (
    installed_model_ids,
    models_roots,
    needs_setup,
    read_raw_config,
    server_bin_status,
    set_server_bin,
)


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_read_raw_config_sans_modele_ne_leve_pas(tmp_path):
    defaults = _write(tmp_path / "defaults.toml", '[server]\nbin = "llama-server"\n')
    raw = read_raw_config(defaults, tmp_path / "local.toml")  # local absent
    assert raw["server"]["bin"] == "llama-server"


def test_read_raw_config_local_ecrase(tmp_path):
    defaults = _write(
        tmp_path / "defaults.toml", '[server]\nbin = "llama-server"\nport = 8080\n'
    )
    local = _write(tmp_path / "local.toml", '[server]\nbin = "C:/x/llama-server.exe"\n')
    raw = read_raw_config(defaults, local)
    assert raw["server"]["bin"] == "C:/x/llama-server.exe"
    assert raw["server"]["port"] == 8080  # fusion, pas remplacement


def test_server_bin_status_fichier_absolu(tmp_path):
    exe = _write(tmp_path / "llama-server.exe", "")
    present, name = server_bin_status({"server": {"bin": str(exe)}})
    assert present is True and name == str(exe)


def test_server_bin_status_via_path(monkeypatch):
    monkeypatch.setattr("loom.setup.steps.shutil.which", lambda n: "/usr/bin/" + n)
    present, name = server_bin_status({})  # défaut "llama-server"
    assert present is True and name == "llama-server"


def test_server_bin_status_absent(monkeypatch):
    monkeypatch.setattr("loom.setup.steps.shutil.which", lambda n: None)
    present, _ = server_bin_status({"server": {"bin": "introuvable-xyz"}})
    assert present is False


def test_models_roots_defaut_et_liste(tmp_path):
    assert models_roots({}, tmp_path) == [tmp_path.resolve()]
    raw = {"storage": {"models_root": [str(tmp_path / "a"), str(tmp_path / "b")]}}
    assert models_roots(raw, tmp_path) == [
        (tmp_path / "a").resolve(),
        (tmp_path / "b").resolve(),
    ]


def test_installed_model_ids(tmp_path):
    _write(
        tmp_path / "local" / "text" / "mon-modele" / "model.toml",
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 1\nsize_mb = 1\n',
    )
    # _TEMPLATE ignoré (préfixe _), dossier sans model.toml ignoré
    _write(tmp_path / "local" / "text" / "_TEMPLATE" / "model.toml", 'repo = "x"\n')
    (tmp_path / "local" / "text" / "vide").mkdir()
    assert installed_model_ids({}, tmp_path) == ["mon-modele"]


def test_installed_model_ids_racine_absente(tmp_path):
    assert installed_model_ids({}, tmp_path / "nexiste-pas") == []


def test_needs_setup(tmp_path, monkeypatch):
    monkeypatch.setattr("loom.setup.steps.shutil.which", lambda n: None)
    exe = _write(tmp_path / "llama-server.exe", "")
    _write(
        tmp_path / "local" / "text" / "m1" / "model.toml",
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 1\nsize_mb = 1\n',
    )
    ok_cfg = {"server": {"bin": str(exe)}}
    assert needs_setup(ok_cfg, tmp_path) is False  # binaire + modèle -> prêt
    assert needs_setup({}, tmp_path) is True  # binaire absent
    assert needs_setup(ok_cfg, tmp_path / "vide") is True  # aucun modèle


def test_set_server_bin_cree_local_toml(tmp_path):
    local = tmp_path / "local.toml"
    exe = tmp_path / "bin" / "llama-server.exe"
    set_server_bin(local, exe)
    raw = tomllib.loads(local.read_text(encoding="utf-8"))
    assert raw["server"]["bin"] == str(exe.resolve()).replace("\\", "/")


def test_set_server_bin_preserve_l_existant(tmp_path):
    local = _write(
        tmp_path / "local.toml",
        "# mon commentaire\n[tools]\n"
        'workspace_dir = "C:/projets"\n\n[server]\nswap_bin = "swap.exe"\n',
    )
    set_server_bin(local, tmp_path / "llama-server.exe")
    text = local.read_text(encoding="utf-8")
    assert "# mon commentaire" in text  # commentaires préservés (tomlkit)
    raw = tomllib.loads(text)
    assert raw["tools"]["workspace_dir"] == "C:/projets"
    assert raw["server"]["swap_bin"] == "swap.exe"
    assert raw["server"]["bin"].endswith("llama-server.exe")
