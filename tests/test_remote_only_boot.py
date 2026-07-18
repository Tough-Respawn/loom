# Boot « remote-only » : load_config tolère un parc local vide quand un modèle
# DISTANT existe ([[remote_models]] de local.toml OU store UI remote_models.json),
# et maybe_bootstrap(remote_ok=True) ne force pas l'installeur dans ce cas.
# Machine vraiment vierge (ni local ni distant) : comportement historique conservé
# (ValueError côté load_config, installeur côté bootstrap).
from __future__ import annotations

import json

import pytest

from loom.config import load_config, remote_store_path
from loom.setup.steps import has_remote_models

REMOTE_TOML = """
[[remote_models]]
id = "glm-distant"
base_url = "https://api.exemple/v4"
model = "glm-4.6"
"""


def _write_defaults(tmp_path, extra: str = "", default_model: str = "") -> str:
    """defaults.toml minimal, tout isolé sous tmp_path (dont le store distant, dérivé
    de chat.history_path — sinon le test lirait le var/remote_models.json du repo)."""
    p = tmp_path / "defaults.toml"
    dm = f'default_model = "{default_model}"\n' if default_model else ""
    p.write_text(
        f"""
[server]
context = 8192
port = 8080
bin = "llama-server"

[chat]
history_path = "{(tmp_path / 'var' / 'conversation.json').as_posix()}"
{dm}
[storage]
models_root = "{(tmp_path / 'models').as_posix()}"
{extra}
""",
        encoding="utf-8",
    )
    return str(p)


def _install_local_model(tmp_path, mid: str = "local-1") -> None:
    d = tmp_path / "models" / "local" / "text" / mid
    d.mkdir(parents=True)
    (d / "model.toml").write_text(
        'repo = "org/x-GGUF"\nfilename = "x.gguf"\nn_layers = 10\nsize_mb = 100\n',
        encoding="utf-8",
    )


def _write_store(tmp_path, models: list[dict]) -> None:
    store = tmp_path / "var" / "remote_models.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(models), encoding="utf-8")


def test_boot_remote_only_via_toml(tmp_path):
    cfg = load_config(_write_defaults(tmp_path, REMOTE_TOML))
    assert cfg.models == []
    assert [rm.id for rm in cfg.remote_models] == ["glm-distant"]
    assert cfg.default_model == "glm-distant"


def test_boot_remote_only_via_store(tmp_path):
    _write_store(
        tmp_path,
        [{"id": "glm-ui", "base_url": "https://api.exemple/v4", "model": "glm-4.6"}],
    )
    cfg = load_config(_write_defaults(tmp_path))
    assert cfg.models == []
    assert [rm.id for rm in cfg.remote_models] == ["glm-ui"]
    assert cfg.default_model == "glm-ui"


def test_store_ecrase_le_toml_par_id(tmp_path):
    _write_store(
        tmp_path,
        [{"id": "glm-distant", "base_url": "https://autre.api/v4", "model": "glm-5"}],
    )
    cfg = load_config(_write_defaults(tmp_path, REMOTE_TOML))
    assert len(cfg.remote_models) == 1
    assert cfg.remote_models[0].base_url == "https://autre.api/v4"
    assert cfg.remote_models[0].model == "glm-5"


def test_boot_vide_leve_toujours(tmp_path):
    with pytest.raises(ValueError, match="aucun modèle"):
        load_config(_write_defaults(tmp_path))


def test_default_model_local_disparu_retombe_sur_le_distant(tmp_path):
    # default_model pointe un modèle local supprimé : en remote-only, on retombe sur
    # le premier distant plutôt que d'envoyer la session dans le vide (404 llama-swap).
    cfg = load_config(
        _write_defaults(tmp_path, REMOTE_TOML, default_model="local-disparu")
    )
    assert cfg.default_model == "glm-distant"


def test_local_present_garde_le_comportement_historique(tmp_path):
    _install_local_model(tmp_path, "local-1")
    cfg = load_config(_write_defaults(tmp_path, REMOTE_TOML))
    assert [m.id for m in cfg.models] == ["local-1"]
    assert cfg.default_model == "local-1"  # le local reste le défaut


def test_cfg_model_sans_local_erreur_claire(tmp_path):
    cfg = load_config(_write_defaults(tmp_path, REMOTE_TOML))
    with pytest.raises(ValueError, match="aucun modèle local"):
        _ = cfg.model


def test_has_remote_models(tmp_path):
    hist = {"chat": {"history_path": str(tmp_path / "var" / "conversation.json")}}
    assert has_remote_models({"remote_models": [{"id": "glm"}], **hist}) is True
    assert has_remote_models(dict(hist)) is False
    _write_store(
        tmp_path,
        [{"id": "glm-ui", "base_url": "https://api.exemple/v4", "model": "glm-4.6"}],
    )
    assert has_remote_models(dict(hist)) is True


def test_remote_store_path_derive_de_l_historique(tmp_path):
    hist = tmp_path / "var" / "conversation.json"
    assert remote_store_path(hist) == tmp_path / "var" / "remote_models.json"


def test_maybe_bootstrap_remote_ok_saute_l_installeur(tmp_path, monkeypatch):
    from loom.runtime import serve
    from loom.setup import steps

    monkeypatch.setattr(serve, "_log", lambda msg: None)
    monkeypatch.setattr(steps, "needs_setup", lambda raw, pkg: True)
    monkeypatch.setattr(
        steps, "read_raw_config", lambda a, b: {"remote_models": [{"id": "glm"}]}
    )
    # Un distant existe : loom.web (remote_ok) boote sans installeur…
    assert serve.maybe_bootstrap(remote_ok=True) is None
    # …mais serve (moteur local) exige toujours le setup (non-tty en tests -> code 1).
    assert serve.maybe_bootstrap() == 1
    # Ni local ni distant : même remote_ok déclenche le chemin installeur.
    # (history_path isolé sous tmp_path : sans lui, le check du store distant
    # lirait le var/remote_models.json du repo de dev.)
    vierge = {"chat": {"history_path": str(tmp_path / "var" / "conversation.json")}}
    monkeypatch.setattr(steps, "read_raw_config", lambda a, b: vierge)
    assert serve.maybe_bootstrap(remote_ok=True) == 1
