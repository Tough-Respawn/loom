# Une édition dans l'UI régénère llama-swap.yaml : elle doit garder les réglages
# machine mesurés (ubatch/batch/checkpoint), comme le lancement initial.
from types import SimpleNamespace

from loom.runtime import serve


def test_regenerate_garde_les_reglages_machine(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        models=[],
        server_bin="llama-server",
        models_dir=tmp_path,
        context=8192,
        override_n_gpu_layers=None,
        override_threads=10,
        n_parallel=1,
        default_ubatch=2048,
        default_batch=4096,
        default_checkpoint_min_step=1024,
        server_log_verbosity=4,
    )
    seen = {}
    monkeypatch.setattr(serve, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(serve, "detect_hardware", lambda *a, **k: object())
    monkeypatch.setattr(
        serve, "build_swap_config", lambda *a, **k: seen.update(k) or {"models": {}}
    )
    monkeypatch.setattr(serve, "write_swap_yaml", lambda *a, **k: None)

    assert serve.regenerate_swap_yaml("d", "l", tmp_path / "s.yaml") is not None
    assert seen["default_ubatch"] == 2048
    assert seen["default_batch"] == 4096
    assert seen["default_checkpoint_min_step"] == 1024
    assert seen["log_verbosity"] == 4  # journal llama-server conservé à la régénération
    assert seen["override_threads"] == 10  # threads mesurés par le bench, comme serve.py


def _fake_config(monkeypatch, tmp_path, models: dict):
    cfg = SimpleNamespace(
        models=list(models),
        server_bin="llama-server",
        models_dir=tmp_path,
        context=8192,
        override_n_gpu_layers=None,
        override_threads=None,
        n_parallel=1,
        default_ubatch=None,
        default_batch=None,
        default_checkpoint_min_step=None,
        server_log_verbosity=None,
    )
    monkeypatch.setattr(serve, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(serve, "detect_hardware", lambda *a, **k: object())
    monkeypatch.setattr(
        serve,
        "build_swap_config",
        lambda models, *a, **k: {"models": {m: {"cmd": models_cmd[m]} for m in models}},
    )
    models_cmd = dict(models)


def test_sync_differe_quand_le_modele_actif_ne_change_pas(monkeypatch, tmp_path):
    # Supprimer B pendant que A tourne : le yaml surveillé ne bouge pas (llama-swap
    # rechargerait TOUT, A compris), la nouvelle version attend la prochaine transition.
    from loom.runtime import swap_sync

    y = tmp_path / "llama-swap.yaml"
    _fake_config(monkeypatch, tmp_path, {"A": "cmd-a", "B": "cmd-b"})
    assert serve.sync_swap_yaml("d", "l", y, running=set()) == "written"
    applied = y.read_text(encoding="utf-8")
    _fake_config(monkeypatch, tmp_path, {"A": "cmd-a"})
    assert serve.sync_swap_yaml("d", "l", y, running={"A"}) == "deferred"
    assert y.read_text(encoding="utf-8") == applied
    assert "B" not in swap_sync.pending_path(y).read_text(encoding="utf-8")


def test_sync_config_illisible_renvoie_none(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise ValueError("toml cassé")

    monkeypatch.setattr(serve, "load_config", boom)
    assert serve.sync_swap_yaml("d", "l", tmp_path / "s.yaml", running=set()) is None


def test_lancement_de_llama_swap_efface_l_attente(monkeypatch, tmp_path):
    # Le lancement écrit une config fraîche : une version en attente est obsolète.
    from loom.runtime import swap_sync

    y = tmp_path / "llama-swap.yaml"
    monkeypatch.setattr(serve, "SWAP_YAML", y)
    swap_sync.pending_path(y).write_text("ancienne attente", encoding="utf-8")
    monkeypatch.setattr(serve, "_swap_config", lambda cfg, profile: {"models": {}})
    monkeypatch.setattr(serve, "_run", lambda args, *a, **k: 0)
    cfg = SimpleNamespace(swap_bin="llama-swap", port=8080)
    assert serve.launch_swap(cfg, object()) == 0
    assert y.exists()
    assert not swap_sync.pending_path(y).exists()


def test_aucun_test_n_ecrit_le_vrai_yaml_de_llama_swap():
    # Garde-fou (conftest) : vécu le 2026-10-14, un test du parcours /remove-model a
    # écrasé le VRAI var/cache/llama-swap.yaml par sa config factice.
    assert serve.SWAP_YAML != serve.REPO_ROOT / "var" / "cache" / "llama-swap.yaml"


def test_le_chemin_par_defaut_est_lu_a_l_appel(monkeypatch, tmp_path):
    # Un défaut figé à la définition (`out_path=SWAP_YAML`) échappait à toute
    # redirection : c'est ce qui a laissé le test écrire dans le vrai fichier.
    y = tmp_path / "ailleurs.yaml"
    monkeypatch.setattr(serve, "SWAP_YAML", y)
    _fake_config(monkeypatch, tmp_path, {"A": "cmd-a"})
    monkeypatch.setattr(serve, "write_swap_yaml", lambda cfg, path: seen.append(path))
    seen = []
    assert serve.regenerate_swap_yaml("d", "l") == y
    assert seen == [y]
    assert serve.sync_swap_yaml("d", "l", running=set()) == "written"
    assert y.exists()
