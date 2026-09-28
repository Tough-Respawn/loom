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
        n_parallel=1,
        default_ubatch=2048,
        default_batch=4096,
        default_checkpoint_min_step=1024,
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
