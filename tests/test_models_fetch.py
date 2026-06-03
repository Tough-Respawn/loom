# tests/test_models_fetch.py
from pathlib import Path
from unittest.mock import patch

from loom.models_fetch import ensure_model


def test_returns_existing_without_download(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    existing = models_dir / "model.gguf"
    existing.write_bytes(b"GGUF")
    with patch("loom.models_fetch.hf_hub_download") as dl:
        result = ensure_model("some/repo", "model.gguf", models_dir)
    dl.assert_not_called()
    assert result == existing


def test_downloads_when_absent(tmp_path):
    models_dir = tmp_path / "models"
    target = models_dir / "model.gguf"

    def fake_download(repo_id, filename, local_dir, **kwargs):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / filename).write_bytes(b"GGUF")
        return str(Path(local_dir) / filename)

    with patch("loom.models_fetch.hf_hub_download", side_effect=fake_download) as dl:
        result = ensure_model("some/repo", "model.gguf", models_dir)
    dl.assert_called_once()
    assert result == target
    assert target.exists()
