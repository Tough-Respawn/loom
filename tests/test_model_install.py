# Installation locale /add-model : fonctions pures (id, reco quant, model.toml)
# puis download en thread + finalisation — SANS réseau (ensure_model faké).
import tomllib

from loom.runtime import model_install
from loom.runtime.model_install import (
    derive_model_id,
    finalize_model_toml,
    recommend_quant,
    start_download,
    write_model_toml,
)


def test_derive_model_id():
    assert derive_model_id("unsloth/Qwen3-30B-A3B-GGUF") == "qwen3-30b-a3b"
    assert derive_model_id("org/Modele_GGUF") == "modele"
    assert derive_model_id("org/déjà!!") == "d-j"


def test_recommend_quant_le_plus_gros_qui_tient():
    files = [
        {"filename": "q4.gguf", "size_mb": 10_000},
        {"filename": "q8.gguf", "size_mb": 20_000},
        {"filename": "f16.gguf", "size_mb": 40_000},
    ]
    # budget = 6000 VRAM + 24000 RAM - 4096 marge = 25904 Mo -> q8 recommandé, f16 ne tient pas
    out = recommend_quant(files, vram_free_mb=6000, ram_mb=24000)
    by = {f["filename"]: f for f in out}
    assert by["q8.gguf"]["recommended"] is True
    assert by["q4.gguf"] == dict(files[0], fits=True, recommended=False)
    assert by["f16.gguf"]["fits"] is False
    # l'entrée d'origine n'est PAS mutée
    assert "fits" not in files[0]


def test_recommend_quant_rien_ne_tient():
    out = recommend_quant([{"filename": "f16.gguf", "size_mb": 40_000}], 2000, 1000)
    assert out[0]["fits"] is False and out[0]["recommended"] is False


def test_write_model_toml(tmp_path):
    d = tmp_path / "mon-modele"
    p = write_model_toml(
        d, "org/repo-GGUF", "m.Q4_K_M.gguf", 5000, mmproj_filename="mmproj-F16.gguf"
    )
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    assert raw == {
        "repo": "org/repo-GGUF",
        "filename": "m.Q4_K_M.gguf",
        "size_mb": 5000,
        "mmproj_filename": "mmproj-F16.gguf",
    }
    assert (d / "profile.md").exists()  # stub (spec §4)


def test_write_model_toml_sans_mmproj(tmp_path):
    p = write_model_toml(tmp_path / "m", "org/r", "m.gguf", 100)
    assert "mmproj" not in p.read_text(encoding="utf-8")


def test_finalize_model_toml(tmp_path, monkeypatch):
    d = tmp_path / "m"
    write_model_toml(d, "org/r", "m.gguf", 100)
    monkeypatch.setattr(
        model_install,
        "read_gguf_meta",
        lambda p: {
            "architecture": "qwen3moe",
            "n_layers": 48,
            "context_length": 262144,
            "expert_count": 128,
        },
    )
    meta = finalize_model_toml(d, d / "m.gguf")
    raw = tomllib.loads((d / "model.toml").read_text(encoding="utf-8"))
    assert raw["n_layers"] == 48
    assert raw["cpu_moe"] is True  # MoE détecté -> experts en RAM par défaut
    assert meta["expert_count"] == 128


def test_finalize_gguf_illisible_est_best_effort(tmp_path, monkeypatch):
    d = tmp_path / "m"
    write_model_toml(d, "org/r", "m.gguf", 100)

    def _boom(p):
        raise ValueError("pas un fichier GGUF")

    monkeypatch.setattr(model_install, "read_gguf_meta", _boom)
    assert finalize_model_toml(d, d / "m.gguf") == {}
    raw = tomllib.loads((d / "model.toml").read_text(encoding="utf-8"))
    assert "n_layers" not in raw  # toml intact


def test_download_job_succes(tmp_path, monkeypatch):
    # ensure_model FAKE : écrit le fichier (aucun réseau), le job doit finir done
    # sans erreur et déclencher on_done.
    def fake_ensure(repo, fn, dest):
        p = tmp_path / fn
        p.write_bytes(b"x" * 2048)
        return p

    monkeypatch.setattr(model_install, "ensure_model", fake_ensure)
    seen = []
    job = start_download(
        "org/r",
        ["a.gguf", "b.gguf"],
        tmp_path,
        total_mb=1,
        on_done=lambda j: seen.append(j.error),
    )
    job._thread.join(timeout=10)
    assert job.done is True and job.error is None
    assert seen == [None]
    assert job.progress_mb() == 0  # 4096 octets -> 0 Mo entier : compteur en Mo


def test_download_job_on_done_avant_done(tmp_path, monkeypatch):
    """on_done doit s'exécuter AVANT que `done` devienne vrai : les observateurs
    de `done` (le flux SSE qui pousse « models » au front) ne doivent se réveiller
    qu'une fois la finalisation/montage de on_done TERMINÉE — sinon le sélecteur
    se recharge avant le montage et rate le nouveau modèle."""
    monkeypatch.setattr(model_install, "ensure_model", lambda r, f, d: None)
    observed = []
    job = start_download(
        "org/r", ["a.gguf"], tmp_path, 1, on_done=lambda j: observed.append(j.done)
    )
    job._thread.join(timeout=10)
    assert observed == [False]  # vu depuis on_done : pas encore done
    assert job.done is True  # et done à la toute fin


def test_download_job_echec_message_actionnable(tmp_path, monkeypatch):
    from loom.runtime.models_fetch import ModelUnavailable

    def fake_ensure(repo, fn, dest):
        raise ModelUnavailable("Modèle indisponible : pose le fichier ici : X")

    monkeypatch.setattr(model_install, "ensure_model", fake_ensure)
    job = start_download("org/r", ["a.gguf"], tmp_path, total_mb=1)
    job._thread.join(timeout=10)
    assert job.done is True
    assert "pose le fichier" in job.error
