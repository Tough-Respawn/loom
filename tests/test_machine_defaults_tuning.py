# Repli MACHINE des réglages mesurés, sur le modèle de `context`.
#
# Un modèle ajouté par /add-model n'est JAMAIS benché : seuls `n_layers` et `cpu_moe`
# sont lus dans le GGUF. `context` avait déjà un repli machine ([server].context) ;
# `ubatch`, `batch` et `checkpoint_min_step` n'en avaient aucun et tombaient sur des
# constantes aveugles (512 / 2048 / défaut serveur) alors que la machine avait ses
# valeurs mesurées — 61 % de prefill perdus, mesuré le 2026-07-21.
#
# Règle : le model.toml GAGNE toujours ; la machine n'est qu'un défaut informé.
from loom.config import ModelConfig
from loom.runtime.hardware import HardwareProfile
from loom.runtime.swap import _model_cmd

_PROFILE = HardwareProfile(True, "GPU", 8000, 16, vram_total_mb=8000, backend="Vulkan")


def _cmd(model, **defaults):
    return _model_cmd(
        model, _PROFILE, "llama-server", "/models", 8192, n_parallel=1, **defaults
    )


def _m(**kw):
    return ModelConfig(
        id="m", filename="m.gguf", repo="r/m", n_layers=40, size_mb=1000, **kw
    )


def test_machine_sert_de_repli_quand_le_modele_ne_dit_rien():
    cmd = _cmd(
        _m(),
        default_ubatch=2048,
        default_batch=4096,
        default_checkpoint_min_step=2048,
    )
    assert "-ub 2048" in cmd
    assert "-b 4096" in cmd
    assert "--checkpoint-min-step 2048" in cmd


def test_le_modele_garde_la_priorite_sur_la_machine():
    cmd = _cmd(
        _m(ubatch=512, batch=1024, checkpoint_min_step=8192),
        default_ubatch=2048,
        default_batch=4096,
        default_checkpoint_min_step=2048,
    )
    assert "-ub 512" in cmd
    assert "-b 1024" in cmd
    assert "--checkpoint-min-step 8192" in cmd


def test_sans_repli_machine_le_comportement_est_inchange():
    """Aucune régression pour une machine jamais benchée : on retombe sur les
    constantes historiques de build_server_args."""
    cmd = _cmd(_m())
    assert "-ub 512" in cmd
    assert "-b 2048" in cmd
    assert "--checkpoint-min-step" not in cmd


def test_repli_partiel_chaque_reglage_est_independant():
    cmd = _cmd(_m(ubatch=1024), default_ubatch=2048, default_batch=4096)
    assert "-ub 1024" in cmd, "le modèle gagne sur ubatch"
    assert "-b 4096" in cmd, "la machine comble batch"
