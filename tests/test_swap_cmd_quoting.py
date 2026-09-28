# La commande llama-swap est une CHAÎNE que llama-swap redécoupe (shlex.Windows /
# shlex.Posix, internal/config/commands.go) : un chemin avec espace doit être entre
# guillemets doubles, sinon llama-server reçoit un chemin de modèle tronqué.
import shlex

from loom.config import ModelConfig
from loom.runtime.hardware import HardwareProfile
from loom.runtime.swap import _model_cmd

_PROFILE = HardwareProfile(True, "GPU", 8000, 16, vram_total_mb=8000, backend="Vulkan")


def test_chemins_avec_espaces_restent_un_seul_argument():
    model = ModelConfig(
        id="m",
        filename="mon modèle.gguf",
        repo="r/m",
        n_layers=40,
        size_mb=1000,
        mmproj_filename="mmproj x.gguf",
    )
    cmd = _model_cmd(
        model,
        _PROFILE,
        "C:\Program Files\llama\llama-server.exe",
        "C:/Users/Jean Dupont/models",
        8192,
        slot_save_dir="C:/Users/Jean Dupont/slots",
        n_parallel=1,
    )
    args = shlex.split(cmd)
    assert args[0] == "C:/Program Files/llama/llama-server.exe"
    assert "C:/Users/Jean Dupont/models/mon modèle.gguf" in args
    assert (
        args[args.index("--mmproj") + 1] == "C:/Users/Jean Dupont/models/mmproj x.gguf"
    )
    assert "C:/Users/Jean Dupont/slots" in " ".join(args)
    # Le placeholder de port reste intact pour la substitution de llama-swap.
    assert "${PORT}" in args


def test_sans_espace_la_commande_ne_change_pas():
    model = ModelConfig(
        id="m", filename="m.gguf", repo="r/m", n_layers=40, size_mb=1000
    )
    cmd = _model_cmd(model, _PROFILE, "llama-server", "/models", 8192, n_parallel=1)
    assert '"' not in cmd
    assert cmd.startswith("llama-server ")
