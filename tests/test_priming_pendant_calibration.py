# Pendant un /rebench, le banc ARRÊTE le serveur modèle pour récupérer la VRAM.
# L'amorçage du cache KV n'en savait rien : `running_local()` répond « vivant » dès
# que llama-swap — le routeur, toujours debout — décroche, donc le garde
# `require_running` passait et le warm_context suivant échouait en rafale
# (`WARM_CTX_ERR` / `Connection error`, constaté 2026-08-03).
#
# Pire : deux appels d'amorçage passent `wait_server=90`, qui REDÉMARRE le serveur.
# En pleine calibration, il dispute la RAM au banc et fausse la mesure.
import time

import pytest

from loom.web.routes import priming
from loom.web.routes.priming import _calibration_en_cours


class _Job:
    def __init__(self, done=False):
        self.done = done


@pytest.fixture(autouse=True)
def _reset():
    from loom.web.routes.rebench import _REBENCH

    avant = _REBENCH.get("job")
    yield
    _REBENCH["job"] = avant


def _set_job(job):
    from loom.web.routes.rebench import _REBENCH

    _REBENCH["job"] = job


def test_calibration_en_cours_est_detectee():
    _set_job(_Job(done=False))
    assert _calibration_en_cours() is True


def test_calibration_terminee_ne_bloque_plus():
    _set_job(_Job(done=True))
    assert _calibration_en_cours() is False


def test_aucune_calibration_ne_bloque_pas():
    _set_job(None)
    assert _calibration_en_cours() is False


class _Sess:
    id = "s1"
    workspace = None

    class conversation:
        model = "ornith-q5"
        thinking = False
        active_tools = []


class _S:
    remote_model_ids = image_model_ids = video_model_ids = set()
    client = object()


def test_amorcage_saute_pendant_une_calibration(capsys, monkeypatch):
    """Le garde doit tomber AVANT tout démarrage de serveur : c'est le point clé,
    `wait_server` relancerait un serveur contre le banc."""
    _set_job(_Job(done=False))
    appels = []
    monkeypatch.setattr(
        priming,
        "_ensure_local_server",
        lambda *a, **k: appels.append("start") or True,
    )

    priming._prime_async(_S(), _Sess(), wait_server=90.0)
    time.sleep(0.3)  # le thread est daemon : lui laisser le temps de sortir

    assert appels == [], "aucun serveur ne doit être démarré pendant une calibration"
    assert "calibration en cours" in capsys.readouterr().out
