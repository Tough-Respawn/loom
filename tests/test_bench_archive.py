# tests/test_bench_archive.py
"""Archive DURABLE et automatique des benchs (revue du 2026-10-10).

Jusqu'ici les mesures ne vivaient que dans [bench] de local.toml (écrasé au bench
suivant), dans des commentaires de model.toml et dans l'état d'application d'une
session /rebench (consommé ou annulé). Le 3e run Ornith a perdu ses échantillons bruts
quand le verdict a été annulé. Chaque bench écrit désormais un JSON horodaté sous
var/bench/<modèle>/, complété par la trace de l'application quand elle a lieu.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loom.setup.archive import archive_bench, note_application
from loom.setup.placement import Placement

_NOW = datetime(2026, 10, 10, 13, 0, 0)


def test_archive_ecrit_un_json_horodate_par_modele(tmp_path):
    payload = {
        "verdict": {"placement": Placement("gpu_total", 999, ubatch=512, batch=2048)},
        "gguf": Path("C:/models/m.gguf"),
        "mesures": {
            "gpu_total@ub512@b2048": {"tg_ts": 11.3, "echantillons": [{"tg_ts": 11.3}]}
        },
        "rien": None,
    }
    p = archive_bench("ornith-1.5-35b-a3b", payload, root=tmp_path, now=_NOW)
    assert p == tmp_path / "ornith-1.5-35b-a3b" / "20261010-130000.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["version"] == 1 and d["date"] == "2026-10-10T13:00:00"
    assert d["model_id"] == "ornith-1.5-35b-a3b"
    # Dataclass -> dict avec sa clé ; Path -> str ; None conservé (JSON le permet).
    assert d["verdict"]["placement"]["key"] == "gpu_total@ub512@b2048"
    assert d["verdict"]["placement"]["ubatch"] == 512
    assert d["gguf"] == "C:/models/m.gguf" and d["rien"] is None
    assert d["mesures"]["gpu_total@ub512@b2048"]["echantillons"][0]["tg_ts"] == 11.3


def test_archive_ne_s_ecrase_pas_a_la_meme_seconde(tmp_path):
    a = archive_bench("m", {"n": 1}, root=tmp_path, now=_NOW)
    b = archive_bench("m", {"n": 2}, root=tmp_path, now=_NOW)
    assert a != b and a.exists() and b.exists()
    assert json.loads(a.read_text(encoding="utf-8"))["n"] == 1
    assert json.loads(b.read_text(encoding="utf-8"))["n"] == 2


def test_note_application_complete_l_archive(tmp_path):
    p = archive_bench("m", {"verdict": {"context": 65536}}, root=tmp_path, now=_NOW)
    note_application(
        p, {"context": 65536, "placement": "gpu_total", "ubatch": 512}, now=_NOW
    )
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["application"]["date"] == "2026-10-10T13:00:00"
    assert d["application"]["context"] == 65536 and d["application"]["ubatch"] == 512
    assert d["verdict"]["context"] == 65536  # le reste est intact


def test_compte_rendu_commun_aux_deux_parcours():
    """Revue 2026-10-10 (P2) : l'archive /rebench omettait matériel, binaire, flags,
    profil et plan. Un SCHÉMA commun, clés fixes, pour loom-setup et /rebench."""
    from loom.setup.archive import bench_payload

    p = bench_payload(source="/rebench", materiel={"gpu_name": "GPU"}, etape="fin")
    for cle in (
        "source",
        "etape",
        "gguf",
        "server_bin",
        "build",
        "materiel",
        "flags",
        "profil",
        "contexte_utile",
        "kv_estime_mb",
        "llama_bench",
        "isolation",
        "plan",
        "placement",
        "placement_avant",
        "calibration",
        "ubatch",
        "final",
        "cache",
        "ecrit",
        "verdict_texte",
        "verdict",
        "echec",
    ):
        assert cle in p
    assert p["materiel"] == {"gpu_name": "GPU"} and p["build"] is None
    # Une trace progressive peut porter des clés hors schéma : conservées, pas perdues.
    assert bench_payload(source="x", couples=[(512, 2048)])["couples"] == [(512, 2048)]


def test_note_application_dit_si_elle_a_echoue(tmp_path, monkeypatch):
    """Revue 2026-10-10 : l'annotation absorbait les erreurs — configuration appliquée
    sans trace, à l'insu de l'utilisateur. Elle renvoie désormais True/False."""
    from loom.setup import archive as archive_mod

    p = archive_bench("m", {"verdict": {"context": 65536}}, root=tmp_path, now=_NOW)
    assert note_application(p, {"context": 65536}, now=_NOW) is True
    assert (
        note_application(tmp_path / "absent.json", {"context": 1}) is False
    )  # ne lève pas

    def _plein(*a, **k):
        raise OSError("disque plein")

    monkeypatch.setattr(archive_mod, "atomic_write_text", _plein)
    assert note_application(p, {"context": 1}) is False
