# tests/test_eval_artifacts.py
"""Conservation des preuves d'éval : deux campagnes ne s'écrasent JAMAIS.

Régression visée : out/<variante>/ était écrasé à chaque campagne — l'échec
windows_shell 2/3 du 2026-07-24 (a644e00) est resté indiagnosticable faute de
transcript. Chaque campagne écrit désormais sous out/runs/<sha>_<horodatage>[-N].
"""

from __future__ import annotations

import pytest

from evals import run_eval
from evals.run_eval import (
    Trajectory,
    _save_transcript,
    new_campaign_dir,
    write_campaign_meta,
)


@pytest.fixture()
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "_OUT", tmp_path)
    return tmp_path


def _traj() -> Trajectory:
    return Trajectory(
        tool_calls=[("run_shell", {"command": "Get-ChildItem"})],
        tool_results=[{"name": "run_shell", "ok": True, "preview": "3"}],
        final_text="3 fichiers",
    )


def test_meme_commit_meme_seconde_dossiers_distincts(out):
    d1 = new_campaign_dir(sha="abc1234", stamp="20260809-120000")
    d2 = new_campaign_dir(sha="abc1234", stamp="20260809-120000")
    assert d1 != d2 and d1.exists() and d2.exists()
    assert d1.parent == out / "runs" and d2.parent == out / "runs"


def test_deux_campagnes_successives_conservent_leurs_transcripts(out):
    checks_ok = {"pas d'unix-isme dans run_shell": True}
    checks_ko = {"pas d'unix-isme dans run_shell": False}

    d1 = new_campaign_dir(sha="abc1234", stamp="20260809-120000")
    _save_transcript(d1, "new", "windows_shell", 0, _traj(), checks_ok, None)

    d2 = new_campaign_dir(sha="abc1234", stamp="20260809-120000")
    _save_transcript(d2, "new", "windows_shell", 0, _traj(), checks_ko, None)

    t1 = d1 / "new" / "windows_shell_run1.md"
    t2 = d2 / "new" / "windows_shell_run1.md"
    # Les DEUX transcripts existent (donc ceux des cas échoués aussi) et sont
    # bien ceux de leur campagne respective.
    assert t1.exists() and t2.exists()
    assert "[x]" in t1.read_text(encoding="utf-8")
    assert "[ ]" in t2.read_text(encoding="utf-8")


def test_campaign_json_porte_les_conditions_du_run(out):
    import json

    d = new_campaign_dir(sha="abc1234", stamp="20260809-140000")
    write_campaign_meta(
        d,
        sha="abc1234",
        dirty=True,
        model="ornith-1.0-35b",
        variants=["old", "new"],
        runs=3,
        cases={"windows_shell", "edit_block"},
        judge=True,
    )
    meta = json.loads((d / "campaign.json").read_text(encoding="utf-8"))
    assert meta["sha"] == "abc1234"
    assert meta["dirty"] is True
    assert meta["model"] == "ornith-1.0-35b"
    assert meta["variants"] == ["old", "new"]
    assert meta["runs"] == 3
    assert meta["cases"] == ["edit_block", "windows_shell"]  # triés, reproductibles
    assert meta["judge"] is True
    assert meta["date"].endswith("Z")  # horodatage UTC
    assert "diff" not in meta  # jamais le diff complet


def test_report_ecrit_dans_le_dossier_de_campagne(out, capsys):
    d = new_campaign_dir(sha="abc1234", stamp="20260809-130000")
    results = {"new": {"windows_shell": []}}
    run_eval.report(results, runs=0, run_dir=d)
    assert (d / "report.json").exists()
    assert (out / "report.json").exists()  # copie « dernier run » conservée
