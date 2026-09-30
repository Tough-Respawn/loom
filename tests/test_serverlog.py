"""Journaux llama-server : arguments, archivage avant écrasement, crochet de swap."""

import os

import pytest

from loom.agent import calltrace
from loom.runtime import serverlog
from loom.runtime.server_args import build_server_args


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(serverlog, "LLAMA_LOGS_DIR", tmp_path / "llama")
    return tmp_path / "llama"


def _args(**kw):
    return build_server_args(
        server_bin="llama-server",
        model_path="m.gguf",
        port=8080,
        context=4096,
        n_gpu_layers=999,
        threads=8,
        **kw,
    )


def test_log_file_et_verbosite_passes_ensemble():
    args = _args(log_file="C:/x/m.log", log_verbosity=4)
    i = args.index("--log-file")
    assert args[i + 1] == "C:/x/m.log" and args[i + 2 : i + 4] == ["-lv", "4"]


def test_pas_de_journal_sans_verbosite():
    assert "--log-file" not in _args(log_file="C:/x/m.log", log_verbosity=0)
    assert "--log-file" not in _args(log_file=None, log_verbosity=4)


def test_chemin_par_modele_nettoye(logs):
    path = serverlog.server_log_path("ternary-bonsai 2/q2")
    assert path.endswith("/ternary-bonsai_2_q2.log") and "\\" not in path


def test_archive_copie_sans_toucher_l_original(logs):
    src = logs / "m.log"
    logs.mkdir(parents=True)
    src.write_text("ligne\n", encoding="utf-8")
    dest = serverlog.archive_server_log("m")
    assert dest is not None and dest.read_text(encoding="utf-8") == "ligne\n"
    assert src.exists()  # copie : le fichier peut être ouvert par un serveur vivant
    assert serverlog.archive_server_log("m") == dest  # pas de doublon identique


def test_archive_ignore_absent_ou_vide(logs):
    assert serverlog.archive_server_log("absent") is None
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "vide.log").write_text("", encoding="utf-8")
    assert serverlog.archive_server_log("vide") is None


def test_elagage_garde_les_plus_recents(logs, monkeypatch):
    monkeypatch.setattr(serverlog, "KEEP_ARCHIVES", 3)
    arch = logs / "archive"
    arch.mkdir(parents=True)
    for i in range(5):
        f = arch / f"m-{i}.log"
        f.write_text(str(i), encoding="utf-8")
        os.utime(f, (1000 + i, 1000 + i))
    serverlog._prune(arch)
    assert sorted(p.name for p in arch.iterdir()) == ["m-2.log", "m-3.log", "m-4.log"]


def test_crochet_appele_seulement_au_changement_de_modele(monkeypatch):
    seen = []
    monkeypatch.setattr(calltrace, "_last_model", None)
    monkeypatch.setattr(calltrace, "_model_switch_hook", seen.append)
    for model in ["a", "a", "b", "b", "a"]:
        calltrace._note_model(model)
    assert seen == ["a", "b", "a"]
