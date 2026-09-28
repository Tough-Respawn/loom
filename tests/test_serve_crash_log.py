# serve.py est lancé sans console (stdout/stderr -> DEVNULL) : un crash inattendu au
# démarrage doit laisser sa trace dans serve.log, le journal que l'UI désigne.
from loom.runtime import serve


def test_crash_au_demarrage_ecrit_la_trace_dans_serve_log(tmp_path, monkeypatch):
    log = tmp_path / "serve.log"
    monkeypatch.setattr(serve, "SERVE_LOG", log)
    monkeypatch.setattr(serve, "maybe_bootstrap", lambda *a, **k: None)

    def broken(*a, **k):
        raise ValueError("local.toml illisible (ligne 12)")

    monkeypatch.setattr(serve, "load_config", broken)
    assert serve.main() == 1
    text = log.read_text(encoding="utf-8")
    assert "local.toml illisible (ligne 12)" in text
    assert "Traceback" in text
