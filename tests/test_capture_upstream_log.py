# `serve.log` ne reçoit QUE les lignes de llama-swap : celui-ci lance llama-server
# lui-même et retient sa sortie dans ses propres tampons, exposés en HTTP (`/logs`).
# Un démarrage raté ne laissait donc aucune trace dans le fichier vers lequel
# l'interface renvoyait l'utilisateur (vécu 2026-08-03 : « cause :
# var/logs/serve.log » sur un fichier de 0 octet).
import io

from loom.runtime import serve


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(body: bytes):
    def _open(url, timeout=None):
        _open.url = url
        return _Resp(body)

    return _open


def test_rapatrie_la_fin_du_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "SERVE_LOG", tmp_path / "serve.log")
    corps = "\n".join(f"ligne {i}" for i in range(200)).encode()
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(corps))

    n = serve.capture_upstream_log("http://127.0.0.1:8080", lines=50)

    assert n == 50
    txt = (tmp_path / "serve.log").read_text(encoding="utf-8")
    assert "journal llama-swap" in txt
    assert "ligne 199" in txt, "la FIN du journal est ce qui intéresse"
    assert "ligne 149" not in txt, "au-delà de la fenêtre demandée"


def test_journal_injoignable_renvoie_zero_sans_lever(tmp_path, monkeypatch):
    """Le serveur est mort : son journal peut l'être aussi. Le diagnostic ne doit
    pas aggraver la panne."""
    monkeypatch.setattr(serve, "SERVE_LOG", tmp_path / "serve.log")

    def _boom(url, timeout=None):
        raise OSError("connexion refusee")

    monkeypatch.setattr("urllib.request.urlopen", _boom)

    assert serve.capture_upstream_log("http://127.0.0.1:8080") == 0


def test_journal_vide_renvoie_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "SERVE_LOG", tmp_path / "serve.log")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(b"\n  \n"))

    assert serve.capture_upstream_log("http://127.0.0.1:8080") == 0
    assert not (tmp_path / "serve.log").exists()


def test_url_interrogee_est_bien_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "SERVE_LOG", tmp_path / "serve.log")
    op = _fake_urlopen(b"une ligne")
    monkeypatch.setattr("urllib.request.urlopen", op)

    serve.capture_upstream_log("http://127.0.0.1:8080/")

    assert op.url == "http://127.0.0.1:8080/logs"
