# Mise à jour du SearXNG géré (2026-10-17) : l'image de juillet, jamais mise à jour, ne tirait
# plus rien de Yahoo, corrigé en amont depuis. Sans Docker réel : `_docker` est simulé.
from types import SimpleNamespace

import loom.runtime.searxng as sx


def _fake_docker(monkeypatch, answers, calls):
    def fake(*args, timeout=30.0):
        calls.append(args)
        out = answers.get(args[0], "")
        return SimpleNamespace(
            returncode=0 if out is not None else 1, stdout=out or "", stderr="boom"
        )

    monkeypatch.setattr(sx, "_docker", fake)


def test_image_installee_lue_depuis_le_conteneur(monkeypatch):
    monkeypatch.setattr(sx, "docker_available", lambda: True)
    monkeypatch.setattr(sx, "container_state", lambda: "running")
    calls = []
    _fake_docker(monkeypatch, {
        "inspect": "sha256:imageid\n",
        "image": '["searxng/searxng@sha256:abc"]|2026-07-10T12:26:01Z\n',
    }, calls)  # fmt: skip
    assert sx.installed_image() == {"digests": ["sha256:abc"], "date": "2026-07-10"}


def test_sans_docker_ni_conteneur_rien_a_proposer(monkeypatch):
    monkeypatch.setattr(sx, "docker_available", lambda: True)
    monkeypatch.setattr(sx, "container_state", lambda: "absent")
    assert sx.installed_image() is None


def test_telechargement_en_echec_ne_touche_pas_au_conteneur(monkeypatch):
    calls = []
    _fake_docker(monkeypatch, {"port": "127.0.0.1:8890\n", "pull": None}, calls)
    monkeypatch.setattr(sx, "install", lambda port: "jamais appelé")
    assert sx.update() is None
    assert not any(c[0] == "rm" for c in calls)


def test_mise_a_jour_garde_le_port_du_conteneur(monkeypatch):
    calls, ports = [], []
    _fake_docker(
        monkeypatch, {"port": "127.0.0.1:8899\n", "pull": "ok", "rm": "ok"}, calls
    )
    monkeypatch.setattr(
        sx, "install", lambda port: ports.append(port) or f"http://127.0.0.1:{port}"
    )
    assert sx.update() == "http://127.0.0.1:8899"
    assert [c[0] for c in calls] == ["port", "pull", "rm"] and ports == [8899]
