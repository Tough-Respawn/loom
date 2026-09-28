# Écriture atomique des fichiers de config/état : jamais de fichier tronqué, ni de
# temporaire partagé entre écrivains concurrents, et tolérance au verrou Windows.
import json
import os
import threading

import pytest

from loom.utils import atomic_write_text


def test_ecrit_et_ne_laisse_aucun_temporaire(tmp_path):
    p = tmp_path / "sub" / "local.toml"
    atomic_write_text(p, "a = 1\n")
    atomic_write_text(p, "a = 2\n")
    assert p.read_text(encoding="utf-8") == "a = 2\n"
    assert [f.name for f in p.parent.iterdir()] == ["local.toml"]


def test_ecrivains_concurrents_jamais_de_json_tronque(tmp_path):
    p = tmp_path / "session.json"
    errors = []

    def writer(n):
        try:
            for i in range(30):
                atomic_write_text(p, json.dumps({"w": n, "i": i, "pad": "x" * 5000}))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert json.loads(p.read_text(encoding="utf-8"))["pad"] == "x" * 5000
    assert [f.name for f in tmp_path.iterdir()] == ["session.json"]


def test_reessaie_si_la_cible_est_verrouillee(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Accès refusé")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    atomic_write_text(p, "ok")
    assert p.read_text(encoding="utf-8") == "ok" and calls["n"] == 3


def test_echec_definitif_garde_l_ancien_contenu(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    p.write_text("ancien", encoding="utf-8")

    def locked(src, dst):
        raise PermissionError(5, "Accès refusé")

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(PermissionError):
        atomic_write_text(p, "nouveau")
    assert p.read_text(encoding="utf-8") == "ancien"
    assert [f.name for f in tmp_path.iterdir()] == ["f.txt"]
