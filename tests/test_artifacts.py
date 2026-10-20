# Images du bac à sable montrées dans le fil : balayage par tour (nouvelles, image
# seulement, jamais deux fois) et route de service (sid strict, extensions image).
from __future__ import annotations

import os
import time

from loom.web.artifacts import new_artifacts


def test_new_artifacts_detecte_les_images_nouvelles_une_seule_fois(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    old = scratch / "ancienne.png"
    old.write_bytes(b"x")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    since = time.time() - 1
    (scratch / "script.py").write_text("print(1)", encoding="utf-8")
    (scratch / "figure.svg").write_text("<svg/>", encoding="utf-8")
    (scratch / "Image.PNG").write_bytes(b"\x89PNG")
    seen: set = set()
    got = new_artifacts(str(scratch), since, seen, "abcdef012345")
    assert [a["name"] for a in got] == ["Image.PNG", "figure.svg"]
    assert got[1]["url"] == "/session/abcdef012345/scratch/figure.svg"
    assert got[0]["size"] == 4
    # Déjà annoncées : plus rien, sauf si le fichier change.
    assert new_artifacts(str(scratch), since, seen, "abcdef012345") == []
    time.sleep(0.02)
    (scratch / "figure.svg").write_text("<svg><rect/></svg>", encoding="utf-8")
    os.utime(scratch / "figure.svg", None)
    assert [
        a["name"] for a in new_artifacts(str(scratch), since, seen, "abcdef012345")
    ] == ["figure.svg"]
    # Dossier absent : rien, pas d'exception.
    assert new_artifacts(str(tmp_path / "nope"), since, set(), "abcdef012345") == []


def test_route_scratch_sert_les_images_seulement(web):
    S = web.application.S
    sid = web.get("/sessions").get_json()["active"]
    scratch = S.session_store.root / sid / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "a.png").write_bytes(b"\x89PNG\r\n")
    (scratch / "a.py").write_text("secret", encoding="utf-8")
    r = web.get(f"/session/{sid}/scratch/a.png")
    assert r.status_code == 200 and r.data.startswith(b"\x89PNG")
    assert web.get(f"/session/{sid}/scratch/a.py").status_code == 404
    assert web.get(f"/session/{sid}/scratch/absente.png").status_code == 404
    assert web.get("/session/../scratch/a.png").status_code == 404
    assert web.get(f"/session/{sid}/scratch/../session.json").status_code == 404
