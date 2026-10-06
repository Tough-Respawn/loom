"""find_files n'entre jamais dans les dossiers ignorés et rend la main à temps : Path.glob
parcourait tout avant de filtrer (`C:/Users/x/**/*.gguf` : 136 s vécus le 2026-10-06)."""

import os

from loom.tools import search
from loom.tools.search import make_find_files


def _tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.gguf").write_text("x")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "b.gguf").write_text("x")
    (tmp_path / "deep" / "x" / "y").mkdir(parents=True)
    (tmp_path / "deep" / "x" / "y" / "c.gguf").write_text("x")


def test_dossiers_ignores_jamais_parcourus(tmp_path, monkeypatch):
    _tree(tmp_path)
    visited = []
    real_walk = os.walk

    def spy(top, *a, **k):
        for dirpath, dirnames, filenames in real_walk(top, *a, **k):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(search.os, "walk", spy)
    out = make_find_files(str(tmp_path)).run({"pattern": "**/*.gguf"})
    assert out.splitlines() == ["deep/x/y/c.gguf", "src/a.gguf"]
    assert not any("node_modules" in v for v in visited)


def test_motif_sans_double_etoile_reste_a_sa_profondeur(tmp_path):
    _tree(tmp_path)
    out = make_find_files(str(tmp_path)).run({"pattern": "src/*.gguf"})
    assert out == "src/a.gguf"


def test_temps_depasse_resultats_partiels(tmp_path):
    _tree(tmp_path)
    out = make_find_files(str(tmp_path), time_budget_s=-1).run({"pattern": "**/*.gguf"})
    assert "résultats PARTIELS" in out
