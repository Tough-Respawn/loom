from loom.explore import ExploreResult, explore


def test_explore_reads_targeted_existing_file(tmp_path):
    (tmp_path / "game.js").write_text("function tick(){}\n", encoding="utf-8")
    res = explore("corrige le bug dans game.js", str(tmp_path))
    assert isinstance(res, ExploreResult)
    assert res.files == ["game.js"]
    assert "function tick(){}" in res.summary
    assert "game.js" in res.summary


def test_explore_ignores_nonexistent_paths(tmp_path):
    res = explore("modifie absent.js et autre.css", str(tmp_path))
    assert res.files == []
    assert res.summary == ""


def test_explore_caps_bytes_per_file(tmp_path):
    (tmp_path / "big.js").write_text("X" * 50_000, encoding="utf-8")
    res = explore("regarde big.js", str(tmp_path), max_bytes=1000)
    assert res.files == ["big.js"]
    assert "…[tronqué]" in res.summary
    assert len(res.summary) < 5000


def test_explore_stops_at_budget(tmp_path):
    # 3 fichiers, contexte minuscule -> le garde-fou budget coupe avant de tout lire
    for name in ("a.js", "b.js", "c.js"):
        (tmp_path / name).write_text("Y" * 8000, encoding="utf-8")
    res = explore(
        "lis a.js b.js c.js",
        str(tmp_path),
        context=2000,
        max_files=3,
        max_bytes=8000,
        budget_ratio=0.6,
    )
    assert len(res.files) < 3  # n'a pas pu tout charger


def test_explore_empty_when_no_paths(tmp_path):
    res = explore("crée un jeu de démineur", str(tmp_path))  # greenfield, aucun path
    assert res.files == []
    assert res.summary == ""


def test_explore_respects_max_files(tmp_path):
    for name in ("a.js", "b.js", "c.js", "d.js"):
        (tmp_path / name).write_text("z\n", encoding="utf-8")
    res = explore("a.js b.js c.js d.js", str(tmp_path), max_files=2)
    assert len(res.files) == 2
