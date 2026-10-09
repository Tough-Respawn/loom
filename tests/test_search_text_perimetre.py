"""search_text : un dossier passé en `glob` est un PÉRIMÈTRE, et un négatif dit combien de
fichiers ont été parcourus. Session c81fcc4bd207 (2026-10-09) : `glob` = chemin d'un
dossier sans joker -> « aucune correspondance » alors qu'AUCUN fichier n'avait été lu
(le motif désignait un fichier nommé comme le dossier) ; le modèle en a conclu « aucun
backend NPU dans l'arbre, c'est un fait » (459 occurrences réelles)."""

from loom.tools import search
from loom.tools.search import make_search_text


def _arbre(tmp_path):
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "a.py").write_text("import npu\n", encoding="utf-8")
    (ws / "src" / "b.py").write_text("rien ici\n", encoding="utf-8")
    (ws / "other").mkdir()
    (ws / "other" / "c.py").write_text("npu aussi\n", encoding="utf-8")
    ext = tmp_path / "ext"
    (ext / "ggml").mkdir(parents=True)
    (ext / "ggml" / "cann.h").write_text("// Ascend NPU backend\n", encoding="utf-8")
    (ext / ".git").mkdir()
    (ext / ".git" / "packed").write_text("NPU dans git\n", encoding="utf-8")
    return ws, ext


def _outil(ws, monkeypatch, **kw):
    monkeypatch.setattr(
        search, "_rg_path", lambda: None
    )  # repli Python, comme sur la VM
    return make_search_text(str(ws), **kw)


def test_dossier_absolu_en_glob_est_un_perimetre(tmp_path, monkeypatch):
    ws, ext = _arbre(tmp_path)
    out = _outil(ws, monkeypatch).run({"pattern": "NPU", "glob": str(ext)})
    assert "cann.h:1:" in out
    assert ".git" not in out  # dossiers ignorés jamais parcourus


def test_dossier_relatif_en_glob_est_un_perimetre(tmp_path, monkeypatch):
    ws, _ = _arbre(tmp_path)
    out = _outil(ws, monkeypatch).run({"pattern": "npu", "glob": "src"})
    assert "src/a.py:1:" in out
    assert "other/c.py" not in out


def test_perimetre_sans_fichier_est_dit_explicitement(tmp_path, monkeypatch):
    ws, _ = _arbre(tmp_path)
    outil = _outil(ws, monkeypatch)
    absent = outil.run({"pattern": "npu", "glob": str(tmp_path / "inexistant")})
    assert "aucun fichier" in absent and "aucune correspondance" not in absent
    vide = outil.run({"pattern": "npu", "glob": "**/*.rs"})
    assert "aucun fichier" in vide and "aucune correspondance" not in vide


def test_zero_correspondance_dit_combien_de_fichiers_ont_ete_lus(tmp_path, monkeypatch):
    ws, _ = _arbre(tmp_path)
    out = _outil(ws, monkeypatch).run({"pattern": "ZZZ", "glob": "src"})
    assert out == "aucune correspondance pour : ZZZ (2 fichiers parcourus)"


def test_temps_depasse_resultats_partiels(tmp_path, monkeypatch):
    ws, _ = _arbre(tmp_path)
    out = _outil(ws, monkeypatch, time_budget_s=-1).run(
        {"pattern": "npu", "glob": "src"}
    )
    assert "PARTIELS" in out
