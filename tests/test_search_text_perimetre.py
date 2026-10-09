"""search_text : un dossier passé en `glob` est un PÉRIMÈTRE, et un négatif dit combien de
fichiers ont été parcourus. Session c81fcc4bd207 (2026-10-09) : `glob` = chemin d'un
dossier sans joker -> « aucune correspondance » alors qu'AUCUN fichier n'avait été lu
(le motif désignait un fichier nommé comme le dossier) ; le modèle en a conclu « aucun
backend NPU dans l'arbre, c'est un fait » (459 occurrences réelles)."""

import os
import subprocess

import pytest

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


def test_sans_glob_les_dossiers_ignores_ne_sont_pas_parcourus(tmp_path, monkeypatch):
    # Sans `glob`, le repli passait par rglob : dossiers ignorés filtrés APRÈS parcours.
    ws, _ = _arbre(tmp_path)
    (ws / "node_modules" / "pkg").mkdir(parents=True)
    (ws / "node_modules" / "pkg" / "n.py").write_text("npu\n", encoding="utf-8")
    visited = []
    real_walk = os.walk

    def spy(top, *a, **k):
        for dirpath, dirnames, filenames in real_walk(top, *a, **k):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(search.os, "walk", spy)
    out = _outil(ws, monkeypatch).run({"pattern": "npu"})
    assert "src/a.py:1:" in out and "node_modules" not in out
    assert visited and not any("node_modules" in v for v in visited)


def test_sans_glob_le_budget_de_temps_s_applique(tmp_path, monkeypatch):
    ws, _ = _arbre(tmp_path)
    out = _outil(ws, monkeypatch, time_budget_s=-1).run({"pattern": "npu"})
    assert "PARTIELS" in out


# ---- chemin ripgrep : même distinction des négatifs -----------------------------------


class _FauxRg:
    """Émule les deux invocations de rg : la recherche (rien trouvé, code 1) et
    `--files` (liste des fichiers du périmètre, une par ligne)."""

    def __init__(self, files):
        self.files = list(files)
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        if "--files" in cmd:
            stdout = "".join(f"{f}\n" for f in self.files)
            return subprocess.CompletedProcess(cmd, 0 if self.files else 1, stdout, "")
        return subprocess.CompletedProcess(cmd, 1, "", "")


def _outil_rg(ws, monkeypatch, faux):
    monkeypatch.setattr(search, "_rg_path", lambda: "rg")
    monkeypatch.setattr(search.subprocess, "run", faux)
    return make_search_text(str(ws))


def test_rg_zero_correspondance_dit_combien_de_fichiers_ont_ete_lus(
    tmp_path, monkeypatch
):
    ws, _ = _arbre(tmp_path)
    faux = _FauxRg(["a.py", "b.py"])
    out = _outil_rg(ws, monkeypatch, faux).run({"pattern": "ZZZ", "glob": "src"})
    assert out == "aucune correspondance pour : ZZZ (2 fichiers parcourus)"
    recherche = next(c for c in faux.cmds if "--files" not in c)
    decompte = next(c for c in faux.cmds if "--files" in c)
    # Même périmètre (-g) pour la recherche et le décompte.
    assert recherche[recherche.index("-g") + 1] == decompte[decompte.index("-g") + 1]


def test_rg_perimetre_vide_est_dit_explicitement(tmp_path, monkeypatch):
    ws, _ = _arbre(tmp_path)
    out = _outil_rg(ws, monkeypatch, _FauxRg([])).run(
        {"pattern": "npu", "glob": "**/*.rs"}
    )
    assert "aucun fichier parcouru" in out and "aucune correspondance" not in out


_RG_REEL = search._rg_path()


@pytest.mark.skipif(_RG_REEL is None, reason="ripgrep absent de cette machine")
def test_rg_reel_distingue_les_deux_negatifs(tmp_path):
    ws, _ = _arbre(tmp_path)
    outil = make_search_text(str(ws))
    assert outil.run({"pattern": "ZZZ", "glob": "src"}) == (
        "aucune correspondance pour : ZZZ (2 fichiers parcourus)"
    )
    vide = outil.run({"pattern": "npu", "glob": "**/*.rs"})
    assert "aucun fichier parcouru" in vide and "aucune correspondance" not in vide
    assert "src/a.py:1:" in outil.run({"pattern": "npu", "glob": "src"})


def test_temps_depasse_resultats_partiels(tmp_path, monkeypatch):
    ws, _ = _arbre(tmp_path)
    out = _outil(ws, monkeypatch, time_budget_s=-1).run(
        {"pattern": "npu", "glob": "src"}
    )
    assert "PARTIELS" in out
