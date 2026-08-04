# Un téléchargement de modèle était 100 % invisible : ni interface, ni journal, alors
# que TOUTE l'information existait déjà sur le disque — octets reçus dans le
# `.incomplete` de Hugging Face, taille cible dans `size_mb` du model.toml. Personne
# ne faisait la division (vécu 2026-08-03 : 23 Go téléchargés en tâche de fond,
# serveur allumé, sans le moindre signal).
from dataclasses import dataclass

from loom.runtime.models_fetch import downloads_in_progress


@dataclass
class _M:
    id: str
    filename: str
    dir: str
    size_mb: int = 0


def _mk(tmp_path, mid, *, recu_mo=0, final=False, size_mb=1000):
    d = tmp_path / mid
    (d / ".cache" / "huggingface" / "download").mkdir(parents=True)
    if recu_mo:
        blob = d / ".cache" / "huggingface" / "download" / "abc.incomplete"
        blob.write_bytes(b"\0" * (recu_mo * 1024 * 1024))
    if final:
        (d / "m.gguf").write_bytes(b"\0")
    return _M(id=mid, filename="m.gguf", dir=str(d), size_mb=size_mb)


def test_transfert_en_cours_est_rapporte_avec_son_pourcentage(tmp_path):
    m = _mk(tmp_path, "ornith-q5", recu_mo=250, size_mb=1000)

    (d,) = downloads_in_progress([m])

    assert d["id"] == "ornith-q5"
    assert d["done_mb"] == 250
    assert d["total_mb"] == 1000
    assert d["pct"] == 25.0


def test_gguf_materialise_ne_compte_plus(tmp_path):
    """Le fichier final existe : le transfert est fini, la ligne doit disparaître."""
    m = _mk(tmp_path, "fini", recu_mo=250, final=True)

    assert downloads_in_progress([m]) == []


def test_sans_incomplete_rien_a_signaler(tmp_path):
    assert downloads_in_progress([_mk(tmp_path, "rien")]) == []


def test_taille_cible_inconnue_donne_pct_none(tmp_path):
    """Mieux vaut afficher les Go bruts qu'un pourcentage inventé."""
    m = _mk(tmp_path, "sans-taille", recu_mo=100, size_mb=0)

    (d,) = downloads_in_progress([m])
    assert d["pct"] is None and d["done_mb"] == 100


def test_le_plus_gros_incomplete_fait_foi(tmp_path):
    """Le mmproj se télécharge dans le même dossier : c'est le GGUF principal
    (le plus gros) qui porte la progression, pas la somme."""
    m = _mk(tmp_path, "multi", recu_mo=300, size_mb=1000)
    petit = (
        tmp_path / "multi" / ".cache" / "huggingface" / "download" / "mmproj.incomplete"
    )
    petit.write_bytes(b"\0" * (40 * 1024 * 1024))

    (d,) = downloads_in_progress([m])
    assert d["done_mb"] == 300


def test_dossier_absent_ou_modele_sans_dir_ne_leve_pas(tmp_path):
    assert downloads_in_progress([_M(id="x", filename="m.gguf", dir="")]) == []
    assert downloads_in_progress([_M(id="y", filename="", dir=str(tmp_path))]) == []
    assert (
        downloads_in_progress(
            [_M(id="z", filename="m.gguf", dir=str(tmp_path / "nope"))]
        )
        == []
    )
