# Confinement strict des IDs de session dans SessionStore (revue sécu) : un id
# invalide (vide, traversal, chemin absolu) ne doit JAMAIS faire sortir une opération
# FS de root — ni supprimer (shutil.rmtree), ni lire/charger un session.json ailleurs.
from __future__ import annotations

import pytest

from loom.agent.session import SessionStore


@pytest.fixture()
def store(tmp_path):
    return SessionStore(tmp_path / "sessions", default_system_prompt="p")


def test_delete_id_vide_ne_supprime_pas_la_racine(store):
    s = store.create()
    assert store.root.exists()
    store.delete("")  # id vide -> root / "" == root
    assert store.root.exists(), "la racine des sessions a été supprimée par un id vide"
    assert store.load(s.id) is not None, "la session légitime a disparu"


def test_delete_traversal_ne_supprime_pas_un_dossier_voisin(store, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x", encoding="utf-8")
    store.create()
    store.delete("../victim")
    assert victim.exists(), "un id de traversal a supprimé un dossier voisin"


def test_delete_chemin_absolu_ne_sort_pas_de_la_racine(store, tmp_path):
    victim = tmp_path / "abs_victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x", encoding="utf-8")
    store.create()
    store.delete(str(victim))  # id = chemin absolu -> pathlib remplace la base
    assert victim.exists(), "un id absolu a supprimé un dossier hors racine"


def test_load_traversal_ne_lit_pas_hors_racine(store, tmp_path):
    s = store.create()
    good = (store.root / s.id / "session.json").read_text(encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "session.json").write_text(good, encoding="utf-8")
    assert store.load("../outside") is None, (
        "un id de traversal a chargé un fichier hors racine"
    )


def test_delete_id_valide_fonctionne_toujours(store):
    s = store.create()
    assert store.load(s.id) is not None
    store.delete(s.id)
    assert store.load(s.id) is None


def test_load_ignore_un_id_falsifie_dans_le_fichier(store):
    # Réserve de confinement : un session.json stocké sous un dossier VALIDE peut
    # contenir "id": "../../victim". load doit faire foi du NOM DE DOSSIER (sid), pas du
    # champ id du fichier — sinon un save(sess)/session_dir(sess.id) ressortirait de root.
    import json

    s = store.create()
    f = store.root / s.id / "session.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    data["id"] = "../../victim"
    f.write_text(json.dumps(data), encoding="utf-8")
    loaded = store.load(s.id)
    assert loaded is not None
    assert loaded.id == s.id, "load a fait confiance à l'id falsifié du fichier"


def test_save_refuse_un_id_invalide(store):
    # save() ne doit jamais écrire hors racine : un id invalide (traversal) -> refus.
    s = store.create()
    s.id = "../evil"
    with pytest.raises(ValueError):
        store.save(s)


def test_import_json_valide_mais_mal_type_leve_valueerror(store):
    # Une archive JSON syntaxiquement valide mais mal typée (conversation: null) ne
    # doit pas remonter en AttributeError/TypeError (500) : rejet propre en ValueError
    # (-> 400 côté route).
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # id présent (sinon KeyError, déjà converti) mais conversation null ->
        # Conversation.from_dict(None) lève AttributeError, non attrapée aujourd'hui.
        z.writestr("session.json", '{"id": "abcabcabcabc", "conversation": null}')
    with pytest.raises(ValueError):
        store.import_zip(buf.getvalue())
