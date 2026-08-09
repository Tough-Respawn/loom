# Découverte des modèles LOCAUX : un dossier cassé ne doit JAMAIS empêcher Loom de
# démarrer. Vécu le 2026-08-03 : /add-model écrit `model.toml` AVANT la fin du
# téléchargement (n_layers se lit dans le GGUF), l'installation a été interrompue,
# et le `KeyError: 'n_layers'` remontait jusqu'au chargement de la config — donc
# plus d'interface du tout, y compris pour réparer le modèle fautif.
from loom.config import _discover_models


def _mk_model(root, mid, body):
    d = root / mid
    d.mkdir(parents=True)
    d.joinpath("model.toml").write_text(body, encoding="utf-8")
    return d


_SAIN = 'repo = "r/m"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 100\n'


def test_dossier_incomplet_est_ignore_pas_fatal(tmp_path, capsys):
    _mk_model(tmp_path, "bon", _SAIN)
    # exactement le cas vécu : tout sauf n_layers
    _mk_model(
        tmp_path, "en-cours-de-dl", 'repo = "r/m"\nfilename = "m.gguf"\nsize_mb = 100\n'
    )

    models = _discover_models(tmp_path)

    assert [m.id for m in models] == ["bon"], "le modèle sain doit rester découvert"
    err = capsys.readouterr().err
    assert "en-cours-de-dl" in err, "l'utilisateur doit savoir QUEL modèle est ignoré"
    assert "n_layers" in err, "et POURQUOI il est ignoré"


def test_toml_syntaxiquement_invalide_est_ignore(tmp_path):
    _mk_model(tmp_path, "bon", _SAIN)
    _mk_model(tmp_path, "casse", "ceci n'est pas du toml [[[")

    assert [m.id for m in _discover_models(tmp_path)] == ["bon"]


def test_tous_casses_renvoie_liste_vide_sans_lever(tmp_path):
    """Aucun modèle exploitable : Loom doit démarrer quand même (l'utilisateur
    peut alors basculer sur un modèle distant ou réparer depuis l'interface)."""
    _mk_model(tmp_path, "a", 'repo = "r/m"\n')
    _mk_model(tmp_path, "b", "[[[")

    assert _discover_models(tmp_path) == []


def test_modele_valide_toujours_lu_normalement(tmp_path):
    _mk_model(tmp_path, "ornith-q5", _SAIN + "cpu_moe = true\n")

    (m,) = _discover_models(tmp_path)
    assert (m.id, m.n_layers, m.cpu_moe) == ("ornith-q5", 40, True)
    assert m.dir == str(tmp_path / "ornith-q5")
