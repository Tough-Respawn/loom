"""Non-régression du harnais d'évaluation hors ligne."""

from evals.run_eval import self_test


def test_eval_self_test_reste_executable_apres_refactoring():
    """Le point d'entrée documenté doit fonctionner sans serveur ni modèle."""
    assert self_test() is True
