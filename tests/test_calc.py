# Outil calculate : puissance `^`, priorité correcte, jeu de fonctions étendu.
# Le bug d'origine (2026-07-14) : `^` était parsé comme XOR binaire -> « BinOp non
# supporté » sur une simple mensualité de prêt. On veut un calc « top notch ».
from __future__ import annotations

import math

import pytest

from loom.tools.calc import _normalize_expr, calculate


def _val(expr: str) -> float:
    # calculate renvoie "expr = <résultat>[ (~ …)]" -> on isole le résultat brut.
    out = calculate(expr)
    rhs = out.split(" = ", 1)[1]
    return float(rhs.split(" (~", 1)[0])


# ---------- puissance via ^ ----------


def test_caret_puissance_simple():
    assert _val("2^10") == 1024


def test_caret_priorite_comme_etoile_etoile():
    # ^ doit se comporter comme ** (priorité PLUS FORTE que *), pas comme XOR
    # (priorité plus faible) : (2^10)*3 = 3072, et surtout PAS 2^(30).
    assert _val("2^10*3") == 3072


def test_caret_exposant_negatif():
    assert _val("(1+0.039/12)^(-240)") == pytest.approx(1.00325**-240)


def test_mensualite_pret_240_mois():
    # Le cas réel qui a levé le bug : mensualité d'un prêt 370k€ à 3,9% sur 240 mois.
    got = _val("(0.039/12) / (1 - (1 + 0.039/12)^(-240)) * 370000")
    r = 0.039 / 12
    expected = r / (1 - (1 + r) ** -240) * 370000
    assert got == pytest.approx(expected)


def test_etoile_etoile_marche_toujours():
    assert _val("2**10") == 1024


# ---------- jeu de fonctions étendu ----------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("sin(0)", 0.0),
        ("cos(0)", 1.0),
        ("factorial(5)", 120),
        ("log(100, 10)", 2.0),
        ("degrees(pi)", 180.0),
        ("radians(180)", math.pi),
        ("gcd(12, 18)", 6),
        ("lcm(4, 6)", 12),
        ("hypot(3, 4)", 5.0),
        ("comb(5, 2)", 10),
        ("perm(5, 2)", 20),
        ("cbrt(27)", 3.0),
        ("sign(-5)", -1),
        ("atan2(1, 1)", math.pi / 4),
        ("tau", math.tau),
    ],
)
def test_fonctions_etendues(expr, expected):
    assert _val(expr) == pytest.approx(expected)


# ---------- exposants Unicode (e²², 10⁻³) ----------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2²", 4),
        ("2³", 8),
        ("10⁻³", 0.001),
        ("e²²", math.e**22),
        ("2²*3", 12),  # priorité : (2²)*3
    ],
)
def test_exposants_unicode(expr, expected):
    assert _val(expr) == pytest.approx(expected)


# ---------- symboles Unicode qu'un modèle produit (× ÷ − ·) ----------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("6×7", 42),
        ("84÷2", 42),
        ("10−3", 7),  # U+2212 (moins Unicode), pas le tiret ASCII
        ("6·7", 42),  # point milieu
        ("2×10³", 2000),
        ("π", math.pi),
    ],
)
def test_symboles_unicode(expr, expected):
    assert _val(expr) == pytest.approx(expected)


# ---------- _normalize_expr ne corrompt pas les noms de colonnes ----------


def test_normalize_hors_guillemets():
    assert _normalize_expr("2^3") == "2**3"


def test_normalize_exposant_unicode():
    assert _normalize_expr("e²²") == "e**(22)"


def test_normalize_preserve_les_chaines():
    # Un nom de colonne contenant ^ ou ² (mode fichier) ne doit PAS être réécrit.
    assert _normalize_expr('sum("a^b") + 2^3') == 'sum("a^b") + 2**3'


# ---------- garde-fous inchangés ----------


def test_division_par_zero():
    from loom.tools.base import ToolError

    with pytest.raises(ToolError):
        calculate("1/0")


def test_fonction_inconnue():
    from loom.tools.base import ToolError

    with pytest.raises(ToolError):
        calculate("banana(2)")
