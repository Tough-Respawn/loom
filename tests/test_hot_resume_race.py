# Reprise à chaud vs démarrage du serveur. Vécu le 2026-08-03 : Loom demandait la
# restauration du slot AVANT que llama-server n'écoute (WinError 10061). L'essai
# unique était consommé quand même, donc la reprise était perdue pour toute la
# période froide — alors que le serveur arrivait quelques secondes plus tard.
# Un refus de connexion n'est pas un échec du slot : la tentative n'a pas eu lieu.
import pytest

from loom.agent.client import LoomClient


class _Client(LoomClient):
    """Client nu : on court-circuite tout le réseau pour ne tester QUE la logique
    de décompte de l'essai unique."""

    def __init__(self, echec):
        self.hot_resume_enabled = True
        self.restore_safe = True
        self.hybrid_models = set()
        self._routes = {}
        self._slot_warm = set()
        self._slot_broken = set()
        self._slot_unreachable = False
        self._echec = echec  # "injoignable" | "refus_serveur" | None
        self.appels = 0

    def is_remote(self, model):
        return False

    def _slots_meta_path(self, name):
        return self._meta

    def restore_slot(self, model, name, force=False):
        self.appels += 1
        self._slot_unreachable = self._echec == "injoignable"
        return self._echec is None


@pytest.fixture
def meta(tmp_path):
    p = tmp_path / "turnend.kv.json"
    p.write_text('{"model": "ornith", "session": "s1"}', encoding="utf-8")
    return p


def test_serveur_injoignable_ne_consomme_pas_lessai(meta):
    c = _Client("injoignable")
    c._meta = meta

    assert c.try_hot_resume("ornith", "s1") is False
    # le slot doit être REDEVENU froid : le tour suivant doit pouvoir retenter
    assert "ornith" not in c._slot_warm

    c._echec = None  # le serveur écoute enfin
    assert c.try_hot_resume("ornith", "s1") is True
    assert c.appels == 2, "la seconde tentative doit avoir lieu"


def test_vrai_echec_consomme_lessai(meta):
    """Un refus du serveur (il répond, mais le restore échoue) reste un essai
    consommé : pas de tempête de retries à chaque tour."""
    c = _Client("refus_serveur")
    c._meta = meta

    assert c.try_hot_resume("ornith", "s1") is False
    assert "ornith" in c._slot_warm

    assert c.try_hot_resume("ornith", "s1") is False
    assert c.appels == 1, "aucune seconde tentative après un vrai échec"


def test_succes_marque_le_slot_chaud(meta):
    c = _Client(None)
    c._meta = meta

    assert c.try_hot_resume("ornith", "s1") is True
    assert "ornith" in c._slot_warm
    assert c.try_hot_resume("ornith", "s1") is False  # déjà chaud : rien à faire
    assert c.appels == 1
