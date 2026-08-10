"""recall : le résumeur LLM peut rendre du vide (stop silencieux, vécu session
5f8824b81d7b du 2026-08-10 : « Synthèse mémoire :\\n » sans aucun contenu). Dans ce
cas l'outil doit SE REPLIER sur le rendu brut borné des hits, jamais renvoyer un
en-tête nu ni propager l'exception du résumeur."""

from types import SimpleNamespace

from loom.tools.memory import make_recall


class FakeProvider:
    def __init__(self, hits):
        self._hits = hits

    def recall(self, query, k=5):
        return self._hits[:k]


def _hits(n):
    return [SimpleNamespace(text=f"souvenir {i}", source="test") for i in range(n)]


def test_recall_replie_sur_le_rendu_brut_si_synthese_vide():
    spec = make_recall(
        FakeProvider(_hits(6)),
        summarize=lambda q, h: "  \n",
        threshold=5,
    )
    out = spec.run({"query": "parcours utilisateur"})
    assert "souvenir 0" in out


def test_recall_replie_sur_le_rendu_brut_si_resumeur_leve():
    def boom(q, h):
        raise RuntimeError("api distante morte")

    spec = make_recall(FakeProvider(_hits(6)), summarize=boom, threshold=5)
    out = spec.run({"query": "parcours utilisateur"})
    assert "souvenir 0" in out


def test_recall_garde_la_synthese_quand_elle_a_du_contenu():
    spec = make_recall(
        FakeProvider(_hits(6)),
        summarize=lambda q, h: "Synthèse mémoire :\nleçon condensée.",
        threshold=5,
    )
    out = spec.run({"query": "parcours utilisateur"})
    assert out == "Synthèse mémoire :\nleçon condensée."
