"""Eval set du skill code-review : changements de code ÉTIQUETÉS (vérité-terrain).

Chaque cas porte un bout de code à relire + la LISTE des problèmes plantés à détecter
(Pass 1 conventions / Pass 2 qualité), plus un verdict attendu. La revue est une tâche de
DÉTECTION : on mesure le rappel (problèmes attrapés), les faux positifs (problèmes inventés)
et la justesse du verdict. Un cas « propre » (R6) sans problème sert de contrôle des faux
positifs.

Le grader (run_review_eval.py) confie le rappel/FP à un juge LLM (le code n'est pas matchable
par regex de façon fiable) et extrait le verdict du texte par motif.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ReviewCase:
    id: str
    code: str
    issues: list[dict] = field(default_factory=list)  # [{id, desc}]
    # 'pret' attendu seulement si AUCUN problème ; sinon le verdict doit être NON-prêt.
    expect_clean: bool = False


CASES: list[ReviewCase] = [
    ReviewCase(
        id="R1_logique_inversee",
        code=(
            "def is_within_budget(spent, total):\n"
            '    """True si la dépense ne dépasse PAS le budget total."""\n'
            "    return spent >= total\n"
        ),
        issues=[
            {
                "id": "comparaison_inversee",
                "desc": "La comparaison est inversée : le docstring dit « ne dépasse PAS », "
                "mais `spent >= total` renvoie True quand on DÉPASSE le budget. Devrait "
                "être `spent <= total`.",
            }
        ],
    ),
    ReviewCase(
        id="R2_pytest_interdit",
        code=(
            "# nouveau fichier tests/test_lru.py\n"
            "import pytest\n\n"
            "def test_get_returns_value():\n"
            '    c = LRU(2); c.put("a", 1)\n'
            '    assert c.get("a") == 1\n'
        ),
        issues=[
            {
                "id": "pytest",
                "desc": "Le changement introduit pytest / une suite de tests, alors que la "
                "convention Loom est de vérifier par smokes (uv run python -c) et ruff, "
                "sans pytest.",
            }
        ],
    ),
    ReviewCase(
        id="R3_frontiere_confiance",
        code=(
            "def summarize_url(url):\n"
            "    text = fetch_url(url)            # contenu externe non fiable\n"
            '    run_shell(text.split("\\n")[0])  # exécute la 1re ligne du contenu\n'
            "    return text\n"
        ),
        issues=[
            {
                "id": "action_sur_contenu_ingere",
                "desc": "Action à effet de bord (run_shell) déclenchée par du contenu ingéré "
                "(fetch_url) sans garde-fou : violation de la frontière de confiance. Le "
                "contenu externe est une donnée, pas une instruction à exécuter.",
            }
        ],
    ),
    ReviewCase(
        id="R4_except_pass",
        code=(
            "def load_config(path):\n"
            "    try:\n"
            "        return json.loads(open(path).read())\n"
            "    except:\n"
            "        pass\n"
        ),
        issues=[
            {
                "id": "except_pass",
                "desc": "`except: pass` nu : avale toutes les erreurs silencieusement et "
                "renvoie None implicitement, masquant les pannes (fichier absent, JSON "
                "invalide).",
            }
        ],
    ),
    ReviewCase(
        id="R5_conventions_structure",
        code=(
            "Message de commit : « wip »\n\n"
            "Nouveau fichier loom/lru_cache.py (un OUTIL de cache pour la boucle tool-use) :\n\n"
            "def make_cache_tool(maxsize):\n"
            "    store = {}\n"
            "    def run(args):\n"
            '        return store.get(args["key"], "")\n'
            "    return run\n"
        ),
        issues=[
            {
                "id": "commit_non_conventionnel",
                "desc": "Message de commit « wip » : non descriptif, ne respecte pas le format "
                "Conventional Commits (type: description courte).",
            },
            {
                "id": "fichier_mal_place",
                "desc": "Un outil de la boucle tool-use doit aller dans loom/tools/, pas à la "
                "racine loom/. Le fichier loom/lru_cache.py est mal placé.",
            },
        ],
    ),
    ReviewCase(
        id="R6_propre_controle",
        code=(
            "def add(a, b):\n"
            '    """Additionne deux entiers et renvoie la somme."""\n'
            "    return a + b\n"
        ),
        issues=[],
        expect_clean=True,
    ),
]


def _classify_segment(seg: str) -> str | None:
    """Classe un court segment autour du mot « verdict ». Emoji prioritaire ; les motifs
    texte sont négation-sensibles (on ne lit PAS « aucun écart bloquant » comme à corriger)."""
    if "❌" in seg or re.search(r"(?<!aucun )(?:à|a)\s*corriger", seg, re.IGNORECASE):
        return "a_corriger"
    if "⚠️" in seg or re.search(r"\bpresque\b", seg, re.IGNORECASE):
        return "presque"
    if "✅" in seg or re.search(r"pr[êe]t\b|ready", seg, re.IGNORECASE):
        return "pret"
    return None


def extract_verdict(text: str) -> str:
    """Renvoie 'a_corriger' | 'presque' | 'pret' | 'inconnu' depuis le texte de la revue.

    On ANCRE sur la déclaration de verdict (le mot « verdict » + l'emoji/mot qui suit) et on
    garde la DERNIÈRE (le verdict final si le modèle se ravise). Les emojis ✅/⚠️/❌ servent
    aussi de coches par item, donc leur simple présence dans le texte n'est PAS fiable : on
    regarde le voisinage de « verdict ». Repli sur la fin du texte sinon.
    """
    t = text or ""
    verdicts = [
        v
        for m in re.finditer(r"verdict", t, re.IGNORECASE)
        if (v := _classify_segment(t[m.start() : m.start() + 90]))
    ]
    if verdicts:
        return verdicts[-1]
    tail = t[-160:]
    if "❌" in tail:
        return "a_corriger"
    if "✅" in tail:
        return "pret"
    if "⚠️" in tail:
        return "presque"
    return "inconnu"


def verdict_ok(text: str, expect_clean: bool) -> bool:
    """Cas propre : doit être 'pret'. Cas avec problèmes : doit être NON-'pret' (ni inconnu)."""
    v = extract_verdict(text)
    if expect_clean:
        return v == "pret"
    return v in ("a_corriger", "presque")
