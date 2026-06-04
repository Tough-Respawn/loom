# tests/test_lessons.py
from loom.lessons import LessonStore, distill_lesson


def test_add_recent_and_persist(tmp_path):
    path = tmp_path / "lessons.json"
    store = LessonStore(path)
    store.add("Expose les fonctions sur window.")
    store.add("Charge les scripts dans l'ordre des dépendances.")
    # rechargé depuis le disque : persistance
    again = LessonStore(path)
    assert again.recent() == [
        "Expose les fonctions sur window.",
        "Charge les scripts dans l'ordre des dépendances.",
    ]


def test_add_dedups_and_ignores_empty(tmp_path):
    store = LessonStore(tmp_path / "l.json")
    store.add("Une leçon.")
    store.add("une leçon.")  # même chose insensible à la casse
    store.add("   ")  # vide -> ignoré
    assert store.recent() == ["Une leçon."]


def test_recent_caps_to_n(tmp_path):
    store = LessonStore(tmp_path / "l.json")
    for i in range(10):
        store.add(f"leçon {i}")
    assert store.recent(3) == ["leçon 7", "leçon 8", "leçon 9"]


def test_distill_lesson_returns_text():
    class C:
        def complete(self, messages, system_prompt, **kw):
            assert "renderBoard is not a function" in messages[0]["content"]
            return "Expose les fonctions globales sur window avant de les appeler."

    out = distill_lesson(
        C(), "app.js: renderBoard is not a function", "fais un jeu", model="m"
    )
    assert "window" in out


def test_distill_lesson_empty_defects_no_call():
    class C:
        def complete(self, *a, **k):
            raise AssertionError("ne doit pas appeler le modèle sans défaut")

    assert distill_lesson(C(), "   ", "t", model="m") == ""
