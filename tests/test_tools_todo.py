# tests/test_tools_todo.py
import pytest

from loom.tools.base import ToolError
from loom.tools.todo import TodoStore, make_manage_todos


def test_manage_todos_stores_and_renders():
    store = TodoStore()
    tool = make_manage_todos(store)
    out = tool.run(
        {
            "todos": [
                {"content": "localiser le fichier", "status": "done"},
                {"content": "modifier la valeur", "status": "in_progress"},
                {"content": "lancer les tests", "status": "pending"},
            ]
        }
    )
    # rendu lisible avec marqueurs ASCII + compteur d'avancement
    assert "[x] localiser le fichier" in out
    assert "[~] modifier la valeur" in out
    assert "[ ] lancer les tests" in out
    assert "1/3" in out
    # l'état est persisté dans le store (mémoire de travail externe)
    assert [i["content"] for i in store.items] == [
        "localiser le fichier",
        "modifier la valeur",
        "lancer les tests",
    ]


def test_manage_todos_replaces_whole_list():
    store = TodoStore()
    tool = make_manage_todos(store)
    tool.run({"todos": [{"content": "a", "status": "pending"}]})
    tool.run({"todos": [{"content": "b", "status": "done"}]})
    # sémantique de remplacement total (pas d'accumulation)
    assert [i["content"] for i in store.items] == ["b"]


def test_manage_todos_rejects_unknown_status():
    store = TodoStore()
    with pytest.raises(ToolError, match="statut"):
        make_manage_todos(store).run({"todos": [{"content": "x", "status": "bloque"}]})


def test_manage_todos_rejects_empty_content():
    store = TodoStore()
    with pytest.raises(ToolError, match="content"):
        make_manage_todos(store).run(
            {"todos": [{"content": "  ", "status": "pending"}]}
        )


def test_manage_todos_requires_list():
    store = TodoStore()
    with pytest.raises(ToolError, match="liste"):
        make_manage_todos(store).run({"todos": "pas une liste"})


def test_manage_todos_status_defaults_to_pending():
    store = TodoStore()
    out = make_manage_todos(store).run({"todos": [{"content": "sans statut"}]})
    assert "[ ] sans statut" in out
    assert store.items[0]["status"] == "pending"


def test_manage_todos_caps_list_length():
    store = TodoStore()
    too_many = [{"content": f"t{i}", "status": "pending"} for i in range(40)]
    with pytest.raises(ToolError, match="trop"):
        make_manage_todos(store).run({"todos": too_many})
