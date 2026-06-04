# tests/test_session.py
from loom.session import SessionStore


def test_create_persists_and_lists(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    s = store.create(workspace="C:/proj", title="Calc")
    assert s.id
    assert s.workspace == "C:/proj"
    metas = store.list()
    assert any(m.id == s.id and m.title == "Calc" for m in metas)


def test_create_seeds_default_tools(tmp_path):
    # Sans seeding, la session part avec active_tools=[] -> chat sans `tools=` -> le
    # modèle crache ses appels d'outil en texte. On vérifie que les outils sont armés.
    store = SessionStore(
        tmp_path, default_system_prompt="sys", default_tools=["read_file"]
    )
    s = store.create(workspace=".")
    assert s.conversation.active_tools == ["read_file"]
    assert store.load(s.id).conversation.active_tools == ["read_file"]


def test_create_without_default_tools_stays_empty(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    assert store.create(workspace=".").conversation.active_tools == []


def test_conversation_roundtrip_through_session(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    s = store.create(workspace=".")
    s.conversation.add("user", "bonjour")
    store.save(s)
    loaded = store.load(s.id)
    assert loaded.conversation.messages == [{"role": "user", "content": "bonjour"}]
    assert loaded.conversation.system_prompt == "sys"


def test_active_session_tracked(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    assert store.active() is None
    s = store.create(workspace=".")
    act = store.active()
    assert act is not None and act.id == s.id  # create() focalise la session


def test_set_active_switches(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    a = store.create(workspace=".")
    b = store.create(workspace=".")
    assert store.active().id == b.id  # la dernière créée est active
    store.set_active(a.id)
    assert store.active().id == a.id


def test_delete_removes_session(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    s = store.create(workspace=".")
    store.delete(s.id)
    assert store.load(s.id) is None
    assert all(m.id != s.id for m in store.list())


def test_load_unknown_returns_none(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    assert store.load("does-not-exist") is None


def test_save_atomic_no_tmp_residue(tmp_path):
    store = SessionStore(tmp_path, default_system_prompt="sys")
    s = store.create(workspace=".")
    s.conversation.add("user", "x")
    store.save(s)
    assert list((tmp_path / s.id).glob("*.tmp")) == []
