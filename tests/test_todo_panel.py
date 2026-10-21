from agent.tools.todo import TodoItem, TodoTool
from agent.tui.app import TodoPanel


class _Session:
    id = "todo-panel-test"


class _App:
    def __init__(self, session):
        self._session_obj = session

    def _session(self):
        return self._session_obj


def _panel(session):
    return TodoPanel(_App(session))


def _store(session):
    sid = TodoTool._session_key(session)
    return TodoTool._strong_keys.setdefault(sid, {})


def test_panel_reads_shared_todo_store():
    session = _Session()
    TodoTool.forget_session(session)
    store = _store(session)
    store["t1"] = TodoItem(id="t1", title="first", status="in_progress")
    store["t2"] = TodoItem(id="t2", title="second", status="pending")

    panel = _panel(session)
    panel.refresh_from_store(force=True)

    assert [e["content"] for e in panel._entries] == ["first", "second"]
    assert [e["status"] for e in panel._entries] == ["in_progress", "pending"]
    assert panel._entries_from_store is True
    TodoTool.forget_session(session)


def test_panel_refreshes_when_store_changes():
    session = _Session()
    TodoTool.forget_session(session)
    store = _store(session)
    store["t1"] = TodoItem(id="t1", title="first", status="in_progress")

    panel = _panel(session)
    panel.refresh_from_store(force=True)
    unchanged = panel._store_sig

    panel.refresh_from_store()
    assert panel._store_sig == unchanged

    store["t1"].status = "done"
    store["t1"].updated_at += 1
    panel.refresh_from_store()

    assert panel._entries[0]["status"] == "completed"
    TodoTool.forget_session(session)


def test_pushed_checklist_survives_until_store_changes():
    session = _Session()
    TodoTool.forget_session(session)
    store = _store(session)
    store["t1"] = TodoItem(id="t1", title="stored", status="pending")

    panel = _panel(session)
    panel.refresh_from_store(force=True)
    panel.set_entries([{"content": "model checklist", "status": "pending"}])

    panel.refresh_from_store()
    assert panel._entries[0]["content"] == "model checklist"

    store["t1"].status = "done"
    store["t1"].updated_at += 1
    panel.refresh_from_store()

    assert panel._entries[0]["content"] == "stored"
    assert panel._entries[0]["status"] == "completed"
    TodoTool.forget_session(session)
