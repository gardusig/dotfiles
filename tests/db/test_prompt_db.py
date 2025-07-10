import pytest
from unittest.mock import patch, MagicMock

import builtins

# Patch external dependencies at the top-level of the module under test
patch_target_create_session_file = "crowler.db.prompt_db.create_session_file"
patch_target_HistoryDB = "crowler.db.prompt_db.HistoryDB"


# Import the module under test after patching dependencies
@pytest.fixture(autouse=True)
def patch_external_deps(monkeypatch):
    # Patch create_session_file to return a dummy file path
    monkeypatch.setattr(
        "crowler.db.prompt_db.create_session_file", lambda name: f"/tmp/{name}.db"
    )
    # Patch HistoryDB with a MagicMock
    mock_history_db_cls = MagicMock()
    monkeypatch.setattr("crowler.db.prompt_db.HistoryDB", mock_history_db_cls)
    yield


@pytest.fixture
def prompt_store(monkeypatch):
    # Patch create_session_file and HistoryDB for PromptHistoryStore
    from crowler.db import prompt_db

    # Create a fresh PromptHistoryStore for each test
    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    return store


@pytest.fixture
def mock_db(monkeypatch):
    # Patch HistoryDB instance for fine-grained control
    from crowler.db import prompt_db

    mock_db = MagicMock()
    # Patch the _db attribute of PromptHistoryStore
    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = mock_db
    return store, mock_db


@pytest.mark.parametrize(
    "input_lines,expected",
    [
        (["prompt1", "prompt2"], ["prompt1", "prompt2"]),
        ([], []),
        (["  prompt1  ", "   ", "prompt2"], ["prompt1", "prompt2"]),
    ],
)
def test_snap_returns_clean_copy(monkeypatch, input_lines, expected):
    from crowler.db import prompt_db

    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = MagicMock()
    store._db.latest.return_value = input_lines
    result = store._snap()
    assert result == input_lines
    assert result is not input_lines  # Should be a copy


def test_clear_calls_db_and_prints(monkeypatch, capsys):
    from crowler.db import prompt_db

    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = MagicMock()
    store.clear()
    store._db.clear.assert_called_once()
    out = capsys.readouterr().out
    assert "cleared" in out


@pytest.mark.parametrize(
    "prompt,existing,expected_push,expected_msg",
    [
        ("new prompt", [], ["new prompt"], "Added prompt"),
        ("", [], None, "Empty prompt"),
        ("   ", ["something"], None, "Empty prompt"),
        ("dupe", ["dupe"], None, "already present"),
    ],
)
def test_append_prompt(
    monkeypatch, capsys, prompt, existing, expected_push, expected_msg
):
    from crowler.db import prompt_db

    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = MagicMock()
    store._db.latest.return_value = existing
    store.append(prompt)
    out = capsys.readouterr().out
    if expected_push is not None:
        store._db.push.assert_called_once_with(expected_push)
        assert expected_msg in out
    else:
        store._db.push.assert_not_called()
        assert expected_msg in out


@pytest.mark.parametrize(
    "prompt,existing,expected_push,expected_msg",
    [
        ("to_remove", ["to_remove", "other"], ["other"], "Removed prompt"),
        ("", ["to_remove"], None, "Empty prompt"),
        ("not_present", ["a", "b"], None, "not tracked"),
    ],
)
def test_remove_prompt(
    monkeypatch, capsys, prompt, existing, expected_push, expected_msg
):
    from crowler.db import prompt_db

    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = MagicMock()
    store._db.latest.return_value = existing.copy()
    store.remove(prompt)
    out = capsys.readouterr().out
    if expected_push is not None:
        store._db.push.assert_called_once_with(expected_push)
        assert expected_msg in out
    else:
        store._db.push.assert_not_called()
        assert expected_msg in out


@pytest.mark.parametrize(
    "undo_result,expected_msg",
    [
        (True, "Reverted"),
        (False, "Nothing to undo"),
    ],
)
def test_undo_prompts(monkeypatch, capsys, undo_result, expected_msg):
    from crowler.db import prompt_db

    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = MagicMock()
    store._db.undo.return_value = undo_result
    store.undo()
    out = capsys.readouterr().out
    assert expected_msg in out


def test_summary_delegates(monkeypatch):
    from crowler.db import prompt_db

    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = MagicMock()
    store._db.summary.return_value = "summary!"
    assert store.summary() == "summary!"


def test_latest_delegates(monkeypatch):
    from crowler.db import prompt_db

    store = prompt_db.PromptHistoryStore("test_history", "Test Prompts")
    store._db = MagicMock()
    store._db.latest.return_value = ["a", "b"]
    assert store.latest() == ["a", "b"]


def test_public_api_functions(monkeypatch):
    from crowler.db import prompt_db

    # Patch _prompt_store methods
    monkeypatch.setattr(prompt_db._prompt_store, "clear", MagicMock())
    monkeypatch.setattr(prompt_db._prompt_store, "append", MagicMock())
    monkeypatch.setattr(prompt_db._prompt_store, "remove", MagicMock())
    monkeypatch.setattr(prompt_db._prompt_store, "undo", MagicMock())
    monkeypatch.setattr(
        prompt_db._prompt_store, "summary", MagicMock(return_value="sum")
    )
    monkeypatch.setattr(
        prompt_db._prompt_store, "latest", MagicMock(return_value=["x"])
    )

    prompt_db.clear_prompts()
    prompt_db._prompt_store.clear.assert_called_once()

    prompt_db.append_prompt("foo")
    prompt_db._prompt_store.append.assert_called_once_with("foo")

    prompt_db.remove_prompt("bar")
    prompt_db._prompt_store.remove.assert_called_once_with("bar")

    prompt_db.undo_prompts()
    prompt_db._prompt_store.undo.assert_called_once()

    assert prompt_db.summary_prompts() == "sum"
    assert prompt_db.get_latest_prompts() == ["x"]
