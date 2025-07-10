import pytest
from unittest.mock import patch, MagicMock

import crowler.db.shared_file_db as shared_file_db


@pytest.fixture(autouse=True)
def patch_file_history_store(monkeypatch):
    # Patch FileHistoryStore in the shared_file_db module
    mock_store = MagicMock()
    monkeypatch.setattr(shared_file_db, "_shared_store", mock_store)
    return mock_store


def test_clear_shared_files_calls_clear(patch_file_history_store):
    shared_file_db.clear_shared_files()
    patch_file_history_store.clear.assert_called_once_with()


@pytest.mark.parametrize("path", ["file1.txt", "another/file2.txt"])
def test_append_shared_file_calls_append(patch_file_history_store, path):
    shared_file_db.append_shared_file(path)
    patch_file_history_store.append.assert_called_once_with(path)


@pytest.mark.parametrize("path", ["file1.txt", "another/file2.txt"])
def test_remove_shared_file_calls_remove(patch_file_history_store, path):
    shared_file_db.remove_shared_file(path)
    patch_file_history_store.remove.assert_called_once_with(path)


def test_undo_shared_files_calls_undo(patch_file_history_store):
    shared_file_db.undo_shared_files()
    patch_file_history_store.undo.assert_called_once_with()


def test_summary_shared_files_returns_summary(patch_file_history_store):
    patch_file_history_store.summary.return_value = "summary string"
    result = shared_file_db.summary_shared_files()
    patch_file_history_store.summary.assert_called_once_with()
    assert result == "summary string"


def test_get_shared_files_returns_latest_set(patch_file_history_store):
    patch_file_history_store.latest_set.return_value = {"a.txt", "b.txt"}
    result = shared_file_db.get_shared_files()
    patch_file_history_store.latest_set.assert_called_once_with()
    assert result == {"a.txt", "b.txt"}
