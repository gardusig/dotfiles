import pytest
from typer.testing import CliRunner
from crowler.cli.process_app import process_app
from unittest.mock import patch, call

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_file_operations():
    with (
        patch("crowler.cli.process_app.get_all_files") as mock_get_all_files,
        patch("crowler.cli.process_app.append_processing_file") as mock_append,
        patch("crowler.cli.process_app.remove_processing_file") as mock_remove,
        patch("crowler.cli.process_app.clear_processing_files") as mock_clear,
        patch("crowler.cli.process_app.summary_processing_files") as mock_summary,
        patch("crowler.cli.process_app.undo_processing_files") as mock_undo,
    ):
        mock_get_all_files.return_value = ["file1.txt", "file2.txt"]
        mock_summary.return_value = "Current files: file1.txt, file2.txt"
        yield mock_get_all_files, mock_append, mock_remove, mock_clear, mock_summary, mock_undo


def test_add_file(mock_file_operations):
    mock_get_all_files, mock_append, _, _, _, _ = mock_file_operations
    result = runner.invoke(process_app, ["add", "dummy_path"])
    assert result.exit_code == 0
    mock_get_all_files.assert_called_once_with("dummy_path")
    mock_append.assert_has_calls([call("file1.txt"), call("file2.txt")])


def test_remove_file(mock_file_operations):
    mock_get_all_files, _, mock_remove, _, _, _ = mock_file_operations
    result = runner.invoke(process_app, ["remove", "path"])
    assert result.exit_code == 0
    mock_get_all_files.assert_called_once_with("path")
    mock_remove.assert_has_calls([call("file1.txt"), call("file2.txt")])


def test_clear_files(mock_file_operations):
    _, _, _, mock_clear, _, _ = mock_file_operations
    result = runner.invoke(process_app, ["clear"])
    assert result.exit_code == 0
    mock_clear.assert_called_once()


def test_list_files(mock_file_operations):
    _, _, _, _, mock_summary, _ = mock_file_operations
    result = runner.invoke(process_app, ["list"])
    assert result.exit_code == 0
    assert "Current files: file1.txt, file2.txt" in result.output
    mock_summary.assert_called_once()


def test_undo_file(mock_file_operations):
    _, _, _, _, _, mock_undo = mock_file_operations
    result = runner.invoke(process_app, ["undo"])
    assert result.exit_code == 0
    mock_undo.assert_called_once()
