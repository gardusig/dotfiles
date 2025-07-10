import pytest
from typer.testing import CliRunner
from crowler.cli.file_app import file_app
from unittest.mock import patch

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_dependencies():
    with (
        patch("crowler.cli.file_app.append_shared_file") as mock_append,
        patch("crowler.cli.file_app.remove_shared_file") as mock_remove,
        patch("crowler.cli.file_app.clear_shared_files") as mock_clear,
        patch(
            "crowler.cli.file_app.summary_shared_files",
            return_value="Summary of shared files",
        ) as mock_summary,
        patch("crowler.cli.file_app.undo_shared_files") as mock_undo,
    ):
        yield {
            "append": mock_append,
            "remove": mock_remove,
            "clear": mock_clear,
            "summary": mock_summary,
            "undo": mock_undo,
        }


def test_add_file(mock_dependencies):
    with patch(
        "crowler.cli.file_app.get_all_files",
        return_value=["/path/to/file1", "/path/to/file2"],
    ):
        result = runner.invoke(file_app, ["add", "/path/to"])
        assert result.exit_code == 0
        mock_dependencies["append"].assert_any_call("/path/to/file1")
        mock_dependencies["append"].assert_any_call("/path/to/file2")


def test_remove_file(mock_dependencies):
    result = runner.invoke(file_app, ["remove", "/path/to/file1"])
    assert result.exit_code == 0
    mock_dependencies["remove"].assert_called_once_with("/path/to/file1")


def test_clear_files(mock_dependencies):
    result = runner.invoke(file_app, ["clear"])
    assert result.exit_code == 0
    mock_dependencies["clear"].assert_called_once()


def test_list_files(mock_dependencies):
    result = runner.invoke(file_app, ["list"])
    assert result.exit_code == 0
    assert "Summary of shared files" in result.output
    mock_dependencies["summary"].assert_called_once()


def test_undo_file(mock_dependencies):
    result = runner.invoke(file_app, ["undo"])
    assert result.exit_code == 0
    mock_dependencies["undo"].assert_called_once()
