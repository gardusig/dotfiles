import pyperclip
from typer.testing import CliRunner
from unittest.mock import patch
from crowler.cli.app import app

runner = CliRunner()


def test_preview_command():
    with (
        patch(
            "crowler.cli.app.summary_prompts", return_value="Prompt Summary"
        ) as mock_summary_prompts,
        patch(
            "crowler.cli.app.summary_shared_files",
            return_value="Shared Files Summary",
        ) as mock_summary_shared_files,
        patch(
            "crowler.cli.app.summary_processing_files",
            return_value="Processing Files Summary",
        ) as mock_summary_processing_files,
    ):

        result = runner.invoke(app, ["show"])

        assert result.exit_code == 0
        assert "Prompt Summary" in result.output
        assert "Shared Files Summary" in result.output
        assert "Processing Files Summary" in result.output

        mock_summary_prompts.assert_called_once()
        mock_summary_shared_files.assert_called_once()
        mock_summary_processing_files.assert_called_once()


def test_add_prompt_command():
    with patch("crowler.cli.app.append_prompt") as mock_append_prompt:
        result = runner.invoke(app, ["add", "Test prompt"])

        assert result.exit_code == 0
        mock_append_prompt.assert_called_once_with("Test prompt")


def test_add_prompt_command_empty():
    result = runner.invoke(app, ["add", ""])

    assert result.exit_code != 0
    assert "⚠️  Empty prompt provided." in result.output


def test_clear_all_command():
    with (
        patch("crowler.cli.app.clear_prompts") as mock_clear_prompts,
        patch("crowler.cli.app.clear_shared_files") as mock_clear_shared_files,
        patch("crowler.cli.app.clear_processing_files") as mock_clear_processing_files,
    ):

        result = runner.invoke(app, ["clear"])

        assert result.exit_code == 0
        mock_clear_prompts.assert_called_once()
        mock_clear_shared_files.assert_called_once()
        mock_clear_processing_files.assert_called_once()


def test_add_prompt_from_clipboard():
    with (
        patch(
            "crowler.cli.app._clipboard_get", return_value="Clipboard prompt"
        ) as mock_clipboard_get,
        patch("crowler.cli.app.append_prompt") as mock_append_prompt,
    ):

        result = runner.invoke(app, ["clipboard"])

        assert result.exit_code == 0
        mock_clipboard_get.assert_called_once()
        mock_append_prompt.assert_called_once_with("Clipboard prompt")


def test_clipboard_get_unavailable():
    with patch("pyperclip.paste", side_effect=pyperclip.PyperclipException):
        result = runner.invoke(app, ["clipboard"])
        assert result.exit_code != 0
        assert "Clipboard not available on this system." in result.output
