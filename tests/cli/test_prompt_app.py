import pytest
from typer.testing import CliRunner
from unittest.mock import patch
from crowler.cli.prompt_app import prompt_app

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_dependencies():
    with (
        patch("crowler.cli.prompt_app.append_prompt") as mock_append_prompt,
        patch("crowler.cli.prompt_app.remove_prompt") as mock_remove_prompt,
        patch("crowler.cli.prompt_app.clear_prompts") as mock_clear_prompts,
        patch("crowler.cli.prompt_app.summary_prompts") as mock_summary_prompts,
        patch("crowler.cli.prompt_app.undo_prompts") as mock_undo_prompts,
    ):
        mock_summary_prompts.return_value = "📜 Prompts:\n(none)"
        yield {
            "append_prompt": mock_append_prompt,
            "remove_prompt": mock_remove_prompt,
            "clear_prompts": mock_clear_prompts,
            "summary_prompts": mock_summary_prompts,
            "undo_prompts": mock_undo_prompts,
        }


def test_add_prompt(mock_dependencies):
    result = runner.invoke(prompt_app, ["add", "Test prompt"])
    assert result.exit_code == 0
    mock_dependencies["append_prompt"].assert_called_once_with("Test prompt")


def test_remove_prompt(mock_dependencies):
    result = runner.invoke(prompt_app, ["remove", "Test prompt"])
    assert result.exit_code == 0
    mock_dependencies["remove_prompt"].assert_called_once_with("Test prompt")


def test_clear_prompts(mock_dependencies):
    result = runner.invoke(prompt_app, ["clear"])
    assert result.exit_code == 0
    mock_dependencies["clear_prompts"].assert_called_once()


def test_list_prompts(mock_dependencies):
    mock_dependencies["summary_prompts"].return_value = "📜 Prompts:\n- Test prompt"
    result = runner.invoke(prompt_app, ["list"])
    assert result.exit_code == 0
    assert "Test prompt" in result.output


def test_undo_prompts(mock_dependencies):
    result = runner.invoke(prompt_app, ["undo"])
    assert result.exit_code == 0
    mock_dependencies["undo_prompts"].assert_called_once()
