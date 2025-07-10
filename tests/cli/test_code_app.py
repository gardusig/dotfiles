import pytest
from unittest.mock import patch
from pathlib import Path
from crowler.cli.code_app import create_tests, create_test


@pytest.fixture
def mock_ai_client():
    with patch("crowler.cli.code_app.get_ai_client") as mock:
        yield mock.return_value


@pytest.fixture
def mock_get_processing_files():
    with patch("crowler.cli.code_app.get_processing_files") as mock:
        yield mock


@pytest.fixture
def mock_find_repo_root():
    with patch("crowler.cli.code_app.find_repo_root") as mock:
        yield mock


@pytest.fixture
def mock_source_to_test_path():
    with patch("crowler.cli.code_app.source_to_test_path") as mock:
        yield mock


@pytest.fixture
def mock_rewrite_files():
    with patch("crowler.cli.code_app.rewrite_files") as mock:
        yield mock


@pytest.fixture
def mock_parse_code_response():
    with patch("crowler.cli.code_app.parse_code_response") as mock:
        yield mock


def test_create_tests(
    mock_ai_client,
    mock_get_processing_files,
    mock_find_repo_root,
    mock_source_to_test_path,
    mock_rewrite_files,
    mock_parse_code_response,
):
    mock_get_processing_files.return_value = ["file1.py", "file2.py"]
    mock_find_repo_root.return_value = Path("/repo/root")
    mock_source_to_test_path.side_effect = lambda src, repo_root: Path(
        f"/repo/root/tests/{src.name}"
    )
    mock_ai_client.send_message.return_value = "mock_response"
    mock_parse_code_response.return_value = {
        "/repo/root/tests/file1_test.py": "test content"
    }

    create_tests(force=True)

    assert mock_ai_client.send_message.call_count == 2
    mock_rewrite_files.assert_called()
