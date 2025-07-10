import pytest
from typing import List, Dict, Any

from crowler.ai.openai.openai_client import OpenAIClient


class DummyResponse:
    class DummyChoice:
        class DummyMessage:
            content = "Hello, world!"

        message = DummyMessage()

    choices = [DummyChoice()]


@pytest.fixture
def dummy_openai(monkeypatch):
    class DummyChatCompletions:
        def create(self, **kwargs):
            return DummyResponse()

    class DummyChat:
        completions = DummyChatCompletions()

    class DummyClient:
        chat = DummyChat()

    monkeypatch.setattr(
        "crowler.ai.openai.openai_client.OpenAI", lambda api_key: DummyClient()
    )


@pytest.fixture
def set_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")


def test_init_raises_if_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAIClient()


def test_init_sets_client(monkeypatch, set_openai_api_key):
    dummy_client = object()
    monkeypatch.setattr(
        "crowler.ai.openai.openai_client.OpenAI", lambda api_key: dummy_client
    )
    client = OpenAIClient()
    assert client.client is dummy_client


def test_get_response_returns_content(
    monkeypatch, set_openai_api_key, dummy_openai, capsys
):
    client = OpenAIClient()
    messages: List[Any] = [{"role": "user", "content": "Say hi"}]
    result = client.get_response(messages)
    assert result == "Hello, world!"
    out = capsys.readouterr().out
    assert "Response received from" in out


def test_get_response_returns_empty_string_if_no_content(
    monkeypatch, set_openai_api_key
):
    class DummyResponseNoContent:
        class DummyChoice:
            class DummyMessage:
                content = None

            message = DummyMessage()

        choices = [DummyChoice()]

    class DummyChatCompletions:
        def create(self, **kwargs):
            return DummyResponseNoContent()

    class DummyChat:
        completions = DummyChatCompletions()

    class DummyClient:
        chat = DummyChat()

    monkeypatch.setattr(
        "crowler.ai.openai.openai_client.OpenAI", lambda api_key: DummyClient()
    )
    client = OpenAIClient()
    # Use the correct type for messages to satisfy mypy
    messages: List[Any] = [{"role": "user", "content": "Say hi"}]
    result = client.get_response(messages)
    assert result == ""
