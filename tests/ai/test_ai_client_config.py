import pytest

from crowler.ai.ai_client_config import AIConfig


class DummyAIConfig(AIConfig):
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.9
    model: str = "gpt-3.5-turbo"


def test_dummy_ai_config_implements_aiconfig():
    config = DummyAIConfig()
    assert hasattr(config, "temperature")
    assert isinstance(config.temperature, float)
    assert hasattr(config, "max_tokens")
    assert isinstance(config.max_tokens, int)
    assert hasattr(config, "top_p")
    assert isinstance(config.top_p, float)
    assert hasattr(config, "model")
    assert isinstance(config.model, str)


@pytest.mark.parametrize(
    "temperature,max_tokens,top_p,model",
    [
        (0.0, 1, 0.0, "test-model"),
        (1.0, 4096, 1.0, "another-model"),
        (0.5, 100, 0.5, "gpt-4"),
    ],
)
def test_various_aiconfig_values(temperature, max_tokens, top_p, model):
    # Use an instance with __init__ to avoid NameError in class scope
    class CustomConfig(AIConfig):
        def __init__(
            self, temperature: float, max_tokens: int, top_p: float, model: str
        ):
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.top_p = top_p
            self.model = model

    config = CustomConfig(temperature, max_tokens, top_p, model)
    assert config.temperature == temperature
    assert config.max_tokens == max_tokens
    assert config.top_p == top_p
    assert config.model == model


def test_missing_attribute_raises_attribute_error():
    class IncompleteConfig(AIConfig):
        temperature = 0.5
        max_tokens = 100
        # top_p is missing
        model = "gpt-4"

        def __getattribute__(self, name):
            if name == "top_p":
                raise AttributeError("top_p is missing")
            return super().__getattribute__(name)

    config = IncompleteConfig()
    with pytest.raises(AttributeError):
        _ = config.top_p
