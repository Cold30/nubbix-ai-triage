"""Exercise configuration validation without reading local credentials."""

import pytest

from src.configuration import ConfigurationError, GenerationSettings, environment_integer
from src.provider_factory import create_provider


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in (
        "GEMINI_API_KEY", "GEMINI_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
        "LLM_TEMPERATURE", "LLM_MAX_OUTPUT_TOKENS", "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
@pytest.mark.parametrize("field", ["API_KEY", "MODEL"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_required_provider_settings(provider, field, value, monkeypatch):
    prefix = provider.upper()
    monkeypatch.setenv(f"{prefix}_API_KEY", "fake-secret-never-print")
    monkeypatch.setenv(f"{prefix}_MODEL", "test-model")
    name = f"{prefix}_{field}"
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name) as caught:
        create_provider(provider)
    assert "fake-secret-never-print" not in str(caught.value)


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_factory_passes_selected_model_and_generation_settings(provider, monkeypatch):
    prefix = provider.upper()
    monkeypatch.setenv(f"{prefix}_API_KEY", "fake-key")
    monkeypatch.setenv(f"{prefix}_MODEL", " configured-model ")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    instance = create_provider(provider)
    assert instance.model == "configured-model"
    assert instance.settings == GenerationSettings(0.4, 256, 12.5)


@pytest.mark.parametrize(("name", "value"), [
    ("LLM_TEMPERATURE", ""), ("LLM_TEMPERATURE", "nan"), ("LLM_TEMPERATURE", "inf"),
    ("LLM_TEMPERATURE", "-0.1"), ("LLM_TEMPERATURE", "2.1"),
    ("LLM_MAX_OUTPUT_TOKENS", "0"), ("LLM_MAX_OUTPUT_TOKENS", "2.5"),
    ("LLM_MAX_OUTPUT_TOKENS", ""), ("LLM_TIMEOUT_SECONDS", "0"),
    ("LLM_TIMEOUT_SECONDS", "0.0001"), ("LLM_TIMEOUT_SECONDS", "nan"),
    ("LLM_TIMEOUT_SECONDS", "inf"), ("LLM_TIMEOUT_SECONDS", "no"),
])
def test_invalid_generation_settings(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        GenerationSettings.from_environment()


def test_generation_defaults_and_retry_validation(monkeypatch):
    assert GenerationSettings.from_environment() == GenerationSettings()
    assert environment_integer("LLM_MAX_RETRIES", 2) == 2
    for value in ("", "-1", "1.5"):
        monkeypatch.setenv("LLM_MAX_RETRIES", value)
        with pytest.raises(ConfigurationError, match="LLM_MAX_RETRIES"):
            environment_integer("LLM_MAX_RETRIES", 2)


def test_unknown_provider_has_actionable_error():
    with pytest.raises(ValueError, match="gemini u openrouter"):
        create_provider("unknown")
