"""Validate optional OpenAI configuration and experiment compatibility."""

from dataclasses import replace

import pytest

from evals.runner import build_experiment_identity
from src.configuration import ConfigurationError, OpenAISettings
from src.provider_factory import create_provider


@pytest.fixture(autouse=True)
def clean_openai_environment(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT",
                 "LLM_TEMPERATURE", "LLM_MAX_OUTPUT_TOKENS", "LLM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_REASONING_EFFORT"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_native_openai_requires_only_its_own_explicit_settings(monkeypatch, name, value):
    for key, setting in {"OPENAI_API_KEY": "fake-key", "OPENAI_MODEL": "gpt-6-luna", "OPENAI_REASONING_EFFORT": "low"}.items():
        monkeypatch.setenv(key, setting)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        create_provider("openai")


def test_factory_reads_native_model_and_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_MODEL", " gpt-6-luna ")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", " low ")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "2048")
    provider = create_provider("openai")
    assert provider.model == "gpt-6-luna"
    assert provider.settings.reasoning_effort == "low"
    assert provider.settings.max_output_tokens == 2048


@pytest.mark.parametrize("effort", ["minimal", "ultra", "LOW", "unsupported"])
def test_rejects_unsupported_luna_effort(effort):
    with pytest.raises(ConfigurationError, match="OPENAI_REASONING_EFFORT"):
        OpenAISettings(reasoning_effort=effort)


def test_enforces_documented_model_output_limit():
    with pytest.raises(ConfigurationError, match="128000"):
        OpenAISettings(max_output_tokens=128001)
    assert OpenAISettings(max_output_tokens=128000).max_output_tokens == 128000


@pytest.mark.parametrize("provider_name", ["gemini", "openrouter"])
def test_existing_providers_ignore_missing_or_invalid_openai_settings(monkeypatch, provider_name):
    monkeypatch.setenv(f"{provider_name.upper()}_API_KEY", "fake-key")
    monkeypatch.setenv(f"{provider_name.upper()}_MODEL", "original-model")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "invalid")
    provider = create_provider(provider_name)
    assert provider.model == "original-model"
    assert provider.settings.generation_parameters() == {"temperature": 0.1, "max_output_tokens": 512}


def test_openai_identity_tracks_only_effective_sampling_parameters():
    def identity(settings):
        return build_experiment_identity([], "triage_v1", "openai", "gpt-6-luna", settings=settings)

    settings = OpenAISettings(reasoning_effort="low")
    assert identity(settings) == identity(replace(settings, temperature=0.8))
    assert identity(settings) != identity(replace(settings, reasoning_effort="medium"))
    assert identity(settings) != identity(replace(settings, max_output_tokens=1024))
    no_reasoning = replace(settings, reasoning_effort="none")
    assert identity(no_reasoning) != identity(replace(no_reasoning, temperature=0.8))
    with pytest.raises(ConfigurationError, match="razonamiento"):
        build_experiment_identity([], "triage_v1", "openai", "gpt-6-luna")
