"""Verify SDK settings, error translation, and retry bounds with mocks."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import errors
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.configuration import GenerationSettings
from src.gemini_provider import GeminiProvider
from src.llm_client import PermanentLLMError, TransientLLMError
from src.openrouter_provider import OpenRouterProvider
from src.triage import TriageService


def mock_client(monkeypatch, target):
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(target, constructor)
    return constructor, client


def test_gemini_sends_settings_and_disables_sdk_retries(monkeypatch):
    constructor, client = mock_client(monkeypatch, "src.gemini_provider.genai.Client")
    client.models.generate_content.return_value = SimpleNamespace(text="respuesta", usage_metadata=None)
    provider = GeminiProvider("fake-key", "test-model", settings=GenerationSettings(0.3, 128, 4.5))
    response = provider.generate("Consulta de soporte")
    options = constructor.call_args.kwargs["http_options"]
    assert options.timeout == 4500
    assert options.retry_options.attempts == 1
    request = client.models.generate_content.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["config"].temperature == 0.3
    assert request["config"].max_output_tokens == 128
    assert response.text == "respuesta"
    assert response.total_tokens == 0


def test_openrouter_sends_settings_and_disables_sdk_retries(monkeypatch):
    constructor, client = mock_client(monkeypatch, "src.openrouter_provider.OpenAI")
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="respuesta"))],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6), model="routed-model",
    )
    response = OpenRouterProvider("fake-key", "test-model", settings=GenerationSettings(0.7, 256, 8)).generate("Consulta")
    assert constructor.call_args.kwargs["max_retries"] == 0
    assert constructor.call_args.kwargs["timeout"] == 8
    request = client.chat.completions.create.call_args.kwargs
    assert request["temperature"] == 0.7
    assert request["max_tokens"] == 256
    assert request["model"] == "test-model"
    assert response.total_tokens == 6
    assert response.model == "routed-model"


@pytest.mark.parametrize("error_class", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_gemini_translates_transport_errors(monkeypatch, error_class):
    _, client = mock_client(monkeypatch, "src.gemini_provider.genai.Client")
    client.models.generate_content.side_effect = error_class("simulated")
    with pytest.raises(TransientLLMError):
        GeminiProvider("fake-key", "test-model").generate("Consulta")


@pytest.mark.parametrize("timeout", [False, True])
def test_openrouter_translates_transport_errors(monkeypatch, timeout):
    _, client = mock_client(monkeypatch, "src.openrouter_provider.OpenAI")
    request = httpx.Request("POST", "https://example.invalid")
    error = APITimeoutError(request=request) if timeout else APIConnectionError(request=request)
    client.chat.completions.create.side_effect = error
    with pytest.raises(TransientLLMError):
        OpenRouterProvider("fake-key", "test-model").generate("Consulta")


@pytest.mark.parametrize("provider_name", ["gemini", "openrouter"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 429, 500, 503])
def test_service_retry_count_for_provider_errors(monkeypatch, provider_name, status):
    monkeypatch.setattr("src.retry.time.sleep", lambda _: None)
    if provider_name == "gemini":
        _, client = mock_client(monkeypatch, "src.gemini_provider.genai.Client")
        operation = client.models.generate_content
        operation.side_effect = errors.APIError(status, {"error": {"message": "simulated"}})
        provider = GeminiProvider("fake-key", "test-model")
    else:
        _, client = mock_client(monkeypatch, "src.openrouter_provider.OpenAI")
        operation = client.chat.completions.create
        response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
        operation.side_effect = APIStatusError("simulated", response=response, body=None)
        provider = OpenRouterProvider("fake-key", "test-model")
    transient = status in (408, 429) or status >= 500
    with pytest.raises(TransientLLMError if transient else PermanentLLMError):
        TriageService(provider, max_retries=3).classify("No puedo ingresar a mi cuenta.")
    assert operation.call_count == (3 if transient else 1)


def test_optional_gemini_httpx2_transport(monkeypatch):
    httpx2 = pytest.importorskip("httpx2")
    _, client = mock_client(monkeypatch, "src.gemini_provider.genai.Client")
    client.models.generate_content.side_effect = httpx2.ReadTimeout("simulated")
    with pytest.raises(TransientLLMError):
        GeminiProvider("fake-key", "test-model").generate("Consulta")
