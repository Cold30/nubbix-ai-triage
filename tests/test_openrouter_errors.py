
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIStatusError

from src.llm_client import PermanentLLMError, TransientLLMError
from src.openrouter_provider import OpenRouterProvider


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, PermanentLLMError),
        (429, TransientLLMError),
        (503, TransientLLMError),
    ],
)
def test_openrouter_classifies_http_errors(
    monkeypatch,
    status_code,
    expected_error,
):
    """Verify provider errors without making real API calls."""
    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    response = httpx.Response(status_code, request=request)

    api_error = APIStatusError(
        "Simulated HTTP failure",
        response=response,
        body=None,
    )

    fake_client = Mock()
    fake_client.__enter__ = Mock(return_value=fake_client)
    fake_client.__exit__ = Mock(return_value=False)
    fake_client.chat.completions.create.side_effect = api_error

    monkeypatch.setattr(
        "src.openrouter_provider.OpenAI",
        lambda **kwargs: fake_client,
    )

    provider = OpenRouterProvider(
        api_key="fake-key",
        model="test-model",
    )

    with pytest.raises(expected_error):
        provider.generate("Simulated support ticket")


def test_openrouter_rejects_empty_choices(monkeypatch):
    """Reject provider responses without generated alternatives."""
    fake_client = Mock()
    fake_client.__enter__ = Mock(return_value=fake_client)
    fake_client.__exit__ = Mock(return_value=False)

    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[],
        usage=None,
        model="test-model",
    )

    monkeypatch.setattr(
        "src.openrouter_provider.OpenAI",
        lambda **kwargs: fake_client,
    )

    provider = OpenRouterProvider(
        api_key="fake-key",
        model="test-model",
    )

    with pytest.raises(
        PermanentLLMError,
        match="no response choices",
    ):
        provider.generate("Simulated support ticket")
