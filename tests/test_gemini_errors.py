
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors

from src.gemini_provider import GeminiProvider
from src.llm_client import PermanentLLMError, TransientLLMError


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (400, PermanentLLMError),
        (429, TransientLLMError),
        (503, TransientLLMError),
    ],
)

def test_gemini_classifies_api_errors( monkeypatch, status_code, expected_error, ):
    """Verify Gemini API error handling without network calls."""
    api_error = (
        errors.ServerError(
            code=status_code,
            response_json={"error": {"message": "Simulated failure"}},
        )
        if status_code >= 500
        else errors.ClientError(
            code=status_code,
            response_json={"error": {"message": "Simulated failure"}},
        )
    )

    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False
    fake_client.models.generate_content.side_effect = api_error

    monkeypatch.setattr(
        "src.gemini_provider.genai.Client",
        lambda **kwargs: fake_client,
    )

    provider = GeminiProvider(
        api_key="fake-key",
        model="test-model",
    )

    with pytest.raises(expected_error):
        provider.generate("Simulated support ticket")


def test_gemini_rejects_empty_response(monkeypatch):
    """Reject responses without generated text."""
    fake_client = MagicMock()
    
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False
    
    fake_client.models.generate_content.return_value = SimpleNamespace(
        text="",
        usage_metadata=None,
    )

    monkeypatch.setattr(
        "src.gemini_provider.genai.Client",
        lambda **kwargs: fake_client,
    )

    provider = GeminiProvider(
        api_key="fake-key",
        model="test-model",
    )

    with pytest.raises(
        PermanentLLMError,
        match="respuesta vacía",
    ):
        provider.generate("Simulated support ticket")
