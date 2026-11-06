
import json
import pytest

from types import SimpleNamespace
from pydantic import ValidationError

from src.triage import TriageService
from src.guardrails import GuardrailError

class FakeProvider:
    """Return predefined responses without calling an API."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.call_count = 0

    def generate(self, prompt: str):
        self.call_count += 1
        return SimpleNamespace(text=next(self.responses))


def test_retries_after_invalid_json():
    valid_response = {
        "answer": "Un agente puede revisar el cobro duplicado.",
        "confidence": 0.95,
        "category": "billing",
        "actions": ["Revisar el historial de facturación"],
    }

    provider = FakeProvider([
        "This is not valid JSON",
        json.dumps(valid_response),
    ])

    service = TriageService(provider)

    result = service.classify(
        query="Me cobraron dos veces.",
        prompt_version="triage_v1",
    )

    assert result.category == "billing"
    assert provider.call_count == 2


def test_valid_response_does_not_retry():
    provider = FakeProvider([
        json.dumps({
            "answer": "Un agente puede revisar el problema.",
            "confidence": 0.9,
            "category": "technical",
            "actions": ["Revisar los registros del sistema"],
        })
    ])

    service = TriageService(provider, max_retries=2)

    result = service.classify("La aplicación no funciona.")

    assert result.category == "technical"
    assert provider.call_count == 1


def test_raises_after_exhausting_retries():
    provider = FakeProvider([
        "Invalid JSON",
        "Still invalid JSON",
        "Another invalid response",
    ])

    service = TriageService(provider, max_retries=2)

    with pytest.raises(ValidationError):
        service.classify("No puedo ingresar a mi cuenta.")

    assert provider.call_count == 3

def test_rejects_invalid_input_without_calling_provider():
    provider = FakeProvider([])
    service = TriageService(provider)

    with pytest.raises(GuardrailError):
        service.classify("   ")

    assert provider.call_count == 0
