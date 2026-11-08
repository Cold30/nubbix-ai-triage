"""Test native Responses behavior without credentials, sockets, or SDK retries."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APIResponseValidationError, APIStatusError, APITimeoutError
from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText

from src.configuration import ConfigurationError, GenerationSettings, OpenAISettings
from src.llm_client import PermanentLLMError, TransientLLMError
from src.openai_provider import OpenAIProvider
from src.triage import TriageService


def mock_client(monkeypatch, response=None):
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.responses.create.return_value = response or SimpleNamespace(
        status="completed", output_text="Respuesta de soporte.", error=None,
        model="gpt-6-luna", usage=None,
    )
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("src.openai_provider.OpenAI", constructor)
    return constructor, client


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
def test_responses_parameters_follow_model_compatibility(monkeypatch, effort):
    constructor, client = mock_client(monkeypatch)
    settings = OpenAISettings(temperature=0.3, max_output_tokens=2048, timeout_s=12, reasoning_effort=effort)
    result = OpenAIProvider("fake-key", "gpt-6-luna", settings=settings).generate("No puedo acceder.")
    request = client.responses.create.call_args.kwargs
    assert request == {
        "model": "gpt-6-luna", "input": "No puedo acceder.", **settings.generation_parameters(),
    }
    assert request["reasoning"] == {"effort": effort}
    assert request["max_output_tokens"] == 2048
    assert ("temperature" in request) == (effort == "none")
    assert "top_p" not in request and "max_tokens" not in request
    assert request["store"] is False and request["service_tier"] == "default"
    assert constructor.call_args.kwargs == {
        "api_key": "fake-key", "base_url": "https://api.openai.com/v1", "timeout": 12, "max_retries": 0,
    }
    client.chat.completions.create.assert_not_called()
    assert result.provider == "openai"


def test_normalizes_usage_latency_and_returned_model(monkeypatch):
    response = SimpleNamespace(
        status="completed", output_text="Respuesta.", error=None, model="returned-model-identity",
        usage=SimpleNamespace(input_tokens=100, output_tokens=70, total_tokens=170,
                              output_tokens_details=SimpleNamespace(reasoning_tokens=50)),
    )
    mock_client(monkeypatch, response)
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr("src.openai_provider.time.perf_counter", lambda: next(ticks))
    result = OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (100, 70, 170)
    assert result.latency_s == 0.25
    assert result.model == "returned-model-identity"


def test_aggregates_sdk_output_text_instead_of_indexing_first_item(monkeypatch):
    response = Response.model_construct(
        status="completed", model="gpt-6-luna", error=None, usage=None,
        output=[
            SimpleNamespace(type="reasoning"),
            ResponseOutputMessage.model_construct(type="message", content=[
                ResponseOutputText.model_construct(type="output_text", text="Primera parte. "),
                ResponseOutputText.model_construct(type="output_text", text="Segunda parte."),
            ]),
        ],
    )
    mock_client(monkeypatch, response)
    result = OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")
    assert result.text == "Primera parte. Segunda parte."
    assert result.total_tokens == 0


@pytest.mark.parametrize("text", ["", " \n ", None])
def test_rejects_empty_outputs(monkeypatch, text):
    mock_client(monkeypatch, SimpleNamespace(status="completed", error=None, output_text=text))
    with pytest.raises(PermanentLLMError, match="vacía"):
        OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_rejects_partial_output_even_if_valid_json(monkeypatch, reason):
    mock_client(monkeypatch, SimpleNamespace(
        status="incomplete", error=None, output_text='{"answer":"Parcial"}',
        incomplete_details=SimpleNamespace(reason=reason),
    ))
    with pytest.raises(PermanentLLMError, match="incompleta"):
        OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")


@pytest.mark.parametrize("status", ["queued", "in_progress", "cancelled", "failed"])
def test_rejects_nonfinal_or_failed_responses(monkeypatch, status):
    mock_client(monkeypatch, SimpleNamespace(status=status, error=None))
    with pytest.raises(PermanentLLMError):
        OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")


@pytest.mark.parametrize(("code", "exception"), [
    ("server_error", TransientLLMError), ("rate_limit_exceeded", TransientLLMError),
    ("invalid_prompt", PermanentLLMError), ("misalignment_policy_violation", PermanentLLMError),
])
def test_response_errors_preserve_retry_classification(monkeypatch, code, exception):
    mock_client(monkeypatch, SimpleNamespace(status="failed", error=SimpleNamespace(code=code)))
    with pytest.raises(exception):
        OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 429, 500, 503])
def test_http_error_retry_bounds(monkeypatch, status):
    _, client = mock_client(monkeypatch)
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    client.responses.create.side_effect = APIStatusError("private diagnostic", response=response, body=None)
    monkeypatch.setattr("src.retry.time.sleep", lambda _: None)
    transient = status in (408, 429) or status >= 500
    with pytest.raises(TransientLLMError if transient else PermanentLLMError) as caught:
        TriageService(OpenAIProvider("fake-key", "gpt-6-luna"), max_retries=3).classify("No puedo acceder.")
    assert client.responses.create.call_count == (3 if transient else 1)
    assert "private diagnostic" not in str(caught.value)


@pytest.mark.parametrize("code", [
    "insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
])
def test_billing_errors_are_permanent_even_with_http_429(monkeypatch, code):
    _, client = mock_client(monkeypatch)
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
    client.responses.create.side_effect = APIStatusError(
        "private diagnostic", response=response, body={"error": {"code": code}},
    )
    monkeypatch.setattr("src.retry.time.sleep", lambda _: pytest.fail("Billing errors must not retry"))
    with pytest.raises(PermanentLLMError, match="facturación"):
        TriageService(OpenAIProvider("fake-key", "gpt-6-luna")).classify("Consulta")
    assert client.responses.create.call_count == 1


@pytest.mark.parametrize("timeout", [True, False])
def test_transport_errors_are_transient(monkeypatch, timeout):
    _, client = mock_client(monkeypatch)
    request = httpx.Request("POST", "https://example.invalid")
    client.responses.create.side_effect = APITimeoutError(request=request) if timeout else APIConnectionError(request=request)
    with pytest.raises(TransientLLMError):
        OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")


def test_sdk_response_validation_errors_are_permanent(monkeypatch):
    _, client = mock_client(monkeypatch)
    response = httpx.Response(200, request=httpx.Request("POST", "https://example.invalid"))
    client.responses.create.side_effect = APIResponseValidationError(response, {"private": "diagnostic"})
    with pytest.raises(PermanentLLMError, match="incompatible") as caught:
        OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta")
    assert "diagnostic" not in str(caught.value)


def test_unsupported_model_and_generic_settings_fail_before_client_creation(monkeypatch):
    constructor, _ = mock_client(monkeypatch)
    with pytest.raises(ConfigurationError, match="compatibilidad"):
        OpenAIProvider("fake-key", "unknown-model")
    with pytest.raises(ConfigurationError, match="razonamiento"):
        OpenAIProvider("fake-key", "gpt-6-luna", settings=GenerationSettings())
    constructor.assert_not_called()


def test_service_validates_and_logs_native_openai_response(tmp_path, monkeypatch):
    from src.metrics import MetricsLogger

    mock_client(monkeypatch, SimpleNamespace(
        status="completed", error=None, model="gpt-6-luna",
        output_text=json.dumps({"answer": "Un agente puede revisar el acceso.", "confidence": 0.9,
                                "category": "account", "actions": ["Revisar permisos"]}),
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150),
    ))
    path = tmp_path / "metrics.jsonl"
    service = TriageService(OpenAIProvider("fake-key", "gpt-6-luna"), metrics_logger=MetricsLogger(str(path)))
    assert service.classify("No puedo entrar.").category == "account"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["provider"] == "openai"
    assert record["estimated_cost_usd"] == pytest.approx(0.000035)


def test_installed_sdk_serializes_responses_request_with_mock_transport(monkeypatch):
    import httpx2
    from openai import DefaultHttpxClient, OpenAI

    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "resp_mock", "object": "response", "created_at": 0,
            "status": "completed", "error": None, "model": "gpt-6-luna",
            "output": [{"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "Respuesta de soporte.", "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 2}},
        })

    client = OpenAI(api_key="fake-key", max_retries=0,
                    http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr("src.openai_provider.OpenAI", lambda **kwargs: client)
    result = OpenAIProvider("fake-key", "gpt-6-luna").generate("Consulta de soporte")
    assert result.total_tokens == 16
    assert result.text == "Respuesta de soporte."
    assert payloads[0]["reasoning"] == {"effort": "low"}
    assert "temperature" not in payloads[0]
