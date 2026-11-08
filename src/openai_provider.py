"""Normalize native OpenAI Responses results using the existing provider contract."""

import time

from openai import APIConnectionError, APIResponseValidationError, APIStatusError, APITimeoutError, OpenAI

from src.configuration import ConfigurationError, OpenAISettings
from src.llm_client import LLMResponse, PermanentLLMError, TransientLLMError


SUPPORTED_MODELS = {"gpt-6-luna"}
TRANSIENT_RESPONSE_CODES = {"server_error", "rate_limit_exceeded"}
BILLING_ERROR_CODES = {
    "insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted",
    "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}


class OpenAIProvider:
    """Generate text with explicitly supported model parameters and no SDK retries."""

    def __init__(self, api_key: str, model: str, *, settings: OpenAISettings | None = None):
        if model not in SUPPORTED_MODELS:
            raise ConfigurationError(
                "La integración nativa admite gpt-6-luna; otros modelos requieren verificar su compatibilidad."
            )
        self.api_key = api_key
        self.model = model
        self.settings = settings or OpenAISettings()
        if not isinstance(self.settings, OpenAISettings):
            raise ConfigurationError("OpenAI requiere una configuración explícita de razonamiento.")

    def generate(self, prompt: str) -> LLMResponse:
        """Generate synchronously and translate failures without exposing response bodies."""
        start_time = time.perf_counter()
        try:
            with OpenAI(
                api_key=self.api_key,
                base_url="https://api.openai.com/v1",
                timeout=self.settings.timeout_s,
                max_retries=0,
            ) as client:
                response = client.responses.create(
                    model=self.model,
                    input=prompt,
                    **self.settings.generation_parameters(),
                )
        except (APITimeoutError, APIConnectionError) as error:
            raise TransientLLMError("OpenAI: fallo temporal de conexión o timeout.") from error
        except APIResponseValidationError as error:
            raise PermanentLLMError("OpenAI devolvió una respuesta incompatible con el SDK instalado.") from error
        except APIStatusError as error:
            body = error.body if isinstance(error.body, dict) else {}
            detail = body.get("error", body)
            detail = detail if isinstance(detail, dict) else {}
            code = detail.get("code") or getattr(error, "code", None)
            error_type = detail.get("type") or getattr(error, "type", None)
            if code in BILLING_ERROR_CODES or error_type == "insufficient_quota":
                raise PermanentLLMError(
                    "OpenAI: revisá los créditos, la facturación y los límites del proyecto API."
                ) from error
            if error.status_code in (408, 429) or 500 <= error.status_code < 600:
                raise TransientLLMError(f"OpenAI: error HTTP temporal {error.status_code}.") from error
            raise PermanentLLMError(f"OpenAI: error HTTP permanente {error.status_code}.") from error

        latency = time.perf_counter() - start_time
        status = response.status
        response_error = getattr(response, "error", None)
        if response_error is not None:
            if getattr(response_error, "code", None) in TRANSIENT_RESPONSE_CODES:
                raise TransientLLMError("OpenAI: la generación falló temporalmente.")
            raise PermanentLLMError("OpenAI: la generación fue rechazada o falló permanentemente.")
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if reason == "max_output_tokens":
                raise PermanentLLMError(
                    "OpenAI: salida incompleta; revisá LLM_MAX_OUTPUT_TOKENS y OPENAI_REASONING_EFFORT."
                )
            raise PermanentLLMError("OpenAI: salida incompleta o filtrada; no se devuelve texto parcial.")
        if status != "completed":
            raise PermanentLLMError("OpenAI no devolvió una generación finalizada.")
        text = response.output_text
        if not isinstance(text, str) or not text.strip():
            raise PermanentLLMError("OpenAI devolvió una respuesta vacía o sin texto utilizable.")

        usage = response.usage
        input_tokens = (usage.input_tokens or 0) if usage else 0
        output_tokens = (usage.output_tokens or 0) if usage else 0
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(usage.total_tokens or input_tokens + output_tokens) if usage else 0,
            latency_s=latency,
            provider="openai",
            model=response.model,
        )
