import time

from google import genai
from google.genai import errors
from google.genai import types
import httpx

from src.configuration import GenerationSettings

try:
    import httpx2
except ImportError:
    httpx2 = None

TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
if httpx2 is not None:
    TRANSPORT_ERRORS += (httpx2.TimeoutException, httpx2.NetworkError, httpx2.RemoteProtocolError)

from src.llm_client import ( LLMResponse, PermanentLLMError, TransientLLMError, )

class GeminiProvider:
    """Handles text generation using the Google Gemini API."""

    def __init__(self, api_key: str, model: str, *, settings: GenerationSettings | None = None):
        """Initialize the provider with its API credentials and model."""
        self.api_key = api_key
        self.model = model
        self.settings = settings or GenerationSettings()

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response and return normalized usage metrics."""
        start_time = time.perf_counter()

        try:
            with genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=round(self.settings.timeout_s * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ) as client:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.settings.temperature,
                        max_output_tokens=self.settings.max_output_tokens,
                    ),
                )

        except TRANSPORT_ERRORS as error:
            raise TransientLLMError("Gemini: fallo temporal de conexión o timeout.") from error
        except errors.APIError as error:
            status_code = getattr(error, "code", None)

            if status_code in (408, 429) or (
                isinstance(status_code, int) and 500 <= status_code < 600
            ):
                raise TransientLLMError(
                    f"Gemini: error HTTP temporal {status_code}."
                ) from error

            raise PermanentLLMError(
                f"Gemini: error permanente de API {status_code}."
            ) from error

        latency = time.perf_counter() - start_time
        
        if not response.text:
            raise PermanentLLMError("Gemini devolvió una respuesta vacía.")
        
        usage = response.usage_metadata

        return LLMResponse(
            text=response.text or "",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            total_tokens=(usage.total_token_count or 0) if usage else 0,
            latency_s=latency,
            provider="gemini",
            model=self.model,
        )
