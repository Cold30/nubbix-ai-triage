import time

from google import genai
from google.genai import errors

from src.llm_client import ( LLMResponse, PermanentLLMError, TransientLLMError, )

class GeminiProvider:
    """Handles text generation using the Google Gemini API."""

    def __init__(self, api_key: str, model: str):
        """Initialize the provider with its API credentials and model."""
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response and return normalized usage metrics."""
        start_time = time.perf_counter()

        try:
            with genai.Client(api_key=self.api_key) as client:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

        except errors.APIError as error:
            status_code = getattr(error, "code", None)

            if status_code == 429 or (
                isinstance(status_code, int) and status_code >= 500
            ):
                raise TransientLLMError(
                    f"Gemini temporary HTTP error: {status_code}"
                ) from error

            raise PermanentLLMError(
                f"Gemini API error: {status_code}"
            ) from error

        latency = time.perf_counter() - start_time
        
        if not response.text:
            raise PermanentLLMError( "Gemini returned an empty response" )
        
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
    pass
