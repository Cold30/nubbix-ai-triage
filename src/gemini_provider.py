import time

from google import genai
from src.llm_client import LLMResponse

class GeminiProvider:
    """Handles text generation using the Google Gemini API."""

    def __init__(self, api_key: str, model: str):
        """Initialize the provider with its API credentials and model."""
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response and return normalized usage metrics."""
        start_time = time.perf_counter()

        with genai.Client(api_key=self.api_key) as client:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                # response_format=,
            )

        latency = time.perf_counter() - start_time
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
