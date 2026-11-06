
import time

from openai import OpenAI

from src.llm_client import LLMResponse


class OpenRouterProvider:
    """Handles text generation through the OpenRouter API."""

    def __init__(self, api_key: str, model: str):
        """Initialize the OpenRouter client configuration."""
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        """Generate text and return normalized usage metrics."""
        start_time = time.perf_counter()

        with OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        ) as client:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

        latency = time.perf_counter() - start_time
        usage = response.usage

        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
            total_tokens=(usage.total_tokens or 0) if usage else 0,
            latency_s=latency,
            provider="openrouter",
            model=response.model,
        )
