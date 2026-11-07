
import time

from openai import OpenAI

from src.llm_client import LLMResponse

from openai import ( APIConnectionError, APIStatusError, APITimeoutError, )

from src.llm_client import ( LLMResponse, PermanentLLMError, TransientLLMError, )

class OpenRouterProvider:
    """Handles text generation through the OpenRouter API."""

    def __init__(self, api_key: str, model: str):
        """Initialize the OpenRouter client configuration."""
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        """Generate text and return normalized usage metrics."""
        start_time = time.perf_counter()

        try:
            with OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            ) as client:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )

        except (APITimeoutError, APIConnectionError) as error:
            raise TransientLLMError(
                "OpenRouter connection or timeout failure"
            ) from error

        except APIStatusError as error:
            if error.status_code == 429 or error.status_code >= 500:
                raise TransientLLMError(
                    f"OpenRouter temporary HTTP error: {error.status_code}"
                ) from error

            raise PermanentLLMError(
                f"OpenRouter HTTP error: {error.status_code}"
            ) from error

        latency = time.perf_counter() - start_time
        usage = response.usage
        
        if not response.choices:
            raise PermanentLLMError( "OpenRouter returned no response choices" )

        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
            total_tokens=(usage.total_tokens or 0) if usage else 0,
            latency_s=latency,
            provider="openrouter",
            model=response.model,
        )
