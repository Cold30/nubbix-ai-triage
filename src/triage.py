from pydantic import ValidationError

from src.models import TriageResponse
from src.prompting import build_prompt
from src.metrics import MetricsLogger
from src.pricing import estimate_cost_usd
from src.guardrails import validate_input, validate_output


class TriageService:
    """Coordinates LLM-based support ticket classification."""

    def __init__( self, provider, max_retries: int = 2, metrics_logger: MetricsLogger | None = None):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.provider = provider
        self.max_retries = max_retries
        self.metrics_logger = metrics_logger

    def classify( self, query: str, prompt_version: str = "triage_v1", ) -> TriageResponse:
        """Classify a ticket and retry invalid structured responses."""
        validated_query = validate_input(query)
        prompt = build_prompt(prompt_version, validated_query)

        for attempt in range(self.max_retries + 1):
            response = self.provider.generate(prompt)
            
            if self.metrics_logger is not None:
                self.metrics_logger.log({
                    "provider": response.provider,
                    "model": response.model,
                    "prompt_version": prompt_version,
                    "attempt": attempt + 1,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                    "latency_s": response.latency_s,
                    "estimated_cost_usd": estimate_cost_usd(
                        model=response.model,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                    ),
                })

            try:
                normalized_text = normalize_json_response(response.text)
                validated_response = TriageResponse.model_validate_json(
                    normalized_text
                )
                return validate_output(validated_response)

            except ValidationError as error:
                if attempt == self.max_retries:
                    raise

                prompt = (
                    f"{prompt}\n\n"
                    "Your previous response did not match the required "
                    "JSON contract. Return only a valid JSON object with "
                    "answer, confidence, category, and actions. "
                    f"Validation error: {error}"
                )

        raise RuntimeError("Unexpected retry loop termination")



def normalize_json_response(text: str) -> str:
    """Remove an optional Markdown JSON code fence."""
    content = text.strip()

    if content.startswith("```json") and content.endswith("```"):
        content = content[len("```json"):-len("```")].strip()

    return content