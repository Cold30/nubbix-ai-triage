
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_s: float
    provider: str
    model: str


class LLMProviderError(Exception):
    """Base exception for LLM provider failures."""


class TransientLLMError(LLMProviderError):
    """A temporary failure that may succeed when retried."""


class PermanentLLMError(LLMProviderError):
    """A failure that should not be retried automatically."""
