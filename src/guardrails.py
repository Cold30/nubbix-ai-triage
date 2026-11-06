
import re

from src.models import TriageResponse

PROMPT_INJECTION_PATTERNS = [
    re.compile(
        r"ignor[aá]\s+(todas\s+)?las\s+instrucciones",
        re.IGNORECASE,
    ),
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"(revel[aá]|mostr[aá])\s+(el\s+)?(prompt|instrucciones)\s+(del\s+)?sistema",
        re.IGNORECASE,
    ),
    re.compile(
        r"reveal\s+(your\s+)?system\s+prompt",
        re.IGNORECASE,
    ),
]

def detect_prompt_injection(query: str) -> bool:
    """Detect common attempts to override system instructions."""
    return any(
        pattern.search(query)
        for pattern in PROMPT_INJECTION_PATTERNS
    )

UNVERIFIED_ACTION_PATTERNS = [
    r"\bhemos (realizado|procesado|emitido|aprobado)\b",
    r"\b(ya|hemos) (reembolsado|devuelto)\b",
    r"\btu contraseña (fue|ha sido) restablecida\b",
]


class GuardrailError(ValueError):
    """Raised when a support request violates a guardrail."""


def validate_input(query: str) -> str:
    """Validate a customer support request before LLM generation."""
    cleaned_query = query.strip()

    if not cleaned_query:
        raise GuardrailError("Support request cannot be empty")

    if len(cleaned_query) > 5000:
        raise GuardrailError("Support request exceeds maximum length")
    
    if detect_prompt_injection(cleaned_query):
        raise GuardrailError("Potential prompt injection detected")

    return cleaned_query

def validate_output(result: TriageResponse) -> TriageResponse:
    """Reject responses claiming unverified completed operations."""
    answer = result.answer.lower()

    for pattern in UNVERIFIED_ACTION_PATTERNS:
        if re.search(pattern, answer):
            raise GuardrailError(
                "Response claims an unverified completed action"
            )

    return result
