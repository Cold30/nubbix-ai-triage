
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
    r"\b(?:el|tu) reembolso (?:ya )?(?:fue|ha sido) (?:procesado|emitido|aprobado)\b",
]

DIRECT_NEGATION_PATTERN = re.compile(
    r"\b(?:no|nunca|jamás)(?:\s+(?:lo|la|los|las|se|te|le|les|ya|todavía|aún))*\s*$"
)


class GuardrailError(ValueError):
    """Raised when a support request violates a guardrail."""


def validate_input(query: str) -> str:
    """Validate a customer support request before LLM generation."""
    cleaned_query = query.strip()

    if not cleaned_query:
        raise GuardrailError("La consulta de soporte no puede estar vacía.")

    if len(cleaned_query) > 5000:
        raise GuardrailError("La consulta de soporte supera los 5000 caracteres.")
    
    if detect_prompt_injection(cleaned_query):
        raise GuardrailError("Se detectó un posible intento de prompt injection.")

    return cleaned_query

def validate_output(result: TriageResponse) -> TriageResponse:
    """Reject responses claiming unverified completed operations."""
    for text in [result.answer, *result.actions]:
        normalized = " ".join(text.lower().split())
        for pattern in UNVERIFIED_ACTION_PATTERNS:
            for match in re.finditer(pattern, normalized):
                # Only direct negation applies; a separate affirmative claim still fails.
                if DIRECT_NEGATION_PATTERN.search(normalized[:match.start()]):
                    continue
                raise GuardrailError(
                    "La respuesta afirma una operación realizada que no se puede verificar."
                )

    return result
