"""Validate environment settings without displaying credential values."""

import math
import os
from dataclasses import dataclass


class ConfigurationError(ValueError):
    """A required setting is missing or invalid."""


def required_environment(name: str) -> str:
    """Read a nonempty environment variable without echoing its value."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"Configurá la variable {name}; no puede estar vacía.")
    return value


def environment_integer(name: str, default: int, minimum: int = 0) -> int:
    """Read an integer setting with an inclusive lower bound."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        raise ConfigurationError(f"{name} debe ser un entero mayor o igual a {minimum}.") from None
    if value < minimum:
        raise ConfigurationError(f"{name} debe ser un entero mayor o igual a {minimum}.")
    return value


@dataclass(frozen=True)
class GenerationSettings:
    """Settings shared by the supported synchronous providers."""

    temperature: float = 0.1
    max_output_tokens: int = 512
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ConfigurationError("LLM_TEMPERATURE debe ser un número finito entre 0 y 2.")
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int) or self.max_output_tokens < 1:
            raise ConfigurationError("LLM_MAX_OUTPUT_TOKENS debe ser un entero positivo.")
        if not math.isfinite(self.timeout_s) or self.timeout_s < 0.001:
            raise ConfigurationError("LLM_TIMEOUT_SECONDS debe ser finito y mayor o igual a 0.001.")

    @classmethod
    def from_environment(cls) -> "GenerationSettings":
        """Read and validate the advertised generation settings."""
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        except ValueError:
            raise ConfigurationError("LLM_TEMPERATURE debe ser un número entre 0 y 2.") from None
        try:
            timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        except ValueError:
            raise ConfigurationError("LLM_TIMEOUT_SECONDS debe ser un número positivo.") from None
        return cls(
            temperature=temperature,
            max_output_tokens=environment_integer("LLM_MAX_OUTPUT_TOKENS", 512, 1),
            timeout_s=timeout_s,
        )

    def generation_parameters(self) -> dict:
        """Return only parameters sent to the generation endpoint."""
        return {"temperature": self.temperature, "max_output_tokens": self.max_output_tokens}
