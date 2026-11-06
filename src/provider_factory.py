
import os

from src.gemini_provider import GeminiProvider
from src.openrouter_provider import OpenRouterProvider


def create_provider(provider_name: str):
    """Create an LLM provider from environment configuration."""
    if provider_name == "gemini":
        return GeminiProvider(
            api_key=os.environ["GEMINI_API_KEY"],
            model=os.environ["GEMINI_MODEL"],
        )

    if provider_name == "openrouter":
        return OpenRouterProvider(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=os.environ["OPENROUTER_MODEL"],
        )

    raise ValueError(f"Unsupported provider: {provider_name}")
