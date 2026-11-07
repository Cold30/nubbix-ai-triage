
from src.gemini_provider import GeminiProvider
from src.openrouter_provider import OpenRouterProvider
from src.configuration import GenerationSettings, required_environment


def create_provider(provider_name: str):
    """Create an LLM provider from environment configuration."""
    providers = {"gemini": GeminiProvider, "openrouter": OpenRouterProvider}
    if provider_name not in providers:
        raise ValueError("Proveedor no compatible. Elegí gemini u openrouter.")
    prefix = provider_name.upper()
    api_key = required_environment(f"{prefix}_API_KEY")
    model = required_environment(f"{prefix}_MODEL")
    return providers[provider_name](
        api_key=api_key,
        model=model,
        settings=GenerationSettings.from_environment(),
    )
