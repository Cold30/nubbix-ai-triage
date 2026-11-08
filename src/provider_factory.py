
from src.gemini_provider import GeminiProvider
from src.openrouter_provider import OpenRouterProvider
from src.openai_provider import OpenAIProvider
from src.configuration import GenerationSettings, OpenAISettings, required_environment


def create_provider(provider_name: str):
    """Create an LLM provider from environment configuration."""
    providers = {"gemini": GeminiProvider, "openrouter": OpenRouterProvider, "openai": OpenAIProvider}
    if provider_name not in providers:
        raise ValueError("Proveedor no compatible. Elegí gemini, openrouter u openai.")
    prefix = provider_name.upper()
    api_key = required_environment(f"{prefix}_API_KEY")
    model = required_environment(f"{prefix}_MODEL")
    settings = OpenAISettings.from_environment() if provider_name == "openai" else GenerationSettings.from_environment()
    return providers[provider_name](
        api_key=api_key,
        model=model,
        settings=settings,
    )
