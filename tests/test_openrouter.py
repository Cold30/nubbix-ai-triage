
import os

from dotenv import load_dotenv

from src.openrouter_provider import OpenRouterProvider


def main():
    """Run a manual integration test against OpenRouter."""
    load_dotenv()

    provider = OpenRouterProvider(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ["OPENROUTER_MODEL"],
    )

    prompt = (
        "You are a Nubbix support agent. "
        "Classify the following ticket as "
        "billing, technical, account, or other. "
        "Return only the category.\n\n"
        "Ticket: I was charged twice for my subscription."
    )

    response = provider.generate(prompt)
    print("Raw response:", repr(response.text))
    print("Normalized response:", response.text.strip())

    print("Response:", response.text)
    print("Provider:", response.provider)
    print("Model:", response.model)
    print("Input tokens:", response.input_tokens)
    print("Output tokens:", response.output_tokens)
    print("Total tokens:", response.total_tokens)
    print("Latency:", round(response.latency_s, 3), "seconds")


if __name__ == "__main__":
    main()
