
import os

from dotenv import load_dotenv

from src.gemini_provider import GeminiProvider


def main():
    """Run a manual integration test against Gemini."""
    load_dotenv()

    provider = GeminiProvider(
        api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ["GEMINI_MODEL"],
    )

    prompt = (
        "You are a Nubbix support agent. "
        "Classify the following ticket as "
        "billing, technical, account, or other. "
        "Return only the category.\n\n"
        "Ticket: I cannot access my account."
    )

    response = provider.generate(prompt)

    print("Response:", response.text)
    print("Provider:", response.provider)
    print("Model:", response.model)
    print("Input tokens:", response.input_tokens)
    print("Output tokens:", response.output_tokens)
    print("Total tokens:", response.total_tokens)
    print("Latency:", response.latency_s)


if __name__ == "__main__":
    main()
