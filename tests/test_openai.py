
import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    """Run a manual integration test against the OpenAI API."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    if not model:
        raise ValueError("OPENAI_MODEL is not configured")

    prompt = (
        "Classify the following support ticket into exactly "
        "one category: billing, technical, account, or other. "
        "Return only the category name.\n\n"
        "Ticket: I was charged twice for my subscription."
    )

    with OpenAI(api_key=api_key) as client:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=100,
        )

    print("Provider: openai")
    print("Model:", response.model)
    print("Response:", response.choices[0].message.content)
    print("Usage:", response.usage)


if __name__ == "__main__":
    main()
