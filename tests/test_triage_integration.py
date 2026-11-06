
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from src.gemini_provider import GeminiProvider
from src.models import TriageResponse
from src.triage import TriageService
from src.prompting import build_prompt


def main():
    """Test the complete LLM-to-Pydantic validation flow."""
    load_dotenv()

    provider = GeminiProvider(
        api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ["GEMINI_MODEL"],
    )

    query = "Me cobraron dos veces la suscripción de este mes."
    service = TriageService(provider)

    result = service.classify(
        query=query,
        prompt_version="triage_v1",
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
