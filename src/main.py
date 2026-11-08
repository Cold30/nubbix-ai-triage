
import argparse
import os
import sys

from dotenv import load_dotenv

from src.provider_factory import create_provider
from src.triage import TriageService
from src.metrics import MetricsLogger
from src.configuration import environment_integer
from src.guardrails import GuardrailError
from src.llm_client import LLMProviderError
from pydantic import ValidationError


def main():
    """Run the Nubbix customer support triage CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Clasificación de tickets de soporte de Nubbix"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    classify_parser = subparsers.add_parser(
        "classify",
        help="Clasificar un ticket de soporte",
    )

    classify_parser.add_argument(
        "query",
        help="Texto de la consulta de soporte",
    )

    classify_parser.add_argument(
        "--provider",
        choices=["gemini", "openrouter", "openai"],
        default=os.getenv("LLM_PROVIDER", "gemini"),
    )

    classify_parser.add_argument(
        "--prompt-version",
        default=os.getenv("PROMPT_VERSION", "triage_v1"),
    )

    args = parser.parse_args()

    if args.command == "classify":
        provider = create_provider(args.provider)

        metrics_logger = MetricsLogger(
            os.getenv("METRICS_PATH", "data/metrics.jsonl")
        )

        service = TriageService(
            provider=provider,
            max_retries=environment_integer("LLM_MAX_RETRIES", 2),
            metrics_logger=metrics_logger,
        )

        result = service.classify(
            query=args.query,
            prompt_version=args.prompt_version,
        )

        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValidationError:
        print("Error: el proveedor no devolvió un JSON válido tras los reintentos.", file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, OSError, LLMProviderError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
