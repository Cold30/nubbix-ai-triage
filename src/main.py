
import argparse
import os

from dotenv import load_dotenv

from src.provider_factory import create_provider
from src.triage import TriageService
from src.metrics import MetricsLogger


def main():
    """Run the Nubbix customer support triage CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Nubbix AI Support Triage"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify a customer support ticket",
    )

    classify_parser.add_argument(
        "query",
        help="Customer support ticket text",
    )

    classify_parser.add_argument(
        "--provider",
        choices=["gemini", "openrouter"],
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
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            metrics_logger=metrics_logger,
        )

        result = service.classify(
            query=args.query,
            prompt_version=args.prompt_version,
        )

        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
