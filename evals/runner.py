import os
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from evals.scoring import calculate_scores
from src.guardrails import GuardrailError
from src.provider_factory import create_provider
from src.triage import TriageService
from src.metrics import MetricsLogger

def save_checkpoint( results: list[dict], output_path: Path, prompt_version: str, ) -> None:
    """Save partial evaluation results to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "prompt_version": prompt_version,
        "results": results,
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
def load_checkpoint( output_path: Path, prompt_version: str, ) -> list[dict]:
    """Load previously completed evaluation results."""
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    if checkpoint["prompt_version"] != prompt_version:
        raise ValueError("Checkpoint prompt version does not match")

    return checkpoint["results"]

def save_checkpoint( results: list[dict], output_path: Path, prompt_version: str, ) -> None: 
    """Save completed evaluation results after each ticket."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "prompt_version": prompt_version,
        "results": results,
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def load_dataset(path: Path) -> list[dict]:
    """Load labeled support tickets from a JSONL dataset."""
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def run_evaluation( service: TriageService, dataset: list[dict], prompt_version: str, output_path: Path, delay_seconds: float = 0.0, ) -> dict:
    """Evaluate classification quality against labeled tickets."""
    results = load_checkpoint(output_path, prompt_version)
    completed_ids = {result["id"] for result in results}

    for index, ticket in enumerate(dataset, start=1):
        if ticket["id"] in completed_ids:
            print( f"Skipping completed ticket: {ticket['id']}", file=sys.stderr, )
            continue
        print( f"Evaluating {index}/{len(dataset)}: {ticket['id']}", file=sys.stderr, )
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            prediction = service.classify(
                query=ticket["query"],
                prompt_version=prompt_version,
            )

            predicted_category = prediction.category
            error_message = None

        except (ValidationError, GuardrailError) as error:
            predicted_category = None
            error_message = str(error)

        results.append({
            "id": ticket["id"],
            "expected_category": ticket["expected_category"],
            "predicted_category": predicted_category,
            "correct": predicted_category == ticket["expected_category"],
            "error": error_message,
        })
        save_checkpoint(results, output_path, prompt_version)
    return {
        "prompt_version": prompt_version,
        "scores": calculate_scores(results),
        "results": results,
    }


def main() -> None:
    """Run the Nubbix evaluation CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate Nubbix support ticket classification"
    )

    parser.add_argument(
        "--provider",
        choices=["gemini", "openrouter"],
        default="gemini",
    )

    parser.add_argument(
        "--prompt-version",
        choices=["triage_v1", "triage_v2"],
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/dataset.jsonl"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results.json"),
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    provider = create_provider(args.provider)
    
    metrics_logger = MetricsLogger(
        os.getenv("METRICS_PATH", "data/metrics.jsonl")
    )

    service = TriageService(
        provider=provider,
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        metrics_logger=metrics_logger,
    )

    report = run_evaluation(
        service=service,
        dataset=dataset,
        prompt_version=args.prompt_version,
        output_path=args.output,
        delay_seconds=6.0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report["scores"], indent=2, ensure_ascii=False))
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
