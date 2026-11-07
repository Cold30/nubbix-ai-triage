import os
import argparse
import json
import sys
import time
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from evals.scoring import calculate_scores
from src.guardrails import GuardrailError
from src.provider_factory import create_provider
from src.triage import TriageService
from src.metrics import MetricsLogger
from src.prompting import PROMPTS_DIR

def save_checkpoint( results: list[dict], output_path: Path, experiment: dict, ) -> None:
    """Save evaluation results with their experiment identity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": experiment,
        "results": results,
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_checkpoint( output_path: Path, experiment: dict, ) -> list[dict]:
    """Load results only when the experiment identity matches."""
    if not output_path.exists():
        return []

    checkpoint = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    if checkpoint.get("experiment") != experiment:
        raise ValueError(
            "Checkpoint experiment does not match the current dataset, "
            "prompt, provider, or model"
        )

    return checkpoint["results"]


def load_dataset(path: Path) -> list[dict]:
    """Load labeled support tickets from a JSONL dataset."""
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def calculate_dataset_hash(dataset: list[dict]) -> str:
    """Return a stable SHA-256 fingerprint of a labeled dataset."""
    serialized = json.dumps(
        dataset,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

def build_experiment_identity( dataset: list[dict], prompt_version: str, provider_name: str, model_name: str, ) -> dict:
    """Build an identity for a reproducible evaluation."""
    prompt_path = PROMPTS_DIR / f"{prompt_version}.yaml"
    prompt_content = prompt_path.read_bytes()

    prompt_hash = hashlib.sha256(prompt_content).hexdigest()

    return {
        "provider": provider_name,
        "model": model_name,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "dataset_hash": calculate_dataset_hash(dataset),
    }


def run_evaluation( service: TriageService, dataset: list[dict], prompt_version: str, output_path: Path, experiment: dict, delay_seconds: float = 0.0,  resume: bool = False, ) -> dict:
    """Evaluate classification quality against labeled tickets."""
    if resume:
        if not output_path.exists():
            raise FileNotFoundError(
                f"No checkpoint to resume: {output_path}"
            )

        results = load_checkpoint(output_path, experiment)
    else:
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing evaluation: {output_path}"
            )

        results = []

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
        save_checkpoint(results, output_path, experiment)
    return {
        "experiment": experiment,
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
    
    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new evaluation without overwriting existing results",
    )

    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing compatible evaluation",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    provider = create_provider(args.provider)
    
    experiment = build_experiment_identity(
        dataset=dataset,
        prompt_version=args.prompt_version,
        provider_name=args.provider,
        model_name=provider.model,
    )
    
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
        experiment=experiment,
        delay_seconds=6.0,
        resume=args.resume,
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
