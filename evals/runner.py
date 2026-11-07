import os
import argparse
import json
import sys
import time
import hashlib
import math
import tempfile
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from evals.scoring import calculate_scores
from src.guardrails import GuardrailError
from src.provider_factory import create_provider
from src.triage import TriageService
from src.metrics import MetricsLogger
from src.prompting import PROMPTS_DIR
from src.configuration import GenerationSettings, environment_integer
from src.llm_client import LLMProviderError


class DatasetTicket(BaseModel):
    """A labeled ticket accepted by the classification runner."""

    model_config = ConfigDict(strict=True, extra="forbid")
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_category: Literal["billing", "technical", "account", "other"]

    @field_validator("id", "query")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only fields without changing dataset contents."""
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


def validate_dataset(dataset: list[dict]) -> list[dict]:
    """Validate all rows and require unique ticket identifiers."""
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("El dataset debe contener al menos un ticket.")
    validated = []
    seen = set()
    for index, ticket in enumerate(dataset, start=1):
        try:
            row = DatasetTicket.model_validate(ticket).model_dump()
        except ValidationError:
            raise ValueError(f"Ticket inválido en la fila {index}: revisá id, query y expected_category.") from None
        if row["id"] in seen:
            raise ValueError(f"ID duplicado en la fila {index} del dataset.")
        seen.add(row["id"])
        validated.append(row)
    return validated


def atomic_write_report(report: dict, output_path: Path) -> None:
    """Replace a report only after its complete JSON has been flushed."""
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

def save_checkpoint( results: list[dict], output_path: Path, experiment: dict, ) -> None:
    """Save evaluation results with their experiment identity."""
    report = {
        "experiment": experiment,
        "results": results,
    }

    atomic_write_report(report, output_path)


def load_checkpoint( output_path: Path, experiment: dict, ) -> list[dict]:
    """Load results only when the experiment identity matches."""
    if not output_path.exists():
        return []

    try:
        checkpoint = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("El checkpoint está dañado; utilizá una copia válida o una salida nueva.") from None
    if not isinstance(checkpoint, dict):
        raise ValueError("El checkpoint debe ser un objeto JSON.")

    if checkpoint.get("experiment") != experiment:
        raise ValueError(
            "El experimento del checkpoint no coincide con dataset, prompt, proveedor, modelo o configuración."
        )

    results = checkpoint.get("results")
    if not isinstance(results, list):
        raise ValueError("El checkpoint no contiene una lista de resultados válida.")
    return results


def load_dataset(path: Path) -> list[dict]:
    """Load labeled support tickets from a JSONL dataset."""
    with path.open("r", encoding="utf-8") as file:
        rows = []
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"JSON inválido en la línea {line_number} del dataset.") from None
        return validate_dataset(rows)

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

def build_experiment_identity(
    dataset: list[dict], prompt_version: str, provider_name: str, model_name: str,
    *, settings: GenerationSettings | None = None, max_retries: int = 2,
) -> dict:
    """Build an identity for a reproducible evaluation."""
    prompt_path = PROMPTS_DIR / f"{prompt_version}.yaml"
    prompt_content = prompt_path.read_bytes()

    prompt_hash = hashlib.sha256(prompt_content).hexdigest()

    settings = settings or GenerationSettings()
    return {
        "identity_version": 2,
        "provider": provider_name,
        "model": model_name,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "dataset_hash": calculate_dataset_hash(dataset),
        "generation": settings.generation_parameters(),
        "request_policy": {
            "timeout_s": settings.timeout_s,
            "sdk_retries": 0,
            "transient_retries": 2,
            "validation_retries": max_retries,
        },
    }


def run_evaluation( service: TriageService, dataset: list[dict], prompt_version: str, output_path: Path, experiment: dict, delay_seconds: float = 0.0,  resume: bool = False, ) -> dict:
    """Evaluate classification quality against labeled tickets."""
    if resume:
        if not output_path.exists():
            raise FileNotFoundError(
                f"No existe un checkpoint para reanudar: {output_path}"
            )

        results = load_checkpoint(output_path, experiment)
    else:
        if output_path.exists():
            raise FileExistsError(
                f"La salida ya existe; usá --resume o elegí otro archivo: {output_path}"
            )

        results = []

    dataset = validate_dataset(dataset)
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("La pausa entre tickets debe ser finita y no negativa.")
    expected = {ticket["id"]: ticket["expected_category"] for ticket in dataset}
    completed_ids = set()
    for result in results:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("id"), str)
            or result["id"] not in expected
            or result["id"] in completed_ids
            or result.get("expected_category") != expected[result["id"]]
            or not isinstance(result.get("predicted_category"), (str, type(None)))
            or result.get("predicted_category") not in {"billing", "technical", "account", "other", None}
            or "predicted_category" not in result
            or type(result.get("correct")) is not bool
            or result["correct"] != (result["predicted_category"] == result["expected_category"])
            or "error" not in result
            or (result["error"] is not None and not isinstance(result["error"], str))
        ):
            raise ValueError("El checkpoint contiene resultados inválidos, duplicados o ajenos al dataset.")
        completed_ids.add(result["id"])

    if experiment.get("dataset_hash") != calculate_dataset_hash(dataset) or experiment.get("prompt_version") != prompt_version:
        raise ValueError("La identidad del experimento no corresponde al dataset o prompt solicitado.")

    for index, ticket in enumerate(dataset, start=1):
        if ticket["id"] in completed_ids:
            print(f"Omitiendo ticket completado: {ticket['id']}", file=sys.stderr)
            continue
        print(f"Evaluando {index}/{len(dataset)}: {ticket['id']}", file=sys.stderr)
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
            error_message = type(error).__name__

        results.append({
            "id": ticket["id"],
            "expected_category": ticket["expected_category"],
            "predicted_category": predicted_category,
            "correct": predicted_category == ticket["expected_category"],
            "error": error_message,
        })
        completed_ids.add(ticket["id"])
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
        description="Evaluar la clasificación de tickets de Nubbix"
    )

    parser.add_argument(
        "--provider",
        choices=["gemini", "openrouter"],
        default=os.getenv("LLM_PROVIDER", "gemini"),
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
        help="Iniciar una evaluación sin sobrescribir resultados existentes",
    )

    mode.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar una evaluación existente y compatible",
    )
    parser.add_argument("--delay-seconds", type=float, default=6.0, help="Pausa entre tickets en segundos")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    provider = create_provider(args.provider)
    max_retries = environment_integer("LLM_MAX_RETRIES", 2)
    
    experiment = build_experiment_identity(
        dataset=dataset,
        prompt_version=args.prompt_version,
        provider_name=args.provider,
        model_name=provider.model,
        settings=provider.settings,
        max_retries=max_retries,
    )
    
    metrics_logger = MetricsLogger(
        os.getenv("METRICS_PATH", "data/metrics.jsonl")
    )

    service = TriageService(
        provider=provider,
        max_retries=max_retries,
        metrics_logger=metrics_logger,
    )

    report = run_evaluation(
        service=service,
        dataset=dataset,
        prompt_version=args.prompt_version,
        output_path=args.output,
        experiment=experiment,
        delay_seconds=args.delay_seconds,
        resume=args.resume,
    )
    

    atomic_write_report(report, args.output)

    print(json.dumps(report["scores"], indent=2, ensure_ascii=False))
    print(f"\nReporte guardado en: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, LLMProviderError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
