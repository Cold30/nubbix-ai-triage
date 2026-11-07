"""Evaluate the deterministic input guardrail without loading credentials or SDKs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src import guardrails


class AdversarialCase(BaseModel):
    """A manually labeled attack or benign quoted support request."""

    model_config = ConfigDict(strict=True, extra="forbid")
    id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=5000)
    attack_type: Literal[
        "prompt_injection", "role_spoofing", "output_manipulation",
        "exfiltration", "jailbreak", "benign_quote",
    ]
    expected_block: bool
    rationale: str = Field(min_length=1)

    @field_validator("id", "query", "rationale")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Require meaningful labels and text."""
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


def load_cases(path: Path) -> list[dict]:
    """Load independently labeled cases and reject duplicates."""
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = AdversarialCase.model_validate_json(line).model_dump()
        except ValidationError:
            raise ValueError(f"Caso adversarial inválido en la línea {line_number}.") from None
        if case["id"] in seen:
            raise ValueError(f"ID adversarial duplicado en la línea {line_number}.")
        if case["expected_block"] == (case["attack_type"] == "benign_quote"):
            raise ValueError(f"Etiqueta adversarial inconsistente en la línea {line_number}.")
        seen.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError("El dataset adversarial está vacío.")
    return cases


def detection_scores(results: list[dict]) -> dict:
    """Calculate binary detection metrics with explicit zero-denominator handling."""
    tp = sum(row["expected_block"] and row["blocked"] for row in results)
    fn = sum(row["expected_block"] and not row["blocked"] for row in results)
    fp = sum(not row["expected_block"] and row["blocked"] for row in results)
    tn = sum(not row["expected_block"] and not row["blocked"] for row in results)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "total": len(results), "true_positives": tp, "false_negatives": fn,
        "false_positives": fp, "true_negatives": tn,
        "attack_detection_rate": ratio(tp, tp + fn),
        "false_negative_rate": ratio(fn, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "precision": ratio(tp, tp + fp),
        "accuracy": ratio(tp + tn, len(results)),
    }


def evaluate_cases(cases: list[dict]) -> dict:
    """Measure input rejection only; accepting an attack does not prove model compromise."""
    results = []
    for case in cases:
        try:
            guardrails.validate_input(case["query"])
            blocked = False
        except guardrails.GuardrailError:
            blocked = True
        results.append({
            **case, "blocked": blocked,
            "correct": blocked == case["expected_block"],
            "mitigation": "Rechazo antes de generar" if blocked else "Sin bloqueo de entrada",
        })
    dataset_bytes = json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "mode": "offline_input_guardrail",
        "dataset_hash": hashlib.sha256(dataset_bytes).hexdigest(),
        "guardrail_hash": hashlib.sha256(Path(guardrails.__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Mide el bloqueo de entrada; no mide jailbreaks exitosos contra un LLM.",
            "No evalúa el guardrail de salida ni demuestra resistencia completa a prompt injection.",
            "Las etiquetas son manuales y el conjunto es pequeño; regex puede omitir ataques y bloquear citas legítimas.",
        ],
        "scores": detection_scores(results),
        "per_attack_type": {
            attack_type: detection_scores([row for row in results if row["attack_type"] == attack_type])
            for attack_type in sorted({row["attack_type"] for row in results})
        },
        "results": results,
    }


def main() -> None:
    """Print an offline report and optionally save it without overwriting evidence."""
    parser = argparse.ArgumentParser(description="Evaluar guardrails adversariales sin llamadas a APIs")
    parser.add_argument("--dataset", type=Path, default=Path("evals/adversarial.jsonl"))
    parser.add_argument("--output", type=Path, help="Guardar en un archivo nuevo; no sobrescribe")
    args = parser.parse_args()
    report = evaluate_cases(load_cases(args.dataset))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            file.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
