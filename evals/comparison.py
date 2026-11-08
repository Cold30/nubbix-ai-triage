"""Audit saved classification results and response metrics without provider calls."""

from collections import Counter
from datetime import datetime
import hashlib
import json
import math

from evals.scoring import calculate_scores
from src.pricing import estimate_cost_usd


CATEGORIES = {"billing", "technical", "account", "other"}


def _number(value, name: str, *, integer: bool = False) -> float | int:
    """Reject missing, negative, nonfinite, and boolean numeric fields."""
    allowed = (int,) if integer else (int, float)
    if type(value) not in allowed or value < 0 or not math.isfinite(value):
        raise ValueError(f"Valor numérico inválido: {name}.")
    return value


def summarize_metrics(metrics: list[dict], experiment: dict) -> dict:
    """Summarize recorded responses, keeping missing costs distinct from zero."""
    for row in metrics:
        if any(row.get(key) != experiment[key] for key in ("provider", "model", "prompt_version")):
            raise ValueError("Las métricas no corresponden al proveedor, modelo o prompt del experimento.")
        for key in ("input_tokens", "output_tokens", "total_tokens", "attempt"):
            _number(row.get(key), key, integer=True)
        if row["attempt"] < 1:
            raise ValueError("El número de intento debe ser positivo.")
        _number(row.get("latency_s"), "latency_s")
        if row.get("estimated_cost_usd") is not None:
            _number(row["estimated_cost_usd"], "estimated_cost_usd")

    count = len(metrics)
    totals = {
        key: sum(row[key] for row in metrics)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    totals["latency_s"] = math.fsum(row["latency_s"] for row in metrics)
    known_costs = [row["estimated_cost_usd"] for row in metrics if row.get("estimated_cost_usd") is not None]
    totals["estimated_cost_usd"] = math.fsum(known_costs) if count and len(known_costs) == count else None
    timestamps = []
    for row in metrics:
        try:
            timestamp = datetime.fromisoformat(row["timestamp"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Marca temporal de métricas inválida.") from None
        if timestamp.utcoffset() is None:
            raise ValueError("Las marcas temporales deben incluir zona horaria.")
        timestamps.append(timestamp)

    cost_mismatches = 0
    unknown_price = 0
    for row in metrics:
        recomputed = estimate_cost_usd(row["model"], row["input_tokens"], row["output_tokens"])
        if recomputed is None:
            unknown_price += 1
        elif row.get("estimated_cost_usd") is not None and not math.isclose(
            row["estimated_cost_usd"], recomputed, rel_tol=1e-9, abs_tol=1e-12,
        ):
            cost_mismatches += 1

    return {
        "responses": count,
        "first_attempt_responses": sum(row["attempt"] == 1 for row in metrics),
        "validation_retry_responses": sum(row["attempt"] > 1 for row in metrics),
        "max_attempt": max((row["attempt"] for row in metrics), default=None),
        "totals": totals,
        "means": {key: value / count if count and value is not None else None for key, value in totals.items()},
        "unknown_cost_records": count - len(known_costs),
        "known_cost_subtotal_usd": math.fsum(known_costs),
        "cost_mismatches": cost_mismatches,
        "unknown_price_records": unknown_price,
        "token_balance_mismatches": sum(row["input_tokens"] + row["output_tokens"] != row["total_tokens"] for row in metrics),
        "timestamp_duplicates": len(timestamps) - len(set(timestamps)),
        "timestamps_monotonic": timestamps == sorted(timestamps),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
    }


def _validate_results(report: dict, expected: dict[str, str], dataset_hash: str) -> dict:
    """Recompute scores and verify labels, identifiers, and stored score fields."""
    experiment = report["experiment"]
    required = {"identity_version", "provider", "model", "prompt_version", "prompt_hash", "dataset_hash", "generation", "request_policy"}
    if not required <= experiment.keys() or experiment["dataset_hash"] != dataset_hash:
        raise ValueError("Identidad incompleta o hash del dataset incompatible.")
    if not {"temperature", "max_output_tokens"} <= experiment["generation"].keys():
        raise ValueError("Faltan parámetros de generación Gemini.")
    if not {"timeout_s", "sdk_retries", "transient_retries", "validation_retries"} <= experiment["request_policy"].keys():
        raise ValueError("Faltan parámetros de timeout o reintentos.")
    if report.get("prompt_version") != experiment["prompt_version"]:
        raise ValueError("La versión de prompt del reporte es inconsistente.")
    rows = report["results"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("Los resultados no contienen exactamente los IDs únicos del dataset.")
    for row in rows:
        predicted = row.get("predicted_category")
        if row["expected_category"] != expected[row["id"]]:
            raise ValueError("Una etiqueta esperada difiere del dataset.")
        if predicted is not None and predicted not in CATEGORIES:
            raise ValueError("Predicción de categoría inválida.")
        if type(row.get("correct")) is not bool or row["correct"] != (row["expected_category"] == predicted):
            raise ValueError("El indicador correct es inconsistente con la predicción.")
    scores = calculate_scores(rows)
    if report.get("scores") != scores:
        raise ValueError("Los scores guardados no coinciden con el recálculo.")
    return scores


def compare_runs(first_report: dict, first_metrics: list[dict], second_report: dict,
                 second_metrics: list[dict], dataset: list[dict], *, expected_count: int = 50) -> dict:
    """Compare Gemini v1/v2 using only saved artifacts and the frozen labeled dataset."""
    expected = {row["id"]: row["expected_category"] for row in dataset}
    if len(dataset) != expected_count or len(expected) != expected_count:
        raise ValueError("El dataset no tiene la cantidad esperada de IDs únicos.")
    if not set(expected.values()) <= CATEGORIES:
        raise ValueError("El dataset contiene categorías desconocidas.")
    canonical = json.dumps(dataset, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    dataset_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    reports = (first_report, second_report)
    summaries = []
    for report, metrics, version in zip(reports, (first_metrics, second_metrics), ("triage_v1", "triage_v2")):
        scores = _validate_results(report, expected, dataset_hash)
        experiment = report["experiment"]
        if experiment["provider"] != "gemini" or experiment["prompt_version"] != version:
            raise ValueError("Se esperaban Gemini triage_v1 y triage_v2, en ese orden.")
        metrics_summary = summarize_metrics(metrics, experiment)
        summaries.append({
            "scores": scores, "metrics": metrics_summary,
            "metric_count_difference": len(metrics) - len(expected),
            "reported_errors": sum(row.get("error") is not None for row in report["results"]),
            "unclassified": sum(row.get("predicted_category") is None for row in report["results"]),
        })
    left, right = (report["experiment"] for report in reports)
    differences = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
    if differences != ["prompt_hash", "prompt_version"]:
        raise ValueError("Las corridas difieren en parámetros ajenos al prompt o no cambian ambos identificadores de prompt.")
    predictions = [{row["id"]: row["predicted_category"] for row in report["results"]} for report in reports]
    errors = [{identifier for identifier in expected if prediction[identifier] != expected[identifier]} for prediction in predictions]
    deltas = {}
    for key, first_value in summaries[0]["metrics"]["totals"].items():
        second_value = summaries[1]["metrics"]["totals"][key]
        deltas[key] = {
            "absolute": second_value - first_value if first_value is not None and second_value is not None else None,
            "percent": (second_value / first_value - 1) * 100 if first_value and second_value is not None else None,
        }
    return {
        "dataset_hash": dataset_hash,
        "distribution": dict(Counter(expected.values())),
        "identity_differences": differences,
        "v1": summaries[0], "v2": summaries[1], "v2_minus_v1": deltas,
        "shared_errors": sorted(errors[0] & errors[1]),
        "v1_only_errors": sorted(errors[0] - errors[1]),
        "v2_only_errors": sorted(errors[1] - errors[0]),
        "prediction_disagreements": sorted(identifier for identifier in expected if predictions[0][identifier] != predictions[1][identifier]),
        "incorrect_tickets": [
            {"id": identifier, "expected_category": expected[identifier],
             "v1": predictions[0][identifier], "v2": predictions[1][identifier]}
            for identifier in sorted(errors[0] | errors[1])
        ],
    }
