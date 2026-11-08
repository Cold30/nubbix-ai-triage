"""Verify saved-report calculations with synthetic, offline-only fixtures."""

from copy import deepcopy
import hashlib
import json

import pytest

from evals.comparison import compare_runs, summarize_metrics
from evals.scoring import calculate_scores


@pytest.fixture
def artifacts():
    dataset = [
        {"id": f"T{index:03d}", "query": f"Consulta sintética {index}", "expected_category": category}
        for index, category in enumerate(("billing", "technical", "account", "other"), 1)
    ]
    fingerprint = hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    reports, metrics = [], []
    for version, predictions in enumerate((
        ("billing", "technical", "technical", "other"),
        ("billing", "technical", "account", "technical"),
    ), 1):
        experiment = {
            "identity_version": 2, "provider": "gemini", "model": "gemini-3.5-flash-lite",
            "prompt_version": f"triage_v{version}", "prompt_hash": str(version) * 64,
            "dataset_hash": fingerprint, "generation": {"temperature": 0.1, "max_output_tokens": 512},
            "request_policy": {"timeout_s": 30, "sdk_retries": 0, "transient_retries": 2, "validation_retries": 3},
        }
        rows = [
            {"id": ticket["id"], "expected_category": ticket["expected_category"],
             "predicted_category": predicted, "correct": predicted == ticket["expected_category"], "error": None}
            for ticket, predicted in zip(dataset, predictions)
        ]
        reports.append({"experiment": experiment, "prompt_version": experiment["prompt_version"],
                        "results": rows, "scores": calculate_scores(rows)})
        metrics.append([
            {"provider": "gemini", "model": experiment["model"], "prompt_version": experiment["prompt_version"],
             "attempt": 1, "input_tokens": 100 if version == 1 else 150,
             "output_tokens": 10 if version == 1 else 20, "total_tokens": 110 if version == 1 else 170,
             "latency_s": index + version, "estimated_cost_usd": 0.000055 if version == 1 else 0.000095,
             "timestamp": f"2026-11-08T00:00:0{index}+00:00"}
            for index in range(4)
        ])
    return reports[0], metrics[0], reports[1], metrics[1], dataset


def compare(artifacts):
    return compare_runs(*artifacts, expected_count=4)


def test_recalculates_scores_tokens_latency_cost_and_error_overlap(artifacts):
    result = compare(artifacts)
    assert result["v1"]["scores"]["accuracy"] == 0.75
    assert result["v1"]["scores"]["per_category"]["account"]["accuracy"] == 0
    assert result["v2"]["scores"]["per_category"]["account"]["accuracy"] == 1
    assert result["v1"]["metrics"]["totals"]["input_tokens"] == 400
    assert result["v1"]["metrics"]["means"]["latency_s"] == 2.5
    assert result["v1"]["metrics"]["totals"]["estimated_cost_usd"] == pytest.approx(0.00022)
    assert result["v2_minus_v1"]["input_tokens"]["percent"] == 50
    assert result["v2_minus_v1"]["estimated_cost_usd"]["absolute"] == pytest.approx(0.00016)
    assert result["v1_only_errors"] == ["T003"]
    assert result["v2_only_errors"] == ["T004"]
    assert result["shared_errors"] == []
    assert result["prediction_disagreements"] == ["T003", "T004"]
    assert result["v1"]["metrics"]["cost_mismatches"] == 0


def test_identifies_shared_failures(artifacts):
    first, _, second, _, _ = artifacts
    second["results"] = deepcopy(first["results"])
    second["scores"] = deepcopy(first["scores"])
    result = compare(artifacts)
    assert result["shared_errors"] == ["T003"]
    assert result["prediction_disagreements"] == []


@pytest.mark.parametrize("field", ["temperature", "max_output_tokens"])
def test_rejects_different_generation_parameters(artifacts, field):
    artifacts[2]["experiment"]["generation"][field] = 0.5 if field == "temperature" else 1024
    with pytest.raises(ValueError, match="parámetros ajenos"):
        compare(artifacts)


@pytest.mark.parametrize("field", ["timeout_s", "sdk_retries", "transient_retries", "validation_retries"])
def test_rejects_different_request_policy(artifacts, field):
    artifacts[2]["experiment"]["request_policy"][field] += 1
    with pytest.raises(ValueError, match="parámetros ajenos"):
        compare(artifacts)


def test_rejects_duplicate_result_ids(artifacts):
    artifacts[0]["results"][1]["id"] = "T001"
    with pytest.raises(ValueError, match="IDs únicos"):
        compare(artifacts)


def test_rejects_different_dataset_hash(artifacts):
    artifacts[2]["experiment"]["dataset_hash"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        compare(artifacts)


def test_rejects_changed_labels_and_stale_scores(artifacts):
    artifacts[0]["results"][0]["expected_category"] = "other"
    with pytest.raises(ValueError, match="etiqueta"):
        compare(artifacts)
    artifacts[0]["results"][0]["expected_category"] = "billing"
    artifacts[0]["scores"]["accuracy"] = 1
    with pytest.raises(ValueError, match="scores"):
        compare(artifacts)


def test_counts_missing_metrics_and_validation_retries(artifacts):
    artifacts[1].pop()
    artifacts[1][1]["attempt"] = 2
    result = compare(artifacts)
    assert result["v1"]["metric_count_difference"] == -1
    assert result["v1"]["metrics"]["validation_retry_responses"] == 1


def test_unknown_cost_is_not_reported_as_free(artifacts):
    artifacts[1][0]["estimated_cost_usd"] = None
    result = compare(artifacts)["v1"]["metrics"]
    assert result["unknown_cost_records"] == 1
    assert result["totals"]["estimated_cost_usd"] is None
    assert result["means"]["estimated_cost_usd"] is None
    assert result["known_cost_subtotal_usd"] == pytest.approx(0.000165)


@pytest.mark.parametrize("field", ["provider", "model", "prompt_version"])
def test_rejects_mixed_metric_sources(artifacts, field):
    artifacts[1][0][field] = "another-run"
    with pytest.raises(ValueError, match="métricas"):
        compare(artifacts)


@pytest.mark.parametrize(("field", "value"), [
    ("input_tokens", -1), ("output_tokens", True), ("latency_s", float("nan")),
    ("estimated_cost_usd", float("inf")), ("attempt", 0),
])
def test_rejects_invalid_numeric_metrics(artifacts, field, value):
    artifacts[1][0][field] = value
    with pytest.raises(ValueError):
        compare(artifacts)


def test_flags_unbalanced_tokens_and_inconsistent_costs(artifacts):
    artifacts[1][0]["total_tokens"] = 999
    artifacts[1][0]["estimated_cost_usd"] = 100
    summary = compare(artifacts)["v1"]["metrics"]
    assert summary["token_balance_mismatches"] == 1
    assert summary["cost_mismatches"] == 1


def test_summarizes_empty_metrics_without_division_by_zero(artifacts):
    summary = summarize_metrics([], artifacts[0]["experiment"])
    assert summary["responses"] == 0
    assert summary["means"]["latency_s"] is None
    assert summary["totals"]["estimated_cost_usd"] is None
