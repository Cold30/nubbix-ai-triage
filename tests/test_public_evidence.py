"""Check published prediction evidence without credentials or private source files."""

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads((ROOT / "docs/evidence/classification_runs_v1.json").read_text(encoding="utf-8"))
LABELS = {row["id"]: row["expected_category"] for row in (
    json.loads(line) for line in (ROOT / "evals/dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
)}


def test_public_evidence_has_exact_run_manifest():
    assert set(EVIDENCE) == {"schema_version", "scope", "runs"}
    assert EVIDENCE["schema_version"] == 1
    assert EVIDENCE["scope"] == "historical_classification_predictions_only"
    assert [run["run_id"] for run in EVIDENCE["runs"]] == [
        "gemini_50_v1", "gemini_50_v2", "openrouter_50_v1", "openrouter_50_v2",
    ]


@pytest.mark.parametrize("run", EVIDENCE["runs"], ids=lambda run: run["run_id"])
def test_predictions_labels_counts_and_sanitization(run):
    common = {"run_id", "status", "source_sha256", "experiment", "expected_total",
              "recorded_total", "correct_recorded", "results"}
    extra = {"accuracy", "per_category"} if run["status"] == "completed" else {
        "pending_total", "next_pending_id", "reported_incident",
    }
    assert set(run) == common | extra
    assert re.fullmatch(r"[0-9a-f]{64}", run["source_sha256"])
    experiment = run["experiment"]
    assert set(experiment) == {"identity_version", "provider", "model", "prompt_version",
                               "prompt_hash", "dataset_hash", "generation", "request_policy"}
    assert set(experiment["generation"]) == {"temperature", "max_output_tokens"}
    assert set(experiment["request_policy"]) == {"timeout_s", "sdk_retries", "transient_retries", "validation_retries"}
    assert experiment["provider"] in {"gemini", "openrouter"}
    assert experiment["model"] in {"gemini-3.5-flash-lite", "nvidia/nemotron-3.5-lightning:free"}
    assert experiment["prompt_version"] in {"triage_v1", "triage_v2"}
    for key in ("dataset_hash", "prompt_hash"):
        assert re.fullmatch(r"[0-9a-f]{64}", experiment[key])
    rows = run["results"]
    assert len({row["id"] for row in rows}) == len(rows) == run["recorded_total"]
    for row in rows:
        assert set(row) == {"id", "expected_category", "predicted_category"}
        assert row["expected_category"] == LABELS[row["id"]]
        assert row["predicted_category"] in {"billing", "technical", "account", "other"}
    assert run["expected_total"] == len(LABELS) == 50
    assert run["correct_recorded"] == sum(row["expected_category"] == row["predicted_category"] for row in rows)
    if run["status"] == "completed":
        assert {row["id"] for row in rows} == set(LABELS)
        assert run["correct_recorded"] == 47
        assert run["accuracy"] == 47 / 50
        assert set(run["per_category"]) == set(LABELS.values())
        for category, scores in run["per_category"].items():
            assert scores == {
                "correct": sum(row["expected_category"] == category == row["predicted_category"] for row in rows),
                "total": sum(row["expected_category"] == category for row in rows),
            }
    else:
        assert [row["id"] for row in rows] == [f"T{number:03}" for number in range(1, 30)]
        assert run["recorded_total"] == run["correct_recorded"] == 29
        assert run["pending_total"] == 21
        assert run["next_pending_id"] == "T030"
        assert run["reported_incident"] == {
            "type": "HTTP 429", "source": "user_reported", "recorded_in_checkpoint": False,
        }


def test_generation_identity_and_observed_mistakes():
    first, second, third, partial = EVIDENCE["runs"]
    assert first["experiment"]["dataset_hash"] == second["experiment"]["dataset_hash"] == third["experiment"]["dataset_hash"] == partial["experiment"]["dataset_hash"]
    for before, after in ((first, second), (third, partial)):
        assert {key: value for key, value in before["experiment"].items() if key not in {"prompt_version", "prompt_hash"}} == {
            key: value for key, value in after["experiment"].items() if key not in {"prompt_version", "prompt_hash"}
        }
    for run in (first, second, third):
        assert {row["id"] for row in run["results"] if row["expected_category"] != row["predicted_category"]} == {"T045", "T049", "T050"}
