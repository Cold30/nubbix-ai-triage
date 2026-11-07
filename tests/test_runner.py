import pytest
from types import SimpleNamespace

from evals.runner import calculate_dataset_hash, run_evaluation, save_checkpoint
from evals.runner import build_experiment_identity


class FakeService:
    """Simulate ticket classification without calling an LLM."""

    def __init__(self):
        self.calls = []

    def classify(self, query: str, prompt_version: str):
        self.calls.append(query)
        return SimpleNamespace(category="technical")


def test_resume_skips_completed_tickets(tmp_path):
    output_path = tmp_path / "results.json"

    dataset = [
        {
            "id": "T001",
            "query": "My invoice is incorrect.",
            "expected_category": "billing",
        },
        {
            "id": "T002",
            "query": "The application is not responding.",
            "expected_category": "technical",
        },
    ]

    save_checkpoint(
        results=[
            {
                "id": "T001",
                "expected_category": "billing",
                "predicted_category": "billing",
                "correct": True,
                "error": None,
            }
        ],
        output_path=output_path,
        prompt_version="triage_v1",
    )

    service = FakeService()

    report = run_evaluation(
        service=service,
        dataset=dataset,
        prompt_version="triage_v1",
        output_path=output_path,
    )

    assert len(service.calls) == 1
    assert service.calls[0] == "The application is not responding."
    assert report["scores"]["total"] == 2
    assert report["scores"]["correct"] == 2

def test_resume_rejects_modified_dataset(tmp_path):
    output_path = tmp_path / "results.json"

    original_results = [
        {
            "id": "T001",
            "expected_category": "billing",
            "predicted_category": "billing",
            "correct": True,
            "error": None,
        }
    ]

    save_checkpoint(
        results=original_results,
        output_path=output_path,
        prompt_version="triage_v1",
    )

    modified_dataset = [
        {
            "id": "T001",
            "query": "The application crashes when opening reports.",
            "expected_category": "technical",
        }
    ]

    service = FakeService()

    with pytest.raises(ValueError, match="dataset"):
        run_evaluation(
            service=service,
            dataset=modified_dataset,
            prompt_version="triage_v1",
            output_path=output_path,
        )

    assert service.calls == []

def test_dataset_hash_is_stable():
    first = [
        {
            "id": "T001",
            "query": "Invoice issue",
            "expected_category": "billing",
        }
    ]

    same_data = [
        {
            "expected_category": "billing",
            "query": "Invoice issue",
            "id": "T001",
        }
    ]

    assert calculate_dataset_hash(first) == calculate_dataset_hash(same_data)


def test_dataset_hash_changes_when_ticket_changes():
    original = [
        {
            "id": "T001",
            "query": "Invoice issue",
            "expected_category": "billing",
        }
    ]

    modified = [
        {
            "id": "T001",
            "query": "Application error",
            "expected_category": "technical",
        }
    ]

    assert calculate_dataset_hash(original) != calculate_dataset_hash(modified)

def test_experiment_identity_is_stable(tmp_path, monkeypatch):
    prompt_path = tmp_path / "triage_v1.yaml"
    prompt_path.write_text("system: Test prompt", encoding="utf-8")

    monkeypatch.setattr("evals.runner.PROMPTS_DIR", tmp_path)

    dataset = [
        {
            "id": "T001",
            "query": "Invoice issue",
            "expected_category": "billing",
        }
    ]

    first = build_experiment_identity(
        dataset=dataset,
        prompt_version="triage_v1",
        provider_name="gemini",
        model_name="test-model",
    )

    second = build_experiment_identity(
        dataset=dataset,
        prompt_version="triage_v1",
        provider_name="gemini",
        model_name="test-model",
    )

    assert first == second

def test_experiment_identity_changes_when_prompt_changes(tmp_path, monkeypatch):
    prompt_path = tmp_path / "triage_v1.yaml"
    prompt_path.write_text("system: Original prompt", encoding="utf-8")

    monkeypatch.setattr("evals.runner.PROMPTS_DIR", tmp_path)

    dataset = [
        {
            "id": "T001",
            "query": "Invoice issue",
            "expected_category": "billing",
        }
    ]

    original = build_experiment_identity(
        dataset, "triage_v1", "gemini", "test-model"
    )

    prompt_path.write_text("system: Updated prompt", encoding="utf-8")

    modified = build_experiment_identity(
        dataset, "triage_v1", "gemini", "test-model"
    )

    assert original["prompt_hash"] != modified["prompt_hash"]


def test_experiment_identity_changes_with_provider(tmp_path, monkeypatch):
    (tmp_path / "triage_v1.yaml").write_text(
        "system: Test prompt",
        encoding="utf-8",
    )

    monkeypatch.setattr("evals.runner.PROMPTS_DIR", tmp_path)

    dataset = [
        {
            "id": "T001",
            "query": "Invoice issue",
            "expected_category": "billing",
        }
    ]

    gemini = build_experiment_identity(
        dataset, "triage_v1", "gemini", "test-model"
    )

    openrouter = build_experiment_identity(
        dataset, "triage_v1", "openrouter", "test-model"
    )

    assert gemini != openrouter
