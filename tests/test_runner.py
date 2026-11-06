
from types import SimpleNamespace

from evals.runner import run_evaluation, save_checkpoint


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
