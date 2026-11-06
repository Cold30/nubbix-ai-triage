
import pytest

from evals.scoring import calculate_scores


def test_calculate_scores():
    results = [
        {
            "expected_category": "billing",
            "predicted_category": "billing",
        },
        {
            "expected_category": "billing",
            "predicted_category": "other",
        },
        {
            "expected_category": "technical",
            "predicted_category": "technical",
        },
    ]

    scores = calculate_scores(results)

    assert scores["total"] == 3
    assert scores["correct"] == 2
    assert scores["accuracy"] == pytest.approx(2 / 3)
    assert scores["per_category"]["billing"]["accuracy"] == 0.5
    assert scores["per_category"]["technical"]["accuracy"] == 1.0
