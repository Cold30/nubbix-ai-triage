
from collections import defaultdict


def calculate_scores(results: list[dict]) -> dict:
    """Calculate overall and per-category classification accuracy."""
    total = len(results)

    if total == 0:
        raise ValueError("Evaluation results cannot be empty")

    correct = sum(
        result["expected_category"] == result["predicted_category"]
        for result in results
    )

    category_stats = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for result in results:
        category = result["expected_category"]
        category_stats[category]["total"] += 1

        if result["predicted_category"] == category:
            category_stats[category]["correct"] += 1

    per_category = {
        category: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"],
        }
        for category, stats in category_stats.items()
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "per_category": per_category,
    }
