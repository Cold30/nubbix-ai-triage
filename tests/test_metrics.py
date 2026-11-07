import json
from datetime import datetime, timezone

from src.metrics import MetricsLogger


def test_metrics_logger_appends_records(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(str(metrics_path))

    logger.log({
        "provider": "gemini",
        "total_tokens": 100,
    })

    logger.log({
        "provider": "openrouter",
        "total_tokens": 200,
    })

    lines = metrics_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    first_record = json.loads(lines[0])
    second_record = json.loads(lines[1])

    assert first_record["provider"] == "gemini"
    assert second_record["total_tokens"] == 200

    for record in (first_record, second_record):
        assert "timestamp" in record

        timestamp = datetime.fromisoformat(record["timestamp"])

        assert timestamp.tzinfo is not None
        assert timestamp.utcoffset().total_seconds() == 0


def test_metrics_logger_does_not_mutate_input(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(str(metrics_path))

    metrics = {
        "provider": "gemini",
        "total_tokens": 100,
    }

    logger.log(metrics)

    assert "timestamp" not in metrics