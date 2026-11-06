import json

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
    assert json.loads(lines[0])["provider"] == "gemini"
    assert json.loads(lines[1])["total_tokens"] == 200
