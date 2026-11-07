import json
from datetime import datetime, timezone
from pathlib import Path


class MetricsLogger:
    """Append timestamped metrics to a JSON Lines file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict) -> None:
        """Append one metrics record with a UTC timestamp."""
        record = {
            **metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
