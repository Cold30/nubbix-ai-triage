
import json
from pathlib import Path


class MetricsLogger:
    """Append structured metrics to a JSON Lines file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict) -> None:
        """Append one metrics record without overwriting existing data."""
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
