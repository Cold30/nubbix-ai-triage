
from pathlib import Path

import yaml


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt_file(version: str) -> dict:
    path = PROMPTS_DIR / f"triage_{version}.yaml"

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_few_shot_preserves_zero_shot_instructions():
    v1 = load_prompt_file("v1")
    v2 = load_prompt_file("v2")

    assert v1["system"].strip() in v2["system"]
    assert v1["user_template"] == v2["user_template"]


def test_prompt_strategies_are_distinct():
    v1 = load_prompt_file("v1")
    v2 = load_prompt_file("v2")

    assert v1["strategy"] == "zero-shot"
    assert v2["strategy"] == "few-shot"
    assert "Ejemplo 1:" not in v1["system"]
    assert "Ejemplo 1:" in v2["system"]
