
from pathlib import Path

import yaml


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(version: str) -> dict:
    """Load a versioned prompt configuration from YAML."""
    prompt_path = PROMPTS_DIR / f"{version}.yaml"

    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt version not found: {version}"
        )

    with prompt_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prompt(version: str, query: str) -> str:
    """Build a prompt using the selected template and customer query."""
    config = load_prompt(version)

    system_prompt = config["system"]
    user_prompt = config["user_template"].format(query=query)

    return f"{system_prompt}\n\n{user_prompt}"
