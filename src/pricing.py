MODEL_PRICING = {
    "gemini-3.5-flash-lite": {
        "input": 0.30,
        "output": 2.50,
    },
    "nvidia/nemotron-3.5-lightning:free": {
        "input": 0.0,
        "output": 0.0,
    },
}


def estimate_cost_usd( model: str, input_tokens: int, output_tokens: int, ) -> float | None:
    """Estimate generation cost using USD per million tokens."""
    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return None

    return (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
    ) / 1_000_000
