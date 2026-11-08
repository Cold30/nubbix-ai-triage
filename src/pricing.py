MODEL_PRICING = {
    "gpt-6-luna": {
        "input": 0.10,
        "output": 0.50,
    },
    "gemini-3.5-flash-lite": {
        "input": 0.30,
        "output": 2.50,
    },
    "nvidia/nemotron-3.5-lightning:free": {
        "input": 0.0,
        "output": 0.0,
    },
}

# Official Standard text rates verified on 2026-11-07. Cache effects are excluded.
OPENAI_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-luna"
OPENAI_PRICING_VERIFIED_ON = "2026-11-07"


def estimate_cost_usd( model: str, input_tokens: int, output_tokens: int, ) -> float | None:
    """Estimate generation cost using USD per million tokens."""
    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return None

    long_context = model == "gpt-6-luna" and input_tokens > 272_000
    return (
        input_tokens * pricing["input"] * (2 if long_context else 1)
        + output_tokens * pricing["output"] * (1.5 if long_context else 1)
    ) / 1_000_000
