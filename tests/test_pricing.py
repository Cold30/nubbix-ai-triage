
import pytest

from src.pricing import estimate_cost_usd


def test_gemini_cost_estimation():
    cost = estimate_cost_usd(
        "gemini-3.5-flash-lite",
        input_tokens=168,
        output_tokens=157,
    )

    assert cost == pytest.approx(0.0004429)


def test_free_model_cost():
    cost = estimate_cost_usd(
        "nvidia/nemotron-3.5-lightning:free",
        input_tokens=56,
        output_tokens=317,
    )

    assert cost == 0.0


def test_unknown_model():
    cost = estimate_cost_usd(
        "unknown-model",
        input_tokens=100,
        output_tokens=100,
    )

    assert cost is None
