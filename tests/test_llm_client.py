
import pytest

from src.llm_client import (
    LLMProviderError,
    PermanentLLMError,
    TransientLLMError,
)


@pytest.mark.parametrize(
    "error_type",
    [TransientLLMError, PermanentLLMError],
)
def test_provider_errors_share_common_base(error_type):
    error = error_type("Simulated provider failure")

    assert isinstance(error, LLMProviderError)


def test_transient_and_permanent_errors_are_distinct():
    transient = TransientLLMError("Temporary failure")
    permanent = PermanentLLMError("Invalid credentials")

    assert not isinstance(transient, PermanentLLMError)
    assert not isinstance(permanent, TransientLLMError)
