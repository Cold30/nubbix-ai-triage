
import pytest

from src.llm_client import PermanentLLMError, TransientLLMError
from src.retry import retry_transient


def test_retry_succeeds_after_transient_failure(monkeypatch):
    delays = []
    monkeypatch.setattr("src.retry.time.sleep", delays.append)

    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise TransientLLMError("Temporary failure")

        return "success"

    result = retry_transient(operation, max_retries=2)

    assert result == "success"
    assert attempts == 3
    assert delays == [1.0, 2.0]


def test_retry_does_not_repeat_permanent_errors(monkeypatch):
    monkeypatch.setattr(
        "src.retry.time.sleep",
        lambda _: pytest.fail("Unexpected sleep"),
    )

    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise PermanentLLMError("Invalid credentials")

    with pytest.raises(PermanentLLMError):
        retry_transient(operation)

    assert attempts == 1


def test_retry_stops_after_limit(monkeypatch):
    delays = []
    monkeypatch.setattr("src.retry.time.sleep", delays.append)

    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise TransientLLMError("Service unavailable")

    with pytest.raises(TransientLLMError):
        retry_transient(operation, max_retries=2)

    assert attempts == 3
    assert delays == [1.0, 2.0]
