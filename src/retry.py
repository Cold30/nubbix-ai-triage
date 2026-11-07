
import time
from typing import Callable, TypeVar

from src.llm_client import TransientLLMError


T = TypeVar("T")


def retry_transient( operation: Callable[[], T], max_retries: int = 2, base_delay: float = 1.0, ) -> T:
    """Retry an operation when a transient provider error occurs."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except TransientLLMError:
            if attempt == max_retries:
                raise

            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

    raise RuntimeError("Unexpected retry loop termination")
