"""Retry with exponential backoff for HTTP requests."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
    **kwargs,
) -> T:
    """Call fn with exponential backoff retry.

    Retries on exceptions matching retryable_exceptions.
    Delay doubles each retry: base_delay, base_delay*2, base_delay*4, ...
    Capped at max_delay.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}"
                )
                time.sleep(delay)
    raise last_exception
