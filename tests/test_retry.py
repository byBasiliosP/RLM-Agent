"""Tests for retry_with_backoff utility."""

from unittest.mock import patch

import pytest

from scholaragent.utils.retry import retry_with_backoff


@patch("scholaragent.utils.retry.time.sleep")
def test_succeeds_on_first_try(mock_sleep):
    """No retry needed when function succeeds immediately."""
    result = retry_with_backoff(lambda: 42, max_retries=3, base_delay=1.0)
    assert result == 42
    mock_sleep.assert_not_called()


@patch("scholaragent.utils.retry.time.sleep")
def test_retries_then_succeeds(mock_sleep):
    """Should retry on failure and return result on eventual success."""
    call_count = 0

    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return "ok"

    result = retry_with_backoff(
        flaky,
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "ok"
    assert call_count == 3
    assert mock_sleep.call_count == 2


@patch("scholaragent.utils.retry.time.sleep")
def test_raises_after_max_retries(mock_sleep):
    """Should raise the last exception after all retries are exhausted."""

    def always_fail():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError, match="down"):
        retry_with_backoff(
            always_fail,
            max_retries=2,
            base_delay=1.0,
            retryable_exceptions=(ConnectionError,),
        )
    # 1 initial + 2 retries = 3 calls, 2 sleeps
    assert mock_sleep.call_count == 2


@patch("scholaragent.utils.retry.time.sleep")
def test_only_retries_specified_exceptions(mock_sleep):
    """Should not retry on exception types not in retryable_exceptions."""

    def raise_value_error():
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        retry_with_backoff(
            raise_value_error,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(ConnectionError,),
        )
    mock_sleep.assert_not_called()


@patch("scholaragent.utils.retry.time.sleep")
def test_exponential_backoff_delays(mock_sleep):
    """Delays should double each retry: base, base*2, base*4, ..."""
    call_count = 0

    def fail_three_times():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ConnectionError("fail")
        return "done"

    result = retry_with_backoff(
        fail_three_times,
        max_retries=3,
        base_delay=2.0,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "done"
    delays = [c.args[0] for c in mock_sleep.call_args_list]
    assert delays == [2.0, 4.0, 8.0]


@patch("scholaragent.utils.retry.time.sleep")
def test_delay_capped_at_max_delay(mock_sleep):
    """Delay should not exceed max_delay."""
    call_count = 0

    def fail_four_times():
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            raise ConnectionError("fail")
        return "done"

    result = retry_with_backoff(
        fail_four_times,
        max_retries=5,
        base_delay=5.0,
        max_delay=15.0,
        retryable_exceptions=(ConnectionError,),
    )
    assert result == "done"
    delays = [c.args[0] for c in mock_sleep.call_args_list]
    # 5, 10, 15 (capped), 15 (capped)
    assert delays == [5.0, 10.0, 15.0, 15.0]


@patch("scholaragent.utils.retry.time.sleep")
def test_passes_args_and_kwargs(mock_sleep):
    """Should forward positional and keyword arguments to the function."""

    def add(a, b, extra=0):
        return a + b + extra

    result = retry_with_backoff(add, 1, 2, extra=10, max_retries=1)
    assert result == 13
