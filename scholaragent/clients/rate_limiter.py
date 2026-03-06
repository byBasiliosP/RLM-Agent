"""Sliding-window rate limiter for API calls."""

from __future__ import annotations

import time
import threading
from collections import deque

PROVIDER_DEFAULTS: dict[str, dict[str, int]] = {
    "openai": {"rpm": 60, "tpm": 90_000},
    "anthropic": {"rpm": 50, "tpm": 80_000},
}

WINDOW_SECONDS = 60.0


class RateLimiter:
    """Thread-safe sliding-window rate limiter with RPM and TPM limits."""

    def __init__(self, rpm: int = 60, tpm: int = 90_000):
        self.rpm = rpm
        self.tpm = tpm
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._token_records: deque[tuple[float, int]] = deque()

    def _prune_window(self, now: float) -> None:
        """Remove entries older than WINDOW_SECONDS."""
        cutoff = now - WINDOW_SECONDS
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        while self._token_records and self._token_records[0][0] < cutoff:
            self._token_records.popleft()

    def _current_window_tokens(self) -> int:
        """Sum tokens in the current sliding window."""
        now = time.monotonic()
        with self._lock:
            self._prune_window(now)
            return sum(count for _, count in self._token_records)

    def wait_if_needed(self) -> None:
        """Block until under RPM limit."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune_window(now)
                if len(self._request_times) < self.rpm:
                    self._request_times.append(now)
                    return
                # Calculate wait time until oldest request exits window
                wait_until = self._request_times[0] + WINDOW_SECONDS
            sleep_time = max(0.01, wait_until - time.monotonic())
            time.sleep(sleep_time)

    def record_tokens(self, count: int) -> None:
        """Record token usage in the sliding window."""
        with self._lock:
            self._token_records.append((time.monotonic(), count))

    def wait_for_tokens(self, estimated: int) -> None:
        """Block if adding estimated tokens would exceed TPM."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune_window(now)
                current = sum(count for _, count in self._token_records)
                if current + estimated <= self.tpm:
                    return
                if self._token_records:
                    wait_until = self._token_records[0][0] + WINDOW_SECONDS
                else:
                    return
            sleep_time = max(0.01, wait_until - time.monotonic())
            time.sleep(sleep_time)
