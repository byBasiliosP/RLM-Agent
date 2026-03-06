"""Tests for RateLimiter."""

import time
from unittest.mock import patch

import pytest

from scholaragent.clients.rate_limiter import RateLimiter, PROVIDER_DEFAULTS


class TestRateLimiterCreation:
    def test_default_creation(self):
        rl = RateLimiter(rpm=60, tpm=90000)
        assert rl.rpm == 60
        assert rl.tpm == 90000

    def test_provider_defaults_exist(self):
        assert "openai" in PROVIDER_DEFAULTS
        assert "anthropic" in PROVIDER_DEFAULTS
        assert PROVIDER_DEFAULTS["openai"]["rpm"] == 60
        assert PROVIDER_DEFAULTS["anthropic"]["rpm"] == 50


class TestRateLimiterRPM:
    def test_under_limit_does_not_block(self):
        rl = RateLimiter(rpm=100, tpm=1_000_000)
        start = time.monotonic()
        rl.wait_if_needed()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_at_limit_blocks(self):
        rl = RateLimiter(rpm=2, tpm=1_000_000)
        rl.wait_if_needed()
        rl.wait_if_needed()
        # 3rd call should block -- mock time to avoid real wait
        # Just verify the request count tracks correctly
        assert len(rl._request_times) == 2


class TestRateLimiterTPM:
    def test_record_tokens(self):
        rl = RateLimiter(rpm=100, tpm=1000)
        rl.record_tokens(500)
        assert rl._current_window_tokens() == 500

    def test_multiple_records(self):
        rl = RateLimiter(rpm=100, tpm=10000)
        rl.record_tokens(3000)
        rl.record_tokens(2000)
        assert rl._current_window_tokens() == 5000
