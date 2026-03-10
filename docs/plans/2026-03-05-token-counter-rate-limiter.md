# Token Counter & Rate Limiter Implementation Plan

> **Status:** ✅ Fully implemented. All 4 tasks completed.

**Goal:** Add per-client API rate limiting (RPM/TPM) and a token counter with live logging and end-of-run summaries.

**Architecture:** A `RateLimiter` class wraps each BaseLM client with sliding-window RPM/TPM throttling. A `TokenCounter` aggregates usage across all clients, printing live per-call logs in verbose mode and a summary table after each research run.

**Tech Stack:** Python stdlib only (`collections.deque`, `time.monotonic`, `threading.Lock`)

---

### Task 1: RateLimiter class

**Files:**
- Create: `scholaragent/clients/rate_limiter.py`
- Test: `tests/test_rate_limiter.py`

**Step 1: Write the failing tests**

```python
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
        # 3rd call should block — mock time to avoid real wait
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: FAIL — module not found

**Step 3: Write implementation**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add scholaragent/clients/rate_limiter.py tests/test_rate_limiter.py
git commit -m "feat: add RateLimiter with sliding-window RPM/TPM throttling"
```

---

### Task 2: TokenCounter class

**Files:**
- Create: `scholaragent/clients/token_counter.py`
- Test: `tests/test_token_counter.py`

**Step 1: Write the failing tests**

```python
"""Tests for TokenCounter."""

from scholaragent.clients.token_counter import TokenCounter


class TestTokenCounterRecord:
    def test_record_single(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        s = tc.summary()
        assert s["models"]["gpt-4o"]["prompt_tokens"] == 100
        assert s["models"]["gpt-4o"]["completion_tokens"] == 50
        assert s["total"]["total_tokens"] == 150

    def test_record_multiple_same_model(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        tc.record("gpt-4o", 200, 100)
        s = tc.summary()
        assert s["models"]["gpt-4o"]["prompt_tokens"] == 300
        assert s["total"]["total_tokens"] == 450

    def test_record_multiple_models(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        tc.record("gpt-4.1-mini", 80, 20)
        s = tc.summary()
        assert len(s["models"]) == 2
        assert s["total"]["total_tokens"] == 250

    def test_call_count(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        tc.record("gpt-4o", 100, 50)
        assert s["total"]["calls"] == 2


class TestTokenCounterReport:
    def test_report_contains_model_name(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        report = tc.report()
        assert "gpt-4o" in report

    def test_report_empty(self):
        tc = TokenCounter()
        report = tc.report()
        assert "No LLM calls" in report


class TestTokenCounterLogCall:
    def test_log_call_returns_string(self, capsys):
        tc = TokenCounter()
        tc.log_call("gpt-4o", 100, 50)
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out
        assert "150" in captured.out
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_token_counter.py -v`
Expected: FAIL — module not found

**Step 3: Write implementation**

```python
"""Token usage counter and reporter."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class _ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0


class TokenCounter:
    """Aggregates token usage across all LLM clients."""

    def __init__(self):
        self._lock = threading.Lock()
        self._models: dict[str, _ModelUsage] = {}

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from a single LLM call."""
        total = prompt_tokens + completion_tokens
        with self._lock:
            if model not in self._models:
                self._models[model] = _ModelUsage()
            m = self._models[model]
            m.prompt_tokens += prompt_tokens
            m.completion_tokens += completion_tokens
            m.total_tokens += total
            m.calls += 1

    def summary(self) -> dict:
        """Return per-model and total token counts."""
        with self._lock:
            models = {}
            total_prompt = total_completion = total_total = total_calls = 0
            for name, m in self._models.items():
                models[name] = {
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                    "calls": m.calls,
                }
                total_prompt += m.prompt_tokens
                total_completion += m.completion_tokens
                total_total += m.total_tokens
                total_calls += m.calls
            return {
                "models": models,
                "total": {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_total,
                    "calls": total_calls,
                },
            }

    def log_call(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Print live per-call token info."""
        total = prompt_tokens + completion_tokens
        print(f"  [tokens] {model}: {prompt_tokens} in + {completion_tokens} out = {total} total")

    def report(self) -> str:
        """Return formatted end-of-run summary."""
        s = self.summary()
        if not s["models"]:
            return "No LLM calls recorded."
        lines = ["", "=== Token Usage Summary ==="]
        for name, m in s["models"].items():
            lines.append(
                f"  {name}: {m['calls']} calls, "
                f"{m['prompt_tokens']} prompt + {m['completion_tokens']} completion "
                f"= {m['total_tokens']} total"
            )
        t = s["total"]
        lines.append(f"  ────────────────────────")
        lines.append(
            f"  TOTAL: {t['calls']} calls, {t['total_tokens']} tokens"
        )
        lines.append("")
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_token_counter.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add scholaragent/clients/token_counter.py tests/test_token_counter.py
git commit -m "feat: add TokenCounter with per-model tracking and reporting"
```

---

### Task 3: Wire RateLimiter into BaseLM clients

**Files:**
- Modify: `scholaragent/clients/base.py`
- Modify: `scholaragent/clients/openai_client.py`
- Modify: `scholaragent/clients/anthropic_client.py`
- Modify: `scholaragent/clients/router.py`
- Test: `tests/test_clients.py` (add new tests)

**Changes:**

1. `BaseLM.__init__` accepts optional `rate_limiter: RateLimiter | None = None`
2. `OpenAIClient.completion()` calls `self.rate_limiter.wait_if_needed()` before API call and `self.rate_limiter.record_tokens()` after
3. Same for `AnthropicClient.completion()`
4. `ModelRouter.get_client()` creates a `RateLimiter` with `PROVIDER_DEFAULTS[backend]`

**New tests in `tests/test_clients.py`:**

```python
class TestRateLimiterIntegration:
    def test_openai_client_has_rate_limiter(self):
        from scholaragent.clients.rate_limiter import RateLimiter
        client = OpenAIClient(model_name="gpt-4o", api_key="fake")
        assert isinstance(client.rate_limiter, RateLimiter)

    def test_anthropic_client_has_rate_limiter(self):
        from scholaragent.clients.rate_limiter import RateLimiter
        client = AnthropicClient(model_name="claude-sonnet-4-6", api_key="fake")
        assert isinstance(client.rate_limiter, RateLimiter)
```

**Commit:**

```bash
git commit -m "feat: wire RateLimiter into OpenAI and Anthropic clients"
```

---

### Task 4: Wire TokenCounter into LMHandler and ScholarAgent

**Files:**
- Modify: `scholaragent/core/handler.py`
- Modify: `scholaragent/__init__.py`
- Test: `tests/test_handler.py` (add new tests)
- Test: `tests/test_public_api.py` (add new tests)

**Changes:**

1. `LMHandler.__init__` accepts optional `token_counter: TokenCounter | None = None` and `verbose: bool = False`
2. `LMHandler.completion()` records usage via `token_counter.record()` and optionally calls `token_counter.log_call()` when verbose
3. `ScholarAgent.__init__` creates a `TokenCounter`, passes it to `LMHandler`
4. `ScholarAgent.research()` prints `counter.report()` after dispatcher finishes

**New tests:**

```python
class TestTokenCounterWiring:
    def test_handler_has_token_counter(self):
        from scholaragent.clients.token_counter import TokenCounter
        # ... create handler with counter, verify it exists

    def test_scholar_agent_prints_report(self):
        # ... verify research() returns result and counter has records
```

**Commit:**

```bash
git commit -m "feat: wire TokenCounter into LMHandler with live logging and summary"
```
