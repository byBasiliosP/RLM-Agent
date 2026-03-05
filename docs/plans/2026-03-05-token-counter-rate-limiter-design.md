# Token Counter & Rate Limiter Design

**Date:** 2026-03-05
**Status:** Approved

## Goal

Add per-client API rate limiting (RPM/TPM) and a token counter with live logging and end-of-run summaries.

## Components

### 1. RateLimiter (`scholaragent/clients/rate_limiter.py`)

Thread-safe sliding-window rate limiter with two dimensions:

- **RPM** (requests per minute) — sliding deque of timestamps
- **TPM** (tokens per minute) — sliding deque of (timestamp, token_count) pairs

Provider defaults:

| Provider   | RPM | TPM    |
|-----------|-----|--------|
| OpenAI    | 60  | 90,000 |
| Anthropic | 50  | 80,000 |

API:
- `RateLimiter(rpm, tpm)` — constructor with configurable limits
- `wait_if_needed()` — blocks until under RPM limit
- `record_tokens(count)` — records token usage for TPM window
- `wait_for_tokens(estimated)` — blocks if TPM would be exceeded

Uses `collections.deque`, `time.monotonic()`, `threading.Lock`. No external deps.

### 2. BaseLM Integration

Each client (`OpenAIClient`, `AnthropicClient`) accepts an optional `rate_limiter: RateLimiter` parameter. The `completion()` method:

1. Calls `rate_limiter.wait_if_needed()` before the API request
2. Makes the API call
3. Calls `rate_limiter.record_tokens(total_tokens)` after

### 3. TokenCounter (`scholaragent/clients/token_counter.py`)

Aggregates usage across all clients in a session:

- `record(model, prompt_tokens, completion_tokens)` — called after each LLM call
- `summary() -> dict` — per-model and total token counts
- `log_call(model, prompt_tokens, completion_tokens)` — prints live per-call info (verbose mode)
- `report() -> str` — formatted end-of-run summary table

### 4. Wiring

- `ModelRouter.get_client()` creates `RateLimiter` with provider defaults
- `LMHandler` gets a `TokenCounter` instance, records usage after each completion
- `LMHandler.completion()` calls `counter.log_call()` when verbose=True
- `ScholarAgent.research()` prints `counter.report()` after dispatcher finishes

### 5. Configuration

Rate limits configurable via `ScholarAgent` constructor:
```python
agent = ScholarAgent(
    strong_model={...},
    cheap_model={...},
    rpm_limit=60,    # optional override
    tpm_limit=90000, # optional override
    verbose=True,    # enables live token logging
)
```

## Testing

- Unit tests for RateLimiter (RPM blocking, TPM blocking, thread safety)
- Unit tests for TokenCounter (record, summary, report)
- Integration test verifying rate limiter is wired into client calls
