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
