"""Abstract base class for all language model clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from scholaragent.core.types import ModelUsageSummary, UsageSummary

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholaragent.clients.rate_limiter import RateLimiter


class BaseLM(ABC):
    """Base class for all language model clients.

    Provides a uniform interface so agents can call any LLM backend
    without knowing the underlying SDK.
    """

    def __init__(
        self,
        model_name: str,
        timeout: float = 120.0,
        max_tokens: int | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.rate_limiter = rate_limiter
        self._cumulative_usage: dict[str, ModelUsageSummary] = {}
        self._last_usage = ModelUsageSummary(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    @abstractmethod
    def completion(self, prompt: str) -> str:
        """Single synchronous completion."""
        ...

    @abstractmethod
    async def acompletion(self, prompt: str) -> str:
        """Single async completion."""
        ...

    def completion_messages(self, messages: list[dict[str, str]]) -> str:
        """Completion from a structured message list.

        The default implementation flattens messages into a single string
        and delegates to :meth:`completion`.  Subclasses (OpenAI, Anthropic)
        override this to pass the native message array to the API so that
        system / user / assistant roles are preserved.
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        return self.completion("\n\n".join(parts))

    def _record_usage_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update tracked token counts from extracted values."""
        total_tokens = prompt_tokens + completion_tokens
        self._last_usage = ModelUsageSummary(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        if self.model_name in self._cumulative_usage:
            prev = self._cumulative_usage[self.model_name]
            self._cumulative_usage[self.model_name] = ModelUsageSummary(
                prompt_tokens=prev.prompt_tokens + prompt_tokens,
                completion_tokens=prev.completion_tokens + completion_tokens,
                total_tokens=prev.total_tokens + total_tokens,
            )
        else:
            self._cumulative_usage[self.model_name] = ModelUsageSummary(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

    def get_usage_summary(self) -> UsageSummary:
        """Get cumulative usage summary for all calls made by this client."""
        return UsageSummary(model_usage_summaries=dict(self._cumulative_usage))

    def get_last_usage(self) -> ModelUsageSummary:
        """Get the usage summary from the most recent call."""
        return self._last_usage
