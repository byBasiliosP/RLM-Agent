"""OpenAI SDK wrapper implementing BaseLM."""

from __future__ import annotations

import httpx
import openai

from typing import TYPE_CHECKING

from scholaragent.clients.base import BaseLM
from scholaragent.utils.retry import retry_with_backoff

if TYPE_CHECKING:
    from scholaragent.clients.rate_limiter import RateLimiter


class OpenAIClient(BaseLM):
    """Wraps the OpenAI Python SDK for sync and async completions."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int | None = None,
        rate_limiter: "RateLimiter | None" = None,
    ):
        super().__init__(model_name, timeout=timeout, max_tokens=max_tokens, rate_limiter=rate_limiter)
        self._sync_client = openai.OpenAI(
            api_key=api_key, timeout=httpx.Timeout(self.timeout)
        )
        self._async_client = openai.AsyncOpenAI(
            api_key=api_key, timeout=httpx.Timeout(self.timeout)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_usage(self, usage) -> None:  # noqa: ANN001
        """Update tracked token counts from an API response usage object."""
        if usage is None:
            return
        self._record_usage_tokens(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def completion(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"model": self.model_name, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = retry_with_backoff(
            self._sync_client.chat.completions.create,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
            **kwargs,
        )
        self._record_usage(response.usage)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(self._last_usage.total_tokens)
        return response.choices[0].message.content or ""

    def completion_messages(self, messages: list[dict[str, str]]) -> str:
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        kwargs: dict = {"model": self.model_name, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = retry_with_backoff(
            self._sync_client.chat.completions.create,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
            **kwargs,
        )
        self._record_usage(response.usage)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(self._last_usage.total_tokens)
        return response.choices[0].message.content or ""

    async def acompletion(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"model": self.model_name, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = await self._async_client.chat.completions.create(**kwargs)
        self._record_usage(response.usage)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(self._last_usage.total_tokens)
        return response.choices[0].message.content or ""
