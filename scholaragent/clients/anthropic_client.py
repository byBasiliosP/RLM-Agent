"""Anthropic SDK wrapper implementing BaseLM."""

from __future__ import annotations

import anthropic

from typing import TYPE_CHECKING

from scholaragent.clients.base import BaseLM
from scholaragent.utils.retry import retry_with_backoff

if TYPE_CHECKING:
    from scholaragent.clients.rate_limiter import RateLimiter


class AnthropicClient(BaseLM):
    """Wraps the Anthropic Python SDK for sync and async completions."""

    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int | None = None,
        rate_limiter: "RateLimiter | None" = None,
    ):
        super().__init__(model_name, timeout=timeout, max_tokens=max_tokens, rate_limiter=rate_limiter)
        self._sync_client = anthropic.Anthropic(
            api_key=api_key, timeout=self.timeout
        )
        self._async_client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=self.timeout
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_usage(self, usage) -> None:  # noqa: ANN001
        """Update tracked token counts from an API response usage object."""
        if usage is None:
            return
        self._record_usage_tokens(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def completion(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        response = retry_with_backoff(
            self._sync_client.messages.create,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
            model=self.model_name,
            max_tokens=self.max_tokens or self.DEFAULT_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        self._record_usage(response.usage)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(self._last_usage.total_tokens)
        return response.content[0].text if response.content else ""

    def completion_messages(self, messages: list[dict[str, str]]) -> str:
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        system_msg = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                api_messages.append(msg)
        kwargs: dict = {
            "model": self.model_name,
            "max_tokens": self.max_tokens or self.DEFAULT_MAX_TOKENS,
            "messages": api_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        response = retry_with_backoff(
            self._sync_client.messages.create,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
            **kwargs,
        )
        self._record_usage(response.usage)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(self._last_usage.total_tokens)
        return response.content[0].text if response.content else ""

    async def acompletion(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        response = await self._async_client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens or self.DEFAULT_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        self._record_usage(response.usage)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(self._last_usage.total_tokens)
        return response.content[0].text if response.content else ""
