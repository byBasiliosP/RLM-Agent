"""OpenAI SDK wrapper implementing BaseLM."""

from __future__ import annotations

import httpx
import openai

from scholaragent.clients.base import BaseLM
from scholaragent.core.types import ModelUsageSummary, UsageSummary


class OpenAIClient(BaseLM):
    """Wraps the OpenAI Python SDK for sync and async completions."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int | None = None,
    ):
        super().__init__(model_name, timeout=timeout, max_tokens=max_tokens)
        self._sync_client = openai.OpenAI(
            api_key=api_key, timeout=httpx.Timeout(self.timeout)
        )
        self._async_client = openai.AsyncOpenAI(
            api_key=api_key, timeout=httpx.Timeout(self.timeout)
        )
        self._cumulative_usage: dict[str, ModelUsageSummary] = {}
        self._last_usage = ModelUsageSummary(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_usage(self, usage) -> None:  # noqa: ANN001
        """Update tracked token counts from an API response usage object."""
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def completion(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"model": self.model_name, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self._sync_client.chat.completions.create(**kwargs)
        self._record_usage(response.usage)
        return response.choices[0].message.content or ""

    async def acompletion(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"model": self.model_name, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = await self._async_client.chat.completions.create(**kwargs)
        self._record_usage(response.usage)
        return response.choices[0].message.content or ""

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries=dict(self._cumulative_usage))

    def get_last_usage(self) -> ModelUsageSummary:
        return self._last_usage
