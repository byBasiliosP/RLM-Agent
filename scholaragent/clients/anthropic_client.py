"""Anthropic SDK wrapper implementing BaseLM."""

from __future__ import annotations

import anthropic

from scholaragent.clients.base import BaseLM
from scholaragent.core.types import ModelUsageSummary, UsageSummary


class AnthropicClient(BaseLM):
    """Wraps the Anthropic Python SDK for sync and async completions."""

    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__(model_name)
        self._sync_client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
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
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
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
        response = self._sync_client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        self._record_usage(response.usage)
        return response.content[0].text if response.content else ""

    async def acompletion(self, prompt: str) -> str:
        response = await self._async_client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        self._record_usage(response.usage)
        return response.content[0].text if response.content else ""

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries=dict(self._cumulative_usage))

    def get_last_usage(self) -> ModelUsageSummary:
        return self._last_usage
