"""Model router -- assigns LLM clients by agent role."""

from __future__ import annotations

from dataclasses import dataclass

from scholaragent.clients.base import BaseLM

CHEAP_ROLES: frozenset[str] = frozenset({"scout"})


@dataclass
class ModelConfig:
    """Identifies a specific model on a specific backend."""

    backend: str  # "openai", "anthropic", or "lmstudio"
    model_name: str
    max_tokens: int | None = None
    base_url: str | None = None


class ModelRouter:
    """Routes agent roles to the appropriate LLM client.

    Scout agents are assigned a *cheap* (fast/inexpensive) model while all
    other agents receive the *strong* model.
    """

    def __init__(self, strong: ModelConfig, cheap: ModelConfig):
        self.strong = strong
        self.cheap = cheap

    def get_config(self, role: str) -> ModelConfig:
        """Return cheap config for CHEAP_ROLES, strong for others."""
        if role in CHEAP_ROLES:
            return self.cheap
        return self.strong

    def get_client(self, role: str) -> BaseLM:
        """Create and return a BaseLM client for the given role."""
        from scholaragent.clients.rate_limiter import PROVIDER_DEFAULTS, RateLimiter

        config = self.get_config(role)
        kwargs: dict = {"model_name": config.model_name}
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        if config.backend == "openai":
            from scholaragent.clients.openai_client import OpenAIClient

            if config.base_url:
                kwargs["base_url"] = config.base_url
            client = OpenAIClient(**kwargs)
        elif config.backend == "anthropic":
            from scholaragent.clients.anthropic_client import AnthropicClient

            client = AnthropicClient(**kwargs)
        elif config.backend == "lmstudio":
            from scholaragent.clients.openai_client import OpenAIClient

            kwargs["base_url"] = config.base_url or "http://localhost:1234/v1"
            kwargs["api_key"] = "lm-studio"
            client = OpenAIClient(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

        backend_key = "lmstudio" if config.backend == "lmstudio" else config.backend
        if backend_key in PROVIDER_DEFAULTS:
            client.rate_limiter = RateLimiter(**PROVIDER_DEFAULTS[backend_key])
        return client
