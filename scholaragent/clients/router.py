"""Model router -- assigns LLM clients by agent role."""

from __future__ import annotations

from dataclasses import dataclass

from scholaragent.clients.base import BaseLM

CHEAP_ROLES: frozenset[str] = frozenset({"scout"})


@dataclass
class ModelConfig:
    """Identifies a specific model on a specific backend."""

    backend: str  # "openai" or "anthropic"
    model_name: str


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
        config = self.get_config(role)
        if config.backend == "openai":
            from scholaragent.clients.openai_client import OpenAIClient

            return OpenAIClient(model_name=config.model_name)
        elif config.backend == "anthropic":
            from scholaragent.clients.anthropic_client import AnthropicClient

            return AnthropicClient(model_name=config.model_name)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
