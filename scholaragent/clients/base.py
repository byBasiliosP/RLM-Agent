"""Abstract base class for all language model clients."""

from abc import ABC, abstractmethod

from scholaragent.core.types import ModelUsageSummary, UsageSummary


class BaseLM(ABC):
    """Base class for all language model clients.

    Provides a uniform interface so agents can call any LLM backend
    without knowing the underlying SDK.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def completion(self, prompt: str) -> str:
        """Single synchronous completion."""
        ...

    @abstractmethod
    async def acompletion(self, prompt: str) -> str:
        """Single async completion."""
        ...

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """Get cumulative usage summary for all calls made by this client."""
        ...

    @abstractmethod
    def get_last_usage(self) -> ModelUsageSummary:
        """Get the usage summary from the most recent call."""
        ...
