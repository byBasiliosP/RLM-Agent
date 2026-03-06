"""Abstract base class for all language model clients."""

from abc import ABC, abstractmethod

from scholaragent.core.types import ModelUsageSummary, UsageSummary


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
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens

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

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """Get cumulative usage summary for all calls made by this client."""
        ...

    @abstractmethod
    def get_last_usage(self) -> ModelUsageSummary:
        """Get the usage summary from the most recent call."""
        ...
