"""Base environment types and abstract classes for ScholarAgent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class REPLResult:
    """Result from executing code in a REPL environment."""

    output: str
    error: str | None = None
    has_final: bool = False
    final_value: str | None = None


RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "llm_query",
        "call_agent",
        "FINAL_VAR",
        "SHOW_VARS",
        "SHOW_PROGRESS",
        "context",
    }
)


class BaseEnv(ABC):
    """Abstract base class for REPL-like environments."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize the environment namespace and builtins."""
        ...

    @abstractmethod
    def load_context(self, context: str) -> None:
        """Load context data into the environment."""
        ...

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        """Execute code and return the result."""
        ...
