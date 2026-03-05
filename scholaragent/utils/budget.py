"""Budget tracking for agent resource consumption."""

from dataclasses import dataclass


@dataclass
class Budget:
    """Tracks token and iteration budgets for an agent run."""

    max_tokens: int = 100_000
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    tokens_used: int = 0
    iterations_used: int = 0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def iterations_remaining(self) -> int:
        return max(0, self.max_iterations - self.iterations_used)

    @property
    def is_exhausted(self) -> bool:
        return self.tokens_remaining <= 0 or self.iterations_remaining <= 0

    def use_tokens(self, n: int) -> None:
        self.tokens_used += n

    def use_iteration(self) -> None:
        self.iterations_used += 1
