"""Shared test fixtures and helpers."""


class FakeEmbeddings:
    """Deterministic embedding backend for testing."""

    def embed(self, text: str) -> list[float]:
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
