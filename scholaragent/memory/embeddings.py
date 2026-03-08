"""Embedding backends for semantic search. Swappable: start with OpenAI API, replace with local model later."""
from abc import ABC, abstractmethod
import hashlib

import numpy as np
import openai


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddings(EmbeddingBackend):
    def __init__(self, model: str = "text-embedding-3-small", cache_size: int = 512):
        self.model = model
        self._client = openai.OpenAI()
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size

    def embed(self, text: str) -> list[float]:
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self._cache:
            return list(self._cache[cache_key])  # return copy
        response = self._client.embeddings.create(input=[text], model=self.model)
        result = response.data[0].embedding
        if len(self._cache) >= self._cache_size:
            # Evict oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result
        return list(result)  # return copy

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
