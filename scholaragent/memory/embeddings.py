"""Embedding backends for semantic search."""
from abc import ABC, abstractmethod
import hashlib
import os

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
    def __init__(
        self,
        model: str | None = None,
        cache_size: int = 512,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        embedding_backend = os.environ.get("SCHOLAR_EMBEDDING_BACKEND", "openai").lower()
        if embedding_backend == "lmstudio":
            default_model = os.environ.get(
                "SCHOLAR_EMBEDDING_MODEL",
                "text-embedding-nomic-embed-text-v1.5",
            )
            default_base_url = os.environ.get(
                "SCHOLAR_EMBEDDING_BASE_URL",
                os.environ.get("SCHOLAR_LMSTUDIO_URL", "http://localhost:1234/v1"),
            )
            default_api_key = os.environ.get("SCHOLAR_EMBEDDING_API_KEY", "lm-studio")
        else:
            default_model = os.environ.get(
                "SCHOLAR_EMBEDDING_MODEL",
                "text-embedding-3-small",
            )
            default_base_url = os.environ.get("SCHOLAR_EMBEDDING_BASE_URL")
            default_api_key = os.environ.get("SCHOLAR_EMBEDDING_API_KEY")

        self.model = model or default_model
        client_kwargs = {}
        resolved_base_url = base_url or default_base_url
        resolved_api_key = api_key or default_api_key
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        self._client = openai.OpenAI(**client_kwargs)
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
