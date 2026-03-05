"""Embedding backends for semantic search. Swappable: start with OpenAI API, replace with local model later."""
from abc import ABC, abstractmethod

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
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = openai.OpenAI()

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
