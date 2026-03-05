"""Tests for embedding backends."""
import pytest
from unittest.mock import patch, MagicMock


class TestEmbeddingBackendABC:
    def test_cannot_instantiate(self):
        from scholaragent.memory.embeddings import EmbeddingBackend
        with pytest.raises(TypeError):
            EmbeddingBackend()

    def test_has_required_methods(self):
        from scholaragent.memory.embeddings import EmbeddingBackend
        import inspect
        assert hasattr(EmbeddingBackend, "embed")
        assert hasattr(EmbeddingBackend, "embed_batch")
        assert inspect.isabstract(EmbeddingBackend)


class TestOpenAIEmbeddings:
    @patch("openai.OpenAI")
    def test_creation(self, mock_openai):
        from scholaragent.memory.embeddings import OpenAIEmbeddings
        backend = OpenAIEmbeddings(model="text-embedding-3-small")
        assert backend.model == "text-embedding-3-small"

    @patch("openai.OpenAI")
    def test_embed_returns_list(self, mock_openai_cls):
        from scholaragent.memory.embeddings import OpenAIEmbeddings
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response
        backend = OpenAIEmbeddings()
        result = backend.embed("test")
        assert result == [0.1, 0.2, 0.3]

    @patch("openai.OpenAI")
    def test_embed_batch(self, mock_openai_cls):
        from scholaragent.memory.embeddings import OpenAIEmbeddings
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        e1, e2 = MagicMock(), MagicMock()
        e1.embedding, e2.embedding = [0.1, 0.2], [0.3, 0.4]
        mock_response.data = [e1, e2]
        mock_client.embeddings.create.return_value = mock_response
        backend = OpenAIEmbeddings()
        results = backend.embed_batch(["a", "b"])
        assert len(results) == 2


class TestCosineSimilarity:
    def test_identical(self):
        from scholaragent.memory.embeddings import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        from scholaragent.memory.embeddings import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        from scholaragent.memory.embeddings import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
