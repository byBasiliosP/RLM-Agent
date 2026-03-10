"""Tests for embedding backends."""
import pytest
from unittest.mock import patch, MagicMock
import os


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

    @patch.dict(
        os.environ,
        {
            "SCHOLAR_EMBEDDING_BACKEND": "lmstudio",
            "SCHOLAR_LMSTUDIO_URL": "http://localhost:1234/v1",
            "SCHOLAR_EMBEDDING_MODEL": "text-embedding-nomic-embed-text-v1.5",
        },
        clear=False,
    )
    @patch("openai.OpenAI")
    def test_creation_uses_lmstudio_env(self, mock_openai):
        from scholaragent.memory.embeddings import OpenAIEmbeddings

        backend = OpenAIEmbeddings()

        assert backend.model == "text-embedding-nomic-embed-text-v1.5"
        mock_openai.assert_called_once_with(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )


class TestEmbeddingCache:
    def test_embed_cache_avoids_duplicate_api_calls(self):
        """Calling embed() with same text should hit API only once."""
        with patch("scholaragent.memory.embeddings.openai") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            from scholaragent.memory.embeddings import OpenAIEmbeddings
            emb = OpenAIEmbeddings()

            result1 = emb.embed("hello world")
            result2 = emb.embed("hello world")

            assert result1 == result2
            assert mock_client.embeddings.create.call_count == 1

    def test_embed_cache_different_texts_make_separate_calls(self):
        """Different texts should make separate API calls."""
        with patch("scholaragent.memory.embeddings.openai") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            from scholaragent.memory.embeddings import OpenAIEmbeddings
            emb = OpenAIEmbeddings()

            emb.embed("hello")
            emb.embed("world")

            assert mock_client.embeddings.create.call_count == 2

    def test_embed_cache_returns_copy(self):
        """Cached result should be a copy, not the same list object."""
        with patch("scholaragent.memory.embeddings.openai") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            from scholaragent.memory.embeddings import OpenAIEmbeddings
            emb = OpenAIEmbeddings()

            result1 = emb.embed("hello")
            result2 = emb.embed("hello")

            assert result1 == result2
            assert result1 is not result2  # must be different list objects


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
