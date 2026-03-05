"""Tests for the MCP server tool definitions."""

import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class TestMCPToolFunctions:
    """Test the tool handler functions directly (not via MCP transport)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up a temporary memory store for each test."""
        self.db_path = str(tmp_path / "test_memory.db")
        with patch.dict(os.environ, {"SCHOLAR_MEMORY_DB": self.db_path}):
            yield

    def test_memory_store_creates_entry(self):
        from scholaragent.mcp_server import _get_store, _memory_store
        from scholaragent.memory.store import MemoryStore

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        result = _memory_store(
            store=store,
            content="RLHF uses human feedback",
            source="arxiv:2203.02155",
            tags=["rlhf"],
        )
        assert result["status"] == "stored"
        assert store.count() == 1

    def test_memory_lookup_returns_results(self):
        from scholaragent.mcp_server import _memory_lookup
        from scholaragent.memory.store import MemoryStore
        from scholaragent.memory.types import MemoryEntry

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        store.add(MemoryEntry(
            content="Transformers use attention",
            summary="Attention",
            source_type="paper",
            source_ref="ref1",
            tags=["transformers"],
        ))
        result = _memory_lookup(store=store, query="attention mechanisms")
        assert "results" in result
        assert len(result["results"]) > 0

    def test_memory_lookup_empty_store(self):
        from scholaragent.mcp_server import _memory_lookup
        from scholaragent.memory.store import MemoryStore

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        result = _memory_lookup(store=store, query="anything")
        assert result["results"] == []

    def test_memory_forget_by_id(self):
        from scholaragent.mcp_server import _memory_forget
        from scholaragent.memory.store import MemoryStore
        from scholaragent.memory.types import MemoryEntry

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        entry = MemoryEntry(
            content="Delete me",
            summary="Del",
            source_type="paper",
            source_ref="ref",
            tags=[],
        )
        store.add(entry)
        result = _memory_forget(store=store, query_or_id=entry.id)
        assert result["deleted"] == 1
        assert store.count() == 0

    def test_memory_status(self):
        from scholaragent.mcp_server import _memory_status
        from scholaragent.memory.store import MemoryStore

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        result = _memory_status(store=store)
        assert "total_entries" in result
        assert result["total_entries"] == 0
