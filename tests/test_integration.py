"""Integration test: full flow from MCP tool functions through to store."""

import os
import tempfile

import pytest
from unittest.mock import patch


class FakeEmbeddings:
    def embed(self, text):
        # Use character frequency for more realistic-ish embeddings
        vec = [0.0] * 8
        for i, c in enumerate(text.lower()):
            vec[ord(c) % 8] += 1.0
        # Normalize
        total = sum(v * v for v in vec) ** 0.5
        if total > 0:
            vec = [v / total for v in vec]
        return vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class TestFullFlow:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.db_path = str(tmp_path / "integration.db")

    def test_store_then_lookup(self):
        from scholaragent.memory.store import MemoryStore
        from scholaragent.mcp_server import _memory_store, _memory_lookup

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())

        # Store some knowledge
        _memory_store(store, "RLHF uses reward models trained on human preferences", "arxiv:2203.02155", ["rlhf"])
        _memory_store(store, "PPO is the standard optimizer for RLHF fine-tuning", "arxiv:2210.01234", ["rlhf", "ppo"])
        _memory_store(store, "React useState hook manages component state", "https://react.dev/reference/react/useState", ["react"])

        # Lookup should find relevant results
        result = _memory_lookup(store, "reinforcement learning from human feedback")
        assert len(result["results"]) > 0
        assert result["total_indexed"] == 3
        # RLHF entries should score higher than React entry
        rlhf_results = [r for r in result["results"] if "rlhf" in r.get("tags", [])]
        assert len(rlhf_results) > 0

    def test_store_then_forget(self):
        from scholaragent.memory.store import MemoryStore
        from scholaragent.mcp_server import _memory_store, _memory_forget, _memory_status

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())

        result = _memory_store(store, "Temporary note", "manual", ["temp"])
        entry_id = result["id"]

        status = _memory_status(store)
        assert status["total_entries"] == 1

        forget_result = _memory_forget(store, entry_id)
        assert forget_result["deleted"] == 1

        status = _memory_status(store)
        assert status["total_entries"] == 0

    def test_research_then_lookup(self):
        from scholaragent.memory.store import MemoryStore
        from scholaragent.memory.research import ResearchPipeline
        from scholaragent.mcp_server import _memory_research, _memory_lookup

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        pipeline = ResearchPipeline(store=store)

        # IMPORTANT: patch where the functions are USED (in research.py), not where they're defined
        with patch("scholaragent.memory.research.search_arxiv") as mock_arxiv, \
             patch("scholaragent.memory.research.search_semantic_scholar") as mock_s2, \
             patch("scholaragent.memory.research.search_github_code") as mock_gh, \
             patch("scholaragent.memory.research.search_docs") as mock_docs:
            mock_arxiv.return_value = '[{"arxiv_id": "2401.00001", "title": "Attention Is All You Need", "authors": ["Vaswani"], "abstract": "We propose the Transformer architecture based on attention mechanisms", "published": "2017", "categories": ["cs.CL"]}]'
            mock_s2.return_value = '[]'
            mock_gh.return_value = [{"content": "class MultiHeadAttention(nn.Module):", "source_type": "code", "source_ref": "https://github.com/example/transformer"}]
            mock_docs.return_value = []

            research_result = _memory_research(pipeline, "transformer attention", depth="quick", focus="implementation")
            assert research_result["entries_added"] > 0

            # Now lookup should find the research results
            lookup_result = _memory_lookup(store, "attention mechanism in transformers")
            assert len(lookup_result["results"]) > 0
