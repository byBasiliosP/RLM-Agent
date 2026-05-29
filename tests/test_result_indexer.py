"""ResultIndexer — writes MemoryEntries from raw, enriched, and synthesis results."""
from __future__ import annotations

import os
import tempfile

import pytest


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def store():
    from scholaragent.memory.store import MemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        s = MemoryStore(db_path=os.path.join(tmp, "t.db"), embeddings=FakeEmbeddings())
        yield s
        s.close()


class TestResultIndexer:
    def test_index_raw_counts_additions(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        raws = [
            {"content": "P1 body", "source_type": "paper", "source_ref": "arxiv:1"},
            {"content": "P2 body", "source_type": "paper", "source_ref": "arxiv:2"},
        ]
        count = indexer.index_raw(query="rlhf", raw_results=raws)
        assert count == 2
        assert store.count() == 2

    def test_index_raw_applies_query_tag(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        indexer.index_raw(
            query="Reward Model",
            raw_results=[{"content": "X", "source_type": "paper", "source_ref": "arxiv:9"}],
        )
        hits = store.search("X", max_results=1)
        assert hits
        entry, _ = hits[0]
        assert "reward-model" in entry.tags

    def test_index_enriched_appends_reader_and_critic(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        enriched = [{
            "content": "Paper body",
            "source_type": "paper",
            "source_ref": "arxiv:1",
            "reader_findings": "key claim",
            "critic_assessment": "methodologically sound",
        }]
        indexer.index_enriched(query="rlhf", enriched_results=enriched)

        entry, _ = store.search("rlhf", max_results=1)[0]
        assert "--- Reader Analysis ---" in entry.content
        assert "--- Critic Assessment ---" in entry.content
        assert "agent-processed" in entry.tags

    def test_index_enriched_without_reader_or_critic(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        enriched = [{
            "content": "Paper body only",
            "source_type": "paper",
            "source_ref": "arxiv:1",
        }]
        count = indexer.index_enriched(query="rlhf", enriched_results=enriched)
        assert count == 1
        entry, _ = store.search("rlhf", max_results=1)[0]
        assert "--- Reader Analysis ---" not in entry.content
        assert "agent-processed" not in entry.tags

    def test_index_synthesis_uses_synthesized_report_type(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        count = indexer.index_synthesis(
            query="long query about rlhf and dpo",
            synthesis_text="# Report\n\nFull synthesized report body.",
        )
        assert count == 1

        entry, _ = store.search("synthesized", max_results=1)[0]
        assert entry.source_type == "synthesized_report"
        assert entry.source_ref.startswith("deep-research:")
        assert "deep-pipeline" in entry.tags

    def test_index_synthesis_returns_zero_for_empty(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        assert indexer.index_synthesis(query="q", synthesis_text="") == 0
        assert store.count() == 0

    def test_log_research_delegates_to_store(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        indexer.log_research(query="rlhf", depth="quick", focus="implementation", result_count=3)
        logged = store.get_recent_research("rlhf")
        assert len(logged) == 1
        assert logged[0].depth == "quick"
