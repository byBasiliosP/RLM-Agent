"""Tests for paper deduplication in the research pipeline."""

import os
import tempfile

import pytest

from scholaragent.memory.research import ResearchPipeline


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def pipeline():
    from scholaragent.memory.store import MemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
        p = ResearchPipeline(store=store)
        yield p
        store.close()


class TestDeduplication:
    def test_arxiv_id_match_removes_duplicate(self, pipeline):
        results = [
            {"content": "Title: Paper A\n\nAbstract: ...", "source_type": "paper", "source_ref": "arxiv:2401.00001"},
            {"content": "Title: Paper A\n\nAbstract: ...\n\nYear: 2024\nCitations: 10", "source_type": "paper", "source_ref": "s2:abc123"},
        ]
        # S2 entry has same paper but different ref — won't match on arxiv_id here.
        # Only matches if both have same arxiv_id in source_ref.
        deduped = pipeline._deduplicate(results)
        assert len(deduped) == 1  # title-based match
        # Prefer S2 entry (has citation counts)
        assert deduped[0]["source_ref"] == "s2:abc123"

    def test_title_match_removes_duplicate(self, pipeline):
        results = [
            {"content": "Title: Attention Is All You Need\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:1706.03762"},
            {"content": "Title: Attention Is All You Need\n\nAbstract: B\n\nYear: 2017\nCitations: 50000", "source_type": "paper", "source_ref": "s2:xyz"},
        ]
        deduped = pipeline._deduplicate(results)
        assert len(deduped) == 1
        # S2 entry preferred
        assert deduped[0]["source_ref"] == "s2:xyz"

    def test_different_papers_not_deduped(self, pipeline):
        results = [
            {"content": "Title: Paper One\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:111"},
            {"content": "Title: Paper Two\n\nAbstract: B", "source_type": "paper", "source_ref": "s2:222"},
        ]
        deduped = pipeline._deduplicate(results)
        assert len(deduped) == 2

    def test_non_paper_entries_pass_through(self, pipeline):
        results = [
            {"content": "Code snippet", "source_type": "code", "source_ref": "github:org/repo"},
            {"content": "Title: Paper\n\nAbstract: X", "source_type": "paper", "source_ref": "arxiv:123"},
            {"content": "Docs content", "source_type": "docs", "source_ref": "https://docs.python.org"},
        ]
        deduped = pipeline._deduplicate(results)
        assert len(deduped) == 3

    def test_punctuation_insensitive_title_match(self, pipeline):
        results = [
            {"content": "Title: GPT-4: A New Model!\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:aaa"},
            {"content": "Title: GPT4 A New Model\n\nAbstract: B", "source_type": "paper", "source_ref": "s2:bbb"},
        ]
        deduped = pipeline._deduplicate(results)
        assert len(deduped) == 1

    def test_empty_results(self, pipeline):
        assert pipeline._deduplicate([]) == []

    def test_arxiv_only_no_dedup(self, pipeline):
        results = [
            {"content": "Title: Paper\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:111"},
        ]
        deduped = pipeline._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["source_ref"] == "arxiv:111"
