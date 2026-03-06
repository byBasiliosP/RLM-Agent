"""Tests for the research pipeline integration."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch

from scholaragent.memory.types import MemoryEntry


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def store():
    from scholaragent.memory.store import MemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        s = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
        yield s
        s.close()


class TestResearchPipeline:
    def test_creation(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)
        assert pipeline.store is store

    def test_collect_sources_papers(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.memory.research.search_arxiv") as mock_arxiv, \
             patch("scholaragent.memory.research.search_semantic_scholar") as mock_s2:
            mock_arxiv.return_value = '[{"arxiv_id": "2401.00001", "title": "Test Paper", "authors": ["A"], "abstract": "Test abstract", "published": "2024-01-01", "categories": ["cs.AI"]}]'
            mock_s2.return_value = '[{"paper_id": "abc", "title": "Test Paper 2", "authors": ["B"], "abstract": "Abstract 2", "year": 2024, "citation_count": 10, "arxiv_id": ""}]'

            results, errors = pipeline._collect_sources("attention mechanisms", sources=["paper"])
            assert len(results) > 0
            assert all(r["source_type"] == "paper" for r in results)
            assert errors == []

    def test_quick_research_stores_entries(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.memory.research.search_arxiv") as mock_arxiv, \
             patch("scholaragent.memory.research.search_semantic_scholar") as mock_s2, \
             patch("scholaragent.memory.research.search_github_code") as mock_gh:
            mock_arxiv.return_value = '[{"arxiv_id": "123", "title": "Paper", "authors": ["X"], "abstract": "Content", "published": "2024", "categories": []}]'
            mock_s2.return_value = '[]'
            mock_gh.return_value = []

            result = pipeline.run("test query", depth="quick", focus="implementation")
            assert store.count() > 0
            assert result["depth"] == "quick"
            assert result["entries_added"] > 0

    def test_deduplication_check(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        # Log a recent research
        store.log_research("RLHF techniques", "normal", "theory", 5)

        # Check dedup
        recent = pipeline._check_dedup("RLHF techniques")
        assert recent is not None

    def test_no_dedup_for_new_query(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)
        recent = pipeline._check_dedup("completely new topic")
        assert recent is None

    def test_source_failure_appears_in_errors(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.memory.research.search_arxiv") as mock_arxiv, \
             patch("scholaragent.memory.research.search_semantic_scholar") as mock_s2, \
             patch("scholaragent.memory.research.search_github_code") as mock_gh, \
             patch("scholaragent.memory.research.search_docs") as mock_docs:
            mock_arxiv.side_effect = ConnectionError("network down")
            mock_s2.return_value = '[]'
            mock_gh.return_value = []
            mock_docs.return_value = []

            result = pipeline.run("test query", depth="quick")
            assert "errors" in result
            assert len(result["errors"]) == 1
            assert "arXiv" in result["errors"][0]
            assert "ConnectionError" in result["errors"][0]

    def test_pipeline_works_when_one_source_fails(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.memory.research.search_arxiv") as mock_arxiv, \
             patch("scholaragent.memory.research.search_semantic_scholar") as mock_s2, \
             patch("scholaragent.memory.research.search_github_code") as mock_gh, \
             patch("scholaragent.memory.research.search_docs") as mock_docs:
            mock_arxiv.return_value = '[{"arxiv_id": "123", "title": "Paper", "authors": ["X"], "abstract": "Content", "published": "2024", "categories": []}]'
            mock_s2.side_effect = RuntimeError("API limit exceeded")
            mock_gh.return_value = []
            mock_docs.return_value = []

            result = pipeline.run("test query", depth="quick")
            assert result["status"] == "completed"
            assert result["entries_added"] > 0
            assert len(result["errors"]) == 1
            assert "Semantic Scholar" in result["errors"][0]

    def test_all_sources_fail_returns_errors(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.memory.research.search_arxiv") as mock_arxiv, \
             patch("scholaragent.memory.research.search_semantic_scholar") as mock_s2, \
             patch("scholaragent.memory.research.search_github_code") as mock_gh, \
             patch("scholaragent.memory.research.search_docs") as mock_docs:
            mock_arxiv.side_effect = ConnectionError("fail")
            mock_s2.side_effect = TimeoutError("fail")
            mock_gh.side_effect = ValueError("fail")
            mock_docs.side_effect = OSError("fail")

            result = pipeline.run("test query", depth="quick")
            assert result["entries_added"] == 0
            assert len(result["errors"]) == 4
