"""Tests for the research pipeline integration."""

import os
import tempfile
from unittest.mock import MagicMock

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

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        s = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
        yield s
        s.close()


def _mock_collector(
    results=None,
    errors=None,
):
    """Create a mock SourceCollector with configurable return values."""
    collector = MagicMock()
    collector.collect.return_value = (
        results if results is not None else [],
        errors if errors is not None else [],
    )
    collector.deduplicate.side_effect = lambda xs: xs
    return collector


class TestResearchPipeline:
    def test_creation(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)
        assert pipeline.store is store

    def test_quick_research_stores_entries(self, store):
        from scholaragent.memory.research import ResearchPipeline

        collector = _mock_collector(
            results=[{"content": "Title: Paper\n\nAbstract: Content", "source_type": "paper", "source_ref": "arxiv:123"}],
        )
        pipeline = ResearchPipeline(store=store, collector=collector)
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

        collector = _mock_collector(
            results=[],
            errors=["arXiv: ConnectionError: network down"],
        )
        pipeline = ResearchPipeline(store=store, collector=collector)
        result = pipeline.run("test query", depth="quick")

        assert "errors" in result
        assert len(result["errors"]) == 1
        assert "arXiv" in result["errors"][0]
        assert "ConnectionError" in result["errors"][0]

    def test_pipeline_works_when_one_source_fails(self, store):
        from scholaragent.memory.research import ResearchPipeline

        collector = _mock_collector(
            results=[{"content": "Title: Paper\n\nAbstract: Content", "source_type": "paper", "source_ref": "arxiv:123"}],
            errors=["Semantic Scholar: RuntimeError: API limit exceeded"],
        )
        pipeline = ResearchPipeline(store=store, collector=collector)
        result = pipeline.run("test query", depth="quick")

        assert result["status"] == "completed"
        assert result["entries_added"] > 0
        assert len(result["errors"]) == 1
        assert "Semantic Scholar" in result["errors"][0]

    def test_all_sources_fail_returns_errors(self, store):
        from scholaragent.memory.research import ResearchPipeline

        collector = _mock_collector(
            results=[],
            errors=[
                "arXiv: ConnectionError: fail",
                "Semantic Scholar: TimeoutError: fail",
                "GitHub: ValueError: fail",
                "Docs: OSError: fail",
            ],
        )
        pipeline = ResearchPipeline(store=store, collector=collector)
        result = pipeline.run("test query", depth="quick")

        assert result["entries_added"] == 0
        assert len(result["errors"]) == 4


class TestPipelineDependencyInjection:
    def test_pipeline_uses_injected_collector(self, store):
        from scholaragent.memory.research import ResearchPipeline

        fake_collector = MagicMock()
        fake_collector.collect.return_value = (
            [{"content": "X", "source_type": "paper", "source_ref": "arxiv:9"}],
            [],
        )
        fake_collector.deduplicate.side_effect = lambda xs: xs

        pipeline = ResearchPipeline(store=store, collector=fake_collector)
        result = pipeline.run("test", depth="quick")

        assert result["entries_added"] == 1
        fake_collector.collect.assert_called_once()
        fake_collector.deduplicate.assert_called_once()

    def test_pipeline_uses_injected_indexer(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        from scholaragent.memory.research import ResearchPipeline

        real_indexer = ResultIndexer(store)
        pipeline = ResearchPipeline(
            store=store,
            collector=_mock_collector(
                results=[{"content": "X", "source_type": "paper", "source_ref": "arxiv:9"}],
            ),
            indexer=real_indexer,
        )
        result = pipeline.run("test", depth="quick")

        assert result["entries_added"] == 1
        assert store.count() == 1
