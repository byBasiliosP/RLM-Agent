"""Tests for research pipeline depth levels."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch

from scholaragent.memory.research import ResearchPipeline


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


def _patch_sources():
    """Context manager to mock all source fetchers."""
    return patch.multiple(
        "scholaragent.memory.research",
        search_arxiv=MagicMock(return_value='[{"arxiv_id": "123", "title": "Test", "authors": ["A"], "abstract": "Test abstract", "published": "2024", "categories": []}]'),
        search_semantic_scholar=MagicMock(return_value='[]'),
        search_github_code=MagicMock(return_value=[]),
        search_docs=MagicMock(return_value=[]),
    )


class TestHasAgentInfra:
    def test_no_infra_by_default(self, store):
        pipeline = ResearchPipeline(store=store)
        assert not pipeline.has_agent_infra

    def test_has_infra_when_set(self, store):
        pipeline = ResearchPipeline(store=store)
        pipeline.set_agent_infra(
            handler=MagicMock(),
            registry=MagicMock(),
            dispatcher=MagicMock(),
        )
        assert pipeline.has_agent_infra

    def test_constructor_with_infra(self, store):
        pipeline = ResearchPipeline(
            store=store,
            handler=MagicMock(),
            registry=MagicMock(),
            dispatcher=MagicMock(),
        )
        assert pipeline.has_agent_infra


class TestQuickDepth:
    def test_quick_indexes_raw_results(self, store):
        pipeline = ResearchPipeline(store=store)
        with _patch_sources():
            result = pipeline.run("test", depth="quick")
            assert result["depth"] == "quick"
            assert result["entries_added"] > 0
            assert store.count() > 0

    def test_normal_falls_back_to_quick_without_infra(self, store):
        pipeline = ResearchPipeline(store=store)
        with _patch_sources():
            result = pipeline.run("test", depth="normal")
            assert result["status"] == "completed"
            assert result["entries_added"] > 0


class TestNormalDepth:
    def test_normal_runs_scout_reader_critic(self, store):
        mock_handler = MagicMock()
        mock_handler.address = ("127.0.0.1", 0)
        mock_handler.token_counter = None
        mock_handler.completion_messages = MagicMock(return_value="FINAL(done)")

        from scholaragent.core.types import AgentResult

        mock_scout = MagicMock()
        mock_scout.name = "scout"
        mock_scout.run.return_value = AgentResult(
            agent_name="scout", task="test", result="found papers", iterations=1, success=True
        )
        mock_reader = MagicMock()
        mock_reader.name = "reader"
        mock_reader.run.return_value = AgentResult(
            agent_name="reader", task="test", result="analysis", iterations=1, success=True
        )
        mock_critic = MagicMock()
        mock_critic.name = "critic"
        mock_critic.run.return_value = AgentResult(
            agent_name="critic", task="test", result="assessment", iterations=1, success=True
        )

        mock_registry = MagicMock()

        def get_agent(name):
            return {"scout": mock_scout, "reader": mock_reader, "critic": mock_critic}[name]

        mock_registry.get = get_agent

        pipeline = ResearchPipeline(
            store=store,
            handler=mock_handler,
            registry=mock_registry,
            dispatcher=MagicMock(),
        )

        with _patch_sources():
            result = pipeline.run("RLHF", depth="normal")
            assert result["status"] == "completed"
            assert result["depth"] == "normal"
            mock_scout.run.assert_called_once()

    def test_normal_falls_back_on_scout_failure(self, store):
        from scholaragent.core.types import AgentResult

        mock_handler = MagicMock()
        mock_scout = MagicMock()
        mock_scout.run.return_value = AgentResult(
            agent_name="scout", task="test", result="failed", iterations=1, success=False
        )
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_scout

        pipeline = ResearchPipeline(
            store=store,
            handler=mock_handler,
            registry=mock_registry,
            dispatcher=MagicMock(),
        )

        with _patch_sources():
            result = pipeline.run("test", depth="normal")
            assert result["status"] == "completed"
            assert result["entries_added"] >= 0


class TestDeepDepth:
    def test_deep_runs_dispatcher(self, store):
        from scholaragent.core.types import AgentResult

        mock_dispatcher = MagicMock()
        mock_dispatcher.run.return_value = AgentResult(
            agent_name="dispatcher", task="test", result="# Full synthesis report\n\nKey findings...",
            iterations=5, success=True,
        )

        pipeline = ResearchPipeline(
            store=store,
            handler=MagicMock(),
            registry=MagicMock(),
            dispatcher=mock_dispatcher,
        )

        result = pipeline.run("RLHF", depth="deep")
        assert result["depth"] == "deep"
        assert result["entries_added"] == 1
        mock_dispatcher.run.assert_called_once()

    def test_deep_falls_back_on_dispatcher_failure(self, store):
        mock_handler = MagicMock()
        mock_dispatcher = MagicMock()
        mock_dispatcher.run.side_effect = RuntimeError("LLM unavailable")

        from scholaragent.core.types import AgentResult

        mock_scout = MagicMock()
        mock_scout.run.return_value = AgentResult(
            agent_name="scout", task="test", result="found", iterations=1, success=True
        )
        mock_reader = MagicMock()
        mock_reader.run.return_value = AgentResult(
            agent_name="reader", task="test", result="read", iterations=1, success=True
        )
        mock_critic = MagicMock()
        mock_critic.run.return_value = AgentResult(
            agent_name="critic", task="test", result="critique", iterations=1, success=True
        )

        mock_registry = MagicMock()

        def get_agent(name):
            return {"scout": mock_scout, "reader": mock_reader, "critic": mock_critic}[name]

        mock_registry.get = get_agent

        pipeline = ResearchPipeline(
            store=store,
            handler=mock_handler,
            registry=mock_registry,
            dispatcher=mock_dispatcher,
        )

        with _patch_sources():
            result = pipeline.run("test", depth="deep")
            # Should fall back to normal, then complete
            assert result["status"] == "completed"
