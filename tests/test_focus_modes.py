"""Tests for focus mode injection into agent tasks."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, call

from scholaragent.memory.research import ResearchPipeline, FOCUS_HINTS
from scholaragent.core.types import AgentResult


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


class TestFocusHints:
    def test_all_focus_modes_have_hints(self):
        assert "implementation" in FOCUS_HINTS
        assert "theory" in FOCUS_HINTS
        assert "comparison" in FOCUS_HINTS

    def test_focus_injected_into_scout_task(self, store):
        mock_handler = MagicMock()
        mock_scout = MagicMock()
        mock_scout.name = "scout"
        mock_scout.run.return_value = AgentResult(
            agent_name="scout", task="test", result="found", iterations=1, success=True
        )
        mock_reader = MagicMock()
        mock_reader.name = "reader"
        mock_reader.run.return_value = AgentResult(
            agent_name="reader", task="test", result="read", iterations=1, success=True
        )
        mock_critic = MagicMock()
        mock_critic.name = "critic"
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
            dispatcher=MagicMock(),
        )

        from unittest.mock import patch
        with patch.multiple(
            "scholaragent.memory.research",
            search_arxiv=MagicMock(return_value='[]'),
            search_semantic_scholar=MagicMock(return_value='[]'),
            search_github_code=MagicMock(return_value=[]),
            search_docs=MagicMock(return_value=[]),
        ):
            pipeline.run("RLHF", depth="normal", focus="theory")

            # Scout task should contain focus hint
            scout_call = mock_scout.run.call_args
            task_arg = scout_call[1].get("task", scout_call[0][0] if scout_call[0] else "")
            assert FOCUS_HINTS["theory"] in task_arg

    def test_focus_injected_into_deep_task(self, store):
        mock_dispatcher = MagicMock()
        mock_dispatcher.run.return_value = AgentResult(
            agent_name="dispatcher", task="test", result="report", iterations=3, success=True,
        )

        pipeline = ResearchPipeline(
            store=store,
            handler=MagicMock(),
            registry=MagicMock(),
            dispatcher=mock_dispatcher,
        )

        pipeline.run("RLHF", depth="deep", focus="comparison")

        dispatch_call = mock_dispatcher.run.call_args
        task_arg = dispatch_call[1].get("task", dispatch_call[0][0] if dispatch_call[0] else "")
        assert FOCUS_HINTS["comparison"] in task_arg
