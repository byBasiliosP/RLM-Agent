"""Verify that modules emit structured log messages on errors."""

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import httpx
import pytest

from tests.helpers import FakeEmbeddings


@contextmanager
def capture_logs(logger_name):
    """Context manager to capture log messages from a logger."""
    logs = []

    class Handler(logging.Handler):
        def emit(self, record):
            logs.append(self.format(record))

    handler = Handler()
    target_logger = logging.getLogger(logger_name)
    target_logger.addHandler(handler)
    old_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        yield logs
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(old_level)


class TestStructuredLogging:
    def test_github_logs_on_error(self, monkeypatch):
        """GitHub adapter should log a warning on errors."""
        import scholaragent.sources.github as mod

        def fail_get(*args, **kwargs):
            raise httpx.HTTPError("connection failed")

        monkeypatch.setattr(mod, "_http_client", MagicMock())
        mod._http_client.get.side_effect = httpx.HTTPError("connection failed")
        # Also patch retry_with_backoff to just call the function directly
        monkeypatch.setattr(mod, "retry_with_backoff", lambda fn, *a, **kw: fn(*a, **{k: v for k, v in kw.items() if k not in ('max_retries', 'base_delay', 'retryable_exceptions')}))

        with capture_logs("scholaragent.sources.github") as logs:
            result = mod.search_github_code("test")
        assert result == []
        assert any("failed" in msg.lower() for msg in logs)

    def test_docs_logs_on_fetch_error(self, monkeypatch):
        """Docs adapter should log a warning on fetch errors."""
        import scholaragent.sources.docs as mod

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("timeout")
        monkeypatch.setattr(mod, "_http_client", mock_client)

        with capture_logs("scholaragent.sources.docs") as logs:
            result = mod.fetch_docs("http://example.com")
        assert result == []
        assert any("timeout" in msg.lower() or "failed" in msg.lower() for msg in logs)

    def test_research_pipeline_logs_source_errors(self, tmp_path, monkeypatch):
        """ResearchPipeline should log warnings for failed sources."""
        import scholaragent.memory.research as mod
        from scholaragent.memory.store import MemoryStore
        from scholaragent.memory.research import ResearchPipeline

        monkeypatch.setattr(mod, "search_arxiv", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("arxiv down")))
        monkeypatch.setattr(mod, "search_semantic_scholar", lambda *a, **kw: "[]")
        monkeypatch.setattr(mod, "search_github_code", lambda *a, **kw: [])
        monkeypatch.setattr(mod, "search_docs", lambda *a, **kw: [])

        db_path = str(tmp_path / "test.db")
        store = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
        pipeline = ResearchPipeline(store=store)

        with capture_logs("scholaragent.memory.research") as logs:
            pipeline.run("test", depth="quick", focus="implementation", force=True)
        assert any("arxiv down" in msg for msg in logs)
