"""Tests for the MCP server tool definitions."""

import json
import os
import threading

import pytest
from unittest.mock import patch, MagicMock

from tests.helpers import FakeEmbeddings


class TestMCPToolFunctions:
    """Test the tool handler functions directly (not via MCP transport)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up a temporary memory store for each test."""
        self.db_path = str(tmp_path / "test_memory.db")
        with patch.dict(os.environ, {"SCHOLAR_MEMORY_DB": self.db_path}):
            yield

    def _make_store(self):
        from scholaragent.memory.store import MemoryStore

        return MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())

    def _make_entry(self, content="Test content", source_type="paper", source_ref="ref1"):
        from scholaragent.memory.types import MemoryEntry

        return MemoryEntry(
            content=content,
            summary=MemoryEntry.smart_summary(content),
            source_type=source_type,
            source_ref=source_ref,
            tags=["test"],
        )

    # ---- memory_store ----

    def test_memory_store_creates_entry(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(
            store=store,
            content="RLHF uses human feedback",
            source="arxiv:2203.02155",
            tags=["rlhf"],
        )
        assert result["status"] == "stored"
        assert store.count() == 1

    def test_memory_store_infers_paper_from_arxiv(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(store=store, content="Test", source="arxiv:1234", tags=[])
        assert result["source_type"] == "paper"

    def test_memory_store_infers_paper_from_s2(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(store=store, content="Test", source="s2:abc123", tags=[])
        assert result["source_type"] == "paper"

    def test_memory_store_infers_code_from_github_prefix(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(store=store, content="Test", source="github:org/repo", tags=[])
        assert result["source_type"] == "code"

    def test_memory_store_infers_code_from_github_url(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(
            store=store, content="Test", source="https://github.com/org/repo", tags=[]
        )
        assert result["source_type"] == "code"

    def test_memory_store_defaults_to_docs(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(
            store=store, content="Test", source="https://docs.python.org/3/", tags=[]
        )
        assert result["source_type"] == "docs"

    def test_memory_store_returns_entry_id(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(store=store, content="Test", source="ref", tags=[])
        assert "id" in result
        assert len(result["id"]) > 0

    def test_memory_store_rejects_empty_content(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(store=store, content="", source="ref", tags=[])
        assert "error" in result

    def test_memory_store_rejects_whitespace_content(self):
        from scholaragent.mcp_server import _memory_store

        store = self._make_store()
        result = _memory_store(store=store, content="   \n\t  ", source="ref", tags=[])
        assert "error" in result

    # ---- memory_lookup ----

    def test_memory_lookup_returns_results(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        store.add(self._make_entry("Transformers use attention"))
        result = _memory_lookup(store=store, query="attention mechanisms")
        assert "results" in result
        assert len(result["results"]) > 0

    def test_memory_lookup_empty_store(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        result = _memory_lookup(store=store, query="anything")
        assert result["results"] == []

    def test_memory_lookup_includes_total_indexed(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        store.add(self._make_entry("Entry 1"))
        store.add(self._make_entry("Entry 2", source_ref="ref2"))
        result = _memory_lookup(store=store, query="entry")
        assert result["total_indexed"] == 2

    def test_memory_lookup_includes_query_echo(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        result = _memory_lookup(store=store, query="my search query")
        assert result["query"] == "my search query"

    def test_memory_lookup_includes_relevance_scores(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        store.add(self._make_entry("Machine learning algorithms"))
        result = _memory_lookup(store=store, query="machine learning")
        assert "relevance_score" in result["results"][0]
        assert isinstance(result["results"][0]["relevance_score"], float)

    def test_memory_lookup_respects_max_results(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        for i in range(10):
            store.add(self._make_entry(f"Entry number {i}", source_ref=f"ref{i}"))
        result = _memory_lookup(store=store, query="entry", max_results=3)
        assert len(result["results"]) <= 3

    def test_memory_lookup_filters_by_source(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        store.add(self._make_entry("Paper about RLHF", source_type="paper", source_ref="ref1"))
        store.add(self._make_entry("Code for RLHF", source_type="code", source_ref="ref2"))
        result = _memory_lookup(store=store, query="RLHF", sources=["paper"])
        for r in result["results"]:
            assert r["source_type"] == "paper"

    def test_memory_lookup_rejects_max_results_zero(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        result = _memory_lookup(store=store, query="test", max_results=0)
        assert "error" in result

    def test_memory_lookup_rejects_max_results_negative(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        result = _memory_lookup(store=store, query="test", max_results=-1)
        assert "error" in result

    def test_memory_lookup_rejects_max_results_over_50(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        result = _memory_lookup(store=store, query="test", max_results=51)
        assert "error" in result

    def test_memory_lookup_accepts_max_results_boundary(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        result1 = _memory_lookup(store=store, query="test", max_results=1)
        assert "error" not in result1
        result50 = _memory_lookup(store=store, query="test", max_results=50)
        assert "error" not in result50

    def test_memory_lookup_compact_default_excludes_content(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        store.add(self._make_entry("Full content about transformers"))
        result = _memory_lookup(store=store, query="transformers")
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert "content" not in r
            assert "summary" in r

    def test_memory_lookup_compact_false_includes_content(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        store.add(self._make_entry("Full content about RLHF"))
        result = _memory_lookup(store=store, query="RLHF", compact=False)
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert "content" in r
            assert "summary" in r

    def test_memory_lookup_compact_has_relevance_score(self):
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        store.add(self._make_entry("Machine learning algorithms"))
        result = _memory_lookup(store=store, query="machine learning", compact=True)
        assert "relevance_score" in result["results"][0]

    # ---- memory_get ----

    def test_memory_get_returns_full_entry(self):
        from scholaragent.mcp_server import _memory_get

        store = self._make_store()
        entry = self._make_entry("Full content about RLHF reward models")
        store.add(entry)
        result = _memory_get(store=store, entry_id=entry.id)
        assert result["content"] == "Full content about RLHF reward models"
        assert result["id"] == entry.id
        assert "summary" in result

    def test_memory_get_returns_error_for_missing_id(self):
        from scholaragent.mcp_server import _memory_get

        store = self._make_store()
        result = _memory_get(store=store, entry_id="nonexistent-uuid")
        assert "error" in result

    def test_memory_get_lookup_flow(self):
        """Test the browse-then-read pattern: lookup compact, then get full."""
        from scholaragent.mcp_server import _memory_lookup, _memory_get

        store = self._make_store()
        entry = self._make_entry("Detailed content about attention mechanisms in transformers")
        store.add(entry)

        # Step 1: compact lookup
        lookup = _memory_lookup(store=store, query="attention", compact=True)
        assert "content" not in lookup["results"][0]
        entry_id = lookup["results"][0]["id"]

        # Step 2: get full content by ID
        full = _memory_get(store=store, entry_id=entry_id)
        assert full["content"] == "Detailed content about attention mechanisms in transformers"

    # ---- memory_forget ----

    def test_memory_forget_by_id(self):
        from scholaragent.mcp_server import _memory_forget

        store = self._make_store()
        entry = self._make_entry("Delete me")
        store.add(entry)
        result = _memory_forget(store=store, query_or_id=entry.id)
        assert result["deleted"] == 1
        assert store.count() == 0

    def test_memory_forget_returns_zero_for_missing_id(self):
        from scholaragent.mcp_server import _memory_forget

        store = self._make_store()
        result = _memory_forget(store=store, query_or_id="nonexistent-uuid")
        assert result["deleted"] == 0

    def test_memory_forget_echoes_query(self):
        from scholaragent.mcp_server import _memory_forget

        store = self._make_store()
        result = _memory_forget(store=store, query_or_id="test-query")
        assert result["query_or_id"] == "test-query"

    # ---- memory_status ----

    def test_memory_status_empty(self):
        from scholaragent.mcp_server import _memory_status

        store = self._make_store()
        result = _memory_status(store=store)
        assert result["total_entries"] == 0
        assert result["entries_by_source"] == {}
        assert result["research_queries_logged"] == 0

    def test_memory_status_with_entries(self):
        from scholaragent.mcp_server import _memory_status

        store = self._make_store()
        store.add(self._make_entry("Paper 1", source_type="paper", source_ref="ref1"))
        store.add(self._make_entry("Paper 2", source_type="paper", source_ref="ref2"))
        store.add(self._make_entry("Code 1", source_type="code", source_ref="ref3"))
        result = _memory_status(store=store)
        assert result["total_entries"] == 3
        assert result["entries_by_source"]["paper"] == 2
        assert result["entries_by_source"]["code"] == 1

    def test_memory_status_includes_db_path(self):
        from scholaragent.mcp_server import _memory_status

        store = self._make_store()
        result = _memory_status(store=store)
        assert "db_path" in result
        assert result["db_path"] == self.db_path

    def test_memory_status_counts_research_log(self):
        from scholaragent.mcp_server import _memory_status

        store = self._make_store()
        store.log_research(query="test query", depth="quick", focus="implementation", result_count=5)
        result = _memory_status(store=store)
        assert result["research_queries_logged"] == 1

    # ---- memory_research validation ----

    def test_memory_research_rejects_invalid_depth(self):
        from scholaragent.mcp_server import _memory_research

        pipeline = MagicMock()
        result = _memory_research(pipeline=pipeline, query="test", depth="invalid")
        assert "error" in result
        pipeline.run.assert_not_called()

    def test_memory_research_rejects_invalid_focus(self):
        from scholaragent.mcp_server import _memory_research

        pipeline = MagicMock()
        result = _memory_research(pipeline=pipeline, query="test", focus="invalid")
        assert "error" in result
        pipeline.run.assert_not_called()

    def test_memory_research_accepts_all_valid_depths(self):
        from scholaragent.mcp_server import _memory_research, VALID_DEPTHS

        for depth in VALID_DEPTHS:
            pipeline = MagicMock()
            pipeline.run.return_value = {"status": "completed"}
            result = _memory_research(pipeline=pipeline, query="test", depth=depth)
            assert "error" not in result
            pipeline.run.assert_called_once()

    def test_memory_research_accepts_all_valid_focuses(self):
        from scholaragent.mcp_server import _memory_research, VALID_FOCUSES

        for focus in VALID_FOCUSES:
            pipeline = MagicMock()
            pipeline.run.return_value = {"status": "completed"}
            result = _memory_research(pipeline=pipeline, query="test", focus=focus)
            assert "error" not in result
            pipeline.run.assert_called_once()

    def test_memory_research_passes_params_to_pipeline(self):
        from scholaragent.mcp_server import _memory_research

        pipeline = MagicMock()
        pipeline.run.return_value = {"status": "completed"}
        _memory_research(pipeline=pipeline, query="my query", depth="deep", focus="theory")
        pipeline.run.assert_called_once_with(query="my query", depth="deep", focus="theory")


class TestMCPToolJSONResponses:
    """Test the MCP tool wrappers that return JSON strings."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.db_path = str(tmp_path / "test_memory.db")
        self._original_store = None
        with patch.dict(os.environ, {"SCHOLAR_MEMORY_DB": self.db_path}):
            yield
        # Cleanup global state
        import scholaragent.mcp_server as mod
        if self._original_store is not None:
            mod._store = self._original_store

    def _patch_store(self):
        """Patch _get_store to return a test store with fake embeddings."""
        import scholaragent.mcp_server as mod
        from scholaragent.memory.store import MemoryStore

        self._original_store = mod._store
        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        mod._store = store
        return store

    def test_memory_lookup_returns_valid_json(self):
        self._patch_store()
        from scholaragent.mcp_server import memory_lookup

        result = memory_lookup(query="test")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "results" in parsed

    def test_memory_get_returns_valid_json(self):
        self._patch_store()
        from scholaragent.mcp_server import memory_get

        result = memory_get(entry_id="nonexistent")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "error" in parsed

    def test_memory_store_returns_valid_json(self):
        self._patch_store()
        from scholaragent.mcp_server import memory_store

        result = memory_store(content="Test content", source="ref", tags=["t"])
        parsed = json.loads(result)
        assert parsed["status"] == "stored"

    def test_memory_forget_returns_valid_json(self):
        self._patch_store()
        from scholaragent.mcp_server import memory_forget

        result = memory_forget(query_or_id="nonexistent-id")
        parsed = json.loads(result)
        assert "deleted" in parsed

    def test_memory_status_returns_valid_json(self):
        self._patch_store()
        from scholaragent.mcp_server import memory_status

        result = memory_status()
        parsed = json.loads(result)
        assert "total_entries" in parsed

    def test_memory_store_error_returns_valid_json(self):
        self._patch_store()
        from scholaragent.mcp_server import memory_store

        result = memory_store(content="", source="ref", tags=[])
        parsed = json.loads(result)
        assert "error" in parsed


class TestMCPCleanup:
    """Test the atexit cleanup handler."""

    def test_cleanup_function_exists_and_callable(self):
        from scholaragent.mcp_server import _cleanup

        assert callable(_cleanup)

    def test_cleanup_when_store_is_none(self):
        import scholaragent.mcp_server as mod

        original = mod._store
        try:
            mod._store = None
            mod._cleanup()  # should not raise
            assert mod._store is None
        finally:
            mod._store = original

    def test_cleanup_closes_store(self):
        import scholaragent.mcp_server as mod

        mock_store = MagicMock()
        original = mod._store
        try:
            mod._store = mock_store
            mod._cleanup()
            mock_store.close.assert_called_once()
            assert mod._store is None
        finally:
            mod._store = original


class TestMCPValidationConstants:
    """Test that validation constants are properly defined."""

    def test_valid_depths_defined(self):
        from scholaragent.mcp_server import VALID_DEPTHS

        assert "quick" in VALID_DEPTHS
        assert "normal" in VALID_DEPTHS
        assert "deep" in VALID_DEPTHS
        assert len(VALID_DEPTHS) == 3

    def test_valid_focuses_defined(self):
        from scholaragent.mcp_server import VALID_FOCUSES

        assert "implementation" in VALID_FOCUSES
        assert "theory" in VALID_FOCUSES
        assert "comparison" in VALID_FOCUSES
        assert len(VALID_FOCUSES) == 3

    def test_valid_depths_is_frozenset(self):
        from scholaragent.mcp_server import VALID_DEPTHS

        assert isinstance(VALID_DEPTHS, frozenset)

    def test_valid_focuses_is_frozenset(self):
        from scholaragent.mcp_server import VALID_FOCUSES

        assert isinstance(VALID_FOCUSES, frozenset)


class TestMCPThreadSafety:
    """Verify lazy singletons are thread-safe."""

    def test_concurrent_get_store_returns_same_instance(self, tmp_path, monkeypatch):
        """Multiple threads calling _get_store() must get the same instance."""
        import scholaragent.mcp_server as mod

        monkeypatch.setattr(mod, "_store", None)
        monkeypatch.setattr(mod, "_pipeline", None)
        monkeypatch.setattr(mod, "DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(mod, "OpenAIEmbeddings", lambda: FakeEmbeddings())

        stores = []
        barrier = threading.Barrier(4)

        def grab():
            barrier.wait()
            stores.append(mod._get_store())

        threads = [threading.Thread(target=grab) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(stores) == 4
        assert all(s is stores[0] for s in stores)
