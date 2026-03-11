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


class TestContextStreamingWorkflow:
    """End-to-end tests for the browse-then-read on-demand retrieval pattern.

    Verifies that memory_lookup returns compact, high-quality summaries
    and memory_get retrieves full content on demand — keeping the client
    agent's context window lean.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.db_path = str(tmp_path / "test_memory.db")
        with patch.dict(os.environ, {"SCHOLAR_MEMORY_DB": self.db_path}):
            yield

    def _make_store(self):
        from scholaragent.memory.store import MemoryStore

        return MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())

    def _populate_store(self, store):
        """Add realistic entries simulating papers, code, and docs."""
        from scholaragent.memory.types import MemoryEntry

        entries_data = [
            (
                "Reinforcement Learning from Human Feedback (RLHF) has emerged as "
                "a powerful technique for aligning large language models with human "
                "preferences. This paper surveys recent advances in RLHF, including "
                "Direct Preference Optimization (DPO), which eliminates the need for "
                "a separate reward model by directly optimizing the policy using "
                "preference data. We compare RLHF, DPO, and other alignment methods "
                "across multiple benchmarks.",
                "paper",
                "arxiv:2401.12345",
                ["rlhf", "alignment"],
            ),
            (
                "def train_reward_model(dataset, model, epochs=3):\n"
                "    optimizer = AdamW(model.parameters(), lr=1e-5)\n"
                "    for epoch in range(epochs):\n"
                "        for batch in dataset:\n"
                "            chosen, rejected = batch\n"
                "            loss = -torch.log(torch.sigmoid(\n"
                "                model(chosen) - model(rejected)))\n"
                "            loss.backward()\n"
                "            optimizer.step()\n"
                "    return model",
                "code",
                "github:anthropic/rlhf-examples",
                ["rlhf", "training"],
            ),
            (
                "Flash Attention v2 achieves 2-4x speedup over standard attention "
                "by tiling the computation to fit in GPU SRAM, avoiding materialization "
                "of the full NxN attention matrix in HBM. Key insight: recompute "
                "attention weights during backward pass instead of storing them, "
                "trading FLOPs for memory bandwidth. Benchmarks show 3x speedup on "
                "A100 GPUs for sequence lengths above 2048.",
                "paper",
                "arxiv:2307.08691",
                ["attention", "efficiency"],
            ),
        ]

        entries = []
        for content, stype, sref, tags in entries_data:
            entry = MemoryEntry(
                content=content,
                summary=MemoryEntry.smart_summary(content),
                source_type=stype,
                source_ref=sref,
                tags=tags,
            )
            store.add(entry)
            entries.append(entry)
        return entries

    def test_summaries_end_cleanly(self):
        """All smart summaries must end at sentence boundaries or with ellipsis."""
        store = self._make_store()
        entries = self._populate_store(store)
        for entry in entries:
            summary = entry.summary
            assert summary.endswith('.') or summary.endswith('...') or summary.endswith('?') or summary.endswith('!'), \
                f"Summary has unclean ending: ...{summary[-20:]!r}"

    def test_summaries_never_cut_mid_word(self):
        """Summaries must not end with a partial word (no trailing letter fragments)."""
        store = self._make_store()
        entries = self._populate_store(store)
        for entry in entries:
            summary = entry.summary
            # If truncated, should end with '.', '...', '?', or '!'
            if len(entry.content) > 200:
                assert summary[-1] in '.?!', \
                    f"Long content summary should end with punctuation, got: ...{summary[-10:]!r}"

    def test_compact_lookup_excludes_content(self):
        """Compact lookup must not include full content in any result."""
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        self._populate_store(store)
        result = _memory_lookup(store=store, query="RLHF alignment", compact=True)
        for r in result["results"]:
            assert "content" not in r, "Compact results must not contain 'content' field"
            assert "summary" in r
            assert "id" in r
            assert "relevance_score" in r

    def test_compact_is_smaller_than_full(self):
        """Compact JSON payload must be meaningfully smaller than full."""
        from scholaragent.mcp_server import _memory_lookup

        store = self._make_store()
        self._populate_store(store)

        compact = _memory_lookup(store=store, query="RLHF", compact=True)
        full = _memory_lookup(store=store, query="RLHF", compact=False)

        compact_size = len(json.dumps(compact))
        full_size = len(json.dumps(full))
        assert compact_size < full_size, "Compact must be smaller than full"
        # At least 20% reduction
        assert compact_size < full_size * 0.8, \
            f"Compact ({compact_size}) should be at least 20% smaller than full ({full_size})"

    def test_browse_then_read_full_workflow(self):
        """Complete workflow: browse summaries, pick best, read full content."""
        from scholaragent.mcp_server import _memory_lookup, _memory_get

        store = self._make_store()
        entries = self._populate_store(store)

        # Step 1: Browse — compact lookup
        lookup = _memory_lookup(store=store, query="reward model training", compact=True)
        assert len(lookup["results"]) > 0

        # Verify compact format
        for r in lookup["results"]:
            assert "content" not in r
            assert "id" in r
            assert "summary" in r

        # Step 2: Pick — select the best result by score
        best = max(lookup["results"], key=lambda r: r["relevance_score"])
        assert best["relevance_score"] > 0

        # Step 3: Read — fetch full content by ID
        full = _memory_get(store=store, entry_id=best["id"])
        assert "error" not in full
        assert "content" in full
        assert len(full["content"]) > len(best["summary"])

    def test_multiple_reads_accumulate_less_than_batch(self):
        """Reading 1-2 entries on demand should use less payload than batch dump."""
        from scholaragent.mcp_server import _memory_lookup, _memory_get

        store = self._make_store()
        self._populate_store(store)

        # Batch dump
        batch = json.dumps(_memory_lookup(store=store, query="RLHF", compact=False))

        # On-demand: browse + read 1
        browse = json.dumps(_memory_lookup(store=store, query="RLHF", compact=True))
        best_id = max(
            _memory_lookup(store=store, query="RLHF", compact=True)["results"],
            key=lambda r: r["relevance_score"],
        )["id"]
        read_one = json.dumps(_memory_get(store=store, entry_id=best_id))

        on_demand_total = len(browse) + len(read_one)
        assert on_demand_total < len(batch), \
            f"On-demand ({on_demand_total}) should be less than batch ({len(batch)})"

    def test_memory_get_preserves_all_fields(self):
        """memory_get must return all fields including content, tags, created_at."""
        from scholaragent.mcp_server import _memory_get

        store = self._make_store()
        entries = self._populate_store(store)
        entry = entries[0]

        full = _memory_get(store=store, entry_id=entry.id)
        assert full["id"] == entry.id
        assert full["content"] == entry.content
        assert full["summary"] == entry.summary
        assert full["source_type"] == entry.source_type
        assert full["source_ref"] == entry.source_ref
        assert "tags" in full
        assert "created_at" in full
        assert "access_count" in full

    def test_code_summary_uses_ellipsis(self):
        """Code content (no sentence endings) should use word-boundary + ellipsis."""
        from scholaragent.memory.types import MemoryEntry

        code = (
            "def process_batch(data, config):\n"
            "    results = []\n"
            "    for item in data:\n"
            "        transformed = apply_transform(item, config.transform)\n"
            "        validated = validate_output(transformed, config.schema)\n"
            "        results.append(validated)\n"
            "    return aggregate_results(results, config.aggregation_method)"
        )
        summary = MemoryEntry.smart_summary(code)
        assert summary.endswith("..."), f"Code summary should end with '...', got: {summary[-10:]!r}"

    def test_paper_summary_uses_sentence_boundary(self):
        """Paper abstracts should truncate at sentence boundaries."""
        from scholaragent.memory.types import MemoryEntry

        abstract = (
            "We introduce a novel approach to neural network pruning. "
            "Our method uses gradient sensitivity analysis to identify "
            "redundant parameters. Experiments on GPT-2 show 4x compression "
            "with less than 1 percent accuracy loss. This makes deployment "
            "on edge devices practical for the first time."
        )
        summary = MemoryEntry.smart_summary(abstract)
        assert summary.endswith("."), f"Paper summary should end with '.', got: {summary[-10:]!r}"
        # Should not include the last sentence (would exceed 200 chars)
        assert len(summary) <= 200


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
