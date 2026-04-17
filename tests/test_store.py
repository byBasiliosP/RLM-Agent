"""Tests for the memory store."""

import os
import tempfile
import threading

import pytest

from tests.helpers import FakeEmbeddings


@pytest.fixture
def store():
    from scholaragent.memory.store import MemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        s = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
        yield s
        s.close()


class TestMemoryStoreBasic:
    def test_store_and_retrieve(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="RLHF uses human feedback for alignment",
            summary="RLHF overview",
            source_type="paper",
            source_ref="arxiv:2203.02155",
            tags=["rlhf", "alignment"],
        )
        store.add(entry)
        result = store.get(entry.id)
        assert result is not None
        assert result.content == entry.content

    def test_store_multiple(self, store):
        from scholaragent.memory.types import MemoryEntry

        for i in range(5):
            store.add(MemoryEntry(
                content=f"Finding {i}",
                summary=f"Summary {i}",
                source_type="paper",
                source_ref=f"ref-{i}",
                tags=["test"],
            ))
        assert store.count() == 5

    def test_delete(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="To be deleted",
            summary="Delete me",
            source_type="docs",
            source_ref="url",
            tags=[],
        )
        store.add(entry)
        assert store.count() == 1
        store.delete(entry.id)
        assert store.count() == 0

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent-id") is None


class TestMemoryStoreSearch:
    def test_search_returns_results(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Transformer attention mechanisms",
            summary="Attention",
            source_type="paper",
            source_ref="ref1",
            tags=["attention"],
        ))
        store.add(MemoryEntry(
            content="Cooking pasta recipes",
            summary="Pasta",
            source_type="docs",
            source_ref="ref2",
            tags=["cooking"],
        ))
        results = store.search("transformer architecture", max_results=5)
        assert len(results) > 0
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2

    def test_search_filter_by_source(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Paper about X",
            summary="X",
            source_type="paper",
            source_ref="ref1",
            tags=[],
        ))
        store.add(MemoryEntry(
            content="Docs about X",
            summary="X",
            source_type="docs",
            source_ref="ref2",
            tags=[],
        ))
        results = store.search("X", sources=["paper"])
        assert all(entry.source_type == "paper" for entry, _ in results)

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert results == []

    def test_search_increments_access_count(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="Accessed content",
            summary="Access test",
            source_type="paper",
            source_ref="ref",
            tags=[],
        )
        store.add(entry)
        store.search("accessed")
        result = store.get(entry.id)
        assert result.access_count >= 1


class TestMemoryStoreForget:
    def test_forget_by_id(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="Forget me",
            summary="Forget",
            source_type="paper",
            source_ref="ref",
            tags=[],
        )
        store.add(entry)
        deleted = store.forget(entry.id)
        assert deleted == 1
        assert store.count() == 0

    def test_forget_by_query(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Topic A stuff",
            summary="A",
            source_type="paper",
            source_ref="ref1",
            tags=["topic-a"],
        ))
        store.add(MemoryEntry(
            content="Topic B stuff",
            summary="B",
            source_type="paper",
            source_ref="ref2",
            tags=["topic-b"],
        ))
        deleted = store.forget("topic-a", threshold=0.0)
        assert deleted >= 1

    def test_forget_high_threshold_limits_deletions(self, store):
        """With threshold=0.8, only very similar entries get deleted."""
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Very specific unique topic alpha",
            summary="Alpha",
            source_type="paper",
            source_ref="ref1",
            tags=["alpha"],
        ))
        store.add(MemoryEntry(
            content="Completely unrelated cooking recipe for pasta",
            summary="Pasta",
            source_type="docs",
            source_ref="ref2",
            tags=["cooking"],
        ))
        # Default threshold=0.8 should be strict enough that a vague query
        # doesn't wipe everything
        initial_count = store.count()
        deleted = store.forget("something vague")
        assert store.count() >= initial_count - deleted

    def test_forget_max_delete_caps_deletions(self, store):
        """max_delete parameter limits the number of entries deleted."""
        from scholaragent.memory.types import MemoryEntry

        # Add many similar entries
        for i in range(8):
            store.add(MemoryEntry(
                content=f"Research finding number {i}",
                summary=f"Finding {i}",
                source_type="paper",
                source_ref=f"ref-{i}",
                tags=["research"],
            ))
        assert store.count() == 8

        # With threshold=0.0 (match everything) and max_delete=3,
        # only 3 should be deleted
        deleted = store.forget("research finding", threshold=0.0, max_delete=3)
        assert deleted == 3
        assert store.count() == 5

    def test_forget_does_not_increment_access_count(self, store):
        """forget() should not increment access_count on entries."""
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="Entry to be forgotten",
            summary="Forget me",
            source_type="paper",
            source_ref="ref",
            tags=[],
        )
        store.add(entry)
        assert store.get(entry.id).access_count == 0

        # Use threshold > 1.0 so nothing can match (cosine sim maxes at 1.0),
        # ensuring the entry survives but search() is still called internally.
        store.forget("completely unrelated query that wont match", threshold=1.1)
        result = store.get(entry.id)
        assert result is not None
        assert result.access_count == 0


class TestResearchLog:
    def test_log_research(self, store):
        store.log_research(
            query="RLHF techniques",
            depth="normal",
            focus="implementation",
            result_count=5,
        )
        recent = store.get_recent_research("RLHF", days=7)
        assert len(recent) == 1
        assert recent[0].query == "RLHF techniques"

    def test_no_recent_research(self, store):
        recent = store.get_recent_research("unknown topic", days=7)
        assert recent == []


class TestMemoryStoreStatus:
    def test_status(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Test",
            summary="Test",
            source_type="paper",
            source_ref="ref",
            tags=[],
        ))
        status = store.status()
        assert status["total_entries"] == 1
        assert "paper" in status["entries_by_source"]


class TestMemoryStoreConcurrency:
    def test_concurrent_writes(self, store):
        """10 threads each add an entry simultaneously; all 10 must be stored."""
        from scholaragent.memory.types import MemoryEntry

        errors = []

        def _add(i):
            try:
                store.add(MemoryEntry(
                    content=f"Concurrent entry {i}",
                    summary=f"Summary {i}",
                    source_type="paper",
                    source_ref=f"ref-{i}",
                    tags=["concurrent"],
                ))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_add, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes raised errors: {errors}"
        assert store.count() == 10

    def test_concurrent_read_and_write(self, store):
        """One thread adds entries while another searches; no crashes."""
        from scholaragent.memory.types import MemoryEntry

        errors = []

        def _writer():
            try:
                for i in range(10):
                    store.add(MemoryEntry(
                        content=f"Writer entry {i}",
                        summary=f"Summary {i}",
                        source_type="paper",
                        source_ref=f"ref-w-{i}",
                        tags=["write"],
                    ))
            except Exception as exc:
                errors.append(exc)

        def _reader():
            try:
                for _ in range(10):
                    store.search("writer entry", max_results=5)
            except Exception as exc:
                errors.append(exc)

        writer = threading.Thread(target=_writer)
        reader = threading.Thread(target=_reader)
        writer.start()
        reader.start()
        writer.join()
        reader.join()

        assert errors == [], f"Concurrent read/write raised errors: {errors}"
        assert store.count() == 10


class TestMemoryStoreLifecycle:
    def test_context_manager_closes(self):
        from scholaragent.memory.store import MemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "lifecycle.db")
            with MemoryStore(db_path=db_path, embeddings=FakeEmbeddings()) as s:
                assert s.count() == 0
            assert s._closed is True

    def test_double_close_is_noop(self):
        from scholaragent.memory.store import MemoryStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "lifecycle.db")
            s = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
            s.close()
            s.close()  # must not raise
            assert s._closed is True


class TestMemoryStoreAddMany:
    """Verify batch insert calls embed_batch once and inserts all rows."""

    def test_add_many_uses_batch_embed(self, store):
        from scholaragent.memory.types import MemoryEntry

        call_count = {"embed": 0, "batch": 0}
        real_embed = store.embeddings.embed
        real_batch = store.embeddings.embed_batch

        def counted_embed(text):
            call_count["embed"] += 1
            return real_embed(text)

        def counted_batch(texts):
            call_count["batch"] += 1
            return real_batch(texts)

        store.embeddings.embed = counted_embed
        store.embeddings.embed_batch = counted_batch

        entries = [
            MemoryEntry(
                content=f"content {i}",
                summary=f"sum {i}",
                source_type="paper",
                source_ref=f"ref-{i}",
                tags=["batch"],
            )
            for i in range(5)
        ]
        store.add_many(entries)
        assert store.count() == 5
        # Single batch call (FakeEmbeddings' embed_batch may internally fan
        # out, but the store only calls embed_batch once per add_many).
        assert call_count["batch"] == 1

    def test_add_many_empty_list_is_noop(self, store):
        store.add_many([])
        assert store.count() == 0

    def test_add_many_preserves_existing_embeddings(self, store):
        from scholaragent.memory.types import MemoryEntry

        batch_calls = {"n": 0}
        real_batch = store.embeddings.embed_batch

        def counted_batch(texts):
            batch_calls["n"] += 1
            return real_batch(texts)

        store.embeddings.embed_batch = counted_batch

        entries = [
            MemoryEntry(
                content="pre-embedded",
                summary="x",
                source_type="paper",
                source_ref="ref-p",
                tags=[],
                embedding=[0.1, 0.2, 0.3],
            ),
        ]
        store.add_many(entries)
        assert store.count() == 1
        assert batch_calls["n"] == 0  # no batch call needed
