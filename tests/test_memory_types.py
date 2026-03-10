"""Tests for memory data types."""
import pytest

class TestMemoryEntry:
    def test_creation(self):
        from scholaragent.memory.types import MemoryEntry
        entry = MemoryEntry(
            content="Transformers use self-attention",
            summary="Self-attention in transformers",
            source_type="paper",
            source_ref="arxiv:1706.03762",
            tags=["attention"],
        )
        assert entry.content == "Transformers use self-attention"
        assert entry.source_type == "paper"
        assert entry.id  # auto-generated uuid
        assert entry.access_count == 0

    def test_to_dict(self):
        from scholaragent.memory.types import MemoryEntry
        entry = MemoryEntry(content="Test", summary="T", source_type="docs", source_ref="url", tags=["t"])
        d = entry.to_dict()
        assert d["content"] == "Test"
        assert "id" in d
        assert d["access_count"] == 0

    def test_to_compact_dict(self):
        from scholaragent.memory.types import MemoryEntry
        entry = MemoryEntry(content="Full content here", summary="Short summary", source_type="paper", source_ref="arxiv:1234", tags=["ml"])
        d = entry.to_compact_dict()
        assert d["summary"] == "Short summary"
        assert d["source_type"] == "paper"
        assert d["source_ref"] == "arxiv:1234"
        assert d["tags"] == ["ml"]
        assert "id" in d
        assert "content" not in d
        assert "embedding" not in d
        assert "created_at" not in d
        assert "access_count" not in d

    def test_invalid_source_type(self):
        from scholaragent.memory.types import MemoryEntry
        with pytest.raises(ValueError, match="source_type"):
            MemoryEntry(content="x", summary="x", source_type="invalid", source_ref="x", tags=[])

class TestResearchLogEntry:
    def test_creation(self):
        from scholaragent.memory.types import ResearchLogEntry
        log = ResearchLogEntry(query="RLHF", depth="normal", focus="implementation", result_count=5)
        assert log.query == "RLHF"
        assert log.id

    def test_to_dict(self):
        from scholaragent.memory.types import ResearchLogEntry
        log = ResearchLogEntry(query="q", depth="quick", focus="theory", result_count=3)
        d = log.to_dict()
        assert d["query"] == "q"
        assert d["depth"] == "quick"

class TestSourceTypes:
    def test_valid_types(self):
        from scholaragent.memory.types import VALID_SOURCE_TYPES
        assert "paper" in VALID_SOURCE_TYPES
        assert "docs" in VALID_SOURCE_TYPES
        assert "code" in VALID_SOURCE_TYPES
