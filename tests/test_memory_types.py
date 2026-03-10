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


class TestSmartSummary:
    def test_short_content_returned_unchanged(self):
        from scholaragent.memory.types import MemoryEntry
        assert MemoryEntry.smart_summary("Short text.") == "Short text."

    def test_truncates_at_sentence_boundary(self):
        from scholaragent.memory.types import MemoryEntry
        content = "First sentence about RLHF. Second sentence about DPO. Third sentence that goes on and on and on and on and on and on and on to push us well past the two hundred character limit for summaries in the system."
        result = MemoryEntry.smart_summary(content)
        assert result.endswith('.')
        assert len(result) <= 200
        assert 'First sentence' in result

    def test_falls_back_to_word_boundary_for_code(self):
        from scholaragent.memory.types import MemoryEntry
        content = "def train_reward_model(dataset, model, epochs=3):\n    optimizer = AdamW(model.parameters(), lr=1e-5)\n    for epoch in range(epochs):\n        for batch in dataset:\n            chosen, rejected = batch\n            loss = compute_loss(chosen, rejected)"
        result = MemoryEntry.smart_summary(content)
        assert result.endswith('...')
        assert not result.endswith(' ...')  # no trailing space before ellipsis
        assert len(result) <= 203  # 200 + '...'

    def test_does_not_truncate_too_short(self):
        from scholaragent.memory.types import MemoryEntry
        # Sentence boundary is too early (less than 1/3 of max_length)
        content = "Hi. " + "x" * 250
        result = MemoryEntry.smart_summary(content)
        # Should not truncate to just "Hi." (too short), should use word/char fallback
        assert len(result) > 50

    def test_exact_boundary(self):
        from scholaragent.memory.types import MemoryEntry
        content = "A" * 200
        assert MemoryEntry.smart_summary(content) == content
        content201 = "A" * 201
        result = MemoryEntry.smart_summary(content201)
        assert len(result) <= 203  # may have '...'


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
