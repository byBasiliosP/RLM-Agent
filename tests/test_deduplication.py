"""Tests for paper deduplication in SourceCollector."""

import pytest

from scholaragent.memory.source_collector import SourceCollector


@pytest.fixture
def collector():
    return SourceCollector()


class TestDeduplication:
    def test_arxiv_id_match_removes_duplicate(self, collector):
        results = [
            {"content": "Title: Paper A\n\nAbstract: ...", "source_type": "paper", "source_ref": "arxiv:2401.00001"},
            {"content": "Title: Paper A\n\nAbstract: ...\n\nYear: 2024\nCitations: 10", "source_type": "paper", "source_ref": "s2:abc123"},
        ]
        # S2 entry has same paper but different ref — won't match on arxiv_id here.
        # Only matches if both have same arxiv_id in source_ref.
        deduped = collector.deduplicate(results)
        assert len(deduped) == 1  # title-based match
        # Prefer S2 entry (has citation counts)
        assert deduped[0]["source_ref"] == "s2:abc123"

    def test_title_match_removes_duplicate(self, collector):
        results = [
            {"content": "Title: Attention Is All You Need\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:1706.03762"},
            {"content": "Title: Attention Is All You Need\n\nAbstract: B\n\nYear: 2017\nCitations: 50000", "source_type": "paper", "source_ref": "s2:xyz"},
        ]
        deduped = collector.deduplicate(results)
        assert len(deduped) == 1
        # S2 entry preferred
        assert deduped[0]["source_ref"] == "s2:xyz"

    def test_different_papers_not_deduped(self, collector):
        results = [
            {"content": "Title: Paper One\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:111"},
            {"content": "Title: Paper Two\n\nAbstract: B", "source_type": "paper", "source_ref": "s2:222"},
        ]
        deduped = collector.deduplicate(results)
        assert len(deduped) == 2

    def test_non_paper_entries_pass_through(self, collector):
        results = [
            {"content": "Code snippet", "source_type": "code", "source_ref": "github:org/repo"},
            {"content": "Title: Paper\n\nAbstract: X", "source_type": "paper", "source_ref": "arxiv:123"},
            {"content": "Docs content", "source_type": "docs", "source_ref": "https://docs.python.org"},
        ]
        deduped = collector.deduplicate(results)
        assert len(deduped) == 3

    def test_punctuation_insensitive_title_match(self, collector):
        results = [
            {"content": "Title: GPT-4: A New Model!\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:aaa"},
            {"content": "Title: GPT4 A New Model\n\nAbstract: B", "source_type": "paper", "source_ref": "s2:bbb"},
        ]
        deduped = collector.deduplicate(results)
        assert len(deduped) == 1

    def test_empty_results(self, collector):
        assert collector.deduplicate([]) == []

    def test_arxiv_only_no_dedup(self, collector):
        results = [
            {"content": "Title: Paper\n\nAbstract: A", "source_type": "paper", "source_ref": "arxiv:111"},
        ]
        deduped = collector.deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["source_ref"] == "arxiv:111"
