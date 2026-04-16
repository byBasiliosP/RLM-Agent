"""SourceCollector owns raw retrieval from arXiv, S2, GitHub, and docs."""
from __future__ import annotations

from unittest.mock import patch

import pytest


class TestSourceCollectorContract:
    def test_importable(self):
        from scholaragent.memory.source_collector import SourceCollector
        assert SourceCollector is not None

    def test_default_source_types(self):
        from scholaragent.memory.source_collector import SourceCollector
        assert SourceCollector.DEFAULT_SOURCES == ("paper", "code", "docs")

    def test_collect_returns_results_and_errors(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv",
                   return_value='[{"arxiv_id": "1", "title": "T", "authors": [], "abstract": "A"}]'), \
             patch("scholaragent.memory.source_collector.search_semantic_scholar",
                   return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[]), \
             patch("scholaragent.memory.source_collector.search_docs",
                   return_value=[]):
            results, errors = SourceCollector().collect("rlhf")

        assert errors == []
        assert len(results) == 1
        assert results[0]["source_type"] == "paper"
        assert results[0]["source_ref"] == "arxiv:1"

    def test_collect_captures_errors_without_raising(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv",
                   side_effect=RuntimeError("network down")), \
             patch("scholaragent.memory.source_collector.search_semantic_scholar",
                   return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[]), \
             patch("scholaragent.memory.source_collector.search_docs",
                   return_value=[]):
            results, errors = SourceCollector().collect("rlhf")

        assert results == []
        assert len(errors) == 1
        assert "arXiv" in errors[0]
        assert "network down" in errors[0]

    def test_collect_honors_source_type_filter(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv") as arx, \
             patch("scholaragent.memory.source_collector.search_semantic_scholar") as s2, \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[{"content": "x", "source_type": "code", "source_ref": "github:a/b"}]), \
             patch("scholaragent.memory.source_collector.search_docs",
                   return_value=[]):
            arx.return_value = '[]'
            s2.return_value = '[]'
            results, _ = SourceCollector().collect("rlhf", source_types=("code",))

        arx.assert_not_called()
        s2.assert_not_called()
        assert len(results) == 1
        assert results[0]["source_type"] == "code"

    def test_collect_passes_code_language(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv", return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_semantic_scholar", return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[]) as gh, \
             patch("scholaragent.memory.source_collector.search_docs", return_value=[]):
            SourceCollector().collect("rlhf", code_language="rust")

        gh.assert_called_once()
        assert gh.call_args.kwargs["language"] == "rust"


class TestSourceCollectorDeduplicate:
    def test_dedup_keeps_s2_over_arxiv_on_arxiv_id_collision(self):
        from scholaragent.memory.source_collector import SourceCollector

        items = [
            {"content": "Title: Attention\n\nAbstract: old", "source_type": "paper",
             "source_ref": "arxiv:1706.03762"},
            {"content": "Title: Attention\n\nAbstract: newer + citations",
             "source_type": "paper", "source_ref": "s2:1706.03762"},
        ]
        deduped = SourceCollector().deduplicate(items)
        assert len(deduped) == 1
        assert deduped[0]["source_ref"] == "s2:1706.03762"

    def test_dedup_leaves_non_papers_alone(self):
        from scholaragent.memory.source_collector import SourceCollector
        items = [
            {"content": "code", "source_type": "code", "source_ref": "gh:a/b"},
            {"content": "code", "source_type": "code", "source_ref": "gh:a/b"},
        ]
        assert len(SourceCollector().deduplicate(items)) == 2
