"""ResearchResult — typed result of a research run."""
from __future__ import annotations


class TestResearchResult:
    def test_minimum_fields(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="completed",
            query="rlhf",
            requested_depth="normal",
            actual_depth="normal",
            entries_added=3,
        )
        assert r.errors == []
        assert r.fallback_reason is None
        assert r.message == ""

    def test_to_dict_exposes_depth_alias_for_backwards_compat(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="completed",
            query="rlhf",
            requested_depth="deep",
            actual_depth="normal",
            entries_added=3,
            fallback_reason="Dispatcher timed out",
        )
        d = r.to_dict()
        assert d["depth"] == "normal"  # alias for actual_depth
        assert d["requested_depth"] == "deep"
        assert d["actual_depth"] == "normal"
        assert d["fallback_reason"] == "Dispatcher timed out"
        assert d["entries_added"] == 3

    def test_to_dict_omits_empty_fallback_reason(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="completed", query="q", requested_depth="quick",
            actual_depth="quick", entries_added=1,
        )
        d = r.to_dict()
        assert "fallback_reason" not in d

    def test_to_dict_includes_cached_results_when_set(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="cached", query="q", requested_depth="normal",
            actual_depth="normal", entries_added=0, cached_results=5,
        )
        d = r.to_dict()
        assert d["cached_results"] == 5
        assert d["status"] == "cached"

    def test_to_dict_omits_cached_results_when_none(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="completed", query="q", requested_depth="quick",
            actual_depth="quick", entries_added=2,
        )
        assert "cached_results" not in r.to_dict()
