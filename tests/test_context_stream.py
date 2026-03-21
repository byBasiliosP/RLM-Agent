"""Tests for ContextStream data model, persistence, and agent integration."""

import json
import pytest

from scholaragent.core.context import ContextStream, PipelineState, StreamEvent


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    def test_empty_state(self):
        state = PipelineState()
        assert state.papers == []
        assert state.findings == {}
        assert state.assessments == {}
        assert state.themes == {}
        assert state.synthesis == ""

    def test_to_dict(self):
        state = PipelineState(papers=[{"title": "Test"}])
        d = state.to_dict()
        assert d["papers"] == [{"title": "Test"}]
        assert d["findings"] == {}

    def test_from_dict(self):
        d = {"papers": [{"title": "X"}], "findings": {}, "assessments": {}, "themes": {}, "synthesis": "done"}
        state = PipelineState.from_dict(d)
        assert state.papers == [{"title": "X"}]
        assert state.synthesis == "done"


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_creation(self):
        event = StreamEvent(agent="scout", event_type="papers_found", data={"count": 3})
        assert event.agent == "scout"
        assert event.event_type == "papers_found"
        assert event.timestamp  # auto-generated

    def test_to_dict(self):
        event = StreamEvent(agent="scout", event_type="papers_found", data={"count": 3})
        d = event.to_dict()
        assert d["agent"] == "scout"
        assert "timestamp" in d


class TestContextStream:
    """Tests for ContextStream creation and methods."""

    def test_creation(self):
        stream = ContextStream(query="test query")
        assert stream.query == "test query"
        assert stream.id  # UUID generated
        assert stream.state.papers == []
        assert stream.traces == {}
        assert stream.events == []

    def test_push_appends_event(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        assert len(stream.events) == 1
        assert stream.events[0].agent == "scout"
        assert stream.events[0].event_type == "papers_found"

    def test_push_updates_state_papers(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        assert len(stream.state.papers) == 1
        stream.push("scout", "papers_found", {"papers": [{"title": "B"}]})
        assert len(stream.state.papers) == 2

    def test_push_updates_state_findings(self):
        stream = ContextStream(query="test")
        stream.push("reader", "finding_extracted", {"paper_ref": "arxiv:123", "finding": {"key_claims": ["x"]}})
        assert "arxiv:123" in stream.state.findings

    def test_push_updates_state_assessments(self):
        stream = ContextStream(query="test")
        stream.push("critic", "assessment_complete", {"paper_ref": "arxiv:123", "assessment": {"score": 0.9}})
        assert "arxiv:123" in stream.state.assessments

    def test_push_updates_state_themes(self):
        stream = ContextStream(query="test")
        stream.push("analyst", "themes_identified", {"themes": {"gaps": ["x"]}})
        assert stream.state.themes == {"gaps": ["x"]}

    def test_push_updates_state_synthesis(self):
        stream = ContextStream(query="test")
        stream.push("synthesizer", "synthesis_complete", {"synthesis": "Final report"})
        assert stream.state.synthesis == "Final report"

    def test_commit_saves_trace(self):
        stream = ContextStream(query="test")
        messages = [{"role": "system", "content": "You are scout"}, {"role": "user", "content": "find papers"}]
        stream.commit("scout", messages)
        assert stream.traces["scout"] == messages

    def test_commit_overwrites_trace(self):
        stream = ContextStream(query="test")
        stream.commit("scout", [{"role": "user", "content": "v1"}])
        stream.commit("scout", [{"role": "user", "content": "v2"}])
        assert stream.traces["scout"] == [{"role": "user", "content": "v2"}]

    def test_read_all(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        data = stream.read()
        assert "state" in data
        assert "traces" in data
        assert "events" in data

    def test_read_filtered_by_agent(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.push("reader", "finding_extracted", {"paper_ref": "x", "finding": {}})
        stream.commit("scout", [{"role": "user", "content": "scout task"}])
        stream.commit("reader", [{"role": "user", "content": "reader task"}])
        data = stream.read(agent="scout")
        assert "scout" in data["traces"]
        assert "reader" not in data["traces"]
        assert all(e["agent"] == "scout" for e in data["events"])

    def test_to_dict_and_from_dict_roundtrip(self):
        stream = ContextStream(query="roundtrip test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        d = stream.to_dict()
        restored = ContextStream.from_dict(d)
        assert restored.id == stream.id
        assert restored.query == stream.query
        assert restored.state.papers == stream.state.papers
        assert restored.traces == stream.traces
        assert len(restored.events) == len(stream.events)

    def test_updated_at_changes_on_push(self):
        stream = ContextStream(query="test")
        original = stream.updated_at
        stream.push("scout", "papers_found", {"papers": []})
        assert stream.updated_at >= original

    def test_push_with_save_callback(self):
        saved = []
        stream = ContextStream(query="test", on_save=lambda s: saved.append(s.id))
        stream.push("scout", "papers_found", {"papers": []})
        assert len(saved) == 1

    def test_commit_with_save_callback(self):
        saved = []
        stream = ContextStream(query="test", on_save=lambda s: saved.append(s.id))
        stream.commit("scout", [])
        assert len(saved) == 1
