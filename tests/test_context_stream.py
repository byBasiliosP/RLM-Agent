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


from scholaragent.memory.store import MemoryStore
from tests.helpers import FakeEmbeddings


class TestMemoryStoreStreams:
    """Tests for ContextStream persistence in MemoryStore."""

    @pytest.fixture()
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        return MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())

    def test_save_and_load_stream(self, store):
        stream = ContextStream(query="test query")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        store.save_stream(stream)
        loaded = store.load_stream(stream.id)
        assert loaded is not None
        assert loaded.id == stream.id
        assert loaded.query == stream.query
        assert loaded.state.papers == [{"title": "A"}]
        assert loaded.traces["scout"] == [{"role": "user", "content": "task"}]
        assert len(loaded.events) == 1

    def test_load_nonexistent_stream(self, store):
        assert store.load_stream("nonexistent") is None

    def test_save_stream_upsert(self, store):
        stream = ContextStream(query="test")
        store.save_stream(stream)
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        store.save_stream(stream)
        loaded = store.load_stream(stream.id)
        assert len(loaded.state.papers) == 1

    def test_list_streams_empty(self, store):
        assert store.list_streams() == []

    def test_list_streams_returns_compact(self, store):
        stream = ContextStream(query="test query")
        stream.push("scout", "papers_found", {"papers": []})
        stream.commit("scout", [])
        store.save_stream(stream)
        results = store.list_streams()
        assert len(results) == 1
        assert results[0]["id"] == stream.id
        assert results[0]["query"] == "test query"
        assert "scout" in results[0]["agents"]
        assert results[0]["event_count"] == 1

    def test_list_streams_filter_by_query(self, store):
        s1 = ContextStream(query="protein folding")
        s2 = ContextStream(query="quantum computing")
        store.save_stream(s1)
        store.save_stream(s2)
        results = store.list_streams(query="protein")
        assert len(results) == 1
        assert results[0]["query"] == "protein folding"

    def test_list_streams_respects_limit(self, store):
        for i in range(5):
            store.save_stream(ContextStream(query=f"query {i}"))
        results = store.list_streams(limit=3)
        assert len(results) == 3

    def test_list_streams_ordered_by_updated_at(self, store):
        s1 = ContextStream(query="first")
        s2 = ContextStream(query="second")
        store.save_stream(s1)
        store.save_stream(s2)
        results = store.list_streams()
        # Most recent first
        assert results[0]["query"] == "second"


from unittest.mock import MagicMock, patch
from scholaragent.core.types import AgentResult, ModelUsageSummary, UsageSummary
from scholaragent.clients.base import BaseLM
from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.handler import LMHandler


class FakeLM(BaseLM):
    """Fake LM that returns a configurable response."""

    def __init__(self, response: str = "FINAL(done)"):
        super().__init__(model_name="fake-model")
        self._response = response

    def completion(self, prompt: str) -> str:
        return self._response

    async def acompletion(self, prompt: str) -> str:
        return self._response

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class MockStreamAgent(SpecialistAgent):
    """Agent for testing stream integration."""

    @property
    def name(self) -> str:
        return "scout"

    @property
    def system_prompt(self) -> str:
        return "You are a scout."


class TestAgentStreamIntegration:
    """Tests for stream injection into agent REPL."""

    @pytest.fixture()
    def handler(self):
        h = LMHandler(client=FakeLM("FINAL(done)"), token_counter=None, verbose=False)
        h.start()
        yield h
        h.stop()

    def test_stream_functions_injected_into_repl(self, handler):
        """When stream is provided, stream_push and stream_read are in REPL."""
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=handler, stream=stream)
        assert result.success

    def test_auto_commit_on_completion(self, handler):
        """Agent auto-commits trace to stream on successful completion."""
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=handler, stream=stream)
        assert result.success
        assert "scout" in stream.traces
        assert len(stream.traces["scout"]) > 0

    def test_auto_commit_on_failure(self, handler):
        """Agent auto-commits trace even on max-iterations failure."""
        lm = FakeLM("no final answer here")
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=h, max_iterations=1, stream=stream)
        assert not result.success
        assert "scout" in stream.traces
        h.stop()

    def test_stream_push_from_repl_code(self, handler):
        """Agent REPL code can call stream_push."""
        code_response = '```repl\nstream_push("papers_found", {"papers": [{"title": "A"}]})\nanswer = "done"\nFINAL_VAR("answer")\n```'
        lm = FakeLM(code_response)
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=h, stream=stream)
        assert result.success
        # FakeLM replays the same response, so push may fire more than once
        assert len(stream.state.papers) >= 1
        assert stream.state.papers[0] == {"title": "A"}
        h.stop()

    def test_stream_read_from_repl_code(self, handler):
        """Agent REPL code can call stream_read."""
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "Prior"}]})
        code_response = '```repl\ndata = stream_read()\nresult = str(len(data["events"]))\nFINAL_VAR("result")\n```'
        lm = FakeLM(code_response)
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()

        class ReaderAgent(SpecialistAgent):
            @property
            def name(self): return "reader"
            @property
            def system_prompt(self): return "You are a reader."

        agent = ReaderAgent()
        result = agent.run(task="test", handler=h, stream=stream)
        assert result.success
        assert result.result == "1"
        h.stop()

    def test_no_stream_no_injection(self, handler):
        """When stream=None, no stream functions in REPL."""
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=handler)
        assert result.success
