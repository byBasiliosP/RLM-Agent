"""Tests for ContextStream data model, persistence, and agent integration."""

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

    def test_push_batches_saves(self):
        """Pushes batch: save fires every `flush_every` events, not per-push."""
        saved = []
        stream = ContextStream(
            query="test", on_save=lambda s: saved.append(s.id), flush_every=10
        )
        for _ in range(9):
            stream.push("scout", "papers_found", {"papers": []})
        assert saved == []
        stream.push("scout", "papers_found", {"papers": []})
        assert len(saved) == 1

    def test_flush_saves_pending(self):
        saved = []
        stream = ContextStream(
            query="test", on_save=lambda s: saved.append(s.id), flush_every=10
        )
        stream.push("scout", "papers_found", {"papers": []})
        assert saved == []
        stream.flush()
        assert len(saved) == 1

    def test_flush_noop_when_clean(self):
        saved = []
        stream = ContextStream(
            query="test", on_save=lambda s: saved.append(s.id), flush_every=10
        )
        stream.flush()
        assert saved == []

    def test_commit_with_save_callback(self):
        saved = []
        stream = ContextStream(query="test", on_save=lambda s: saved.append(s.id))
        stream.commit("scout", [])
        assert len(saved) == 1

    def test_commit_flushes_pending_pushes(self):
        saved = []
        stream = ContextStream(
            query="test", on_save=lambda s: saved.append(s.id), flush_every=10
        )
        stream.push("scout", "papers_found", {"papers": []})
        stream.commit("scout", [])
        # commit should save exactly once, covering the pending push
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


from unittest.mock import patch

from scholaragent.clients.base import BaseLM
from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.handler import LMHandler
from scholaragent.core.types import ModelUsageSummary, UsageSummary


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


from scholaragent.core.dispatcher import Dispatcher
from scholaragent.core.registry import AgentRegistry


class SimpleAgent(SpecialistAgent):
    """Minimal agent that immediately returns."""

    def __init__(self, agent_name: str = "scout"):
        self._name = agent_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def system_prompt(self) -> str:
        return f"You are {self._name}."


class TestDispatcherStream:
    """Tests for Dispatcher creating and threading ContextStream."""

    @pytest.fixture()
    def registry(self):
        reg = AgentRegistry()
        for name in ["scout", "reader", "critic", "analyst", "synthesizer"]:
            reg.register(SimpleAgent(name))
        return reg

    @pytest.fixture()
    def handler(self):
        h = LMHandler(client=FakeLM("FINAL(done)"), token_counter=None, verbose=False)
        h.start()
        yield h
        h.stop()

    def test_dispatcher_creates_stream(self, registry, handler):
        """Dispatcher creates a ContextStream on run()."""
        dispatcher = Dispatcher(registry=registry, handler=handler)
        result = dispatcher.run(task="test query")
        assert result.success
        assert dispatcher._stream is not None
        assert dispatcher._stream.query == "test query"

    def test_dispatcher_passes_stream_to_child(self, registry, handler):
        """Dispatched agents receive the stream."""
        code = '```repl\nresult = call_agent("scout", "find papers")\nFINAL(result)\n```'
        lm = FakeLM(code)
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()
        dispatcher = Dispatcher(registry=registry, handler=h)
        result = dispatcher.run(task="test")
        assert dispatcher._stream is not None
        # Scout should have committed its trace
        assert "scout" in dispatcher._stream.traces
        h.stop()

    def test_dispatcher_stream_persists_with_store(self, registry, handler, tmp_path):
        """When store is set, stream is persisted on completion."""
        store = MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())
        dispatcher = Dispatcher(registry=registry, handler=handler, store=store)
        result = dispatcher.run(task="persist test")
        assert result.success
        streams = store.list_streams()
        assert len(streams) == 1
        assert streams[0]["query"] == "persist test"


from scholaragent.memory.research import ResearchPipeline


class TestResearchPipelineStream:
    """Tests for ContextStream in ResearchPipeline."""

    @pytest.fixture()
    def store(self, tmp_path):
        return MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())

    @pytest.fixture()
    def pipeline_with_agents(self, store):
        handler = LMHandler(client=FakeLM("FINAL(done)"), token_counter=None, verbose=False)
        handler.start()
        registry = AgentRegistry()
        for name in ["scout", "reader", "critic", "analyst", "synthesizer"]:
            registry.register(SimpleAgent(name))
        dispatcher = Dispatcher(registry=registry, handler=handler, store=store)
        pipeline = ResearchPipeline(store=store)
        pipeline.set_agent_infra(handler, registry, dispatcher)
        return pipeline, handler

    def test_deep_run_creates_stream(self, pipeline_with_agents):
        pipeline, handler = pipeline_with_agents
        result = pipeline.run("test deep", depth="deep", force=True)
        assert result["status"] == "completed"
        streams = pipeline.store.list_streams()
        assert len(streams) >= 1
        handler.stop()

    def test_normal_run_creates_stream(self, pipeline_with_agents):
        pipeline, handler = pipeline_with_agents
        with patch("scholaragent.memory.source_collector.search_arxiv", return_value="[]"), \
             patch("scholaragent.memory.source_collector.search_semantic_scholar", return_value="[]"), \
             patch("scholaragent.memory.source_collector.search_github_code", return_value=[]), \
             patch("scholaragent.memory.source_collector.search_docs", return_value=[]):
            result = pipeline.run("test normal", depth="normal", force=True)
        assert result["status"] == "completed"
        handler.stop()


try:
    import mcp
    _has_mcp = True
except ImportError:
    _has_mcp = False


@pytest.mark.skipif(not _has_mcp, reason="mcp package not installed")
class TestMCPStreamTools:
    """Tests for MCP stream tool handler functions."""

    @pytest.fixture()
    def store(self, tmp_path):
        return MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())

    def test_stream_list_empty(self, store):
        from scholaragent.mcp_server import _memory_stream_list
        result = _memory_stream_list(store)
        assert result["streams"] == []

    def test_stream_list_with_data(self, store):
        from scholaragent.mcp_server import _memory_stream_list
        stream = ContextStream(query="test")
        stream.commit("scout", [{"role": "user", "content": "hi"}])
        store.save_stream(stream)
        result = _memory_stream_list(store)
        assert len(result["streams"]) == 1
        assert result["streams"][0]["query"] == "test"

    def test_stream_list_filter(self, store):
        from scholaragent.mcp_server import _memory_stream_list
        store.save_stream(ContextStream(query="protein folding"))
        store.save_stream(ContextStream(query="quantum computing"))
        result = _memory_stream_list(store, query="protein")
        assert len(result["streams"]) == 1

    def test_stream_get_full(self, store):
        from scholaragent.mcp_server import _memory_stream_get
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        store.save_stream(stream)
        result = _memory_stream_get(store, stream.id)
        assert result["query"] == "test"
        assert "state" in result
        assert "traces" in result
        assert "events" in result

    def test_stream_get_filtered(self, store):
        from scholaragent.mcp_server import _memory_stream_get
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": []})
        stream.push("reader", "finding_extracted", {"paper_ref": "x", "finding": {}})
        stream.commit("scout", [])
        stream.commit("reader", [])
        store.save_stream(stream)
        result = _memory_stream_get(store, stream.id, agent="scout")
        assert "scout" in result["traces"]
        assert "reader" not in result["traces"]
        assert all(e["agent"] == "scout" for e in result["events"])

    def test_stream_get_not_found(self, store):
        from scholaragent.mcp_server import _memory_stream_get
        result = _memory_stream_get(store, "nonexistent")
        assert "error" in result


class TestEndToEnd:
    """End-to-end test: full pipeline with stream persistence."""

    def test_full_pipeline_with_stream(self, tmp_path):
        """Dispatcher -> agents -> stream -> persist -> load -> read."""
        store = MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())

        # Agent that calls scout then returns
        dispatch_code = '```repl\nscout_result = call_agent("scout", "find papers")\nFINAL(scout_result)\n```'
        handler = LMHandler(client=FakeLM(dispatch_code), token_counter=None, verbose=False)
        handler.start()

        registry = AgentRegistry()
        registry.register(SimpleAgent("scout"))
        registry.register(SimpleAgent("reader"))
        registry.register(SimpleAgent("critic"))
        registry.register(SimpleAgent("analyst"))
        registry.register(SimpleAgent("synthesizer"))

        dispatcher = Dispatcher(registry=registry, handler=handler, store=store)
        result = dispatcher.run(task="end to end test")

        # Verify stream was created and persisted
        streams = store.list_streams()
        assert len(streams) == 1
        assert streams[0]["query"] == "end to end test"

        # Verify we can load and read it
        loaded = store.load_stream(streams[0]["id"])
        assert loaded is not None
        assert "dispatcher" in loaded.traces
        data = loaded.read()
        assert "state" in data
        assert "traces" in data
        assert "events" in data

        handler.stop()


class TestQualityState:
    """Tests for quality field in PipelineState and ContextStream."""

    def test_empty_state_has_quality(self):
        state = PipelineState()
        assert state.quality == {"lint": [], "architecture": [], "coverage": []}

    def test_to_dict_includes_quality(self):
        state = PipelineState()
        state.quality["lint"].append({"issue": "unused import"})
        d = state.to_dict()
        assert d["quality"]["lint"] == [{"issue": "unused import"}]

    def test_from_dict_restores_quality(self):
        d = {"papers": [], "findings": {}, "assessments": {}, "themes": {},
             "synthesis": "", "quality": {"lint": [{"x": 1}], "architecture": [], "coverage": []}}
        state = PipelineState.from_dict(d)
        assert state.quality["lint"] == [{"x": 1}]

    def test_from_dict_backward_compat(self):
        """Old dicts without quality field still work."""
        d = {"papers": [], "findings": {}, "assessments": {}, "themes": {}, "synthesis": ""}
        state = PipelineState.from_dict(d)
        assert state.quality == {"lint": [], "architecture": [], "coverage": []}

    def test_push_quality_lint(self):
        stream = ContextStream(query="test")
        stream.push("linter", "quality_lint", {"result": {"issues": ["unused import"]}})
        assert len(stream.state.quality["lint"]) == 1

    def test_push_quality_architecture(self):
        stream = ContextStream(query="test")
        stream.push("architect", "quality_architecture", {"result": {"violations": ["circular dep"]}})
        assert len(stream.state.quality["architecture"]) == 1

    def test_push_quality_coverage(self):
        stream = ContextStream(query="test")
        stream.push("coverage", "quality_coverage", {"result": {"untested": ["module_x"]}})
        assert len(stream.state.quality["coverage"]) == 1

    def test_quality_accumulates(self):
        stream = ContextStream(query="test")
        stream.push("linter", "quality_lint", {"result": {"a": 1}})
        stream.push("linter", "quality_lint", {"result": {"b": 2}})
        stream.push("architect", "quality_architecture", {"result": {"c": 3}})
        assert len(stream.state.quality["lint"]) == 2
        assert len(stream.state.quality["architecture"]) == 1
        assert len(stream.state.quality["coverage"]) == 0
