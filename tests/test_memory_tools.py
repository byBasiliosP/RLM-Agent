"""Tests for MemoryStore integration into SpecialistAgent REPL via memory_tools."""

import pytest

from scholaragent.clients.base import BaseLM
from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.dispatcher import Dispatcher
from scholaragent.core.handler import LMHandler
from scholaragent.core.registry import AgentRegistry
from scholaragent.core.types import AgentResult, ModelUsageSummary, UsageSummary
from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry
from tests.helpers import FakeEmbeddings

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


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


class MockAgent(SpecialistAgent):
    """Concrete agent for testing."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def system_prompt(self) -> str:
        return "You are a mock agent."


class ToolCapturingAgent(SpecialistAgent):
    """Agent that captures its REPL globals on the first run for inspection."""

    captured_tools: dict = {}

    @property
    def name(self) -> str:
        return "tool-capturer"

    @property
    def system_prompt(self) -> str:
        return "You capture tools."

    def run(self, task, handler, max_iterations=10, agent_call_fn=None,
            verbose=False, budget=None, store=None):
        # Build tools the same way SpecialistAgent.run() would
        from scholaragent.core.agent import SpecialistAgent as SA
        tools = self.get_tools()
        if store is not None:
            tools = {**tools, **SA.memory_tools(store)}
        ToolCapturingAgent.captured_tools = tools
        return AgentResult(
            agent_name=self.name, task=task, result="captured", iterations=1, success=True
        )


class EchoAgent(SpecialistAgent):
    """Echoes task back."""

    received_store: object = None

    @property
    def name(self) -> str:
        return "echo"

    @property
    def system_prompt(self) -> str:
        return "You echo."

    def run(self, task, handler, max_iterations=10, agent_call_fn=None,
            verbose=False, budget=None, store=None, stream=None):
        # Capture received store for assertions
        EchoAgent.received_store = store
        return AgentResult(
            agent_name="echo", task=task, result=f"echo: {task}",
            iterations=1, success=True
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_store(tmp_path):
    """An in-memory-backed MemoryStore using FakeEmbeddings."""
    db_path = str(tmp_path / "test_memory.db")
    return MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())


@pytest.fixture
def handler():
    fake = FakeLM(response="FINAL(done)")
    h = LMHandler(client=fake)
    h.start()
    yield h
    h.stop()


# ---------------------------------------------------------------------------
# Tests: SpecialistAgent.memory_tools()
# ---------------------------------------------------------------------------


class TestMemoryTools:
    def test_memory_tools_returns_lookup_and_store(self, mem_store):
        tools = SpecialistAgent.memory_tools(mem_store)
        assert "memory_lookup" in tools
        assert "memory_store" in tools
        assert callable(tools["memory_lookup"])
        assert callable(tools["memory_store"])

    def test_memory_lookup_empty_store(self, mem_store):
        tools = SpecialistAgent.memory_tools(mem_store)
        results = tools["memory_lookup"]("some query")
        assert results == []

    def test_memory_store_persists_entry(self, mem_store):
        tools = SpecialistAgent.memory_tools(mem_store)
        result = tools["memory_store"](
            "Important finding about RLHF",
            "arxiv:2301.00001",
            ["rlhf", "reward-model"],
        )
        assert result.startswith("stored:")
        assert mem_store.count() == 1

    def test_memory_store_infers_source_type_paper(self, mem_store):
        tools = SpecialistAgent.memory_tools(mem_store)
        tools["memory_store"]("content", "arxiv:1234.5678", ["tag"])
        # Retrieve and verify source_type
        results = mem_store.search("content", max_results=1)
        assert len(results) == 1
        assert results[0][0].source_type == "paper"

    def test_memory_store_infers_source_type_code(self, mem_store):
        tools = SpecialistAgent.memory_tools(mem_store)
        tools["memory_store"]("code snippet", "github:org/repo", ["python"])
        results = mem_store.search("code snippet", max_results=1)
        assert results[0][0].source_type == "code"

    def test_memory_store_infers_source_type_docs(self, mem_store):
        tools = SpecialistAgent.memory_tools(mem_store)
        tools["memory_store"]("documentation", "https://docs.python.org", ["python"])
        results = mem_store.search("documentation", max_results=1)
        assert results[0][0].source_type == "docs"

    def test_memory_lookup_returns_compact_dicts_with_score(self, mem_store):
        # First store an entry directly
        entry = MemoryEntry(
            content="Transformer architecture details",
            summary="Transformer summary",
            source_type="paper",
            source_ref="arxiv:1706.03762",
            tags=["transformer", "attention"],
        )
        mem_store.add(entry)

        tools = SpecialistAgent.memory_tools(mem_store)
        results = tools["memory_lookup"]("Transformer architecture")
        assert len(results) == 1
        result = results[0]
        # Check compact dict fields
        assert "id" in result
        assert "summary" in result
        assert "source_type" in result
        assert "source_ref" in result
        assert "tags" in result
        assert "score" in result
        assert isinstance(result["score"], float)
        # Full content should NOT be present
        assert "content" not in result

    def test_memory_lookup_max_results(self, mem_store):
        tools = SpecialistAgent.memory_tools(mem_store)
        # Store 5 entries
        for i in range(5):
            tools["memory_store"](f"Finding number {i}", f"arxiv:{i:04d}.0000", [f"tag{i}"])

        results = tools["memory_lookup"]("Finding", max_results=2)
        assert len(results) <= 2

    def test_memory_lookup_score_rounded(self, mem_store):
        entry = MemoryEntry(
            content="Neural network training",
            summary="NN summary",
            source_type="paper",
            source_ref="arxiv:0000.0001",
            tags=["nn"],
        )
        mem_store.add(entry)
        tools = SpecialistAgent.memory_tools(mem_store)
        results = tools["memory_lookup"]("neural network")
        assert len(results) == 1
        score = results[0]["score"]
        # Score should be rounded to 3 decimal places
        assert score == round(score, 3)


# ---------------------------------------------------------------------------
# Tests: SpecialistAgent.run() with store kwarg
# ---------------------------------------------------------------------------


class TestAgentRunWithStore:
    def test_run_without_store_unchanged(self, handler):
        """run() with no store arg works as before (backward compat)."""
        agent = MockAgent()
        result = agent.run(task="test task", handler=handler)
        assert isinstance(result, AgentResult)

    def test_run_with_store_injects_memory_tools(self, mem_store):
        """run() with store injects memory_lookup and memory_store into REPL."""
        # Use a code response that calls memory_lookup and checks it works
        code = "```repl\nresult = memory_lookup('test query')\nFINAL_VAR(result)\n```"
        fake = FakeLM(response=code)
        h = LMHandler(client=fake)
        h.start()
        try:
            agent = MockAgent()
            result = agent.run(task="test task", handler=h, store=mem_store)
            assert isinstance(result, AgentResult)
            assert result.success is True
            # Result should be a list (possibly empty) since store has no entries
            assert result.result == "[]"
        finally:
            h.stop()

    def test_run_with_store_memory_store_callable(self, mem_store):
        """run() with store allows calling memory_store from REPL."""
        code = (
            "```repl\n"
            "memory_store('test finding', 'docs:test', ['test'])\n"
            "FINAL_VAR(1)\n"
            "```"
        )
        fake = FakeLM(response=code)
        h = LMHandler(client=fake)
        h.start()
        try:
            agent = MockAgent()
            result = agent.run(task="test task", handler=h, store=mem_store)
            assert result.success is True
            assert mem_store.count() == 1
        finally:
            h.stop()

    def test_run_without_store_no_memory_tools(self, handler):
        """run() without store does NOT inject memory tools into REPL."""
        code = "```repl\nresult = 'memory_lookup' in dir()\nFINAL_VAR(result)\n```"
        fake = FakeLM(response=code)
        h = LMHandler(client=fake)
        h.start()
        try:
            agent = MockAgent()
            result = agent.run(task="test task", handler=h)
            assert result.success is True
            # memory_lookup should not exist in globals → calling it would error
        finally:
            h.stop()


# ---------------------------------------------------------------------------
# Tests: Dispatcher with store
# ---------------------------------------------------------------------------


class TestDispatcherWithStore:
    def test_dispatcher_accepts_store_kwarg(self, mem_store, handler):
        registry = AgentRegistry()
        d = Dispatcher(registry=registry, handler=handler, store=mem_store)
        assert d._store is mem_store

    def test_dispatcher_passes_store_to_dispatched_agent(self, mem_store, handler):
        EchoAgent.received_store = None
        registry = AgentRegistry()
        registry.register(EchoAgent())
        d = Dispatcher(registry=registry, handler=handler, store=mem_store)
        d._dispatch_agent("echo", "some task")
        assert EchoAgent.received_store is mem_store

    def test_dispatcher_no_store_passes_none(self, handler):
        EchoAgent.received_store = "sentinel"
        registry = AgentRegistry()
        registry.register(EchoAgent())
        d = Dispatcher(registry=registry, handler=handler)
        d._dispatch_agent("echo", "some task")
        assert EchoAgent.received_store is None

    def test_dispatcher_store_none_by_default(self, handler):
        registry = AgentRegistry()
        d = Dispatcher(registry=registry, handler=handler)
        assert d._store is None
