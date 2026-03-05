"""Tests for the Dispatcher agent."""

import pytest

from scholaragent.clients.base import BaseLM
from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.dispatcher import Dispatcher
from scholaragent.core.handler import LMHandler
from scholaragent.core.registry import AgentRegistry
from scholaragent.core.types import AgentResult, ModelUsageSummary, UsageSummary


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


class EchoAgent(SpecialistAgent):
    """Test agent that simply echoes the task back."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def system_prompt(self) -> str:
        return "You echo."

    def run(
        self,
        task: str,
        handler: LMHandler,
        max_iterations: int = 10,
        agent_call_fn=None,
        verbose: bool = False,
    ) -> AgentResult:
        return AgentResult(
            agent_name="echo",
            task=task,
            result=f"echo: {task}",
            iterations=1,
            success=True,
        )


class FailingAgent(SpecialistAgent):
    """Test agent that always fails."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def system_prompt(self) -> str:
        return "You fail."

    def run(
        self,
        task: str,
        handler: LMHandler,
        max_iterations: int = 10,
        agent_call_fn=None,
        verbose: bool = False,
    ) -> AgentResult:
        return AgentResult(
            agent_name="failing",
            task=task,
            result="something went wrong",
            iterations=1,
            success=False,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_lm():
    return FakeLM()


@pytest.fixture
def handler(fake_lm):
    h = LMHandler(client=fake_lm)
    h.start()
    yield h
    h.stop()


@pytest.fixture
def registry():
    reg = AgentRegistry()
    reg.register(EchoAgent())
    return reg


@pytest.fixture
def dispatcher(registry, handler):
    return Dispatcher(registry=registry, handler=handler)


# ---------------------------------------------------------------------------
# Tests: creation and properties
# ---------------------------------------------------------------------------


class TestDispatcherCreation:
    def test_creation_with_registry_and_handler(self, registry, handler):
        d = Dispatcher(registry=registry, handler=handler)
        assert d is not None

    def test_name(self, dispatcher):
        assert dispatcher.name == "dispatcher"

    def test_system_prompt_contains_agent_names(self, dispatcher):
        prompt = dispatcher.system_prompt
        assert "echo" in prompt
        assert "call_agent" in prompt

    def test_system_prompt_with_multiple_agents(self, handler):
        reg = AgentRegistry()
        reg.register(EchoAgent())
        reg.register(FailingAgent())
        d = Dispatcher(registry=reg, handler=handler)
        prompt = d.system_prompt
        assert "- echo" in prompt
        assert "- failing" in prompt


# ---------------------------------------------------------------------------
# Tests: _dispatch_agent
# ---------------------------------------------------------------------------


class TestDispatchAgent:
    def test_dispatch_calls_correct_agent(self, dispatcher):
        result = dispatcher._dispatch_agent("echo", "hello world")
        assert result == "echo: hello world"

    def test_dispatch_returns_result_string_on_success(self, dispatcher):
        result = dispatcher._dispatch_agent("echo", "some task")
        assert isinstance(result, str)
        assert "echo: some task" == result

    def test_dispatch_returns_error_on_failure(self, handler):
        reg = AgentRegistry()
        reg.register(FailingAgent())
        d = Dispatcher(registry=reg, handler=handler)
        result = d._dispatch_agent("failing", "do something")
        assert "failed" in result.lower()
        assert "failing" in result

    def test_dispatch_unknown_agent_raises(self, dispatcher):
        with pytest.raises(KeyError, match="not found"):
            dispatcher._dispatch_agent("nonexistent", "task")


# ---------------------------------------------------------------------------
# Tests: run()
# ---------------------------------------------------------------------------


class TestDispatcherRun:
    def test_run_with_immediate_final(self, registry):
        """FakeLM returns FINAL(done), run() should succeed immediately."""
        fake = FakeLM(response="FINAL(done)")
        h = LMHandler(client=fake)
        h.start()
        try:
            d = Dispatcher(registry=registry, handler=h)
            result = d.run(task="research topic X")
            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.result == "done"
            assert result.agent_name == "dispatcher"
            assert result.iterations == 1
        finally:
            h.stop()

    def test_run_max_iterations(self, registry):
        """FakeLM returns no final, should exhaust iterations."""
        fake = FakeLM(response="I need to think more...")
        h = LMHandler(client=fake)
        h.start()
        try:
            d = Dispatcher(registry=registry, handler=h)
            result = d.run(task="hard problem", max_iterations=2)
            assert result.success is False
            assert result.iterations == 2
            assert "Max iterations" in result.result
        finally:
            h.stop()
