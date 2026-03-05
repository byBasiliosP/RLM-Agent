"""Tests for SpecialistAgent and AgentRegistry."""

import pytest
from abc import ABC

from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.registry import AgentRegistry
from scholaragent.core.types import AgentResult
from scholaragent.core.handler import LMHandler
from scholaragent.clients.base import BaseLM
from scholaragent.core.types import ModelUsageSummary, UsageSummary


# ---- Helpers ----------------------------------------------------------------


class MockAgent(SpecialistAgent):
    """Concrete agent for testing."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def system_prompt(self) -> str:
        return "You are a mock agent."


class FakeLM(BaseLM):
    """Fake LM that returns a fixed response."""

    def __init__(self, response: str = "FINAL(test answer)"):
        super().__init__(model_name="fake-model")
        self._response = response
        self._call_count = 0

    def completion(self, prompt):
        self._call_count += 1
        return self._response

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(prompt_tokens=0, completion_tokens=0, total_tokens=0)


# ---- Test MockAgent properties ----------------------------------------------


class TestMockAgent:
    def test_name(self):
        agent = MockAgent()
        assert agent.name == "mock"

    def test_system_prompt(self):
        agent = MockAgent()
        assert agent.system_prompt == "You are a mock agent."

    def test_get_tools_returns_empty_dict(self):
        agent = MockAgent()
        assert agent.get_tools() == {}


# ---- Test SpecialistAgent is abstract ---------------------------------------


class TestSpecialistAgentAbstract:
    def test_cannot_instantiate_without_abstract_methods(self):
        with pytest.raises(TypeError):
            SpecialistAgent()  # type: ignore

    def test_is_abc(self):
        assert issubclass(SpecialistAgent, ABC)


# ---- Test AgentRegistry -----------------------------------------------------


class TestAgentRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = MockAgent()
        registry.register(agent)
        assert registry.get("mock") is agent

    def test_list_agents(self):
        registry = AgentRegistry()
        registry.register(MockAgent())
        assert registry.list_agents() == ["mock"]

    def test_get_unknown_raises_key_error(self):
        registry = AgentRegistry()
        with pytest.raises(KeyError, match="Agent 'unknown' not found"):
            registry.get("unknown")

    def test_contains(self):
        registry = AgentRegistry()
        registry.register(MockAgent())
        assert "mock" in registry
        assert "unknown" not in registry

    def test_len(self):
        registry = AgentRegistry()
        assert len(registry) == 0
        registry.register(MockAgent())
        assert len(registry) == 1


# ---- Test run() with FakeLM ------------------------------------------------


class TestAgentRun:
    def test_run_with_immediate_final_answer(self):
        """FakeLM returns FINAL(test answer), run() should succeed on first iteration."""
        agent = MockAgent()
        fake_lm = FakeLM(response="FINAL(test answer)")
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            result = agent.run(task="test task", handler=handler)
            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.result == "test answer"
            assert result.agent_name == "mock"
            assert result.task == "test task"
            assert result.iterations == 1
        finally:
            handler.stop()

    def test_run_max_iterations_reached(self):
        """FakeLM returns no FINAL, should exhaust max iterations."""
        agent = MockAgent()
        # Response with no FINAL and no code blocks
        fake_lm = FakeLM(response="I'm thinking about this problem...")
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            result = agent.run(task="test task", handler=handler, max_iterations=3)
            assert isinstance(result, AgentResult)
            assert result.success is False
            assert result.iterations == 3
            assert "Max iterations" in result.result
        finally:
            handler.stop()

    def test_run_with_code_block_and_final_var(self):
        """FakeLM returns a code block using FINAL_VAR with a computed value.

        FINAL_VAR receives a non-string (int), so it takes the direct-value
        path: str(value) is stored as the final answer immediately.
        """
        agent = MockAgent()
        code_response = "```repl\nresult = 2 + 2\nFINAL_VAR(result)\n```"
        fake_lm = FakeLM(response=code_response)
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            result = agent.run(task="test task", handler=handler)
            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.result == "4"
            assert result.iterations == 1
        finally:
            handler.stop()
