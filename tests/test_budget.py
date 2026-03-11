"""Tests for budget enforcement in agent runs."""

import pytest

from scholaragent.utils.budget import Budget


class TestBudget:
    def test_initial_state(self):
        b = Budget(max_tokens=100_000, max_iterations=10)
        assert b.tokens_remaining == 100_000
        assert b.iterations_remaining == 10
        assert not b.is_exhausted

    def test_use_tokens(self):
        b = Budget(max_tokens=1000)
        b.use_tokens(500)
        assert b.tokens_remaining == 500
        assert not b.is_exhausted

    def test_use_iteration(self):
        b = Budget(max_iterations=3)
        b.use_iteration()
        assert b.iterations_remaining == 2
        b.use_iteration()
        b.use_iteration()
        assert b.is_exhausted

    def test_token_exhaustion(self):
        b = Budget(max_tokens=100)
        b.use_tokens(100)
        assert b.is_exhausted

    def test_tokens_remaining_never_negative(self):
        b = Budget(max_tokens=100)
        b.use_tokens(200)
        assert b.tokens_remaining == 0

    def test_iterations_remaining_never_negative(self):
        b = Budget(max_iterations=2)
        b.use_iteration()
        b.use_iteration()
        b.use_iteration()
        assert b.iterations_remaining == 0


class TestBudgetInAgent:
    def test_agent_stops_on_budget_exhaustion(self):
        """Agent.run() should stop when budget is exhausted."""
        from unittest.mock import MagicMock, patch

        from scholaragent.core.types import AgentResult
        from scholaragent.utils.budget import Budget

        # Create a budget that exhausts after 1 iteration
        budget = Budget(max_tokens=100_000, max_iterations=1)

        mock_handler = MagicMock()
        mock_handler.address = ("127.0.0.1", 0)
        mock_handler.token_counter = None
        mock_handler.completion_messages = MagicMock(return_value="thinking...")

        # Create a simple agent
        from scholaragent.agents.critic import CriticAgent
        agent = CriticAgent()

        with patch("scholaragent.core.agent.LocalREPL") as MockREPL:
            mock_repl = MagicMock()
            MockREPL.return_value = mock_repl

            result = agent.run(
                task="test",
                handler=mock_handler,
                max_iterations=10,
                budget=budget,
            )

            # Should fail due to budget, not max_iterations
            assert not result.success
            assert "Budget exhausted" in result.result


class TestDispatcherBudget:
    def test_dispatcher_creates_sub_budgets(self):
        from unittest.mock import MagicMock, patch
        from scholaragent.core.dispatcher import Dispatcher
        from scholaragent.core.registry import AgentRegistry
        from scholaragent.core.types import AgentResult
        from scholaragent.utils.budget import Budget

        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.name = "scout"
        mock_agent.run.return_value = AgentResult(
            agent_name="scout", task="test", result="done", iterations=1, success=True
        )

        registry = AgentRegistry()
        registry.register(mock_agent)

        handler = MagicMock()
        handler.address = ("127.0.0.1", 0)
        handler.token_counter = None

        budget = Budget(max_tokens=50_000, max_iterations=10)
        dispatcher = Dispatcher(registry=registry, handler=handler, budget=budget)

        # Call _dispatch_agent directly
        result = dispatcher._dispatch_agent("scout", "find papers")

        # The mock agent should have been called with a sub-budget
        call_kwargs = mock_agent.run.call_args[1]
        assert "budget" in call_kwargs
        sub_budget = call_kwargs["budget"]
        assert sub_budget is not None
        assert sub_budget.max_tokens <= budget.tokens_remaining
