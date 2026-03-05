"""Tests for the ScholarAgent public API."""

import pytest
from unittest.mock import patch, MagicMock

from scholaragent.core.types import AgentResult


# Patch both SDK modules so no real API clients are created.
@pytest.fixture()
def scholar():
    with (
        patch("scholaragent.clients.openai_client.openai"),
        patch("scholaragent.clients.anthropic_client.anthropic"),
    ):
        from scholaragent import ScholarAgent

        agent = ScholarAgent(
            strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
            cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
        )
        yield agent


# ---------------------------------------------------------------------------
# 1. Creation
# ---------------------------------------------------------------------------
class TestScholarAgentCreation:
    def test_creates_successfully(self, scholar):
        from scholaragent import ScholarAgent

        assert isinstance(scholar, ScholarAgent)

    def test_stores_max_papers(self, scholar):
        assert scholar.max_papers == 10

    def test_stores_max_iterations(self, scholar):
        assert scholar.max_iterations == 15

    def test_stores_verbose(self, scholar):
        assert scholar.verbose is False

    def test_custom_max_papers(self):
        with (
            patch("scholaragent.clients.openai_client.openai"),
            patch("scholaragent.clients.anthropic_client.anthropic"),
        ):
            from scholaragent import ScholarAgent

            agent = ScholarAgent(
                strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
                cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
                max_papers=20,
                max_iterations=5,
                verbose=True,
            )
            assert agent.max_papers == 20
            assert agent.max_iterations == 5
            assert agent.verbose is True


# ---------------------------------------------------------------------------
# 2. Has research() method
# ---------------------------------------------------------------------------
class TestResearchMethod:
    def test_has_research_method(self, scholar):
        assert hasattr(scholar, "research")
        assert callable(scholar.research)


# ---------------------------------------------------------------------------
# 3. Registry contains all 5 agents
# ---------------------------------------------------------------------------
class TestRegistryAgents:
    def test_registry_has_5_agents(self, scholar):
        assert len(scholar.registry) == 5

    def test_registry_contains_scout(self, scholar):
        assert "scout" in scholar.registry

    def test_registry_contains_reader(self, scholar):
        assert "reader" in scholar.registry

    def test_registry_contains_critic(self, scholar):
        assert "critic" in scholar.registry

    def test_registry_contains_analyst(self, scholar):
        assert "analyst" in scholar.registry

    def test_registry_contains_synthesizer(self, scholar):
        assert "synthesizer" in scholar.registry

    def test_list_agents_returns_all(self, scholar):
        agents = scholar.registry.list_agents()
        assert set(agents) == {"scout", "reader", "critic", "analyst", "synthesizer"}


# ---------------------------------------------------------------------------
# 4. Handler is created and has clients registered
# ---------------------------------------------------------------------------
class TestHandler:
    def test_handler_exists(self, scholar):
        from scholaragent.core.handler import LMHandler

        assert isinstance(scholar.handler, LMHandler)

    def test_handler_has_default_client(self, scholar):
        assert scholar.handler.default_client is not None

    def test_handler_has_multiple_clients(self, scholar):
        # Should have at least the strong (default) and the cheap client
        assert len(scholar.handler.clients) >= 2


# ---------------------------------------------------------------------------
# 5. Dispatcher is created with the registry
# ---------------------------------------------------------------------------
class TestDispatcher:
    def test_dispatcher_exists(self, scholar):
        from scholaragent.core.dispatcher import Dispatcher

        assert isinstance(scholar.dispatcher, Dispatcher)

    def test_dispatcher_has_registry(self, scholar):
        assert scholar.dispatcher._registry is scholar.registry

    def test_dispatcher_has_handler(self, scholar):
        assert scholar.dispatcher._handler is scholar.handler


# ---------------------------------------------------------------------------
# 6. __repr__ returns meaningful string
# ---------------------------------------------------------------------------
class TestRepr:
    def test_repr_contains_agents(self, scholar):
        r = repr(scholar)
        assert "ScholarAgent" in r

    def test_repr_contains_model_names(self, scholar):
        r = repr(scholar)
        assert "claude-sonnet-4-6" in r
        assert "gpt-4o-mini" in r

    def test_repr_contains_agent_list(self, scholar):
        r = repr(scholar)
        for name in ("scout", "reader", "critic", "analyst", "synthesizer"):
            assert name in r


# ---------------------------------------------------------------------------
# 7. research() calls dispatcher.run()
# ---------------------------------------------------------------------------
class TestResearchCallsDispatcher:
    def test_research_delegates_to_dispatcher(self, scholar):
        expected = AgentResult(
            agent_name="dispatcher",
            task="test query",
            result="some findings",
            iterations=3,
            success=True,
        )
        with patch.object(scholar.dispatcher, "run", return_value=expected) as mock_run:
            result = scholar.research("test query")

        mock_run.assert_called_once_with(
            task="test query",
            max_iterations=scholar.max_iterations,
            verbose=scholar.verbose,
        )
        assert result is expected
        assert result.success is True
        assert result.agent_name == "dispatcher"
