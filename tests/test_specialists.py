"""Tests for all five specialist agents."""

import pytest

from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.registry import AgentRegistry

from scholaragent.agents.scout import ScoutAgent
from scholaragent.agents.reader import ReaderAgent
from scholaragent.agents.critic import CriticAgent
from scholaragent.agents.analyst import AnalystAgent
from scholaragent.agents.synthesizer import SynthesizerAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scout():
    return ScoutAgent()

@pytest.fixture
def reader():
    return ReaderAgent()

@pytest.fixture
def critic():
    return CriticAgent()

@pytest.fixture
def analyst():
    return AnalystAgent()

@pytest.fixture
def synthesizer():
    return SynthesizerAgent()


# ---------------------------------------------------------------------------
# ScoutAgent tests
# ---------------------------------------------------------------------------

class TestScoutAgent:
    def test_is_specialist(self, scout):
        assert isinstance(scout, SpecialistAgent)

    def test_name(self, scout):
        assert scout.name == "scout"

    def test_system_prompt_keywords(self, scout):
        prompt = scout.system_prompt
        assert "Scout" in prompt
        assert "search_arxiv" in prompt
        assert "search_semantic_scholar" in prompt
        assert "FINAL_VAR" in prompt

    def test_tools(self, scout):
        tools = scout.get_tools()
        assert len(tools) == 6
        assert "search_arxiv" in tools
        assert "search_semantic_scholar" in tools
        assert "get_citations" in tools
        assert "get_references" in tools
        assert "fetch_arxiv_pdf" in tools
        assert "search_github_code" in tools
        # All values should be callable
        for fn in tools.values():
            assert callable(fn)


# ---------------------------------------------------------------------------
# ReaderAgent tests
# ---------------------------------------------------------------------------

class TestReaderAgent:
    def test_is_specialist(self, reader):
        assert isinstance(reader, SpecialistAgent)

    def test_name(self, reader):
        assert reader.name == "reader"

    def test_system_prompt_keywords(self, reader):
        prompt = reader.system_prompt
        assert "Reader" in prompt
        assert "methodology" in prompt.lower()
        assert "FINAL" in prompt

    def test_tools(self, reader):
        tools = reader.get_tools()
        assert len(tools) == 1
        assert "fetch_arxiv_pdf" in tools


# ---------------------------------------------------------------------------
# CriticAgent tests
# ---------------------------------------------------------------------------

class TestCriticAgent:
    def test_is_specialist(self, critic):
        assert isinstance(critic, SpecialistAgent)

    def test_name(self, critic):
        assert critic.name == "critic"

    def test_system_prompt_keywords(self, critic):
        prompt = critic.system_prompt
        assert "Critic" in prompt
        assert "methodology" in prompt.lower()
        assert "FINAL" in prompt

    def test_tools_empty(self, critic):
        assert critic.get_tools() == {}


# ---------------------------------------------------------------------------
# AnalystAgent tests
# ---------------------------------------------------------------------------

class TestAnalystAgent:
    def test_is_specialist(self, analyst):
        assert isinstance(analyst, SpecialistAgent)

    def test_name(self, analyst):
        assert analyst.name == "analyst"

    def test_system_prompt_keywords(self, analyst):
        prompt = analyst.system_prompt
        assert "Analyst" in prompt
        assert "themes" in prompt.lower()
        assert "FINAL" in prompt

    def test_tools_empty(self, analyst):
        assert analyst.get_tools() == {}


# ---------------------------------------------------------------------------
# SynthesizerAgent tests
# ---------------------------------------------------------------------------

class TestSynthesizerAgent:
    def test_is_specialist(self, synthesizer):
        assert isinstance(synthesizer, SpecialistAgent)

    def test_name(self, synthesizer):
        assert synthesizer.name == "synthesizer"

    def test_system_prompt_keywords(self, synthesizer):
        prompt = synthesizer.system_prompt
        assert "Synthesizer" in prompt
        assert "literature review" in prompt.lower()
        assert "FINAL" in prompt

    def test_tools_empty(self, synthesizer):
        assert synthesizer.get_tools() == {}


# ---------------------------------------------------------------------------
# Registry integration test
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_register_all_agents(self):
        registry = AgentRegistry()
        agents = [ScoutAgent(), ReaderAgent(), CriticAgent(), AnalystAgent(), SynthesizerAgent()]
        for agent in agents:
            registry.register(agent)

        assert len(registry) == 5
        expected_names = {"scout", "reader", "critic", "analyst", "synthesizer"}
        assert set(registry.list_agents()) == expected_names

    def test_retrieve_registered_agents(self):
        registry = AgentRegistry()
        agents = [ScoutAgent(), ReaderAgent(), CriticAgent(), AnalystAgent(), SynthesizerAgent()]
        for agent in agents:
            registry.register(agent)

        for agent in agents:
            retrieved = registry.get(agent.name)
            assert retrieved is agent
