"""Agent Registry - maps agent names to SpecialistAgent instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholaragent.core.agent import SpecialistAgent


class AgentRegistry:
    """Simple registry mapping agent names to agent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, SpecialistAgent] = {}

    def register(self, agent: SpecialistAgent) -> None:
        """Register an agent under its ``name`` property."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> SpecialistAgent:
        """Retrieve a registered agent by name.

        Raises:
            KeyError: If the agent name is not registered.
        """
        if name not in self._agents:
            raise KeyError(
                f"Agent '{name}' not found. Available: {list(self._agents.keys())}"
            )
        return self._agents[name]

    def list_agents(self) -> list[str]:
        """Return a list of all registered agent names."""
        return list(self._agents.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)
