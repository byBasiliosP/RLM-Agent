"""Dispatcher Agent - orchestrates specialist agents via REPL code generation."""

from collections.abc import Callable

from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.handler import LMHandler
from scholaragent.core.registry import AgentRegistry
from scholaragent.core.types import AgentResult
from scholaragent.utils.prompts import DISPATCHER_SYSTEM_PROMPT


class Dispatcher(SpecialistAgent):
    """The orchestrator agent that dispatches to specialist agents.

    The Dispatcher writes Python code that calls specialist agents through
    ``call_agent(name, task)``.  It injects its own ``_dispatch_agent``
    method as the ``call_agent`` function inside the REPL so that agent
    look-ups go through the :class:`AgentRegistry`.
    """

    def __init__(self, registry: AgentRegistry, handler: LMHandler):
        self._registry = registry
        self._handler = handler

    @property
    def name(self) -> str:
        return "dispatcher"

    @property
    def system_prompt(self) -> str:
        agent_list = "\n".join(
            f"- {name}" for name in self._registry.list_agents()
        )
        return DISPATCHER_SYSTEM_PROMPT.format(agent_list=agent_list)

    def _dispatch_agent(self, agent_name: str, task: str) -> str:
        """Called from REPL code as ``call_agent(name, task)``.

        Looks up the agent in the registry, runs it, and returns the
        result string.
        """
        agent = self._registry.get(agent_name)
        result = agent.run(
            task=task,
            handler=self._handler,
            max_iterations=10,
        )
        if result.success:
            return result.result
        return f"Agent '{agent_name}' failed: {result.result}"

    def run(
        self,
        task: str,
        handler: LMHandler | None = None,
        max_iterations: int = 15,
        agent_call_fn: Callable | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Override run() to inject ``_dispatch_agent`` as the ``call_agent`` function."""
        return super().run(
            task=task,
            handler=self._handler,
            max_iterations=max_iterations,
            agent_call_fn=self._dispatch_agent,
            verbose=verbose,
        )
