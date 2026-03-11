"""Dispatcher Agent - orchestrates specialist agents via REPL code generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from scholaragent.core.agent import SpecialistAgent
from scholaragent.core.handler import LMHandler
from scholaragent.core.registry import AgentRegistry
from scholaragent.core.types import AgentResult
from scholaragent.utils.prompts import DISPATCHER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from scholaragent.utils.budget import Budget


class Dispatcher(SpecialistAgent):
    """The orchestrator agent that dispatches to specialist agents.

    The Dispatcher writes Python code that calls specialist agents through
    ``call_agent(name, task)``.  It injects its own ``_dispatch_agent``
    method as the ``call_agent`` function inside the REPL so that agent
    look-ups go through the :class:`AgentRegistry`.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        handler: LMHandler,
        budget: Budget | None = None,
    ):
        self._registry = registry
        self._handler = handler
        self._budget = budget

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
        result string.  When a budget is configured, creates a sub-budget
        for each dispatched agent.
        """
        agent = self._registry.get(agent_name)

        # Create sub-budget for the child agent if dispatcher has a budget
        sub_budget = None
        if self._budget is not None and not self._budget.is_exhausted:
            from scholaragent.utils.budget import Budget

            sub_budget = Budget(
                max_tokens=self._budget.tokens_remaining // 2,
                max_iterations=min(10, self._budget.iterations_remaining),
            )

        result = agent.run(
            task=task,
            handler=self._handler,
            max_iterations=10,
            budget=sub_budget,
        )

        # Roll up sub-budget usage to dispatcher budget
        if sub_budget is not None and self._budget is not None:
            self._budget.use_tokens(sub_budget.tokens_used)

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
        budget: Budget | None = None,
    ) -> AgentResult:
        """Override run() to inject ``_dispatch_agent`` as the ``call_agent`` function."""
        # Use the provided budget or the one from __init__
        if budget is not None:
            self._budget = budget
        return super().run(
            task=task,
            handler=self._handler,
            max_iterations=max_iterations,
            agent_call_fn=self._dispatch_agent,
            verbose=verbose,
            budget=self._budget,
        )
