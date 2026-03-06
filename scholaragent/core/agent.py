"""Base SpecialistAgent abstraction for ScholarAgent.

Each specialist agent (Scout, Reader, Critic, Scribe) extends SpecialistAgent
and provides its own name, system prompt, and optional custom tools.

The core loop follows the RLM pattern:
  prompt LLM -> parse code blocks -> execute in REPL -> check for final answer.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from scholaragent.core.handler import LMHandler
from scholaragent.core.types import AgentResult
from scholaragent.environments.local_repl import LocalREPL
from scholaragent.utils.parsing import find_code_blocks, find_final_answer, format_iteration_output


class SpecialistAgent(ABC):
    """Base class for all specialist agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name (e.g., 'scout', 'reader')."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this agent's LLM."""
        ...

    def get_tools(self) -> dict[str, Any]:
        """Return custom tools for this agent's REPL. Override in subclasses."""
        return {}

    def run(
        self,
        task: str,
        handler: LMHandler,
        max_iterations: int = 10,
        agent_call_fn: Callable | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Run the agent's REPL loop.

        1. Create LocalREPL with tools
        2. Build initial messages (system prompt + task)
        3. Loop up to max_iterations:
           a. Call LLM with messages
           b. Parse code blocks from response
           c. If code blocks: execute in REPL, append output to messages
           d. Check for FINAL() in response or FINAL_VAR in REPL
           e. If final found: return AgentResult
        4. If max_iterations reached: return with success=False
        """
        # Create REPL with agent's tools.
        # Do NOT pass call_agent as a custom_tool (it's in RESERVED_NAMES).
        custom_tools = self.get_tools()
        repl = LocalREPL(handler_address=handler.address, custom_tools=custom_tools)

        # If an agent_call_fn was provided, inject it directly into the REPL
        # globals, bypassing the reserved-name check in LocalREPL.__init__.
        if agent_call_fn is not None:
            repl.globals["call_agent"] = agent_call_fn
            repl._call_agent = agent_call_fn  # so _restore_scaffold preserves it

        repl.load_context(task)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n\n"
                    "Use the REPL to work on this task. "
                    "The task is available as the `context` variable."
                ),
            },
        ]

        for i in range(max_iterations):
            # Get LLM response
            llm_response = handler.completion_messages(messages)

            # Check for inline FINAL()
            final_answer = find_final_answer(llm_response)
            if final_answer:
                return AgentResult(
                    agent_name=self.name,
                    task=task,
                    result=final_answer,
                    iterations=i + 1,
                    success=True,
                )

            # Parse and execute code blocks
            code_blocks = find_code_blocks(llm_response)
            repl_output = ""
            has_final = False
            final_value = None

            for block in code_blocks:
                result = repl.execute_code(block.code)
                repl_output += result.output
                if result.error:
                    repl_output += f"\nError: {result.error}"
                if result.has_final:
                    has_final = True
                    final_value = result.final_value

            if has_final and final_value is not None:
                return AgentResult(
                    agent_name=self.name,
                    task=task,
                    result=final_value,
                    iterations=i + 1,
                    success=True,
                )

            # Add to conversation
            messages.append({"role": "assistant", "content": llm_response})
            if repl_output:
                formatted = format_iteration_output(i, repl_output)
                messages.append({"role": "user", "content": formatted})
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Continue working on the task. Use ```repl code blocks to make progress.",
                    }
                )

        # Max iterations reached
        return AgentResult(
            agent_name=self.name,
            task=task,
            result=f"Max iterations ({max_iterations}) reached without final answer.",
            iterations=max_iterations,
            success=False,
        )

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format message list into a single prompt string for the LLM."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        return "\n\n".join(parts)
