"""Parsing utilities for extracting code blocks and final answers from LLM responses."""

import re

from scholaragent.core.types import CodeBlock


def find_code_blocks(text: str) -> list[CodeBlock]:
    """Extract ```repl and ```python code blocks from LLM response text.

    Pattern: ```repl\\n<code>\\n``` (or ```python\\n<code>\\n```)
    Returns list of CodeBlock(code=..., language="repl" or "python").
    """
    pattern = r"```(repl|python)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [CodeBlock(code=code.strip(), language=lang) for lang, code in matches]


def find_final_answer(text: str) -> str | None:
    """Extract FINAL(...) answer from LLM response.

    Looks for FINAL(answer text here) -- NOT FINAL_VAR (that's handled by REPL).
    Returns the answer text or None if no FINAL() found.
    """
    # Match FINAL(...) but not FINAL_VAR(...)
    pattern = r"FINAL\((?!VAR)(.*?)\)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_iteration_output(
    iteration: int, repl_output: str, max_length: int = 2000
) -> str:
    """Format REPL output for inclusion in conversation history.

    Truncates long outputs and adds iteration context.
    """
    if len(repl_output) > max_length:
        repl_output = (
            repl_output[:max_length]
            + f"\n... [truncated, {len(repl_output)} total chars]"
        )
    return f"[Iteration {iteration}] REPL output:\n{repl_output}"
