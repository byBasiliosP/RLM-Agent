"""Tests for parsing utilities and budget tracking."""

import pytest

from scholaragent.core.types import CodeBlock
from scholaragent.utils.parsing import (
    find_code_blocks,
    find_final_answer,
    format_iteration_output,
)
from scholaragent.utils.budget import Budget


# ---------------------------------------------------------------------------
# find_code_blocks
# ---------------------------------------------------------------------------


class TestFindCodeBlocks:
    def test_single_repl_block(self):
        text = 'Here is some code:\n```repl\nresults = search_arxiv("protein")\nprint(results)\n```\nDone.'
        blocks = find_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].language == "repl"
        assert 'search_arxiv("protein")' in blocks[0].code

    def test_multiple_blocks(self):
        text = (
            "Step 1:\n```repl\nx = 1\n```\n"
            "Step 2:\n```repl\ny = 2\n```\n"
        )
        blocks = find_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].code == "x = 1"
        assert blocks[1].code == "y = 2"

    def test_no_blocks(self):
        text = "Just some plain text with no code."
        blocks = find_code_blocks(text)
        assert blocks == []

    def test_python_block(self):
        text = "```python\nprint('hello')\n```"
        blocks = find_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert blocks[0].code == "print('hello')"

    def test_returns_code_block_instances(self):
        text = "```repl\na = 1\n```"
        blocks = find_code_blocks(text)
        assert isinstance(blocks[0], CodeBlock)

    def test_ignores_other_languages(self):
        text = "```json\n{}\n```"
        blocks = find_code_blocks(text)
        assert blocks == []


# ---------------------------------------------------------------------------
# find_final_answer
# ---------------------------------------------------------------------------


class TestFindFinalAnswer:
    def test_final_found(self):
        text = "After analysis, FINAL(The key finding is diffusion models outperform GANs.)"
        answer = find_final_answer(text)
        assert answer == "The key finding is diffusion models outperform GANs."

    def test_no_final(self):
        text = "Still working on the analysis."
        answer = find_final_answer(text)
        assert answer is None

    def test_final_var_not_matched(self):
        text = 'FINAL_VAR(my_result, "some value")'
        answer = find_final_answer(text)
        assert answer is None

    def test_multiline_final(self):
        text = "FINAL(Line one.\nLine two.)"
        answer = find_final_answer(text)
        assert answer == "Line one.\nLine two."

    def test_final_with_surrounding_text(self):
        text = "Let me conclude.\nFINAL(The answer is 42.)\nThat's it."
        answer = find_final_answer(text)
        assert answer == "The answer is 42."


# ---------------------------------------------------------------------------
# format_iteration_output
# ---------------------------------------------------------------------------


class TestFormatIterationOutput:
    def test_normal_output(self):
        result = format_iteration_output(1, "Hello world")
        assert result == "[Iteration 1] REPL output:\nHello world"

    def test_truncated_output(self):
        long_output = "x" * 3000
        result = format_iteration_output(2, long_output, max_length=2000)
        assert "[truncated, 3000 total chars]" in result
        assert result.startswith("[Iteration 2] REPL output:\n")
        # The body (after header) should be at most 2000 chars + truncation notice
        body = result.split("\n", 1)[1]
        assert body.startswith("x" * 2000)

    def test_exact_max_length_not_truncated(self):
        output = "a" * 2000
        result = format_iteration_output(1, output, max_length=2000)
        assert "truncated" not in result


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class TestBudget:
    def test_default_creation(self):
        b = Budget()
        assert b.max_tokens == 100_000
        assert b.max_iterations == 10
        assert b.timeout_seconds == 300.0
        assert b.tokens_used == 0
        assert b.iterations_used == 0

    def test_custom_creation(self):
        b = Budget(max_tokens=50_000, max_iterations=5, timeout_seconds=60.0)
        assert b.max_tokens == 50_000
        assert b.max_iterations == 5
        assert b.timeout_seconds == 60.0

    def test_tokens_remaining(self):
        b = Budget(max_tokens=1000)
        b.tokens_used = 300
        assert b.tokens_remaining == 700

    def test_iterations_remaining(self):
        b = Budget(max_iterations=5)
        b.iterations_used = 3
        assert b.iterations_remaining == 2

    def test_tokens_remaining_never_negative(self):
        b = Budget(max_tokens=100)
        b.tokens_used = 200
        assert b.tokens_remaining == 0

    def test_is_exhausted_tokens(self):
        b = Budget(max_tokens=100)
        assert not b.is_exhausted
        b.tokens_used = 100
        assert b.is_exhausted

    def test_is_exhausted_iterations(self):
        b = Budget(max_iterations=3)
        assert not b.is_exhausted
        b.iterations_used = 3
        assert b.is_exhausted

    def test_use_tokens(self):
        b = Budget()
        b.use_tokens(500)
        assert b.tokens_used == 500
        b.use_tokens(300)
        assert b.tokens_used == 800

    def test_use_iteration(self):
        b = Budget()
        b.use_iteration()
        assert b.iterations_used == 1
        b.use_iteration()
        assert b.iterations_used == 2
