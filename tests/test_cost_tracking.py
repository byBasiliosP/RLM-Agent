"""Tests for token cost tracking."""

import pytest

from scholaragent.utils.cost import PRICING, estimate_cost
from scholaragent.clients.token_counter import TokenCounter


class TestCostEstimation:
    def test_known_model_cost(self):
        # gpt-4o: $2.50/1M input, $10.00/1M output
        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(12.50)

    def test_small_usage(self):
        cost = estimate_cost("gpt-4o", 1000, 500)
        expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_zero(self):
        assert estimate_cost("unknown-model", 10000, 5000) == 0.0

    def test_zero_tokens(self):
        assert estimate_cost("gpt-4o", 0, 0) == 0.0

    def test_anthropic_models_in_pricing(self):
        assert "claude-sonnet-4-6" in PRICING
        assert "claude-haiku-3-5" in PRICING
        assert "claude-opus-4-6" in PRICING

    def test_openai_models_in_pricing(self):
        assert "gpt-4o" in PRICING
        assert "gpt-4o-mini" in PRICING
        assert "gpt-4.1" in PRICING


class TestTokenCounterCosts:
    def test_cost_summary_empty(self):
        tc = TokenCounter()
        summary = tc.cost_summary()
        assert summary["total_cost_usd"] == 0.0
        assert summary["model_costs"] == {}

    def test_cost_summary_with_usage(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 10000, 5000)
        summary = tc.cost_summary()
        assert summary["total_cost_usd"] > 0
        assert "gpt-4o" in summary["model_costs"]

    def test_cost_summary_multiple_models(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 10000, 5000)
        tc.record("gpt-4o-mini", 50000, 10000)
        summary = tc.cost_summary()
        assert len(summary["model_costs"]) == 2
        assert summary["total_cost_usd"] > 0

    def test_report_includes_costs(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 10000, 5000)
        report = tc.report()
        assert "$" in report
        assert "gpt-4o" in report

    def test_report_empty(self):
        tc = TokenCounter()
        assert tc.report() == "No LLM calls recorded."
