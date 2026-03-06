"""Tests for TokenCounter."""

from scholaragent.clients.token_counter import TokenCounter


class TestTokenCounterRecord:
    def test_record_single(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        s = tc.summary()
        assert s["models"]["gpt-4o"]["prompt_tokens"] == 100
        assert s["models"]["gpt-4o"]["completion_tokens"] == 50
        assert s["total"]["total_tokens"] == 150

    def test_record_multiple_same_model(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        tc.record("gpt-4o", 200, 100)
        s = tc.summary()
        assert s["models"]["gpt-4o"]["prompt_tokens"] == 300
        assert s["total"]["total_tokens"] == 450

    def test_record_multiple_models(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        tc.record("gpt-4.1-mini", 80, 20)
        s = tc.summary()
        assert len(s["models"]) == 2
        assert s["total"]["total_tokens"] == 250

    def test_call_count(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        tc.record("gpt-4o", 100, 50)
        s = tc.summary()
        assert s["total"]["calls"] == 2


class TestTokenCounterReport:
    def test_report_contains_model_name(self):
        tc = TokenCounter()
        tc.record("gpt-4o", 100, 50)
        report = tc.report()
        assert "gpt-4o" in report

    def test_report_empty(self):
        tc = TokenCounter()
        report = tc.report()
        assert "No LLM calls" in report


class TestTokenCounterLogCall:
    def test_log_call_returns_string(self, capsys):
        tc = TokenCounter()
        tc.log_call("gpt-4o", 100, 50)
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out
        assert "150" in captured.out
