"""Tests for wire protocol (comms.py) and LM handler (handler.py)."""

import socket

import pytest

from scholaragent.clients.base import BaseLM
from scholaragent.core.comms import socket_recv, socket_request, socket_send
from scholaragent.core.handler import LMHandler
from scholaragent.core.types import ModelUsageSummary, UsageSummary


# ---------------------------------------------------------------------------
# FakeLM test double
# ---------------------------------------------------------------------------


class FakeLM(BaseLM):
    """Deterministic LM client for testing."""

    def __init__(self, model_name: str = "fake-model"):
        super().__init__(model_name)

    def completion(self, prompt: str) -> str:
        return f"echo: {prompt}"

    async def acompletion(self, prompt: str) -> str:
        return f"echo: {prompt}"

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(prompt_tokens=0, completion_tokens=0, total_tokens=0)


# ---------------------------------------------------------------------------
# Wire protocol tests
# ---------------------------------------------------------------------------


class TestWireProtocol:
    """Test socket_send / socket_recv round-trip using socketpair."""

    def test_round_trip_simple(self):
        a, b = socket.socketpair()
        try:
            data = {"hello": "world", "n": 42}
            socket_send(a, data)
            result = socket_recv(b)
            assert result == data
        finally:
            a.close()
            b.close()

    def test_round_trip_empty_dict(self):
        a, b = socket.socketpair()
        try:
            socket_send(a, {})
            assert socket_recv(b) == {}
        finally:
            a.close()
            b.close()

    def test_round_trip_nested(self):
        a, b = socket.socketpair()
        try:
            data = {"list": [1, 2, 3], "nested": {"a": True}}
            socket_send(a, data)
            assert socket_recv(b) == data
        finally:
            a.close()
            b.close()

    def test_connection_closed_raises(self):
        a, b = socket.socketpair()
        a.close()
        with pytest.raises(Exception):
            socket_recv(b)
        b.close()


# ---------------------------------------------------------------------------
# LMHandler unit tests
# ---------------------------------------------------------------------------


class TestLMHandler:
    def test_start_stop(self):
        handler = LMHandler(FakeLM())
        addr = handler.start()
        assert addr[0] == "127.0.0.1"
        assert addr[1] > 0
        handler.stop()

    def test_completion_direct(self):
        handler = LMHandler(FakeLM())
        result = handler.completion("hi")
        assert result == "echo: hi"

    def test_get_client_default(self):
        fake = FakeLM()
        handler = LMHandler(fake)
        assert handler.get_client() is fake

    def test_get_client_by_name(self):
        default = FakeLM("default")
        other = FakeLM("other")
        handler = LMHandler(default)
        handler.register_client("other", other)
        assert handler.get_client("other") is other
        assert handler.get_client("unknown") is default

    def test_context_manager(self):
        fake = FakeLM()
        with LMHandler(fake) as handler:
            assert handler.port > 0
        # After exit, server should be stopped
        assert handler._server is None


# ---------------------------------------------------------------------------
# Full round-trip integration test
# ---------------------------------------------------------------------------


class TestLMHandlerRoundTrip:
    def test_socket_request_round_trip(self):
        with LMHandler(FakeLM()) as handler:
            response = socket_request(handler.address, {"prompt": "test"})
            assert response["status"] == "ok"
            assert response["content"] == "echo: test"

    def test_socket_request_with_model(self):
        default = FakeLM("default-model")
        other = FakeLM("other-model")
        handler = LMHandler(default)
        handler.register_client("other-model", other)
        with handler:
            response = socket_request(
                handler.address, {"prompt": "hello", "model": "other-model"}
            )
            assert response["status"] == "ok"
            assert response["content"] == "echo: hello"

    def test_socket_request_missing_prompt(self):
        with LMHandler(FakeLM()) as handler:
            response = socket_request(handler.address, {})
            assert response["status"] == "error"

    def test_socket_request_error_handling(self):
        """Verify that exceptions in the client are returned as error responses."""

        class BrokenLM(FakeLM):
            def completion(self, prompt: str) -> str:
                raise RuntimeError("boom")

        with LMHandler(BrokenLM()) as handler:
            response = socket_request(handler.address, {"prompt": "hi"})
            assert response["status"] == "error"
            assert "boom" in response["error"]
