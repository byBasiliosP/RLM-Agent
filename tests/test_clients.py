"""Tests for LLM client abstraction and model router."""

import pytest
from unittest.mock import patch, MagicMock

from scholaragent.clients.base import BaseLM
from scholaragent.clients.openai_client import OpenAIClient
from scholaragent.clients.anthropic_client import AnthropicClient
from scholaragent.clients.router import ModelConfig, ModelRouter, CHEAP_ROLES
from scholaragent.core.types import ModelUsageSummary, UsageSummary


# ---------------------------------------------------------------------------
# 1. BaseLM is abstract -- cannot be instantiated
# ---------------------------------------------------------------------------
class TestBaseLM:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLM(model_name="test-model")  # type: ignore[abstract]

    def test_has_required_abstract_methods(self):
        abstract_methods = BaseLM.__abstractmethods__
        assert "completion" in abstract_methods
        assert "acompletion" in abstract_methods

    def test_has_concrete_usage_methods(self):
        """get_usage_summary and get_last_usage are now concrete in BaseLM."""
        assert hasattr(BaseLM, "get_usage_summary")
        assert hasattr(BaseLM, "get_last_usage")
        assert "get_usage_summary" not in BaseLM.__abstractmethods__
        assert "get_last_usage" not in BaseLM.__abstractmethods__


# ---------------------------------------------------------------------------
# 2. ModelConfig dataclass
# ---------------------------------------------------------------------------
class TestModelConfig:
    def test_creation(self):
        cfg = ModelConfig(backend="openai", model_name="gpt-4o-mini")
        assert cfg.backend == "openai"
        assert cfg.model_name == "gpt-4o-mini"

    def test_creation_anthropic(self):
        cfg = ModelConfig(backend="anthropic", model_name="claude-3-5-haiku-latest")
        assert cfg.backend == "anthropic"
        assert cfg.model_name == "claude-3-5-haiku-latest"


# ---------------------------------------------------------------------------
# 3. ModelRouter.get_config -- cheap for "scout", strong for others
# ---------------------------------------------------------------------------
class TestModelRouterGetConfig:
    @pytest.fixture()
    def router(self):
        return ModelRouter(
            strong=ModelConfig(backend="openai", model_name="gpt-4o"),
            cheap=ModelConfig(backend="openai", model_name="gpt-4o-mini"),
        )

    def test_scout_gets_cheap(self, router: ModelRouter):
        cfg = router.get_config("scout")
        assert cfg.model_name == "gpt-4o-mini"

    def test_analyst_gets_strong(self, router: ModelRouter):
        cfg = router.get_config("analyst")
        assert cfg.model_name == "gpt-4o"

    def test_critic_gets_strong(self, router: ModelRouter):
        cfg = router.get_config("critic")
        assert cfg.model_name == "gpt-4o"

    def test_synthesizer_gets_strong(self, router: ModelRouter):
        cfg = router.get_config("synthesizer")
        assert cfg.model_name == "gpt-4o"

    def test_cheap_roles_is_frozenset(self):
        assert isinstance(CHEAP_ROLES, frozenset)
        assert "scout" in CHEAP_ROLES


# ---------------------------------------------------------------------------
# 4. ModelRouter.get_client -- creates correct client type
# ---------------------------------------------------------------------------
class TestModelRouterGetClient:
    def test_get_client_openai(self):
        router = ModelRouter(
            strong=ModelConfig(backend="openai", model_name="gpt-4o"),
            cheap=ModelConfig(backend="openai", model_name="gpt-4o-mini"),
        )
        with patch("scholaragent.clients.openai_client.openai"):
            client = router.get_client("analyst")
        assert isinstance(client, OpenAIClient)
        assert isinstance(client, BaseLM)
        assert client.model_name == "gpt-4o"

    def test_get_client_anthropic(self):
        router = ModelRouter(
            strong=ModelConfig(backend="anthropic", model_name="claude-sonnet-4-20250514"),
            cheap=ModelConfig(backend="anthropic", model_name="claude-3-5-haiku-latest"),
        )
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = router.get_client("scout")
        assert isinstance(client, AnthropicClient)
        assert isinstance(client, BaseLM)
        assert client.model_name == "claude-3-5-haiku-latest"

    def test_get_client_unknown_backend_raises(self):
        router = ModelRouter(
            strong=ModelConfig(backend="unknown", model_name="x"),
            cheap=ModelConfig(backend="unknown", model_name="y"),
        )
        with pytest.raises(ValueError, match="Unknown backend"):
            router.get_client("analyst")


# ---------------------------------------------------------------------------
# 5. Concrete clients are BaseLM subclasses
# ---------------------------------------------------------------------------
class TestClientSubclassing:
    def test_openai_is_base_lm_subclass(self):
        assert issubclass(OpenAIClient, BaseLM)

    def test_anthropic_is_base_lm_subclass(self):
        assert issubclass(AnthropicClient, BaseLM)

    def test_openai_client_has_model_name(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o")
        assert client.model_name == "gpt-4o"

    def test_anthropic_client_has_model_name(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514")
        assert client.model_name == "claude-sonnet-4-20250514"

    def test_openai_client_initial_usage_summary(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o")
        summary = client.get_usage_summary()
        assert isinstance(summary, UsageSummary)

    def test_anthropic_client_initial_usage_summary(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514")
        summary = client.get_usage_summary()
        assert isinstance(summary, UsageSummary)

    def test_openai_client_initial_last_usage(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o")
        usage = client.get_last_usage()
        assert isinstance(usage, ModelUsageSummary)
        assert usage.total_tokens == 0

    def test_anthropic_client_initial_last_usage(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514")
        usage = client.get_last_usage()
        assert isinstance(usage, ModelUsageSummary)
        assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# 6. Timeout and max_tokens configuration
# ---------------------------------------------------------------------------
class TestClientTimeoutAndMaxTokens:
    def test_openai_default_timeout(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o", api_key="fake")
        assert client.timeout == 120.0

    def test_anthropic_default_timeout(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514", api_key="fake")
        assert client.timeout == 120.0

    def test_openai_custom_timeout(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="test", api_key="fake", timeout=30.0)
        assert client.timeout == 30.0

    def test_anthropic_custom_timeout(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="test", api_key="fake", timeout=60.0)
        assert client.timeout == 60.0

    def test_openai_default_max_tokens_is_none(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="test", api_key="fake")
        assert client.max_tokens is None

    def test_anthropic_default_max_tokens_is_none(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="test", api_key="fake")
        assert client.max_tokens is None

    def test_openai_custom_max_tokens(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="test", api_key="fake", max_tokens=2048)
        assert client.max_tokens == 2048

    def test_anthropic_custom_max_tokens(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="test", api_key="fake", max_tokens=8192)
        assert client.max_tokens == 8192

    def test_model_config_max_tokens_default_none(self):
        cfg = ModelConfig(backend="openai", model_name="gpt-4o")
        assert cfg.max_tokens is None

    def test_model_config_max_tokens_set(self):
        cfg = ModelConfig(backend="openai", model_name="gpt-4o", max_tokens=8192)
        assert cfg.max_tokens == 8192

    def test_router_passes_max_tokens(self):
        router = ModelRouter(
            strong=ModelConfig(backend="openai", model_name="gpt-4o", max_tokens=4096),
            cheap=ModelConfig(backend="openai", model_name="gpt-4o-mini"),
        )
        with patch("scholaragent.clients.openai_client.openai"):
            client = router.get_client("analyst")
        assert client.max_tokens == 4096

    def test_router_no_max_tokens(self):
        router = ModelRouter(
            strong=ModelConfig(backend="openai", model_name="gpt-4o"),
            cheap=ModelConfig(backend="openai", model_name="gpt-4o-mini"),
        )
        with patch("scholaragent.clients.openai_client.openai"):
            client = router.get_client("analyst")
        assert client.max_tokens is None


# ---------------------------------------------------------------------------
# 7. completion_messages() -- preserves role structure
# ---------------------------------------------------------------------------
class TestCompletionMessages:
    """Tests for the completion_messages method on BaseLM subclasses."""

    _messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Thanks"},
    ]

    def test_completion_messages_exists_on_openai(self):
        assert hasattr(OpenAIClient, "completion_messages")

    def test_completion_messages_exists_on_anthropic(self):
        assert hasattr(AnthropicClient, "completion_messages")

    def test_openai_completion_messages_passes_messages_directly(self):
        with patch("scholaragent.clients.openai_client.openai") as mock_openai:
            client = OpenAIClient(model_name="gpt-4o", max_tokens=100)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response text"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        client._sync_client.chat.completions.create = MagicMock(return_value=mock_response)

        result = client.completion_messages(self._messages)

        call_kwargs = client._sync_client.chat.completions.create.call_args
        assert call_kwargs[1]["messages"] == self._messages
        assert call_kwargs[1]["model"] == "gpt-4o"
        assert call_kwargs[1]["max_tokens"] == 100
        assert result == "response text"

    def test_openai_completion_messages_omits_max_tokens_when_none(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1

        client._sync_client.chat.completions.create = MagicMock(return_value=mock_response)
        client.completion_messages(self._messages)

        call_kwargs = client._sync_client.chat.completions.create.call_args[1]
        assert "max_tokens" not in call_kwargs

    def test_anthropic_completion_messages_extracts_system(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514")

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "response text"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        client._sync_client.messages.create = MagicMock(return_value=mock_response)

        result = client.completion_messages(self._messages)

        call_kwargs = client._sync_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."
        # system message should NOT appear in the messages list
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"
        assert len(call_kwargs["messages"]) == 3
        assert call_kwargs["messages"][0] == {"role": "user", "content": "Hello"}
        assert result == "response text"

    def test_anthropic_completion_messages_no_system(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514")

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "ok"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1

        client._sync_client.messages.create = MagicMock(return_value=mock_response)

        msgs_no_system = [m for m in self._messages if m["role"] != "system"]
        client.completion_messages(msgs_no_system)

        call_kwargs = client._sync_client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_openai_completion_messages_records_usage(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10

        client._sync_client.chat.completions.create = MagicMock(return_value=mock_response)
        client.completion_messages(self._messages)

        usage = client.get_last_usage()
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 30

    def test_anthropic_completion_messages_records_usage(self):
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514")

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "ok"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 8

        client._sync_client.messages.create = MagicMock(return_value=mock_response)
        client.completion_messages(self._messages)

        usage = client.get_last_usage()
        assert usage.prompt_tokens == 15
        assert usage.completion_tokens == 8
        assert usage.total_tokens == 23


# ---------------------------------------------------------------------------
# 8. RateLimiter integration
# ---------------------------------------------------------------------------
class TestRateLimiterIntegration:
    def test_openai_client_has_rate_limiter_from_router(self):
        from scholaragent.clients.rate_limiter import RateLimiter

        router = ModelRouter(
            strong=ModelConfig(backend="openai", model_name="gpt-4o"),
            cheap=ModelConfig(backend="openai", model_name="gpt-4o-mini"),
        )
        with patch("scholaragent.clients.openai_client.openai"):
            client = router.get_client("analyst")
        assert isinstance(client.rate_limiter, RateLimiter)
        assert client.rate_limiter.rpm == 60

    def test_anthropic_client_has_rate_limiter_from_router(self):
        from scholaragent.clients.rate_limiter import RateLimiter

        router = ModelRouter(
            strong=ModelConfig(backend="anthropic", model_name="claude-sonnet-4-6"),
            cheap=ModelConfig(backend="anthropic", model_name="claude-haiku-3"),
        )
        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = router.get_client("scout")
        assert isinstance(client.rate_limiter, RateLimiter)
        assert client.rate_limiter.rpm == 50

    def test_base_client_rate_limiter_defaults_none(self):
        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o", api_key="fake")
        assert client.rate_limiter is None


# ---------------------------------------------------------------------------
# 9. Client error handling and retry
# ---------------------------------------------------------------------------
class TestClientErrorHandling:
    @patch("scholaragent.utils.retry.time.sleep")  # skip actual delays
    def test_openai_retries_on_rate_limit(self, mock_sleep):
        """OpenAI client should retry on RateLimitError."""
        import openai as openai_mod

        call_count = 0
        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise openai_mod.RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            return mock_resp

        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o-mini")
        client._sync_client = MagicMock()
        client._sync_client.chat.completions.create = fake_create

        result = client.completion("test")
        assert result == "ok"
        assert call_count == 3  # 2 failures + 1 success

    @patch("scholaragent.utils.retry.time.sleep")
    def test_openai_raises_after_max_retries(self, mock_sleep):
        """After max retries, the error should propagate."""
        import openai as openai_mod

        def always_fail(**kwargs):
            raise openai_mod.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )

        with patch("scholaragent.clients.openai_client.openai"):
            client = OpenAIClient(model_name="gpt-4o-mini")
        client._sync_client = MagicMock()
        client._sync_client.chat.completions.create = always_fail

        with pytest.raises(openai_mod.RateLimitError):
            client.completion("test")

    @patch("scholaragent.utils.retry.time.sleep")
    def test_anthropic_retries_on_rate_limit(self, mock_sleep):
        """Anthropic client should retry on RateLimitError."""
        import anthropic as anthropic_mod

        call_count = 0
        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise anthropic_mod.RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            mock_resp = MagicMock()
            mock_resp.content = [MagicMock(text="ok")]
            mock_resp.usage = MagicMock(input_tokens=10, output_tokens=5)
            return mock_resp

        with patch("scholaragent.clients.anthropic_client.anthropic"):
            client = AnthropicClient(model_name="claude-sonnet-4-20250514")
        client._sync_client = MagicMock()
        client._sync_client.messages.create = fake_create

        result = client.completion("test")
        assert result == "ok"
        assert call_count == 3


# ---------------------------------------------------------------------------
# 10. LM Studio backend support
# ---------------------------------------------------------------------------
class TestLMStudioBackend:
    def test_model_config_base_url(self):
        cfg = ModelConfig(backend="lmstudio", model_name="llama-3.2-3b-instruct")
        assert cfg.base_url is None

    def test_model_config_custom_base_url(self):
        cfg = ModelConfig(
            backend="lmstudio",
            model_name="llama-3.2-3b-instruct",
            base_url="http://localhost:5000/v1",
        )
        assert cfg.base_url == "http://localhost:5000/v1"

    def test_openai_client_accepts_base_url(self):
        with patch("scholaragent.clients.openai_client.openai") as mock_openai:
            client = OpenAIClient(
                model_name="test-model",
                api_key="fake",
                base_url="http://localhost:1234/v1",
            )
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:1234/v1"
        assert call_kwargs["api_key"] == "fake"

    def test_router_creates_lmstudio_client(self):
        router = ModelRouter(
            strong=ModelConfig(backend="lmstudio", model_name="kimi-dev-72b"),
            cheap=ModelConfig(backend="lmstudio", model_name="llama-3.2-3b-instruct"),
        )
        with patch("scholaragent.clients.openai_client.openai"):
            client = router.get_client("analyst")
        assert isinstance(client, OpenAIClient)
        assert client.model_name == "kimi-dev-72b"

    def test_router_lmstudio_default_base_url(self):
        router = ModelRouter(
            strong=ModelConfig(backend="lmstudio", model_name="test-model"),
            cheap=ModelConfig(backend="lmstudio", model_name="test-model"),
        )
        with patch("scholaragent.clients.openai_client.openai") as mock_openai:
            router.get_client("analyst")
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:1234/v1"
        assert call_kwargs["api_key"] == "lm-studio"

    def test_router_lmstudio_custom_base_url(self):
        router = ModelRouter(
            strong=ModelConfig(
                backend="lmstudio",
                model_name="test-model",
                base_url="http://192.168.1.100:1234/v1",
            ),
            cheap=ModelConfig(backend="lmstudio", model_name="test-model"),
        )
        with patch("scholaragent.clients.openai_client.openai") as mock_openai:
            router.get_client("analyst")
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://192.168.1.100:1234/v1"

    def test_lmstudio_rate_limiter_defaults(self):
        from scholaragent.clients.rate_limiter import PROVIDER_DEFAULTS, RateLimiter

        assert "lmstudio" in PROVIDER_DEFAULTS
        assert PROVIDER_DEFAULTS["lmstudio"]["rpm"] == 1000

        router = ModelRouter(
            strong=ModelConfig(backend="lmstudio", model_name="test-model"),
            cheap=ModelConfig(backend="lmstudio", model_name="test-model"),
        )
        with patch("scholaragent.clients.openai_client.openai"):
            client = router.get_client("scout")
        assert isinstance(client.rate_limiter, RateLimiter)
        assert client.rate_limiter.rpm == 1000
