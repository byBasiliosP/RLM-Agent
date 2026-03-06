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
        assert "get_usage_summary" in abstract_methods
        assert "get_last_usage" in abstract_methods


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
