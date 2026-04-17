"""Tests for the central ScholarConfig module."""


import pytest

from scholaragent.config import (
    VALID_BACKENDS,
    VALID_EMBEDDING_BACKENDS,
    ScholarConfig,
)


class TestScholarConfigDefaults:
    def test_defaults_match_previous_behavior(self, monkeypatch):
        # Clear all SCHOLAR_ env vars that affect defaults
        for var in (
            "SCHOLAR_STRONG_BACKEND",
            "SCHOLAR_CHEAP_BACKEND",
            "SCHOLAR_STRONG_MODEL",
            "SCHOLAR_CHEAP_MODEL",
            "SCHOLAR_LMSTUDIO_URL",
            "SCHOLAR_EMBEDDING_BACKEND",
            "SCHOLAR_EMBEDDING_MODEL",
            "SCHOLAR_LLM_CACHE_DISABLE",
            "SCHOLAR_CONTEXT_FLUSH_EVERY",
            "SCHOLAR_MEMORY_DIR",
            "SCHOLAR_MEMORY_DB",
        ):
            monkeypatch.delenv(var, raising=False)

        cfg = ScholarConfig.from_env()
        assert cfg.strong_backend == "anthropic"
        assert cfg.strong_model == "claude-sonnet-4-6"
        assert cfg.cheap_backend == "openai"
        assert cfg.cheap_model == "gpt-4o-mini"
        assert cfg.embedding_backend == "openai"
        assert cfg.lmstudio_url == "http://localhost:1234/v1"
        assert cfg.llm_cache_disable is False
        assert cfg.context_flush_every == 10


class TestScholarConfigValidation:
    def test_invalid_strong_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_STRONG_BACKEND", "lmstudioo")
        with pytest.raises(ValueError, match="SCHOLAR_STRONG_BACKEND"):
            ScholarConfig.from_env()

    def test_invalid_cheap_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_CHEAP_BACKEND", "gpt")
        with pytest.raises(ValueError, match="SCHOLAR_CHEAP_BACKEND"):
            ScholarConfig.from_env()

    def test_invalid_embedding_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_EMBEDDING_BACKEND", "cohere")
        with pytest.raises(ValueError, match="SCHOLAR_EMBEDDING_BACKEND"):
            ScholarConfig.from_env()

    def test_valid_lmstudio_accepted(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_STRONG_BACKEND", "lmstudio")
        monkeypatch.setenv("SCHOLAR_CHEAP_BACKEND", "lmstudio")
        cfg = ScholarConfig.from_env()
        assert cfg.strong_backend == "lmstudio"
        assert cfg.cheap_backend == "lmstudio"

    def test_flush_every_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_CONTEXT_FLUSH_EVERY", "0")
        with pytest.raises(ValueError, match="SCHOLAR_CONTEXT_FLUSH_EVERY"):
            ScholarConfig.from_env()

    def test_flush_every_non_integer_rejected(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_CONTEXT_FLUSH_EVERY", "abc")
        with pytest.raises(ValueError, match="SCHOLAR_CONTEXT_FLUSH_EVERY"):
            ScholarConfig.from_env()


class TestScholarConfigModelConfigDict:
    def test_anthropic_openai_no_base_url(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_STRONG_BACKEND", "anthropic")
        monkeypatch.setenv("SCHOLAR_CHEAP_BACKEND", "openai")
        cfg = ScholarConfig.from_env()
        mc = cfg.model_config_dict()
        assert "base_url" not in mc["strong"]
        assert "base_url" not in mc["cheap"]

    def test_lmstudio_sets_base_url(self, monkeypatch):
        monkeypatch.setenv("SCHOLAR_STRONG_BACKEND", "lmstudio")
        monkeypatch.setenv("SCHOLAR_LMSTUDIO_URL", "http://example:9999/v1")
        cfg = ScholarConfig.from_env()
        mc = cfg.model_config_dict()
        assert mc["strong"]["base_url"] == "http://example:9999/v1"


class TestScholarConfigConstants:
    def test_backends_are_frozensets(self):
        assert isinstance(VALID_BACKENDS, frozenset)
        assert isinstance(VALID_EMBEDDING_BACKENDS, frozenset)

    def test_expected_backends(self):
        assert "anthropic" in VALID_BACKENDS
        assert "openai" in VALID_BACKENDS
        assert "lmstudio" in VALID_BACKENDS
