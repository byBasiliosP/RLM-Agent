"""RuntimeContainer owns MCP server lifecycle: store, pipeline, agent infra.

All tests are synchronous. Verified via direct Python execution (5/5 pass).
If pytest-asyncio's strict mode causes hangs when running this file standalone,
use: pytest tests/test_runtime_container.py -p no:asyncio
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]


class TestRuntimeContainer:
    def test_container_constructs(self, tmp_path):
        from scholaragent.runtime import RuntimeContainer

        container = RuntimeContainer(
            data_dir=tmp_path,
            db_path=str(tmp_path / "t.db"),
            model_config={
                "strong": {"backend": "anthropic", "model_name": "x"},
                "cheap": {"backend": "openai", "model_name": "y"},
            },
            embeddings=FakeEmbeddings(),
        )

        store = container.get_store()
        assert store is container.get_store()  # same instance
        assert store.count() == 0
        container.close()

    def test_pipeline_lazy_init(self, tmp_path):
        from scholaragent.runtime import RuntimeContainer
        from scholaragent.memory.research import ResearchPipeline

        container = RuntimeContainer(
            data_dir=tmp_path,
            db_path=str(tmp_path / "t.db"),
            model_config={
                "strong": {"backend": "anthropic", "model_name": "x"},
                "cheap": {"backend": "openai", "model_name": "y"},
            },
            embeddings=FakeEmbeddings(),
        )
        pipeline = container.get_pipeline()
        assert isinstance(pipeline, ResearchPipeline)
        assert pipeline is container.get_pipeline()
        container.close()

    def test_close_idempotent(self, tmp_path):
        from scholaragent.runtime import RuntimeContainer

        container = RuntimeContainer(
            data_dir=tmp_path,
            db_path=str(tmp_path / "t.db"),
            model_config={
                "strong": {"backend": "anthropic", "model_name": "x"},
                "cheap": {"backend": "openai", "model_name": "y"},
            },
            embeddings=FakeEmbeddings(),
        )
        container.get_store()
        container.close()
        container.close()  # must not raise

    def test_model_config_accessible(self, tmp_path):
        from scholaragent.runtime import RuntimeContainer

        config = {
            "strong": {"backend": "anthropic", "model_name": "claude"},
            "cheap": {"backend": "openai", "model_name": "gpt-mini"},
        }
        container = RuntimeContainer(
            data_dir=tmp_path,
            db_path=str(tmp_path / "t.db"),
            model_config=config,
            embeddings=FakeEmbeddings(),
        )
        assert container.model_config == config
        container.close()

    def test_token_counter_none_before_agent_init(self, tmp_path):
        from scholaragent.runtime import RuntimeContainer

        container = RuntimeContainer(
            data_dir=tmp_path,
            db_path=str(tmp_path / "t.db"),
            model_config={
                "strong": {"backend": "anthropic", "model_name": "x"},
                "cheap": {"backend": "openai", "model_name": "y"},
            },
            embeddings=FakeEmbeddings(),
        )
        assert container.get_token_counter() is None
        container.close()
