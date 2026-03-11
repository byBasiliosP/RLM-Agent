"""Tests for LLM response cache."""

import os
import tempfile
import time

import pytest

from scholaragent.utils.cache import LLMCache


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        c = LLMCache(cache_dir=os.path.join(tmpdir, "cache"), ttl_seconds=10)
        yield c


class TestLLMCache:
    def test_put_and_get(self, cache):
        messages = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4o", messages, "world")
        assert cache.get("gpt-4o", messages) == "world"

    def test_cache_miss(self, cache):
        messages = [{"role": "user", "content": "unknown"}]
        assert cache.get("gpt-4o", messages) is None

    def test_different_models_different_keys(self, cache):
        messages = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4o", messages, "answer-a")
        cache.put("claude-sonnet-4-6", messages, "answer-b")
        assert cache.get("gpt-4o", messages) == "answer-a"
        assert cache.get("claude-sonnet-4-6", messages) == "answer-b"

    def test_different_messages_different_keys(self, cache):
        msg1 = [{"role": "user", "content": "hello"}]
        msg2 = [{"role": "user", "content": "goodbye"}]
        cache.put("gpt-4o", msg1, "answer-1")
        cache.put("gpt-4o", msg2, "answer-2")
        assert cache.get("gpt-4o", msg1) == "answer-1"
        assert cache.get("gpt-4o", msg2) == "answer-2"

    def test_clear(self, cache):
        messages = [{"role": "user", "content": "test"}]
        cache.put("gpt-4o", messages, "value")
        assert cache.get("gpt-4o", messages) == "value"
        cache.clear()
        assert cache.get("gpt-4o", messages) is None

    def test_expired_entry_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            c = LLMCache(cache_dir=os.path.join(tmpdir, "cache"), ttl_seconds=0)
            messages = [{"role": "user", "content": "test"}]
            c.put("gpt-4o", messages, "value")
            time.sleep(0.1)
            assert c.get("gpt-4o", messages) is None

    def test_prune_expired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            c = LLMCache(cache_dir=cache_dir, ttl_seconds=0)
            messages = [{"role": "user", "content": "test"}]
            c.put("gpt-4o", messages, "value")
            time.sleep(0.1)
            removed = c.prune_expired()
            assert removed >= 1
            # Verify file was actually deleted
            json_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
            assert len(json_files) == 0

    def test_cache_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "nested", "cache")
            c = LLMCache(cache_dir=cache_dir)
            assert os.path.isdir(cache_dir)
