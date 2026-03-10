# ScholarAgent Gaps & Optimizations Implementation Plan

> **Status:** ✅ Fully implemented. All 7 tasks completed.

**Goal:** Fix the 7 highest-impact gaps in ScholarAgent: thread-safe MCP init, concurrent source collection, HTTP connection pooling, embedding caching, LLM client error handling, structured logging, and `_html_to_text` test coverage.

**Architecture:** Each task is independent and can be committed separately. We prioritize production robustness (race conditions, error paths) first, then performance (concurrency, caching), then test coverage.

**Tech Stack:** Python 3.12+, httpx, threading, functools.lru_cache, logging, pytest

---

### Task 1: Thread-safe MCP global state initialization

The `_get_store()` and `_get_pipeline()` functions in `mcp_server.py` have a TOCTOU race — two concurrent tool calls can both see `_store is None` and double-initialize. Fix with a module-level `threading.Lock`.

**Files:**
- Modify: `scholaragent/mcp_server.py:37-72`
- Test: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

Add to `tests/test_mcp_server.py`:

```python
import threading

class TestMCPThreadSafety:
    """Verify lazy singletons are thread-safe."""

    def test_concurrent_get_store_returns_same_instance(self, tmp_path, monkeypatch):
        """Multiple threads calling _get_store() must get the same instance."""
        import scholaragent.mcp_server as mod
        monkeypatch.setattr(mod, "_store", None)
        monkeypatch.setattr(mod, "_pipeline", None)
        monkeypatch.setattr(mod, "DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(mod, "OpenAIEmbeddings", lambda: FakeEmbeddings())

        stores = []
        barrier = threading.Barrier(4)

        def grab():
            barrier.wait()
            stores.append(mod._get_store())

        threads = [threading.Thread(target=grab) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads must get the exact same object
        assert len(stores) == 4
        assert all(s is stores[0] for s in stores)
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_mcp_server.py::TestMCPThreadSafety -v`
Expected: FAIL (race condition — may intermittently pass, but the fix is still needed for correctness)

**Step 3: Write minimal implementation**

In `scholaragent/mcp_server.py`, add a lock and use it in both getters:

```python
# After line 40 (_pipeline declaration), add:
_init_lock = threading.Lock()
```

Add `import threading` to imports.

Replace `_get_store()`:

```python
def _get_store() -> MemoryStore:
    """Lazy-init the global memory store (thread-safe)."""
    global _store
    if _store is not None:
        return _store
    with _init_lock:
        if _store is not None:          # double-check after acquiring lock
            return _store
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        embeddings = OpenAIEmbeddings()
        _store = MemoryStore(db_path=DB_PATH, embeddings=embeddings)
        return _store
```

Replace `_get_pipeline()`:

```python
def _get_pipeline() -> ResearchPipeline:
    """Lazy-init the research pipeline (thread-safe)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _init_lock:
        if _pipeline is not None:
            return _pipeline
        _pipeline = ResearchPipeline(store=_get_store())
        return _pipeline
```

**Step 4: Run test to verify it passes**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_mcp_server.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add scholaragent/mcp_server.py tests/test_mcp_server.py
git commit -m "fix: thread-safe lazy init for MCP global state"
```

---

### Task 2: Concurrent source collection in ResearchPipeline

`_collect_sources()` calls arXiv, Semantic Scholar, GitHub, and docs sequentially. Use `concurrent.futures.ThreadPoolExecutor` to run them in parallel (~3-4x speedup).

**Files:**
- Modify: `scholaragent/memory/research.py:94-152`
- Test: `tests/test_research_pipeline.py`

**Step 1: Write the failing test**

Add to `tests/test_research_pipeline.py`:

```python
import time

class TestConcurrentCollection:

    def test_sources_collected_concurrently(self, store, monkeypatch):
        """Source adapters should run in parallel, not sequentially."""
        call_times = []

        def slow_arxiv(query, max_results=10):
            call_times.append(("arxiv_start", time.monotonic()))
            time.sleep(0.2)
            call_times.append(("arxiv_end", time.monotonic()))
            return "[]"

        def slow_s2(query, limit=10):
            call_times.append(("s2_start", time.monotonic()))
            time.sleep(0.2)
            call_times.append(("s2_end", time.monotonic()))
            return "[]"

        def slow_github(query, language=None, max_results=10):
            call_times.append(("github_start", time.monotonic()))
            time.sleep(0.2)
            call_times.append(("github_end", time.monotonic()))
            return []

        def slow_docs(query, max_results=5):
            call_times.append(("docs_start", time.monotonic()))
            time.sleep(0.2)
            call_times.append(("docs_end", time.monotonic()))
            return []

        import scholaragent.memory.research as mod
        monkeypatch.setattr(mod, "search_arxiv", slow_arxiv)
        monkeypatch.setattr(mod, "search_semantic_scholar", slow_s2)
        monkeypatch.setattr(mod, "search_github_code", slow_github)
        monkeypatch.setattr(mod, "search_docs", slow_docs)

        pipeline = ResearchPipeline(store=store)
        t0 = time.monotonic()
        pipeline._collect_sources("test query")
        elapsed = time.monotonic() - t0

        # 4 sources × 0.2s each = 0.8s sequential.
        # Concurrent should complete in ~0.25s (0.2s + overhead).
        assert elapsed < 0.5, f"Sources collected sequentially ({elapsed:.2f}s)"
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_research_pipeline.py::TestConcurrentCollection -v`
Expected: FAIL with "Sources collected sequentially (0.8Xs)"

**Step 3: Write minimal implementation**

Rewrite `_collect_sources` in `scholaragent/memory/research.py`:

```python
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# ... (existing imports stay) ...

class ResearchPipeline:
    # ... __init__ and run stay ...

    def _collect_sources(
        self,
        query: str,
        sources: list[str] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Collect raw results from all source adapters concurrently."""
        sources = sources or ["paper", "code", "docs"]
        results = []
        errors = []

        def _fetch_arxiv():
            arxiv_json = search_arxiv(query, max_results=10)
            arxiv_papers = json.loads(arxiv_json)
            items = []
            if isinstance(arxiv_papers, list):
                for paper in arxiv_papers:
                    items.append({
                        "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nAuthors: {', '.join(paper.get('authors', []))}",
                        "source_type": "paper",
                        "source_ref": f"arxiv:{paper.get('arxiv_id', '')}",
                    })
            return items

        def _fetch_s2():
            s2_json = search_semantic_scholar(query, limit=10)
            s2_papers = json.loads(s2_json)
            items = []
            if isinstance(s2_papers, list):
                for paper in s2_papers:
                    items.append({
                        "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nYear: {paper.get('year', 'N/A')}\nCitations: {paper.get('citation_count', 0)}",
                        "source_type": "paper",
                        "source_ref": f"s2:{paper.get('paper_id', '')}",
                    })
            return items

        def _fetch_github():
            return search_github_code(query, language="python", max_results=5)

        def _fetch_docs():
            return search_docs(query, max_results=3)

        tasks = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            if "paper" in sources:
                tasks[executor.submit(_fetch_arxiv)] = "arXiv"
                tasks[executor.submit(_fetch_s2)] = "Semantic Scholar"
            if "code" in sources:
                tasks[executor.submit(_fetch_github)] = "GitHub"
            if "docs" in sources:
                tasks[executor.submit(_fetch_docs)] = "Docs"

            for future in as_completed(tasks):
                label = tasks[future]
                try:
                    items = future.result(timeout=60)
                    results.extend(items)
                except Exception as e:
                    errors.append(f"{label}: {type(e).__name__}: {e}")

        return results, errors
```

**Step 4: Run all tests to verify**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add scholaragent/memory/research.py tests/test_research_pipeline.py
git commit -m "perf: concurrent source collection in ResearchPipeline (~3-4x speedup)"
```

---

### Task 3: HTTP connection pooling in source adapters

Each `httpx.get()` call creates a new TCP connection. Add module-level `httpx.Client` instances with connection pooling.

**Files:**
- Modify: `scholaragent/tools/arxiv.py`
- Modify: `scholaragent/tools/semantic_scholar.py`
- Modify: `scholaragent/sources/github.py`
- Modify: `scholaragent/sources/docs.py`
- Test: `tests/test_tools.py`, `tests/test_sources.py`

**Step 1: Write the failing test**

Add to `tests/test_tools.py`:

```python
class TestConnectionPooling:
    def test_arxiv_uses_shared_client(self):
        """arxiv module should expose a module-level httpx.Client."""
        import scholaragent.tools.arxiv as mod
        assert hasattr(mod, "_http_client")
        assert isinstance(mod._http_client, httpx.Client)

    def test_semantic_scholar_uses_shared_client(self):
        import scholaragent.tools.semantic_scholar as mod
        assert hasattr(mod, "_http_client")
        assert isinstance(mod._http_client, httpx.Client)
```

Add to `tests/test_sources.py`:

```python
class TestConnectionPooling:
    def test_github_uses_shared_client(self):
        import scholaragent.sources.github as mod
        assert hasattr(mod, "_http_client")
        assert isinstance(mod._http_client, httpx.Client)

    def test_docs_uses_shared_client(self):
        import scholaragent.sources.docs as mod
        assert hasattr(mod, "_http_client")
        assert isinstance(mod._http_client, httpx.Client)
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_tools.py::TestConnectionPooling tests/test_sources.py::TestConnectionPooling -v`
Expected: FAIL with "has no attribute '_http_client'"

**Step 3: Write minimal implementation**

In each module, add a module-level `httpx.Client` and replace bare `httpx.get()` calls with `_http_client.get()`.

**`scholaragent/tools/arxiv.py`** — add after imports:
```python
_http_client = httpx.Client(timeout=30.0, follow_redirects=True)
```
Replace `httpx.get(url, ...)` inside `retry_with_backoff` with `_http_client.get(url, ...)` (remove the `timeout=` kwarg since it's on the client).

**`scholaragent/tools/semantic_scholar.py`** — add after imports:
```python
_http_client = httpx.Client(timeout=30.0)
```
Replace `httpx.get(...)` calls with `_http_client.get(...)`.

**`scholaragent/sources/github.py`** — add after imports:
```python
_http_client = httpx.Client(timeout=30.0)
```
Replace `httpx.get(...)` inside retry_with_backoff with `_http_client.get(...)`.

**`scholaragent/sources/docs.py`** — add after imports:
```python
_http_client = httpx.Client(timeout=30.0, follow_redirects=True)
```
Replace `httpx.get(url, timeout=timeout, follow_redirects=True)` with `_http_client.get(url, timeout=timeout)`.

**Step 4: Run all tests to verify**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add scholaragent/tools/arxiv.py scholaragent/tools/semantic_scholar.py scholaragent/sources/github.py scholaragent/sources/docs.py tests/test_tools.py tests/test_sources.py
git commit -m "perf: HTTP connection pooling in all source adapters"
```

---

### Task 4: Embedding cache with content hash

Every `embed()` call hits the OpenAI API, even for duplicate content. Add an LRU cache keyed on content hash to `OpenAIEmbeddings`.

**Files:**
- Modify: `scholaragent/memory/embeddings.py`
- Test: `tests/test_embeddings.py`

**Step 1: Write the failing test**

Add to `tests/test_embeddings.py`:

```python
from unittest.mock import MagicMock, patch

class TestEmbeddingCache:
    def test_embed_cache_avoids_duplicate_api_calls(self):
        """Calling embed() with same text should hit API only once."""
        with patch("scholaragent.memory.embeddings.openai") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            from scholaragent.memory.embeddings import OpenAIEmbeddings
            emb = OpenAIEmbeddings()

            result1 = emb.embed("hello world")
            result2 = emb.embed("hello world")

            assert result1 == result2
            assert mock_client.embeddings.create.call_count == 1

    def test_embed_cache_different_texts_make_separate_calls(self):
        """Different texts should make separate API calls."""
        with patch("scholaragent.memory.embeddings.openai") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            from scholaragent.memory.embeddings import OpenAIEmbeddings
            emb = OpenAIEmbeddings()

            emb.embed("hello")
            emb.embed("world")

            assert mock_client.embeddings.create.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_embeddings.py::TestEmbeddingCache -v`
Expected: FAIL — `call_count == 2` (no cache, both calls hit API)

**Step 3: Write minimal implementation**

In `scholaragent/memory/embeddings.py`, add caching to `OpenAIEmbeddings`:

```python
import hashlib
from functools import lru_cache

class OpenAIEmbeddings(EmbeddingBackend):
    def __init__(self, model: str = "text-embedding-3-small", cache_size: int = 512):
        self.model = model
        self._client = openai.OpenAI()
        self._cache_size = cache_size

        # Build a cached inner function bound to this instance
        @lru_cache(maxsize=cache_size)
        def _cached_embed(content_hash: str, text: str) -> tuple[float, ...]:
            response = self._client.embeddings.create(input=[text], model=self.model)
            return tuple(response.data[0].embedding)

        self._cached_embed = _cached_embed

    def embed(self, text: str) -> list[float]:
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return list(self._cached_embed(content_hash, text))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Batch calls bypass cache (they're already efficient)
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
```

**Step 4: Run all tests to verify**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add scholaragent/memory/embeddings.py tests/test_embeddings.py
git commit -m "perf: LRU cache for embeddings avoids duplicate API calls"
```

---

### Task 5: LLM client error handling and retry

The OpenAI and Anthropic clients have zero error handling — any transient API error (429 RateLimitError, 500 InternalServerError, network timeout) crashes the entire pipeline. Add `retry_with_backoff` around API calls and proper exception wrapping.

**Files:**
- Modify: `scholaragent/clients/openai_client.py`
- Modify: `scholaragent/clients/anthropic_client.py`
- Test: `tests/test_clients.py`

**Step 1: Write the failing test**

Add to `tests/test_clients.py`:

```python
from unittest.mock import patch, MagicMock

class TestClientErrorHandling:
    def test_openai_retries_on_rate_limit(self, monkeypatch):
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

        with patch("scholaragent.clients.openai_client.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = fake_create
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.RateLimitError = openai_mod.RateLimitError
            mock_openai.APIConnectionError = openai_mod.APIConnectionError
            mock_openai.InternalServerError = openai_mod.InternalServerError

            from scholaragent.clients.openai_client import OpenAIClient
            client = OpenAIClient(model_name="gpt-4o-mini")
            result = client.completion("test")

            assert result == "ok"
            assert call_count == 3  # 2 failures + 1 success

    def test_openai_raises_after_max_retries(self, monkeypatch):
        """After max retries, the error should propagate."""
        import openai as openai_mod

        def always_fail(**kwargs):
            raise openai_mod.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )

        with patch("scholaragent.clients.openai_client.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = always_fail
            mock_openai.OpenAI.return_value = mock_client
            mock_openai.RateLimitError = openai_mod.RateLimitError
            mock_openai.APIConnectionError = openai_mod.APIConnectionError
            mock_openai.InternalServerError = openai_mod.InternalServerError

            from scholaragent.clients.openai_client import OpenAIClient
            client = OpenAIClient(model_name="gpt-4o-mini")

            with pytest.raises(openai_mod.RateLimitError):
                client.completion("test")
```

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_clients.py::TestClientErrorHandling -v`
Expected: FAIL — no retry logic exists, first error propagates immediately

**Step 3: Write minimal implementation**

In `scholaragent/clients/openai_client.py`, wrap API calls with `retry_with_backoff`:

```python
import openai
from scholaragent.utils.retry import retry_with_backoff

# In completion():
    def completion(self, prompt: str) -> str:
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict = {"model": self.model_name, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = retry_with_backoff(
            self._sync_client.chat.completions.create,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
            **kwargs,
        )
        self._record_usage(response.usage)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(self._last_usage.total_tokens)
        return response.choices[0].message.content or ""
```

Apply the same pattern to `completion_messages()` and `acompletion()`.

Do the same in `scholaragent/clients/anthropic_client.py` with:
```python
import anthropic
retryable_exceptions=(
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)
```

**Step 4: Run all tests to verify**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add scholaragent/clients/openai_client.py scholaragent/clients/anthropic_client.py tests/test_clients.py
git commit -m "fix: retry with backoff on transient LLM API errors (429, 500, connection)"
```

---

### Task 6: Structured logging

The entire codebase has zero logging except `retry.py`. Add `logging.getLogger(__name__)` to key modules so errors are diagnosable. This is NOT about adding print statements — it's about replacing silent `except Exception: return []` patterns with logged warnings.

**Files:**
- Modify: `scholaragent/sources/github.py`
- Modify: `scholaragent/sources/docs.py`
- Modify: `scholaragent/memory/research.py`
- Modify: `scholaragent/mcp_server.py`
- Test: `tests/test_logging.py` (new)

**Step 1: Write the failing test**

Create `tests/test_logging.py`:

```python
"""Verify that modules emit structured log messages on errors."""

import logging

class TestStructuredLogging:
    def test_github_logs_on_http_error(self, monkeypatch):
        """GitHub adapter should log a warning on HTTP errors."""
        import scholaragent.sources.github as mod
        import httpx
        monkeypatch.setattr(mod, "retry_with_backoff", lambda *a, **kw: (_ for _ in ()).throw(httpx.HTTPError("fail")))

        with logging.captureWarnings(True):
            logger = logging.getLogger("scholaragent.sources.github")
            with self._capture_logs(logger) as logs:
                result = mod.search_github_code("test")
            assert result == []
            assert any("fail" in msg for msg in logs)

    def test_docs_logs_on_fetch_error(self, monkeypatch):
        """Docs adapter should log a warning on fetch errors."""
        import scholaragent.sources.docs as mod
        import httpx
        monkeypatch.setattr(mod._http_client, "get", lambda *a, **kw: (_ for _ in ()).throw(httpx.HTTPError("fail")))

        logger = logging.getLogger("scholaragent.sources.docs")
        with self._capture_logs(logger) as logs:
            result = mod.fetch_docs("http://example.com")
        assert result == []
        assert any("fail" in msg for msg in logs)

    def test_research_pipeline_logs_source_errors(self, store, monkeypatch):
        """ResearchPipeline should log warnings for failed sources."""
        import scholaragent.memory.research as mod
        monkeypatch.setattr(mod, "search_arxiv", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("arxiv down")))
        monkeypatch.setattr(mod, "search_semantic_scholar", lambda *a, **kw: "[]")
        monkeypatch.setattr(mod, "search_github_code", lambda *a, **kw: [])
        monkeypatch.setattr(mod, "search_docs", lambda *a, **kw: [])

        from scholaragent.memory.research import ResearchPipeline
        pipeline = ResearchPipeline(store=store)
        logger = logging.getLogger("scholaragent.memory.research")
        with self._capture_logs(logger) as logs:
            result = pipeline.run("test", depth="quick", focus="implementation", force=True)
        assert any("arxiv down" in msg for msg in logs)

    @staticmethod
    from contextlib import contextmanager

    @staticmethod
    @contextmanager
    def _capture_logs(logger):
        """Context manager to capture log messages from a logger."""
        logs = []

        class Handler(logging.Handler):
            def emit(self, record):
                logs.append(self.format(record))

        handler = Handler()
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            yield logs
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
```

Note: The test above has a deliberate syntax issue with `@staticmethod` and `from contextlib` — the subagent must fix this by making `_capture_logs` a proper `@staticmethod` with `contextmanager` imported at the module level.

**Step 2: Run test to verify it fails**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_logging.py -v`
Expected: FAIL — no logging configured in those modules

**Step 3: Write minimal implementation**

In each module, add at the top after imports:

```python
import logging
logger = logging.getLogger(__name__)
```

Then replace silent `except Exception: return []` with:

**`github.py`** (line 50-51):
```python
    except Exception as e:
        logger.warning("GitHub code search failed: %s", e)
        return []
```

**`docs.py`** (line 33-34):
```python
    except Exception as e:
        logger.warning("Failed to fetch docs from %s: %s", url, e)
        return []
```

**`research.py`** — log in the concurrent error handler:
```python
                except Exception as e:
                    logger.warning("Source %s failed: %s", label, e)
                    errors.append(f"{label}: {type(e).__name__}: {e}")
```

**`mcp_server.py`** — add logger and log init:
```python
logger = logging.getLogger(__name__)

# In _get_store():
        logger.info("Initialized memory store at %s", DB_PATH)
```

**Step 4: Run all tests to verify**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/ -x -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add scholaragent/sources/github.py scholaragent/sources/docs.py scholaragent/memory/research.py scholaragent/mcp_server.py tests/test_logging.py
git commit -m "feat: structured logging in source adapters and pipeline"
```

---

### Task 7: Test coverage for `_html_to_text` and docs adapter

The `_html_to_text()` function in `docs.py` has zero tests despite handling script/style stripping, entity decoding, and whitespace normalization.

**Files:**
- Test: `tests/test_sources.py`

**Step 1: Write the tests**

Add to `tests/test_sources.py`:

```python
from scholaragent.sources.docs import _html_to_text, MAX_DOC_CONTENT_LENGTH

class TestHtmlToText:
    def test_strips_script_tags(self):
        html = "<html><script>alert('xss')</script><p>hello</p></html>"
        assert "alert" not in _html_to_text(html)
        assert "hello" in _html_to_text(html)

    def test_strips_style_tags(self):
        html = "<style>.x { color: red; }</style><p>content</p>"
        assert "color" not in _html_to_text(html)
        assert "content" in _html_to_text(html)

    def test_decodes_html_entities(self):
        html = "<p>a &amp; b &lt; c &gt; d &quot;e&quot; f&nbsp;g</p>"
        text = _html_to_text(html)
        assert "a & b" in text
        assert "< c >" in text
        assert '"e"' in text

    def test_collapses_whitespace(self):
        html = "<p>  hello   \n\n   world  </p>"
        text = _html_to_text(html)
        assert text == "hello world"

    def test_empty_html(self):
        assert _html_to_text("") == ""

    def test_no_tags(self):
        assert _html_to_text("plain text") == "plain text"

    def test_nested_tags(self):
        html = "<div><p><span>deep</span></p></div>"
        assert _html_to_text(html) == "deep"

    def test_case_insensitive_script_removal(self):
        html = "<SCRIPT>bad()</SCRIPT><p>good</p>"
        text = _html_to_text(html)
        assert "bad" not in text
        assert "good" in text


class TestFetchDocsEdgeCases:
    def test_truncates_long_content(self, monkeypatch):
        """Content should be truncated to MAX_DOC_CONTENT_LENGTH."""
        import scholaragent.sources.docs as mod
        long_html = "<p>" + "a" * 20_000 + "</p>"
        mock_response = MagicMock()
        mock_response.text = long_html
        mock_response.raise_for_status = lambda: None
        monkeypatch.setattr(mod._http_client, "get", lambda *a, **kw: mock_response)

        result = mod.fetch_docs("http://example.com")
        assert len(result) == 1
        assert len(result[0]["content"]) == MAX_DOC_CONTENT_LENGTH

    def test_empty_html_returns_empty_list(self, monkeypatch):
        """HTML with no text content should return empty list."""
        import scholaragent.sources.docs as mod
        mock_response = MagicMock()
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = lambda: None
        monkeypatch.setattr(mod._http_client, "get", lambda *a, **kw: mock_response)

        result = mod.fetch_docs("http://example.com")
        assert result == []
```

**Step 2: Run tests**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && uv run pytest tests/test_sources.py -v`
Expected: ALL PASS (these are new tests for existing working code)

**Step 3: Commit**

```bash
git add tests/test_sources.py
git commit -m "test: comprehensive tests for _html_to_text and docs adapter edge cases"
```

---

## Summary

| Task | Category | Impact |
|------|----------|--------|
| 1. Thread-safe MCP init | Robustness | Fixes race condition in concurrent tool calls |
| 2. Concurrent source collection | Performance | ~3-4x faster research pipeline |
| 3. HTTP connection pooling | Performance | ~20-50% latency reduction per request |
| 4. Embedding cache | Cost/Performance | Eliminates duplicate OpenAI API calls |
| 5. LLM client error handling | Robustness | Prevents pipeline crashes on transient errors |
| 6. Structured logging | Observability | Replaces silent failures with diagnosable warnings |
| 7. HTML-to-text tests | Coverage | Tests critical untested parsing function |
