# refactor/research-runtime-split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split [scholaragent/memory/research.py](../../../scholaragent/memory/research.py) (467 lines, god-class `ResearchPipeline`) into three focused collaborators — `SourceCollector`, `ResultIndexer`, `ResearchCoordinator` — returning a typed `ResearchResult` that makes fallback explicit, and decouple [scholaragent/mcp_server.py](../../../scholaragent/mcp_server.py)'s module-level globals into a `RuntimeContainer` for testability.

**Architecture:** Internal-only refactor. `ResearchPipeline` is kept as a backwards-compat alias so `from scholaragent.memory.research import ResearchPipeline` keeps working. Dependencies are injected through constructors (not patched), which lets tests pass mocks for `SourceCollector` and `ResultIndexer` instead of monkey-patching module imports. `mcp_server.py` becomes a thin FastMCP adapter over a `RuntimeContainer` that owns lifecycle.

**Tech Stack:** Python 3.12, pytest, FastMCP, SQLite, httpx.

**Prerequisites (already landed):**
- `fix/installer-surface-drift` — `scholaragent._manifest.MCP_TOOLS` as single source of truth.
- `fix/research-taxonomy` — `synthesized_report` is a valid source_type and deep-research uses it.

**Out of scope for this branch:**
- Removing the `language="python"` hardcode in GitHub search (user-plan P1-6). `SourceCollector` will accept `code_language` as a parameter with default `"python"` so behavior is preserved; changing the default is a follow-up PR.
- Store.py rewrite / vector backend refactor (user-plan P0-3).
- Docs search provider expansion (user-plan P1-5).

---

## File Structure

**Created:**
- `scholaragent/memory/source_collector.py` — `SourceCollector` class owning raw source retrieval + dedup.
- `scholaragent/memory/indexer.py` — `ResultIndexer` class owning `MemoryEntry` construction + store writes.
- `scholaragent/memory/research_result.py` — `ResearchResult` dataclass with `requested_depth` and `actual_depth`.
- `scholaragent/runtime.py` — `RuntimeContainer` class owning store/pipeline/agent-infra lifecycle for the MCP server.
- `tests/test_source_collector.py` — SourceCollector unit tests.
- `tests/test_result_indexer.py` — ResultIndexer unit tests.
- `tests/test_research_result.py` — ResearchResult dataclass tests.
- `tests/test_runtime_container.py` — RuntimeContainer unit tests.

**Modified:**
- `scholaragent/memory/research.py` — `ResearchPipeline` becomes a thin coordinator that composes `SourceCollector` + `ResultIndexer`, returns `ResearchResult`.
- `scholaragent/mcp_server.py` — globals replaced with a single `RuntimeContainer`; tool handlers read from it.
- `tests/test_research_pipeline.py` — patches of `scholaragent.memory.research.search_*` replaced with injected `SourceCollector` mocks.
- `tests/test_depth_levels.py` — same migration.
- `tests/test_deduplication.py` — same migration (tests now target `SourceCollector.deduplicate` directly).
- `tests/test_mcp_server.py` — fixture switches to `RuntimeContainer` injection instead of module-global monkey-patching.

---

## Task 0: Baseline and audit

**Files:**
- Read-only pass.

- [ ] **Step 1: Snapshot the current test count on main**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ --ignore=tests/test_installer.py --ignore=tests/test_web_tools_live.py -q 2>&1 | tail -3
```
Expected: `586 passed` (or whatever the current baseline is — record this number; every subsequent task must match or exceed it).

- [ ] **Step 2: Enumerate every patch site that targets research.py module globals**

Run:
```bash
grep -rn 'patch("scholaragent.memory.research\.' tests/
```
Expected output: list of `search_arxiv`, `search_semantic_scholar`, `search_github_code`, `search_docs` patch sites in `tests/test_research_pipeline.py`, `tests/test_depth_levels.py`, `tests/test_deduplication.py`. Save the full list — every one of these becomes either an injected-collector migration (Phase A, Task 4) or a patch-target rewrite (`scholaragent.memory.source_collector.search_*`).

- [ ] **Step 3: Confirm the dispatcher API will cooperate**

Run:
```bash
grep -n "pipeline.run\|set_agent_infra\|has_agent_infra" scholaragent/ tests/ -r
```
Expected: confirms that `pipeline.run(...)` is the only public entry, `set_agent_infra` is only called from `mcp_server.py` and `tests/`. This is the surface we must keep backwards-compatible.

---

## Phase A — SourceCollector

### Task 1: RED — SourceCollector contract tests

**Files:**
- Create: `tests/test_source_collector.py`

- [ ] **Step 1: Write failing tests for SourceCollector**

```python
# tests/test_source_collector.py
"""SourceCollector owns raw retrieval from arXiv, S2, GitHub, and docs."""
from __future__ import annotations

from unittest.mock import patch

import pytest


class TestSourceCollectorContract:
    def test_importable(self):
        from scholaragent.memory.source_collector import SourceCollector
        assert SourceCollector is not None

    def test_default_source_types(self):
        from scholaragent.memory.source_collector import SourceCollector
        assert SourceCollector.DEFAULT_SOURCES == ("paper", "code", "docs")

    def test_collect_returns_results_and_errors(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv",
                   return_value='[{"arxiv_id": "1", "title": "T", "authors": [], "abstract": "A"}]'), \
             patch("scholaragent.memory.source_collector.search_semantic_scholar",
                   return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[]), \
             patch("scholaragent.memory.source_collector.search_docs",
                   return_value=[]):
            results, errors = SourceCollector().collect("rlhf")

        assert errors == []
        assert len(results) == 1
        assert results[0]["source_type"] == "paper"
        assert results[0]["source_ref"] == "arxiv:1"

    def test_collect_captures_errors_without_raising(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv",
                   side_effect=RuntimeError("network down")), \
             patch("scholaragent.memory.source_collector.search_semantic_scholar",
                   return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[]), \
             patch("scholaragent.memory.source_collector.search_docs",
                   return_value=[]):
            results, errors = SourceCollector().collect("rlhf")

        assert results == []
        assert len(errors) == 1
        assert "arXiv" in errors[0]
        assert "network down" in errors[0]

    def test_collect_honors_source_type_filter(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv") as arx, \
             patch("scholaragent.memory.source_collector.search_semantic_scholar") as s2, \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[{"content": "x", "source_type": "code", "source_ref": "github:a/b"}]), \
             patch("scholaragent.memory.source_collector.search_docs",
                   return_value=[]):
            arx.return_value = '[]'
            s2.return_value = '[]'
            results, _ = SourceCollector().collect("rlhf", source_types=("code",))

        arx.assert_not_called()
        s2.assert_not_called()
        assert len(results) == 1
        assert results[0]["source_type"] == "code"

    def test_collect_passes_code_language(self):
        from scholaragent.memory.source_collector import SourceCollector

        with patch("scholaragent.memory.source_collector.search_arxiv", return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_semantic_scholar", return_value='[]'), \
             patch("scholaragent.memory.source_collector.search_github_code",
                   return_value=[]) as gh, \
             patch("scholaragent.memory.source_collector.search_docs", return_value=[]):
            SourceCollector().collect("rlhf", code_language="rust")

        gh.assert_called_once()
        assert gh.call_args.kwargs["language"] == "rust"


class TestSourceCollectorDeduplicate:
    def test_dedup_keeps_s2_over_arxiv_on_arxiv_id_collision(self):
        from scholaragent.memory.source_collector import SourceCollector

        items = [
            {"content": "Title: Attention\n\nAbstract: old", "source_type": "paper",
             "source_ref": "arxiv:1706.03762"},
            {"content": "Title: Attention\n\nAbstract: newer + citations",
             "source_type": "paper", "source_ref": "s2:1706.03762"},
        ]
        deduped = SourceCollector().deduplicate(items)
        assert len(deduped) == 1
        assert deduped[0]["source_ref"] == "s2:1706.03762"

    def test_dedup_leaves_non_papers_alone(self):
        from scholaragent.memory.source_collector import SourceCollector
        items = [
            {"content": "code", "source_type": "code", "source_ref": "gh:a/b"},
            {"content": "code", "source_type": "code", "source_ref": "gh:a/b"},
        ]
        assert len(SourceCollector().deduplicate(items)) == 2
```

- [ ] **Step 2: Run and verify RED**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_source_collector.py -x --tb=short
```
Expected: `ModuleNotFoundError: No module named 'scholaragent.memory.source_collector'`.

### Task 2: GREEN — Implement SourceCollector

**Files:**
- Create: `scholaragent/memory/source_collector.py`

- [ ] **Step 1: Implement the module**

```python
# scholaragent/memory/source_collector.py
"""SourceCollector — raw retrieval from paper, code, and docs adapters.

Extracted from scholaragent.memory.research._collect_sources. Sequential
retrieval is preserved because the underlying source adapters each use a
module-level httpx.Client that is not thread-safe. Concurrency is a
separate concern (fix the adapters first).
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

logger = logging.getLogger(__name__)

from scholaragent.tools.arxiv import search_arxiv
from scholaragent.tools.semantic_scholar import search_semantic_scholar
from scholaragent.sources.github import search_github_code
from scholaragent.sources.docs import search_docs


class SourceCollector:
    """Retrieves raw results from paper, code, and docs sources."""

    DEFAULT_SOURCES: tuple[str, ...] = ("paper", "code", "docs")

    def __init__(self, default_code_language: str = "python"):
        self._default_code_language = default_code_language

    def collect(
        self,
        query: str,
        source_types: Sequence[str] | None = None,
        code_language: str | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Collect raw results. Returns (results, errors)."""
        source_types = tuple(source_types) if source_types else self.DEFAULT_SOURCES
        language = code_language or self._default_code_language
        results: list[dict] = []
        errors: list[str] = []

        if "paper" in source_types:
            self._collect_arxiv(query, results, errors)
            self._collect_s2(query, results, errors)

        if "code" in source_types:
            try:
                results.extend(search_github_code(query, language=language, max_results=5))
            except Exception as e:
                logger.warning("GitHub search failed: %s", e)
                errors.append(f"GitHub: {type(e).__name__}: {e}")

        if "docs" in source_types:
            try:
                results.extend(search_docs(query, max_results=3))
            except Exception as e:
                logger.warning("Docs search failed: %s", e)
                errors.append(f"Docs: {type(e).__name__}: {e}")

        return results, errors

    def _collect_arxiv(self, query: str, results: list[dict], errors: list[str]) -> None:
        try:
            papers = json.loads(search_arxiv(query, max_results=10))
            if isinstance(papers, list):
                for p in papers:
                    results.append({
                        "content": (
                            f"Title: {p.get('title', '')}\n\n"
                            f"Abstract: {p.get('abstract', '')}\n\n"
                            f"Authors: {', '.join(p.get('authors', []))}"
                        ),
                        "source_type": "paper",
                        "source_ref": f"arxiv:{p.get('arxiv_id', '')}",
                    })
        except Exception as e:
            logger.warning("arXiv search failed: %s", e)
            errors.append(f"arXiv: {type(e).__name__}: {e}")

    def _collect_s2(self, query: str, results: list[dict], errors: list[str]) -> None:
        try:
            papers = json.loads(search_semantic_scholar(query, limit=10))
            if isinstance(papers, list):
                for p in papers:
                    results.append({
                        "content": (
                            f"Title: {p.get('title', '')}\n\n"
                            f"Abstract: {p.get('abstract', '')}\n\n"
                            f"Year: {p.get('year', 'N/A')}\n"
                            f"Citations: {p.get('citation_count', 0)}"
                        ),
                        "source_type": "paper",
                        "source_ref": f"s2:{p.get('paper_id', '')}",
                    })
        except Exception as e:
            logger.warning("S2 search failed: %s", e)
            errors.append(f"Semantic Scholar: {type(e).__name__}: {e}")

    def deduplicate(self, results: list[dict]) -> list[dict]:
        """Dedup papers by arxiv_id, then by normalized title. Prefer S2 on collision.

        Non-paper entries pass through unchanged.
        """
        seen_arxiv: dict[str, int] = {}
        seen_titles: dict[str, int] = {}
        deduped: list[dict] = []

        for item in results:
            if item.get("source_type") != "paper":
                deduped.append(item)
                continue

            ref = item.get("source_ref", "")
            content = item.get("content", "")

            arxiv_id = ref[6:] if ref.startswith("arxiv:") else ""

            title_norm = ""
            for line in content.split("\n"):
                if line.startswith("Title: "):
                    title_norm = re.sub(r"[^\w\s]", "", line[7:]).lower().strip()
                    break

            if arxiv_id and arxiv_id in seen_arxiv:
                idx = seen_arxiv[arxiv_id]
                if ref.startswith("s2:"):
                    deduped[idx] = item
                continue

            if title_norm and title_norm in seen_titles:
                idx = seen_titles[title_norm]
                if ref.startswith("s2:"):
                    deduped[idx] = item
                continue

            idx = len(deduped)
            deduped.append(item)
            if arxiv_id:
                seen_arxiv[arxiv_id] = idx
            if title_norm:
                seen_titles[title_norm] = idx

        return deduped
```

- [ ] **Step 2: Run and verify GREEN**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_source_collector.py -v
```
Expected: `8 passed`.

- [ ] **Step 3: Commit**

```bash
git add scholaragent/memory/source_collector.py tests/test_source_collector.py
git commit -m "feat(memory): add SourceCollector for raw retrieval"
```

### Task 3: Rewire research.py through SourceCollector

**Files:**
- Modify: `scholaragent/memory/research.py:343-460` (remove `_collect_sources` + `_deduplicate`)
- Modify: `scholaragent/memory/research.py:41-114` (constructor takes optional `collector`)
- Modify: `tests/test_research_pipeline.py` — patches become injection
- Modify: `tests/test_depth_levels.py` — same
- Modify: `tests/test_deduplication.py` — same

- [ ] **Step 1: Write a new test proving the pipeline uses an injected collector**

Add to `tests/test_research_pipeline.py`:

```python
class TestPipelineDependencyInjection:
    def test_pipeline_uses_injected_collector(self, store):
        from scholaragent.memory.research import ResearchPipeline
        from unittest.mock import MagicMock

        fake_collector = MagicMock()
        fake_collector.collect.return_value = (
            [{"content": "X", "source_type": "paper", "source_ref": "arxiv:9"}],
            [],
        )
        fake_collector.deduplicate.side_effect = lambda xs: xs

        pipeline = ResearchPipeline(store=store, collector=fake_collector)
        result = pipeline.run("test", depth="quick")

        assert result["entries_added"] == 1
        fake_collector.collect.assert_called_once()
        fake_collector.deduplicate.assert_called_once()
```

- [ ] **Step 2: Run test, verify RED**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_research_pipeline.py::TestPipelineDependencyInjection -x --tb=short
```
Expected: `TypeError: ResearchPipeline.__init__() got an unexpected keyword argument 'collector'` (or similar).

- [ ] **Step 3: Update ResearchPipeline.__init__ and delete inline collection**

In `scholaragent/memory/research.py`, replace the top imports section:

```python
# Before (lines 19-24):
from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry, ResearchLogEntry
from scholaragent.tools.arxiv import search_arxiv
from scholaragent.tools.semantic_scholar import search_semantic_scholar
from scholaragent.sources.github import search_github_code
from scholaragent.sources.docs import search_docs
```

with:

```python
from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry, ResearchLogEntry
from scholaragent.memory.source_collector import SourceCollector
```

Replace the constructor (lines 49-59):

```python
def __init__(
    self,
    store: MemoryStore,
    handler: LMHandler | None = None,
    registry: AgentRegistry | None = None,
    dispatcher: Dispatcher | None = None,
    collector: SourceCollector | None = None,
):
    self.store = store
    self.handler = handler
    self.registry = registry
    self.dispatcher = dispatcher
    self._collector = collector or SourceCollector()
```

Replace every call site of `self._collect_sources(...)` and `self._deduplicate(...)`:

```python
# Before
raw_results, errors = self._collect_sources(query, sources=source_types)
raw_results = self._deduplicate(raw_results)

# After
raw_results, errors = self._collector.collect(query, source_types=source_types)
raw_results = self._collector.deduplicate(raw_results)
```

Delete the `_collect_sources` and `_deduplicate` methods (lines 343-460).

- [ ] **Step 4: Run the DI test to verify GREEN**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_research_pipeline.py::TestPipelineDependencyInjection -v
```
Expected: PASS.

- [ ] **Step 5: Migrate legacy patch-based tests**

The legacy tests monkey-patch `scholaragent.memory.research.search_arxiv` etc. Those patches no longer intercept anything because `search_arxiv` has moved to `source_collector.py`.

Two equally valid migrations — pick per test:

**Option A — inject a mock collector (preferred for pipeline-level tests):**

```python
# Before
with patch("scholaragent.memory.research.search_arxiv") as mock_arxiv, \
     patch("scholaragent.memory.research.search_semantic_scholar") as mock_s2:
    mock_arxiv.return_value = '[{"arxiv_id": "1", "title": "T", ...}]'
    mock_s2.return_value = '[]'
    pipeline = ResearchPipeline(store=store)
    result = pipeline.run("test", depth="quick")

# After
from unittest.mock import MagicMock
fake_collector = MagicMock()
fake_collector.collect.return_value = (
    [{"content": "T...", "source_type": "paper", "source_ref": "arxiv:1"}], []
)
fake_collector.deduplicate.side_effect = lambda xs: xs
pipeline = ResearchPipeline(store=store, collector=fake_collector)
result = pipeline.run("test", depth="quick")
```

**Option B — update the patch target (only for tests that really need to exercise the adapter code path):**

```python
# Before
patch("scholaragent.memory.research.search_arxiv")
# After
patch("scholaragent.memory.source_collector.search_arxiv")
```

Apply across `tests/test_research_pipeline.py`, `tests/test_depth_levels.py` (the `_patch_sources` helper), and `tests/test_deduplication.py`. In `tests/test_deduplication.py`, swap to testing `SourceCollector().deduplicate(...)` directly.

- [ ] **Step 6: Run the full suite and verify baseline**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ --ignore=tests/test_installer.py --ignore=tests/test_web_tools_live.py -q 2>&1 | tail -3
```
Expected: same or higher pass count as Task 0 baseline.

- [ ] **Step 7: Commit**

```bash
git add scholaragent/memory/research.py tests/test_research_pipeline.py tests/test_depth_levels.py tests/test_deduplication.py
git commit -m "refactor(memory): rewire ResearchPipeline through injected SourceCollector"
```

---

## Phase B — ResultIndexer

### Task 4: RED — ResultIndexer contract tests

**Files:**
- Create: `tests/test_result_indexer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_result_indexer.py
"""ResultIndexer — writes MemoryEntries from raw, enriched, and synthesis results."""
from __future__ import annotations

import os
import tempfile

import pytest


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]


@pytest.fixture
def store():
    from scholaragent.memory.store import MemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        s = MemoryStore(db_path=os.path.join(tmp, "t.db"), embeddings=FakeEmbeddings())
        yield s
        s.close()


class TestResultIndexer:
    def test_index_raw_counts_additions(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        raws = [
            {"content": "P1 body", "source_type": "paper", "source_ref": "arxiv:1"},
            {"content": "P2 body", "source_type": "paper", "source_ref": "arxiv:2"},
        ]
        count = indexer.index_raw(query="rlhf", raw_results=raws)
        assert count == 2
        assert store.count() == 2

    def test_index_raw_applies_query_tag(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        indexer.index_raw(
            query="Reward Model",
            raw_results=[{"content": "X", "source_type": "paper", "source_ref": "arxiv:9"}],
        )
        hits = store.search("X", max_results=1)
        assert hits
        entry, _ = hits[0]
        assert "reward-model" in entry.tags

    def test_index_enriched_appends_reader_and_critic(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        enriched = [{
            "content": "Paper body",
            "source_type": "paper",
            "source_ref": "arxiv:1",
            "reader_findings": "key claim",
            "critic_assessment": "methodologically sound",
        }]
        indexer.index_enriched(query="rlhf", enriched_results=enriched)

        entry, _ = store.search("rlhf", max_results=1)[0]
        assert "--- Reader Analysis ---" in entry.content
        assert "--- Critic Assessment ---" in entry.content
        assert "agent-processed" in entry.tags

    def test_index_synthesis_uses_synthesized_report_type(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        count = indexer.index_synthesis(
            query="long query about rlhf and dpo",
            synthesis_text="# Report\n\nFull synthesized report body.",
        )
        assert count == 1

        entry, _ = store.search("synthesized", max_results=1)[0]
        assert entry.source_type == "synthesized_report"
        assert entry.source_ref.startswith("deep-research:")
        assert "deep-pipeline" in entry.tags

    def test_log_research_delegates_to_store(self, store):
        from scholaragent.memory.indexer import ResultIndexer
        indexer = ResultIndexer(store)
        indexer.log_research(query="rlhf", depth="quick", focus="implementation", result_count=3)
        logged = store.get_recent_research("rlhf")
        assert len(logged) == 1
        assert logged[0].depth == "quick"
```

- [ ] **Step 2: Run and verify RED**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_result_indexer.py -x --tb=short
```
Expected: `ModuleNotFoundError: No module named 'scholaragent.memory.indexer'`.

### Task 5: GREEN — Implement ResultIndexer

**Files:**
- Create: `scholaragent/memory/indexer.py`

- [ ] **Step 1: Implement the module**

```python
# scholaragent/memory/indexer.py
"""ResultIndexer — persists research results into the MemoryStore.

Extracted from ResearchPipeline._run_quick / _run_normal / _run_deep.
Keeps all MemoryEntry construction and tagging in one place so that
research.py can stay focused on depth orchestration.
"""
from __future__ import annotations

from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry


class ResultIndexer:
    """Writes MemoryEntries into the store with consistent tagging."""

    def __init__(self, store: MemoryStore):
        self._store = store

    def index_raw(self, query: str, raw_results: list[dict]) -> int:
        """Index raw source results. Returns number of entries added."""
        added = 0
        tag = query.lower().replace(" ", "-")
        for raw in raw_results:
            entry = MemoryEntry(
                content=raw["content"],
                summary=MemoryEntry.smart_summary(raw["content"]),
                source_type=raw["source_type"],
                source_ref=raw["source_ref"],
                tags=[tag],
            )
            self._store.add(entry)
            added += 1
        return added

    def index_enriched(self, query: str, enriched_results: list[dict]) -> int:
        """Index agent-enriched results with reader/critic output concatenated."""
        added = 0
        tag = query.lower().replace(" ", "-")
        for item in enriched_results:
            content = item["content"]
            if item.get("reader_findings"):
                content += f"\n\n--- Reader Analysis ---\n{item['reader_findings']}"
            if item.get("critic_assessment"):
                content += f"\n\n--- Critic Assessment ---\n{item['critic_assessment']}"

            tags = [tag]
            if item.get("reader_findings") or item.get("critic_assessment"):
                tags.append("agent-processed")

            entry = MemoryEntry(
                content=content,
                summary=MemoryEntry.smart_summary(content),
                source_type=item["source_type"],
                source_ref=item["source_ref"],
                tags=tags,
            )
            self._store.add(entry)
            added += 1
        return added

    def index_synthesis(self, query: str, synthesis_text: str) -> int:
        """Index a deep-pipeline synthesized report. Returns 1 or 0."""
        if not synthesis_text:
            return 0
        entry = MemoryEntry(
            content=synthesis_text,
            summary=MemoryEntry.smart_summary(synthesis_text),
            source_type="synthesized_report",
            source_ref=f"deep-research:{query[:50]}",
            tags=[query.lower().replace(" ", "-"), "deep-pipeline", "synthesized"],
        )
        self._store.add(entry)
        return 1

    def log_research(
        self, query: str, depth: str, focus: str, result_count: int
    ) -> None:
        self._store.log_research(
            query=query, depth=depth, focus=focus, result_count=result_count
        )
```

- [ ] **Step 2: Run and verify GREEN**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_result_indexer.py -v
```
Expected: `5 passed`.

- [ ] **Step 3: Commit**

```bash
git add scholaragent/memory/indexer.py tests/test_result_indexer.py
git commit -m "feat(memory): add ResultIndexer for store writes"
```

### Task 6: Rewire research.py through ResultIndexer

**Files:**
- Modify: `scholaragent/memory/research.py` — constructor takes optional `indexer`, replace inline `MemoryEntry(...)` construction with `self._indexer.index_*(...)` calls.

- [ ] **Step 1: Write a pipeline test asserting indexer use**

Add to `tests/test_research_pipeline.py`:

```python
class TestPipelineUsesIndexer:
    def test_pipeline_delegates_indexing(self, store):
        from scholaragent.memory.research import ResearchPipeline
        from scholaragent.memory.indexer import ResultIndexer
        from unittest.mock import MagicMock

        fake_collector = MagicMock()
        fake_collector.collect.return_value = (
            [{"content": "X", "source_type": "paper", "source_ref": "arxiv:9"}],
            [],
        )
        fake_collector.deduplicate.side_effect = lambda xs: xs

        real_indexer = ResultIndexer(store)
        pipeline = ResearchPipeline(
            store=store, collector=fake_collector, indexer=real_indexer
        )
        result = pipeline.run("test", depth="quick")

        assert result["entries_added"] == 1
        assert store.count() == 1
```

- [ ] **Step 2: Run and verify RED**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_research_pipeline.py::TestPipelineUsesIndexer -x --tb=short
```
Expected: `TypeError: ... unexpected keyword argument 'indexer'`.

- [ ] **Step 3: Update ResearchPipeline**

In `scholaragent/memory/research.py`:

Update imports:
```python
from scholaragent.memory.indexer import ResultIndexer
from scholaragent.memory.source_collector import SourceCollector
```

Update constructor:
```python
def __init__(
    self,
    store: MemoryStore,
    handler: LMHandler | None = None,
    registry: AgentRegistry | None = None,
    dispatcher: Dispatcher | None = None,
    collector: SourceCollector | None = None,
    indexer: ResultIndexer | None = None,
):
    self.store = store
    self.handler = handler
    self.registry = registry
    self.dispatcher = dispatcher
    self._collector = collector or SourceCollector()
    self._indexer = indexer or ResultIndexer(store)
```

Replace the `_run_quick` body:
```python
def _run_quick(self, query: str, depth: str, focus: str) -> dict:
    raw_results, errors = self._collector.collect(query)
    raw_results = self._collector.deduplicate(raw_results)
    entries_added = self._indexer.index_raw(query=query, raw_results=raw_results)
    self._indexer.log_research(
        query=query, depth=depth, focus=focus, result_count=entries_added
    )
    return {
        "status": "completed",
        "depth": depth,
        "query": query,
        "entries_added": entries_added,
        "errors": errors,
        "message": f"Research complete. {entries_added} entries indexed.",
    }
```

Replace the indexing loop in `_run_normal` (lines 186-208) with:
```python
entries_added = self._indexer.index_enriched(query=query, enriched_results=enriched)
self._indexer.log_research(
    query=query, depth="normal", focus=focus, result_count=entries_added
)
```

Replace the synthesis block in `_run_deep` (lines 235-250) with:
```python
entries_added = 0
if result.success and result.result:
    entries_added = self._indexer.index_synthesis(
        query=query, synthesis_text=result.result
    )
else:
    logger.warning("Deep pipeline returned no result, falling back to normal")
    return self._run_normal(query, focus)

self._indexer.log_research(
    query=query, depth="deep", focus=focus, result_count=entries_added
)
```

- [ ] **Step 4: Verify GREEN**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_research_pipeline.py tests/test_depth_levels.py tests/test_result_indexer.py -q
```
Expected: all pass.

- [ ] **Step 5: Full-suite baseline check**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ --ignore=tests/test_installer.py --ignore=tests/test_web_tools_live.py -q 2>&1 | tail -3
```
Expected: >= Task 0 baseline.

- [ ] **Step 6: Commit**

```bash
git add scholaragent/memory/research.py tests/test_research_pipeline.py
git commit -m "refactor(memory): rewire ResearchPipeline through injected ResultIndexer"
```

---

## Phase C — ResearchResult

### Task 7: RED+GREEN — ResearchResult dataclass

**Files:**
- Create: `scholaragent/memory/research_result.py`
- Create: `tests/test_research_result.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_research_result.py
"""ResearchResult — typed result of a research run."""
from __future__ import annotations


class TestResearchResult:
    def test_minimum_fields(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="completed",
            query="rlhf",
            requested_depth="normal",
            actual_depth="normal",
            entries_added=3,
        )
        assert r.errors == []
        assert r.fallback_reason is None
        assert r.message == ""

    def test_to_dict_exposes_depth_alias_for_backwards_compat(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="completed",
            query="rlhf",
            requested_depth="deep",
            actual_depth="normal",
            entries_added=3,
            fallback_reason="Dispatcher timed out",
        )
        d = r.to_dict()
        assert d["depth"] == "normal"  # alias for actual_depth
        assert d["requested_depth"] == "deep"
        assert d["actual_depth"] == "normal"
        assert d["fallback_reason"] == "Dispatcher timed out"
        assert d["entries_added"] == 3

    def test_to_dict_omits_empty_fallback_reason(self):
        from scholaragent.memory.research_result import ResearchResult
        r = ResearchResult(
            status="completed", query="q", requested_depth="quick",
            actual_depth="quick", entries_added=1,
        )
        assert "fallback_reason" not in r.to_dict()
```

- [ ] **Step 2: Run and verify RED**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_research_result.py -x --tb=short
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement ResearchResult**

```python
# scholaragent/memory/research_result.py
"""Typed result of a research pipeline run.

Before this type existed, ResearchPipeline returned plain dicts and
silently collapsed the requested depth into whatever actually ran after
fallback. ResearchResult keeps both depths visible so the client sees
what happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResearchResult:
    status: str  # "completed" | "cached" | "failed"
    query: str
    requested_depth: str  # what the user asked for
    actual_depth: str  # what actually ran (may differ under fallback)
    entries_added: int
    errors: list[str] = field(default_factory=list)
    message: str = ""
    fallback_reason: str | None = None
    cached_results: int | None = None  # set when status == "cached"

    def to_dict(self) -> dict:
        d: dict = {
            "status": self.status,
            "query": self.query,
            "depth": self.actual_depth,  # backwards-compat alias
            "requested_depth": self.requested_depth,
            "actual_depth": self.actual_depth,
            "entries_added": self.entries_added,
            "errors": list(self.errors),
            "message": self.message,
        }
        if self.fallback_reason is not None:
            d["fallback_reason"] = self.fallback_reason
        if self.cached_results is not None:
            d["cached_results"] = self.cached_results
        return d
```

- [ ] **Step 4: Verify GREEN**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_research_result.py -v
```
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add scholaragent/memory/research_result.py tests/test_research_result.py
git commit -m "feat(memory): add ResearchResult dataclass"
```

### Task 8: Return ResearchResult from ResearchPipeline

**Files:**
- Modify: `scholaragent/memory/research.py` — every return path becomes a `ResearchResult` converted via `.to_dict()`
- Modify: `scholaragent/mcp_server.py:214-231` — `_memory_research` already returns a dict; no caller change needed because `.to_dict()` exposes the `depth` alias.
- Modify: `tests/test_depth_levels.py` — add a new assertion for fallback visibility.

- [ ] **Step 1: Write a fallback-visibility test**

Add to `tests/test_depth_levels.py::TestDeepDepth`:

```python
def test_deep_fallback_exposes_both_depths(self, store):
    from scholaragent.core.types import AgentResult
    mock_handler = MagicMock()
    mock_dispatcher = MagicMock()
    mock_dispatcher.run.side_effect = RuntimeError("LLM unavailable")

    mock_scout = MagicMock()
    mock_scout.run.return_value = AgentResult(
        agent_name="scout", task="test", result="found", iterations=1, success=True
    )
    mock_registry = MagicMock()
    mock_registry.get = lambda name: mock_scout

    pipeline = ResearchPipeline(
        store=store,
        handler=mock_handler,
        registry=mock_registry,
        dispatcher=mock_dispatcher,
    )

    with _patch_sources():
        result = pipeline.run("fallback-visibility", depth="deep")

    assert result["status"] == "completed"
    assert result["requested_depth"] == "deep"
    assert result["actual_depth"] in ("normal", "quick")
    assert result["fallback_reason"]
```

- [ ] **Step 2: Run and verify RED**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_depth_levels.py::TestDeepDepth::test_deep_fallback_exposes_both_depths -x --tb=short
```
Expected: `KeyError: 'requested_depth'`.

- [ ] **Step 3: Update every return site in research.py**

In each of `_run_quick`, `_run_normal`, `_run_deep`, the cached branch, and the fallback branches, build a `ResearchResult` and return `result.to_dict()`. Fallback sites pass `fallback_reason`:

```python
# _run_quick return
return ResearchResult(
    status="completed",
    query=query,
    requested_depth=depth,
    actual_depth=depth,
    entries_added=entries_added,
    errors=errors,
    message=f"Research complete. {entries_added} entries indexed.",
).to_dict()

# _run_normal fallback-to-quick site
logger.warning("Scout failed, falling back to quick: %s", scout_result.result)
quick = self._run_quick(query, "quick", focus)
quick["requested_depth"] = "normal"
quick["fallback_reason"] = f"Scout failed: {scout_result.result}"
return quick

# _run_deep fallback-to-normal site
except Exception as e:
    logger.warning("Deep pipeline failed, falling back to normal: %s", e)
    normal = self._run_normal(query, focus)
    normal["requested_depth"] = "deep"
    normal["fallback_reason"] = f"Dispatcher failed: {type(e).__name__}: {e}"
    return normal
```

Add the import at the top:
```python
from scholaragent.memory.research_result import ResearchResult
```

- [ ] **Step 4: Verify GREEN**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_depth_levels.py tests/test_research_pipeline.py -v
```
Expected: all pass, including the new fallback test.

- [ ] **Step 5: Full-suite baseline check**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ --ignore=tests/test_installer.py --ignore=tests/test_web_tools_live.py -q 2>&1 | tail -3
```
Expected: >= baseline.

- [ ] **Step 6: Commit**

```bash
git add scholaragent/memory/research.py tests/test_depth_levels.py
git commit -m "refactor(memory): return ResearchResult with explicit fallback depths"
```

---

## Phase D — RuntimeContainer

### Task 9: RED — RuntimeContainer contract tests

**Files:**
- Create: `tests/test_runtime_container.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_runtime_container.py
"""RuntimeContainer owns MCP server lifecycle: store, pipeline, agent infra."""
from __future__ import annotations

import os
import tempfile

import pytest


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]


class TestRuntimeContainer:
    def test_container_constructs_with_injected_embeddings(self, tmp_path):
        from scholaragent.runtime import RuntimeContainer

        container = RuntimeContainer(
            data_dir=tmp_path,
            db_path=str(tmp_path / "t.db"),
            model_config={
                "strong": {"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
                "cheap": {"backend": "openai", "model_name": "gpt-4o-mini"},
            },
            embeddings=FakeEmbeddings(),
        )

        store = container.get_store()
        assert store is container.get_store()  # same instance on repeat access
        assert store.count() == 0

        container.close()

    def test_container_pipeline_lazy_init(self, tmp_path):
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

    def test_close_is_idempotent(self, tmp_path):
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
        container.close()  # should not raise
```

- [ ] **Step 2: Run and verify RED**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_runtime_container.py -x --tb=short
```
Expected: `ModuleNotFoundError: No module named 'scholaragent.runtime'`.

### Task 10: GREEN — Implement RuntimeContainer

**Files:**
- Create: `scholaragent/runtime.py`

- [ ] **Step 1: Implement the container**

```python
# scholaragent/runtime.py
"""Runtime container for MCP server state.

Replaces module-level globals in scholaragent.mcp_server with a single
injectable object. Tests can instantiate a container with FakeEmbeddings
and a tmp db, and the MCP tool handlers read from `container.get_store()`
instead of reaching into globals.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from scholaragent.memory.store import MemoryStore
from scholaragent.memory.embeddings import EmbeddingBackend, OpenAIEmbeddings
from scholaragent.memory.research import ResearchPipeline

if TYPE_CHECKING:
    from scholaragent.core.handler import LMHandler
    from scholaragent.core.registry import AgentRegistry
    from scholaragent.core.dispatcher import Dispatcher


class RuntimeContainer:
    """Owns the MCP server's long-lived objects and their lifecycle."""

    def __init__(
        self,
        data_dir: Path,
        db_path: str,
        model_config: dict,
        embeddings: EmbeddingBackend | None = None,
    ):
        self.data_dir = data_dir
        self.db_path = db_path
        self._model_config = model_config
        self._embeddings_override = embeddings
        self._store: MemoryStore | None = None
        self._pipeline: ResearchPipeline | None = None
        self._agent_handler: LMHandler | None = None
        self._agent_registry: AgentRegistry | None = None
        self._agent_dispatcher: Dispatcher | None = None
        self._init_lock = threading.Lock()
        self._agent_lock = threading.Lock()
        self._closed = False

    def get_store(self) -> MemoryStore:
        if self._store is not None:
            return self._store
        with self._init_lock:
            if self._store is not None:
                return self._store
            self.data_dir.mkdir(parents=True, exist_ok=True)
            embeddings = self._embeddings_override or OpenAIEmbeddings()
            self._store = MemoryStore(db_path=self.db_path, embeddings=embeddings)
            logger.info("Initialized memory store at %s", self.db_path)
            return self._store

    def get_pipeline(self) -> ResearchPipeline:
        if self._pipeline is not None:
            return self._pipeline
        with self._init_lock:
            if self._pipeline is not None:
                return self._pipeline
            self._pipeline = ResearchPipeline(store=self.get_store())
            return self._pipeline

    def get_agent_infra(self):
        if self._agent_handler is not None:
            return self._agent_handler, self._agent_registry, self._agent_dispatcher

        with self._agent_lock:
            if self._agent_handler is not None:
                return (
                    self._agent_handler,
                    self._agent_registry,
                    self._agent_dispatcher,
                )

            from scholaragent.clients.router import ModelConfig, ModelRouter
            from scholaragent.clients.token_counter import TokenCounter
            from scholaragent.core.handler import LMHandler
            from scholaragent.core.registry import AgentRegistry
            from scholaragent.core.dispatcher import Dispatcher
            from scholaragent.agents.scout import ScoutAgent
            from scholaragent.agents.reader import ReaderAgent
            from scholaragent.agents.critic import CriticAgent
            from scholaragent.agents.analyst import AnalystAgent
            from scholaragent.agents.synthesizer import SynthesizerAgent

            router = ModelRouter(
                strong=ModelConfig(**self._model_config["strong"]),
                cheap=ModelConfig(**self._model_config["cheap"]),
            )
            token_counter = TokenCounter()
            strong_client = router.get_client("dispatcher")
            handler = LMHandler(
                client=strong_client, token_counter=token_counter, verbose=False
            )
            cheap_client = router.get_client("scout")
            handler.register_client(cheap_client.model_name, cheap_client)
            handler.start()

            registry = AgentRegistry()
            for agent in (
                ScoutAgent(),
                ReaderAgent(),
                CriticAgent(),
                AnalystAgent(),
                SynthesizerAgent(),
            ):
                registry.register(agent)

            dispatcher = Dispatcher(
                registry=registry, handler=handler, store=self.get_store()
            )

            self._agent_handler = handler
            self._agent_registry = registry
            self._agent_dispatcher = dispatcher
            logger.info("Initialized agent infrastructure: %s", registry.list_agents())
            return handler, registry, dispatcher

    def ensure_pipeline_agents(self) -> None:
        pipeline = self.get_pipeline()
        if pipeline.has_agent_infra:
            return
        handler, registry, dispatcher = self.get_agent_infra()
        pipeline.set_agent_infra(handler, registry, dispatcher)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._agent_handler is not None:
            try:
                self._agent_handler.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Error stopping agent handler")
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing store")
```

- [ ] **Step 2: Verify GREEN**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_runtime_container.py -v
```
Expected: `3 passed`.

- [ ] **Step 3: Commit**

```bash
git add scholaragent/runtime.py tests/test_runtime_container.py
git commit -m "feat(runtime): add RuntimeContainer for MCP lifecycle"
```

### Task 11: Migrate mcp_server.py to RuntimeContainer

**Files:**
- Modify: `scholaragent/mcp_server.py` — replace globals with a container accessor.
- Modify: `tests/test_mcp_server.py` — swap monkey-patched globals for container injection.

- [ ] **Step 1: Write a test proving tool handlers read from the container**

Add to `tests/test_mcp_server.py`:

```python
class TestMcpServerContainerInjection:
    def test_memory_lookup_handler_uses_container_store(self, tmp_path):
        from scholaragent.runtime import RuntimeContainer
        from scholaragent.mcp_server import _memory_lookup
        from tests.test_result_indexer import FakeEmbeddings  # or a local shim

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
        # Seed one entry
        from scholaragent.memory.types import MemoryEntry
        store.add(MemoryEntry(
            content="RLHF paper body",
            summary="RLHF summary",
            source_type="paper",
            source_ref="arxiv:1",
            tags=["rlhf"],
        ))

        result = _memory_lookup(store, "rlhf", sources=None, max_results=5, compact=True)
        assert result["total_indexed"] == 1
        assert result["results"]
        container.close()
```

(The handler function already takes `store` as a parameter — this test just proves that end-to-end wiring via a container works without touching module globals.)

- [ ] **Step 2: Run test to verify current state**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_mcp_server.py::TestMcpServerContainerInjection -v
```
Expected: PASS immediately (the handlers are already parameter-driven). This test exists as a regression guard for the next step.

- [ ] **Step 3: Replace globals with a container accessor**

In `scholaragent/mcp_server.py`:

Replace the globals block (lines 49-57):
```python
_container: "RuntimeContainer | None" = None
_container_lock = threading.Lock()

DATA_DIR = Path(os.environ.get("SCHOLAR_MEMORY_DIR", Path.home() / ".scholaragent"))
DB_PATH = os.environ.get("SCHOLAR_MEMORY_DB", str(DATA_DIR / "memory.db"))


def _get_container() -> "RuntimeContainer":
    """Lazy-init the MCP runtime container (thread-safe)."""
    global _container
    if _container is not None:
        return _container
    with _container_lock:
        if _container is not None:
            return _container
        from scholaragent.runtime import RuntimeContainer
        _container = RuntimeContainer(
            data_dir=DATA_DIR,
            db_path=DB_PATH,
            model_config=_build_model_config(),
        )
        return _container


def _cleanup():
    global _container
    if _container is not None:
        _container.close()
        _container = None


atexit.register(_cleanup)
```

Delete `_get_store`, `_get_pipeline`, `_get_agent_infra`, `_ensure_pipeline_agents`.

Update every `_get_store()` call to `_get_container().get_store()` (and same for `_get_pipeline()` → `_get_container().get_pipeline()`). Update `_ensure_pipeline_agents(pipeline)` to `_get_container().ensure_pipeline_agents()`.

In `_memory_status`, replace `if _agent_handler is not None` with:
```python
container = _get_container()
if container._agent_handler is not None:  # noqa: SLF001 — intentional
    tc = container._agent_handler.token_counter
    ...
```

Or add a `get_token_counter(self) -> TokenCounter | None` accessor to `RuntimeContainer` (cleaner).

- [ ] **Step 4: Run the full suite**

Run:
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ --ignore=tests/test_installer.py --ignore=tests/test_web_tools_live.py -q 2>&1 | tail -3
```
Expected: >= baseline.

- [ ] **Step 5: Commit**

```bash
git add scholaragent/mcp_server.py tests/test_mcp_server.py
git commit -m "refactor(mcp): route server state through RuntimeContainer"
```

---

## Self-Review Checklist

- [ ] Every spec requirement from the conversation has a task:
  - ✅ Extract SourceCollector → Tasks 1-3
  - ✅ Extract ResultIndexer → Tasks 4-6
  - ✅ ResearchResult with requested/actual depth → Tasks 7-8
  - ✅ RuntimeContainer for MCP globals → Tasks 9-11
  - ✅ Backwards-compat alias `ResearchPipeline` → preserved (rename deferred)
  - ⚠️  Language hardcode removal → deferred to follow-up (parameterized with `code_language="python"` default)

- [ ] No placeholders remain (every step has concrete code or a concrete command).

- [ ] Type/name consistency:
  - `SourceCollector.collect(query, source_types, code_language)` — used consistently in Tasks 1-3 and 6.
  - `ResultIndexer.index_raw / index_enriched / index_synthesis / log_research` — used consistently in Tasks 4-6.
  - `ResearchResult.to_dict()` — the single conversion point used in Task 8.
  - `RuntimeContainer.get_store / get_pipeline / get_agent_infra / ensure_pipeline_agents / close` — used consistently in Tasks 9-11.

## Rollback notes

- Every phase is an independent commit. `git revert <sha>` per commit unwinds that step without touching siblings.
- The backwards-compat alias `ResearchPipeline = ResearchCoordinator` means existing importers keep working. If the DI migration in Task 3 breaks a consumer, reverting Task 3 alone restores the module-global imports.

## Execution

After the plan lands, the user picks:
1. **Subagent-driven** — each task in a fresh subagent, review between tasks (recommended for plans this size).
2. **Inline execution** — run tasks in one session with checkpoints.
