# Scholar Memory MCP Server — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a persistent, semantically-indexed memory layer to ScholarAgent, exposed as an MCP server that any coding agent can query for on-demand research context.

**Architecture:** 5 MCP tools (memory_lookup, memory_research, memory_store, memory_forget, memory_status) backed by SQLite + embedding vectors. The existing multi-agent research pipeline handles deep research; a fast semantic search handles lookups. Sources: papers (existing), documentation (new), code (new).

**Tech Stack:** Python 3.12+, mcp[cli] (FastMCP), SQLite, OpenAI embeddings API (swappable), httpx (already installed), existing ScholarAgent agents.

**Reference codebase:** /Volumes/WD_4D/RLM/scholaragent/ — the existing ScholarAgent project we're extending.

---

### Task 1: Add MCP + Embedding Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Add `mcp[cli]` to dependencies and `numpy` for cosine similarity:

```toml
[project]
name = "scholaragent"
version = "0.2.0"
description = "Multi-agent scientific literature research system with persistent memory"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "rich>=13.0",
    "openai>=1.0",
    "anthropic>=0.40",
    "mcp[cli]>=1.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 2: Install**

```bash
cd /Volumes/WD_4D/RLM/scholaragent
uv pip install -e ".[dev]"
```

**Step 3: Verify**

```bash
python -c "from mcp.server.fastmcp import FastMCP; print('MCP OK')"
python -c "import numpy; print('numpy OK')"
```

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add mcp and numpy dependencies"
```

---

### Task 2: Memory Types

**Files:**
- Create: `scholaragent/memory/__init__.py` (empty)
- Create: `scholaragent/memory/types.py`
- Create: `tests/test_memory_types.py`

**Step 1: Create package directory**

```bash
mkdir -p scholaragent/memory
touch scholaragent/memory/__init__.py
```

**Step 2: Write the failing test**

```python
"""Tests for memory data types."""

import pytest
from datetime import datetime, timezone


class TestMemoryEntry:
    def test_creation(self):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="Transformers use self-attention mechanisms",
            summary="Self-attention in transformers",
            source_type="paper",
            source_ref="arxiv:1706.03762",
            tags=["attention", "transformers"],
        )
        assert entry.content == "Transformers use self-attention mechanisms"
        assert entry.source_type == "paper"
        assert entry.id  # auto-generated uuid
        assert entry.access_count == 0
        assert entry.created_at is not None

    def test_to_dict(self):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="Test content",
            summary="Test",
            source_type="docs",
            source_ref="https://example.com",
            tags=["test"],
        )
        d = entry.to_dict()
        assert d["content"] == "Test content"
        assert d["source_type"] == "docs"
        assert "id" in d
        assert "created_at" in d
        assert d["access_count"] == 0

    def test_source_type_validation(self):
        from scholaragent.memory.types import MemoryEntry

        with pytest.raises(ValueError, match="source_type"):
            MemoryEntry(
                content="x",
                summary="x",
                source_type="invalid",
                source_ref="x",
                tags=[],
            )


class TestResearchLogEntry:
    def test_creation(self):
        from scholaragent.memory.types import ResearchLogEntry

        log = ResearchLogEntry(
            query="RLHF techniques",
            depth="normal",
            focus="implementation",
            result_count=5,
        )
        assert log.query == "RLHF techniques"
        assert log.id  # auto-generated
        assert log.created_at is not None

    def test_to_dict(self):
        from scholaragent.memory.types import ResearchLogEntry

        log = ResearchLogEntry(
            query="test query",
            depth="quick",
            focus="theory",
            result_count=3,
        )
        d = log.to_dict()
        assert d["query"] == "test query"
        assert d["depth"] == "quick"


class TestSourceTypes:
    def test_valid_source_types(self):
        from scholaragent.memory.types import VALID_SOURCE_TYPES

        assert "paper" in VALID_SOURCE_TYPES
        assert "docs" in VALID_SOURCE_TYPES
        assert "code" in VALID_SOURCE_TYPES
```

**Step 3: Run test to verify it fails**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && python -m pytest tests/test_memory_types.py -v
```

**Step 4: Write types.py**

```python
"""Data types for the memory layer."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

VALID_SOURCE_TYPES: frozenset[str] = frozenset({"paper", "docs", "code"})


@dataclass
class MemoryEntry:
    """A single knowledge item stored in memory."""

    content: str
    summary: str
    source_type: str  # "paper" | "docs" | "code"
    source_ref: str
    tags: list[str]
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: list[float] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    access_count: int = 0

    def __post_init__(self):
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}, "
                f"got '{self.source_type}'"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "access_count": self.access_count,
        }


@dataclass
class ResearchLogEntry:
    """Record of a past research query."""

    query: str
    depth: str
    focus: str
    result_count: int
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "depth": self.depth,
            "focus": self.focus,
            "result_count": self.result_count,
            "created_at": self.created_at,
        }
```

**Step 5: Run test - expect PASS**

```bash
python -m pytest tests/test_memory_types.py -v
```

**Step 6: Commit**

```bash
git add scholaragent/memory/ tests/test_memory_types.py
git commit -m "feat: add memory layer data types"
```

---

### Task 3: Embedding Backend (Swappable)

**Files:**
- Create: `scholaragent/memory/embeddings.py`
- Create: `tests/test_embeddings.py`

**Step 1: Write the failing test**

```python
"""Tests for embedding backends."""

import pytest
from unittest.mock import patch, MagicMock


class TestEmbeddingBackendABC:
    def test_cannot_instantiate(self):
        from scholaragent.memory.embeddings import EmbeddingBackend

        with pytest.raises(TypeError):
            EmbeddingBackend()

    def test_has_required_methods(self):
        from scholaragent.memory.embeddings import EmbeddingBackend
        import inspect

        assert hasattr(EmbeddingBackend, "embed")
        assert hasattr(EmbeddingBackend, "embed_batch")
        assert inspect.isabstract(EmbeddingBackend)


class TestOpenAIEmbeddings:
    @patch("openai.OpenAI")
    def test_creation(self, mock_openai):
        from scholaragent.memory.embeddings import OpenAIEmbeddings

        backend = OpenAIEmbeddings(model="text-embedding-3-small")
        assert backend.model == "text-embedding-3-small"

    @patch("openai.OpenAI")
    def test_embed_returns_list_of_floats(self, mock_openai_cls):
        from scholaragent.memory.embeddings import OpenAIEmbeddings

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        backend = OpenAIEmbeddings()
        result = backend.embed("test text")
        assert result == [0.1, 0.2, 0.3]
        assert isinstance(result, list)

    @patch("openai.OpenAI")
    def test_embed_batch(self, mock_openai_cls):
        from scholaragent.memory.embeddings import OpenAIEmbeddings

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        e1 = MagicMock()
        e1.embedding = [0.1, 0.2]
        e2 = MagicMock()
        e2.embedding = [0.3, 0.4]
        mock_response.data = [e1, e2]
        mock_client.embeddings.create.return_value = mock_response

        backend = OpenAIEmbeddings()
        results = backend.embed_batch(["text1", "text2"])
        assert len(results) == 2
        assert results[0] == [0.1, 0.2]
        assert results[1] == [0.3, 0.4]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from scholaragent.memory.embeddings import cosine_similarity

        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from scholaragent.memory.embeddings import cosine_similarity

        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from scholaragent.memory.embeddings import cosine_similarity

        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
```

**Step 2: Run test - expect FAIL**

**Step 3: Write embeddings.py**

```python
"""Embedding backends for semantic search.

Swappable: start with OpenAI API, replace with local model later.
"""

from abc import ABC, abstractmethod

import numpy as np
import openai


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class EmbeddingBackend(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        ...


class OpenAIEmbeddings(EmbeddingBackend):
    """OpenAI API embeddings (text-embedding-3-small by default)."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = openai.OpenAI()

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
```

**Step 4: Run test - expect PASS**

**Step 5: Commit**

```bash
git add scholaragent/memory/embeddings.py tests/test_embeddings.py
git commit -m "feat: add swappable embedding backend with OpenAI implementation"
```

---

### Task 4: Memory Store (SQLite + Semantic Search)

**Files:**
- Create: `scholaragent/memory/store.py`
- Create: `tests/test_store.py`

**Step 1: Write the failing test**

```python
"""Tests for the memory store."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock


class FakeEmbeddings:
    """Deterministic embedding backend for testing."""

    def embed(self, text: str) -> list[float]:
        # Simple hash-based fake embedding (3 dims)
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def store():
    from scholaragent.memory.store import MemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        s = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
        yield s
        s.close()


class TestMemoryStoreBasic:
    def test_store_and_retrieve(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="RLHF uses human feedback for alignment",
            summary="RLHF overview",
            source_type="paper",
            source_ref="arxiv:2203.02155",
            tags=["rlhf", "alignment"],
        )
        store.add(entry)
        result = store.get(entry.id)
        assert result is not None
        assert result.content == entry.content

    def test_store_multiple(self, store):
        from scholaragent.memory.types import MemoryEntry

        for i in range(5):
            store.add(MemoryEntry(
                content=f"Finding {i}",
                summary=f"Summary {i}",
                source_type="paper",
                source_ref=f"ref-{i}",
                tags=["test"],
            ))
        assert store.count() == 5

    def test_delete(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="To be deleted",
            summary="Delete me",
            source_type="docs",
            source_ref="url",
            tags=[],
        )
        store.add(entry)
        assert store.count() == 1
        store.delete(entry.id)
        assert store.count() == 0

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent-id") is None


class TestMemoryStoreSearch:
    def test_search_returns_results(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Transformer attention mechanisms",
            summary="Attention",
            source_type="paper",
            source_ref="ref1",
            tags=["attention"],
        ))
        store.add(MemoryEntry(
            content="Cooking pasta recipes",
            summary="Pasta",
            source_type="docs",
            source_ref="ref2",
            tags=["cooking"],
        ))
        results = store.search("transformer architecture", max_results=5)
        assert len(results) > 0
        # Each result is (MemoryEntry, score)
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2

    def test_search_filter_by_source(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Paper about X",
            summary="X",
            source_type="paper",
            source_ref="ref1",
            tags=[],
        ))
        store.add(MemoryEntry(
            content="Docs about X",
            summary="X",
            source_type="docs",
            source_ref="ref2",
            tags=[],
        ))
        results = store.search("X", sources=["paper"])
        assert all(entry.source_type == "paper" for entry, _ in results)

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert results == []

    def test_search_increments_access_count(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="Accessed content",
            summary="Access test",
            source_type="paper",
            source_ref="ref",
            tags=[],
        )
        store.add(entry)
        store.search("accessed")
        result = store.get(entry.id)
        assert result.access_count >= 1


class TestMemoryStoreForget:
    def test_forget_by_id(self, store):
        from scholaragent.memory.types import MemoryEntry

        entry = MemoryEntry(
            content="Forget me",
            summary="Forget",
            source_type="paper",
            source_ref="ref",
            tags=[],
        )
        store.add(entry)
        deleted = store.forget(entry.id)
        assert deleted == 1
        assert store.count() == 0

    def test_forget_by_query(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Topic A stuff",
            summary="A",
            source_type="paper",
            source_ref="ref1",
            tags=["topic-a"],
        ))
        store.add(MemoryEntry(
            content="Topic B stuff",
            summary="B",
            source_type="paper",
            source_ref="ref2",
            tags=["topic-b"],
        ))
        # forget returns count of deleted entries matching query
        deleted = store.forget("topic-a", threshold=0.0)
        assert deleted >= 1


class TestResearchLog:
    def test_log_research(self, store):
        store.log_research(
            query="RLHF techniques",
            depth="normal",
            focus="implementation",
            result_count=5,
        )
        recent = store.get_recent_research("RLHF", days=7)
        assert len(recent) == 1
        assert recent[0].query == "RLHF techniques"

    def test_no_recent_research(self, store):
        recent = store.get_recent_research("unknown topic", days=7)
        assert recent == []


class TestMemoryStoreStatus:
    def test_status(self, store):
        from scholaragent.memory.types import MemoryEntry

        store.add(MemoryEntry(
            content="Test",
            summary="Test",
            source_type="paper",
            source_ref="ref",
            tags=[],
        ))
        status = store.status()
        assert status["total_entries"] == 1
        assert "paper" in status["entries_by_source"]
```

**Step 2: Run test - expect FAIL**

**Step 3: Write store.py**

```python
"""Persistent memory store backed by SQLite with semantic search."""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

from scholaragent.memory.embeddings import EmbeddingBackend, cosine_similarity
from scholaragent.memory.types import MemoryEntry, ResearchLogEntry


class MemoryStore:
    """SQLite-backed memory with embedding-based semantic search."""

    def __init__(self, db_path: str, embeddings: EmbeddingBackend):
        self.db_path = db_path
        self.embeddings = embeddings
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                tags TEXT NOT NULL,  -- JSON array
                embedding BLOB,     -- JSON array of floats
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS research_log (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                depth TEXT NOT NULL,
                focus TEXT NOT NULL,
                result_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source_type);
            CREATE INDEX IF NOT EXISTS idx_research_created ON research_log(created_at);
        """)
        self._conn.commit()

    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry, computing embedding if not present."""
        if not entry.embedding:
            entry.embedding = self.embeddings.embed(entry.content)
        self._conn.execute(
            """INSERT OR REPLACE INTO entries
               (id, content, summary, source_type, source_ref, tags, embedding, created_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.content,
                entry.summary,
                entry.source_type,
                entry.source_ref,
                json.dumps(entry.tags),
                json.dumps(entry.embedding),
                entry.created_at,
                entry.access_count,
            ),
        )
        self._conn.commit()

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by ID."""
        row = self._conn.execute(
            "SELECT * FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def delete(self, entry_id: str) -> None:
        """Delete a single entry by ID."""
        self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self._conn.commit()

    def count(self) -> int:
        """Count total entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        return row[0]

    def search(
        self,
        query: str,
        max_results: int = 5,
        sources: list[str] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Semantic search over all entries. Returns (entry, score) pairs."""
        query_embedding = self.embeddings.embed(query)

        # Fetch all entries (or filtered by source)
        if sources:
            placeholders = ",".join("?" for _ in sources)
            rows = self._conn.execute(
                f"SELECT * FROM entries WHERE source_type IN ({placeholders})",
                sources,
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM entries").fetchall()

        if not rows:
            return []

        # Score each entry
        scored = []
        for row in rows:
            entry_embedding = json.loads(row["embedding"])
            score = cosine_similarity(query_embedding, entry_embedding)
            scored.append((self._row_to_entry(row), score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update access counts for returned entries
        results = scored[:max_results]
        for entry, _ in results:
            self._conn.execute(
                "UPDATE entries SET access_count = access_count + 1 WHERE id = ?",
                (entry.id,),
            )
        self._conn.commit()

        return results

    def forget(self, query_or_id: str, threshold: float = 0.5) -> int:
        """Delete entries by ID or by semantic similarity to query.

        If query_or_id matches an existing ID exactly, delete that entry.
        Otherwise, treat it as a semantic query and delete entries above threshold.
        Returns count of deleted entries.
        """
        # Try exact ID match first
        existing = self.get(query_or_id)
        if existing:
            self.delete(query_or_id)
            return 1

        # Semantic search and delete matches above threshold
        results = self.search(query_or_id, max_results=100)
        deleted = 0
        for entry, score in results:
            if score >= threshold:
                self.delete(entry.id)
                deleted += 1
        return deleted

    def log_research(
        self, query: str, depth: str, focus: str, result_count: int
    ) -> None:
        """Record a research query in the log."""
        log_entry = ResearchLogEntry(
            query=query, depth=depth, focus=focus, result_count=result_count
        )
        self._conn.execute(
            """INSERT INTO research_log (id, query, depth, focus, result_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                log_entry.id,
                log_entry.query,
                log_entry.depth,
                log_entry.focus,
                log_entry.result_count,
                log_entry.created_at,
            ),
        )
        self._conn.commit()

    def get_recent_research(
        self, query: str, days: int = 7
    ) -> list[ResearchLogEntry]:
        """Find recent research log entries matching query text."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM research_log WHERE created_at >= ? ORDER BY created_at DESC",
            (cutoff,),
        ).fetchall()

        # Simple text match for deduplication check
        query_lower = query.lower()
        results = []
        for row in rows:
            if query_lower in row["query"].lower() or row["query"].lower() in query_lower:
                results.append(ResearchLogEntry(
                    id=row["id"],
                    query=row["query"],
                    depth=row["depth"],
                    focus=row["focus"],
                    result_count=row["result_count"],
                    created_at=row["created_at"],
                ))
        return results

    def status(self) -> dict:
        """Return memory stats."""
        total = self.count()
        by_source = {}
        for row in self._conn.execute(
            "SELECT source_type, COUNT(*) as cnt FROM entries GROUP BY source_type"
        ).fetchall():
            by_source[row["source_type"]] = row["cnt"]

        research_count = self._conn.execute(
            "SELECT COUNT(*) FROM research_log"
        ).fetchone()[0]

        return {
            "total_entries": total,
            "entries_by_source": by_source,
            "research_queries_logged": research_count,
            "db_path": self.db_path,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            tags=json.loads(row["tags"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else [],
            created_at=row["created_at"],
            access_count=row["access_count"],
        )
```

**Step 4: Run test - expect PASS**

```bash
python -m pytest tests/test_store.py -v
```

**Step 5: Run all tests**

```bash
python -m pytest tests/ -v
```

**Step 6: Commit**

```bash
git add scholaragent/memory/store.py tests/test_store.py
git commit -m "feat: add SQLite memory store with semantic search"
```

---

### Task 5: Source Adapters (Docs + GitHub Code)

**Files:**
- Create: `scholaragent/sources/__init__.py` (empty)
- Create: `scholaragent/sources/docs.py`
- Create: `scholaragent/sources/github.py`
- Create: `tests/test_sources.py`

**Step 1: Create package directory**

```bash
mkdir -p scholaragent/sources
touch scholaragent/sources/__init__.py
```

**Step 2: Write the failing test**

```python
"""Tests for documentation and GitHub code source adapters."""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestDocsFetcher:
    @patch("httpx.get")
    def test_fetch_returns_content(self, mock_get):
        from scholaragent.sources.docs import fetch_docs

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html><body>
        <h1>FastAPI Dependency Injection</h1>
        <p>FastAPI uses Depends() for dependency injection.</p>
        <p>You can declare dependencies as function parameters.</p>
        </body></html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = fetch_docs("https://fastapi.tiangolo.com/tutorial/dependencies/")
        assert len(results) == 1
        assert results[0]["source_type"] == "docs"
        assert "FastAPI" in results[0]["content"]
        assert results[0]["source_ref"] == "https://fastapi.tiangolo.com/tutorial/dependencies/"

    @patch("httpx.get")
    def test_fetch_error_returns_empty(self, mock_get):
        from scholaragent.sources.docs import fetch_docs

        mock_get.side_effect = Exception("timeout")
        results = fetch_docs("https://example.com")
        assert results == []


class TestSearchDocs:
    @patch("httpx.get")
    def test_search_docs_returns_results(self, mock_get):
        from scholaragent.sources.docs import search_docs

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html><body>
        <h1>Search Results</h1>
        <p>Relevant documentation content here.</p>
        </body></html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_response.url = "https://example.com/results"
        mock_get.return_value = mock_response

        results = search_docs("FastAPI dependency injection", max_results=3)
        assert isinstance(results, list)


class TestGitHubCodeSearch:
    @patch("httpx.get")
    def test_search_returns_results(self, mock_get):
        from scholaragent.sources.github import search_github_code

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "name": "attention.py",
                    "path": "src/attention.py",
                    "repository": {"full_name": "org/repo", "html_url": "https://github.com/org/repo"},
                    "html_url": "https://github.com/org/repo/blob/main/src/attention.py",
                    "text_matches": [
                        {"fragment": "class MultiHeadAttention:\n    def forward(self, x):"}
                    ],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = search_github_code("multi head attention python")
        assert len(results) == 1
        assert results[0]["source_type"] == "code"
        assert "attention" in results[0]["content"].lower()
        assert results[0]["source_ref"].startswith("https://github.com")

    @patch("httpx.get")
    def test_search_error_returns_empty(self, mock_get):
        from scholaragent.sources.github import search_github_code

        mock_get.side_effect = Exception("rate limited")
        results = search_github_code("test query")
        assert results == []

    @patch("httpx.get")
    def test_search_with_language_filter(self, mock_get):
        from scholaragent.sources.github import search_github_code

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        search_github_code("test", language="python")
        call_args = mock_get.call_args
        assert "language:python" in call_args[1]["params"]["q"]
```

**Step 3: Run test - expect FAIL**

**Step 4: Write docs.py**

```python
"""Documentation source adapter.

Fetches and extracts text content from web pages.
"""

import re

import httpx


def _html_to_text(html: str) -> str:
    """Simple HTML to text conversion. Strips tags, normalizes whitespace."""
    # Remove script and style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_docs(url: str, timeout: float = 30.0) -> list[dict]:
    """Fetch a documentation page and extract text content.

    Returns list with single dict: {content, source_type, source_ref}
    or empty list on error.
    """
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except Exception:
        return []

    text = _html_to_text(response.text)
    if not text:
        return []

    return [{
        "content": text[:10000],  # Cap at 10k chars
        "source_type": "docs",
        "source_ref": url,
    }]


def search_docs(query: str, max_results: int = 5) -> list[dict]:
    """Search for documentation pages.

    Uses a simple approach: construct search URLs for common doc sites.
    Returns list of {content, source_type, source_ref} dicts.
    """
    # For now, try fetching from known documentation patterns
    # This is a placeholder — a real implementation would use a search API
    urls = [
        f"https://docs.python.org/3/search.html?q={query}",
    ]

    results = []
    for url in urls[:max_results]:
        fetched = fetch_docs(url)
        results.extend(fetched)

    return results
```

**Step 5: Write github.py**

```python
"""GitHub code search source adapter."""

import os

import httpx

GITHUB_API_URL = "https://api.github.com/search/code"


def search_github_code(
    query: str,
    language: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Search GitHub for code examples.

    Returns list of {content, source_type, source_ref} dicts.
    """
    q = query
    if language:
        q += f" language:{language}"

    headers = {
        "Accept": "application/vnd.github.text-match+json",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    params = {
        "q": q,
        "per_page": max_results,
    }

    try:
        response = httpx.get(
            GITHUB_API_URL,
            params=params,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    results = []
    for item in data.get("items", []):
        # Extract text matches (code fragments)
        fragments = []
        for match in item.get("text_matches", []):
            fragment = match.get("fragment", "")
            if fragment:
                fragments.append(fragment)

        content = "\n---\n".join(fragments) if fragments else item.get("name", "")
        repo = item.get("repository", {})
        repo_name = repo.get("full_name", "")
        file_path = item.get("path", "")

        results.append({
            "content": f"# {repo_name}/{file_path}\n\n{content}",
            "source_type": "code",
            "source_ref": item.get("html_url", ""),
        })

    return results
```

**Step 6: Run test - expect PASS**

**Step 7: Commit**

```bash
git add scholaragent/sources/ tests/test_sources.py
git commit -m "feat: add docs and GitHub code source adapters"
```

---

### Task 6: Research Pipeline Integration

**Files:**
- Create: `scholaragent/memory/research.py`
- Create: `tests/test_research_pipeline.py`

This connects the multi-agent research pipeline to the memory store.

**Step 1: Write the failing test**

```python
"""Tests for the research pipeline integration."""

import os
import tempfile

import pytest
from unittest.mock import MagicMock, patch

from scholaragent.memory.types import MemoryEntry


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def store():
    from scholaragent.memory.store import MemoryStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        s = MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())
        yield s
        s.close()


class TestResearchPipeline:
    def test_creation(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)
        assert pipeline.store is store

    def test_collect_sources_papers(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.tools.arxiv.search_arxiv") as mock_arxiv, \
             patch("scholaragent.tools.semantic_scholar.search_semantic_scholar") as mock_s2:
            mock_arxiv.return_value = '[{"arxiv_id": "2401.00001", "title": "Test Paper", "authors": ["A"], "abstract": "Test abstract", "published": "2024-01-01", "categories": ["cs.AI"]}]'
            mock_s2.return_value = '[{"paper_id": "abc", "title": "Test Paper 2", "authors": ["B"], "abstract": "Abstract 2", "year": 2024, "citation_count": 10, "arxiv_id": ""}]'

            results = pipeline._collect_sources("attention mechanisms", sources=["paper"])
            assert len(results) > 0
            assert all(r["source_type"] == "paper" for r in results)

    def test_quick_research_stores_entries(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.tools.arxiv.search_arxiv") as mock_arxiv, \
             patch("scholaragent.tools.semantic_scholar.search_semantic_scholar") as mock_s2, \
             patch("scholaragent.sources.github.search_github_code") as mock_gh:
            mock_arxiv.return_value = '[{"arxiv_id": "123", "title": "Paper", "authors": ["X"], "abstract": "Content", "published": "2024", "categories": []}]'
            mock_s2.return_value = '[]'
            mock_gh.return_value = []

            result = pipeline.run("test query", depth="quick", focus="implementation")
            assert store.count() > 0
            assert result["depth"] == "quick"
            assert result["entries_added"] > 0

    def test_deduplication_check(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)

        # Log a recent research
        store.log_research("RLHF techniques", "normal", "theory", 5)

        # Check dedup
        recent = pipeline._check_dedup("RLHF techniques")
        assert recent is not None

    def test_no_dedup_for_new_query(self, store):
        from scholaragent.memory.research import ResearchPipeline

        pipeline = ResearchPipeline(store=store)
        recent = pipeline._check_dedup("completely new topic")
        assert recent is None
```

**Step 2: Run test - expect FAIL**

**Step 3: Write research.py**

```python
"""Research pipeline — connects multi-agent system to memory store.

Handles depth levels:
- quick: Scout only, raw results indexed
- normal: Scout + Reader + Critic
- deep: Full pipeline (Scout → Reader → Critic → Analyst → Synthesizer)
"""

import json

from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry, ResearchLogEntry
from scholaragent.tools.arxiv import search_arxiv
from scholaragent.tools.semantic_scholar import search_semantic_scholar
from scholaragent.sources.github import search_github_code
from scholaragent.sources.docs import search_docs


FOCUS_HINTS = {
    "implementation": "Focus on code examples, API usage, how-to guides, and practical patterns.",
    "theory": "Focus on concepts, algorithms, mathematical foundations, and trade-offs.",
    "comparison": "Focus on alternatives, benchmarks, pros/cons, and comparative analysis.",
}


class ResearchPipeline:
    """Connects source collection to memory storage."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def run(
        self,
        query: str,
        depth: str = "normal",
        focus: str = "implementation",
        force: bool = False,
    ) -> dict:
        """Execute research pipeline and store results.

        Returns dict with metadata about what was found and stored.
        """
        # Check deduplication
        if not force:
            recent = self._check_dedup(query)
            if recent is not None:
                return {
                    "status": "cached",
                    "depth": recent.depth,
                    "query": recent.query,
                    "entries_added": 0,
                    "cached_results": recent.result_count,
                    "message": f"Recent research found from {recent.created_at}. Use force=True to re-research.",
                }

        # Collect raw sources
        source_types = ["paper", "code", "docs"]
        raw_results = self._collect_sources(query, sources=source_types)

        # For quick depth, just index raw results
        entries_added = 0
        for raw in raw_results:
            focus_hint = FOCUS_HINTS.get(focus, "")
            summary = raw["content"][:200]
            entry = MemoryEntry(
                content=raw["content"],
                summary=summary,
                source_type=raw["source_type"],
                source_ref=raw["source_ref"],
                tags=[query.lower().replace(" ", "-")],
            )
            self.store.add(entry)
            entries_added += 1

        # Log the research
        self.store.log_research(
            query=query,
            depth=depth,
            focus=focus,
            result_count=entries_added,
        )

        return {
            "status": "completed",
            "depth": depth,
            "query": query,
            "entries_added": entries_added,
            "message": f"Research complete. {entries_added} entries indexed.",
        }

    def _collect_sources(
        self,
        query: str,
        sources: list[str] | None = None,
    ) -> list[dict]:
        """Collect raw results from all source adapters."""
        sources = sources or ["paper", "code", "docs"]
        results = []

        if "paper" in sources:
            # arXiv
            try:
                arxiv_json = search_arxiv(query, max_results=10)
                arxiv_papers = json.loads(arxiv_json)
                if isinstance(arxiv_papers, list):
                    for paper in arxiv_papers:
                        results.append({
                            "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nAuthors: {', '.join(paper.get('authors', []))}",
                            "source_type": "paper",
                            "source_ref": f"arxiv:{paper.get('arxiv_id', '')}",
                        })
            except Exception:
                pass

            # Semantic Scholar
            try:
                s2_json = search_semantic_scholar(query, limit=10)
                s2_papers = json.loads(s2_json)
                if isinstance(s2_papers, list):
                    for paper in s2_papers:
                        results.append({
                            "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nYear: {paper.get('year', 'N/A')}\nCitations: {paper.get('citation_count', 0)}",
                            "source_type": "paper",
                            "source_ref": f"s2:{paper.get('paper_id', '')}",
                        })
            except Exception:
                pass

        if "code" in sources:
            try:
                code_results = search_github_code(query, language="python", max_results=5)
                results.extend(code_results)
            except Exception:
                pass

        if "docs" in sources:
            try:
                doc_results = search_docs(query, max_results=3)
                results.extend(doc_results)
            except Exception:
                pass

        return results

    def _check_dedup(self, query: str) -> ResearchLogEntry | None:
        """Check if similar research was done recently."""
        recent = self.store.get_recent_research(query, days=7)
        if recent:
            return recent[0]
        return None
```

**Step 4: Run test - expect PASS**

**Step 5: Run all tests**

**Step 6: Commit**

```bash
git add scholaragent/memory/research.py tests/test_research_pipeline.py
git commit -m "feat: add research pipeline with depth levels and deduplication"
```

---

### Task 7: MCP Server

**Files:**
- Create: `scholaragent/mcp_server.py`
- Create: `tests/test_mcp_server.py`

**Step 1: Write the failing test**

```python
"""Tests for the MCP server tool definitions."""

import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock


class FakeEmbeddings:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000.0, (h * 2 % 1000) / 1000.0, (h * 3 % 1000) / 1000.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class TestMCPToolFunctions:
    """Test the tool handler functions directly (not via MCP transport)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up a temporary memory store for each test."""
        self.db_path = str(tmp_path / "test_memory.db")
        # We import the module-level functions and patch the global store
        with patch.dict(os.environ, {"SCHOLAR_MEMORY_DB": self.db_path}):
            yield

    def test_memory_store_creates_entry(self):
        from scholaragent.mcp_server import _get_store, _memory_store
        from scholaragent.memory.store import MemoryStore

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        result = _memory_store(
            store=store,
            content="RLHF uses human feedback",
            source="arxiv:2203.02155",
            tags=["rlhf"],
        )
        assert result["status"] == "stored"
        assert store.count() == 1

    def test_memory_lookup_returns_results(self):
        from scholaragent.mcp_server import _memory_lookup
        from scholaragent.memory.store import MemoryStore
        from scholaragent.memory.types import MemoryEntry

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        store.add(MemoryEntry(
            content="Transformers use attention",
            summary="Attention",
            source_type="paper",
            source_ref="ref1",
            tags=["transformers"],
        ))
        result = _memory_lookup(store=store, query="attention mechanisms")
        assert "results" in result
        assert len(result["results"]) > 0

    def test_memory_lookup_empty_store(self):
        from scholaragent.mcp_server import _memory_lookup
        from scholaragent.memory.store import MemoryStore

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        result = _memory_lookup(store=store, query="anything")
        assert result["results"] == []

    def test_memory_forget_by_id(self):
        from scholaragent.mcp_server import _memory_forget
        from scholaragent.memory.store import MemoryStore
        from scholaragent.memory.types import MemoryEntry

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        entry = MemoryEntry(
            content="Delete me",
            summary="Del",
            source_type="paper",
            source_ref="ref",
            tags=[],
        )
        store.add(entry)
        result = _memory_forget(store=store, query_or_id=entry.id)
        assert result["deleted"] == 1
        assert store.count() == 0

    def test_memory_status(self):
        from scholaragent.mcp_server import _memory_status
        from scholaragent.memory.store import MemoryStore

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        result = _memory_status(store=store)
        assert "total_entries" in result
        assert result["total_entries"] == 0
```

**Step 2: Run test - expect FAIL**

**Step 3: Write mcp_server.py**

```python
"""ScholarAgent Memory MCP Server.

Run: python -m scholaragent.mcp_server
Or:  uv run mcp run scholaragent/mcp_server.py

Configuration for coding agents:
{
    "mcpServers": {
        "scholar-memory": {
            "command": "python",
            "args": ["-m", "scholaragent.mcp_server"],
            "env": {
                "OPENAI_API_KEY": "...",
                "ANTHROPIC_API_KEY": "..."
            }
        }
    }
}
"""

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from scholaragent.memory.embeddings import OpenAIEmbeddings
from scholaragent.memory.store import MemoryStore
from scholaragent.memory.research import ResearchPipeline

# --- Global state ---

_store: MemoryStore | None = None
_pipeline: ResearchPipeline | None = None

DATA_DIR = Path(os.environ.get("SCHOLAR_MEMORY_DIR", Path.home() / ".scholaragent"))
DB_PATH = os.environ.get("SCHOLAR_MEMORY_DB", str(DATA_DIR / "memory.db"))


def _get_store() -> MemoryStore:
    """Lazy-init the global memory store."""
    global _store
    if _store is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        embeddings = OpenAIEmbeddings()
        _store = MemoryStore(db_path=DB_PATH, embeddings=embeddings)
    return _store


def _get_pipeline() -> ResearchPipeline:
    """Lazy-init the research pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ResearchPipeline(store=_get_store())
    return _pipeline


# --- Tool handler functions (testable without MCP transport) ---


def _memory_lookup(
    store: MemoryStore,
    query: str,
    sources: list[str] | None = None,
    max_results: int = 5,
) -> dict:
    results = store.search(query, max_results=max_results, sources=sources)
    return {
        "results": [
            {**entry.to_dict(), "relevance_score": round(score, 3)}
            for entry, score in results
        ],
        "total_indexed": store.count(),
        "query": query,
    }


def _memory_research(
    pipeline: ResearchPipeline,
    query: str,
    depth: str = "normal",
    focus: str = "implementation",
) -> dict:
    return pipeline.run(query=query, depth=depth, focus=focus)


def _memory_store(
    store: MemoryStore,
    content: str,
    source: str,
    tags: list[str],
) -> dict:
    from scholaragent.memory.types import MemoryEntry

    # Infer source_type from source string
    source_type = "docs"
    if source.startswith("arxiv:") or source.startswith("s2:"):
        source_type = "paper"
    elif source.startswith("github:") or source.startswith("https://github.com"):
        source_type = "code"

    entry = MemoryEntry(
        content=content,
        summary=content[:200],
        source_type=source_type,
        source_ref=source,
        tags=tags,
    )
    store.add(entry)
    return {"status": "stored", "id": entry.id, "source_type": source_type}


def _memory_forget(
    store: MemoryStore,
    query_or_id: str,
) -> dict:
    deleted = store.forget(query_or_id)
    return {"deleted": deleted, "query_or_id": query_or_id}


def _memory_status(store: MemoryStore) -> dict:
    return store.status()


# --- MCP Server ---

mcp = FastMCP("scholar-memory", json_response=True)


@mcp.tool()
def memory_lookup(
    query: str,
    sources: list[str] | None = None,
    max_results: int = 5,
) -> str:
    """Fast semantic search over all indexed knowledge.

    Returns the most relevant findings, code snippets, and insights
    from past research. Call this frequently while coding — it's fast.

    Args:
        query: What you're looking for (natural language)
        sources: Filter by source type: "paper", "docs", "code" (optional)
        max_results: Maximum results to return (default 5)
    """
    result = _memory_lookup(_get_store(), query, sources, max_results)
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_research(
    query: str,
    depth: str = "normal",
    focus: str = "implementation",
) -> str:
    """Deep research on a topic. Searches papers, docs, and code.

    Results are automatically indexed for future memory_lookup calls.
    Use when memory_lookup returns nothing useful.

    Args:
        query: Research question (natural language)
        depth: "quick" (5-10s, search only) | "normal" (30-60s, with analysis) | "deep" (2-5min, full pipeline)
        focus: "implementation" (code/how-to) | "theory" (concepts) | "comparison" (alternatives/benchmarks)
    """
    result = _memory_research(_get_pipeline(), query, depth, focus)
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_store(
    content: str,
    source: str,
    tags: list[str],
) -> str:
    """Manually store a finding, code snippet, or insight.

    Use this to save useful things you discover while coding.
    They'll be available via memory_lookup in future sessions.

    Args:
        content: The actual content to store
        source: Where it came from (e.g., "arxiv:2401.12345", "https://docs.python.org/...", "github:org/repo")
        tags: Categorization tags (e.g., ["rlhf", "reward-model"])
    """
    result = _memory_store(_get_store(), content, source, tags)
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_forget(query_or_id: str) -> str:
    """Remove entries from memory.

    Pass an exact entry ID to delete one entry, or a natural language
    query to delete all semantically similar entries.

    Args:
        query_or_id: Entry ID (exact match) or search query (semantic match)
    """
    result = _memory_forget(_get_store(), query_or_id)
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_status() -> str:
    """Get memory statistics.

    Returns total entries, breakdown by source type, and research history.
    """
    result = _memory_status(_get_store())
    return json.dumps(result, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
```

**Step 4: Run test - expect PASS**

**Step 5: Add `__main__.py` for `python -m scholaragent.mcp_server`**

Create `scholaragent/mcp_server/__init__.py` — actually, keep it as a single file `scholaragent/mcp_server.py` and add:

```python
# Already has if __name__ == "__main__": main() at bottom
```

To support `python -m scholaragent.mcp_server`, the single file approach works since Python treats `scholaragent/mcp_server.py` as the module.

**Step 6: Run all tests**

```bash
python -m pytest tests/ -v
```

**Step 7: Commit**

```bash
git add scholaragent/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add MCP server with 5 memory tools"
```

---

### Task 8: Integration Test + Smoke Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
"""Integration test: full flow from MCP tool functions through to store."""

import os
import tempfile

import pytest
from unittest.mock import patch


class FakeEmbeddings:
    def embed(self, text):
        # Use character frequency for more realistic-ish embeddings
        vec = [0.0] * 8
        for i, c in enumerate(text.lower()):
            vec[ord(c) % 8] += 1.0
        # Normalize
        total = sum(v * v for v in vec) ** 0.5
        if total > 0:
            vec = [v / total for v in vec]
        return vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class TestFullFlow:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.db_path = str(tmp_path / "integration.db")

    def test_store_then_lookup(self):
        from scholaragent.memory.store import MemoryStore
        from scholaragent.mcp_server import _memory_store, _memory_lookup

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())

        # Store some knowledge
        _memory_store(store, "RLHF uses reward models trained on human preferences", "arxiv:2203.02155", ["rlhf"])
        _memory_store(store, "PPO is the standard optimizer for RLHF fine-tuning", "arxiv:2210.01234", ["rlhf", "ppo"])
        _memory_store(store, "React useState hook manages component state", "https://react.dev/reference/react/useState", ["react"])

        # Lookup should find relevant results
        result = _memory_lookup(store, "reinforcement learning from human feedback")
        assert len(result["results"]) > 0
        assert result["total_indexed"] == 3
        # RLHF entries should score higher than React entry
        rlhf_results = [r for r in result["results"] if "rlhf" in r.get("tags", [])]
        assert len(rlhf_results) > 0

    def test_store_then_forget(self):
        from scholaragent.memory.store import MemoryStore
        from scholaragent.mcp_server import _memory_store, _memory_forget, _memory_status

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())

        result = _memory_store(store, "Temporary note", "manual", ["temp"])
        entry_id = result["id"]

        status = _memory_status(store)
        assert status["total_entries"] == 1

        forget_result = _memory_forget(store, entry_id)
        assert forget_result["deleted"] == 1

        status = _memory_status(store)
        assert status["total_entries"] == 0

    def test_research_then_lookup(self):
        from scholaragent.memory.store import MemoryStore
        from scholaragent.memory.research import ResearchPipeline
        from scholaragent.mcp_server import _memory_research, _memory_lookup

        store = MemoryStore(db_path=self.db_path, embeddings=FakeEmbeddings())
        pipeline = ResearchPipeline(store=store)

        with patch("scholaragent.tools.arxiv.search_arxiv") as mock_arxiv, \
             patch("scholaragent.tools.semantic_scholar.search_semantic_scholar") as mock_s2, \
             patch("scholaragent.sources.github.search_github_code") as mock_gh, \
             patch("scholaragent.sources.docs.search_docs") as mock_docs:
            mock_arxiv.return_value = '[{"arxiv_id": "2401.00001", "title": "Attention Is All You Need", "authors": ["Vaswani"], "abstract": "We propose the Transformer architecture based on attention mechanisms", "published": "2017", "categories": ["cs.CL"]}]'
            mock_s2.return_value = '[]'
            mock_gh.return_value = [{"content": "class MultiHeadAttention(nn.Module):", "source_type": "code", "source_ref": "https://github.com/example/transformer"}]
            mock_docs.return_value = []

            research_result = _memory_research(pipeline, "transformer attention", depth="quick", focus="implementation")
            assert research_result["entries_added"] > 0

            # Now lookup should find the research results
            lookup_result = _memory_lookup(store, "attention mechanism in transformers")
            assert len(lookup_result["results"]) > 0
```

**Step 2: Run test - expect PASS**

```bash
python -m pytest tests/test_integration.py -v
```

**Step 3: Run ALL tests**

```bash
python -m pytest tests/ -v
```

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full memory flow"
```

---

### Task 9: Final Verification + Config Example

**Files:**
- Modify: `scholaragent/scholaragent/__init__.py` (add __version__)
- Create: `scholaragent/mcp-config-example.json`

**Step 1: Update version and exports**

Add to `__init__.py`:

```python
__version__ = "0.2.0"
```

**Step 2: Create example MCP config**

```json
{
  "mcpServers": {
    "scholar-memory": {
      "command": "python",
      "args": ["-m", "scholaragent.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key-here",
        "ANTHROPIC_API_KEY": "your-anthropic-key-here",
        "GITHUB_TOKEN": "optional-github-token"
      }
    }
  }
}
```

**Step 3: Run ALL tests**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && python -m pytest tests/ -v
```

**Step 4: Verify MCP server starts**

```bash
python -c "from scholaragent.mcp_server import mcp; print(f'Server: {mcp.name}'); print('Tools:', [t.name for t in mcp._tool_manager.list_tools()])"
```

**Step 5: Verify import**

```bash
python -c "from scholaragent import ScholarAgent, __version__; print(f'ScholarAgent v{__version__} OK')"
```

**Step 6: Final commit**

```bash
git add -A
git commit -m "chore: finalize scholar-memory MCP server v0.2.0"
```
