# Persistent Context Stream — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a persistent `ContextStream` that carries structured pipeline state and conversation traces between agents, with streaming + snapshot writes, persisted to SQLite, and exposed via MCP.

**Architecture:** New `ContextStream` dataclass created per pipeline run, passed to each agent via `run()`. Agents push incremental updates mid-execution and auto-commit traces on completion. Persisted to a `context_streams` SQLite table in the existing memory DB. Two new MCP tools expose streams to coding agents.

**Tech Stack:** Python dataclasses, SQLite (JSON blobs), existing MemoryStore, FastMCP

---

### Task 1: ContextStream Data Model

**Files:**
- Create: `scholaragent/core/context.py`
- Test: `tests/test_context_stream.py`

**Step 1: Write the failing tests**

```python
# tests/test_context_stream.py
"""Tests for ContextStream data model, persistence, and agent integration."""

import json
import pytest

from scholaragent.core.context import ContextStream, PipelineState, StreamEvent


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    def test_empty_state(self):
        state = PipelineState()
        assert state.papers == []
        assert state.findings == {}
        assert state.assessments == {}
        assert state.themes == {}
        assert state.synthesis == ""

    def test_to_dict(self):
        state = PipelineState(papers=[{"title": "Test"}])
        d = state.to_dict()
        assert d["papers"] == [{"title": "Test"}]
        assert d["findings"] == {}

    def test_from_dict(self):
        d = {"papers": [{"title": "X"}], "findings": {}, "assessments": {}, "themes": {}, "synthesis": "done"}
        state = PipelineState.from_dict(d)
        assert state.papers == [{"title": "X"}]
        assert state.synthesis == "done"


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_creation(self):
        event = StreamEvent(agent="scout", event_type="papers_found", data={"count": 3})
        assert event.agent == "scout"
        assert event.event_type == "papers_found"
        assert event.timestamp  # auto-generated

    def test_to_dict(self):
        event = StreamEvent(agent="scout", event_type="papers_found", data={"count": 3})
        d = event.to_dict()
        assert d["agent"] == "scout"
        assert "timestamp" in d


class TestContextStream:
    """Tests for ContextStream creation and methods."""

    def test_creation(self):
        stream = ContextStream(query="test query")
        assert stream.query == "test query"
        assert stream.id  # UUID generated
        assert stream.state.papers == []
        assert stream.traces == {}
        assert stream.events == []

    def test_push_appends_event(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        assert len(stream.events) == 1
        assert stream.events[0].agent == "scout"
        assert stream.events[0].event_type == "papers_found"

    def test_push_updates_state_papers(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        assert len(stream.state.papers) == 1
        stream.push("scout", "papers_found", {"papers": [{"title": "B"}]})
        assert len(stream.state.papers) == 2

    def test_push_updates_state_findings(self):
        stream = ContextStream(query="test")
        stream.push("reader", "finding_extracted", {"paper_ref": "arxiv:123", "finding": {"key_claims": ["x"]}})
        assert "arxiv:123" in stream.state.findings

    def test_push_updates_state_assessments(self):
        stream = ContextStream(query="test")
        stream.push("critic", "assessment_complete", {"paper_ref": "arxiv:123", "assessment": {"score": 0.9}})
        assert "arxiv:123" in stream.state.assessments

    def test_push_updates_state_themes(self):
        stream = ContextStream(query="test")
        stream.push("analyst", "themes_identified", {"themes": {"gaps": ["x"]}})
        assert stream.state.themes == {"gaps": ["x"]}

    def test_push_updates_state_synthesis(self):
        stream = ContextStream(query="test")
        stream.push("synthesizer", "synthesis_complete", {"synthesis": "Final report"})
        assert stream.state.synthesis == "Final report"

    def test_commit_saves_trace(self):
        stream = ContextStream(query="test")
        messages = [{"role": "system", "content": "You are scout"}, {"role": "user", "content": "find papers"}]
        stream.commit("scout", messages)
        assert stream.traces["scout"] == messages

    def test_commit_overwrites_trace(self):
        stream = ContextStream(query="test")
        stream.commit("scout", [{"role": "user", "content": "v1"}])
        stream.commit("scout", [{"role": "user", "content": "v2"}])
        assert stream.traces["scout"] == [{"role": "user", "content": "v2"}]

    def test_read_all(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        data = stream.read()
        assert "state" in data
        assert "traces" in data
        assert "events" in data

    def test_read_filtered_by_agent(self):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.push("reader", "finding_extracted", {"paper_ref": "x", "finding": {}})
        stream.commit("scout", [{"role": "user", "content": "scout task"}])
        stream.commit("reader", [{"role": "user", "content": "reader task"}])
        data = stream.read(agent="scout")
        assert "scout" in data["traces"]
        assert "reader" not in data["traces"]
        assert all(e["agent"] == "scout" for e in data["events"])

    def test_to_dict_and_from_dict_roundtrip(self):
        stream = ContextStream(query="roundtrip test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        d = stream.to_dict()
        restored = ContextStream.from_dict(d)
        assert restored.id == stream.id
        assert restored.query == stream.query
        assert restored.state.papers == stream.state.papers
        assert restored.traces == stream.traces
        assert len(restored.events) == len(stream.events)

    def test_updated_at_changes_on_push(self):
        stream = ContextStream(query="test")
        original = stream.updated_at
        stream.push("scout", "papers_found", {"papers": []})
        assert stream.updated_at >= original

    def test_push_with_save_callback(self):
        saved = []
        stream = ContextStream(query="test", on_save=lambda s: saved.append(s.id))
        stream.push("scout", "papers_found", {"papers": []})
        assert len(saved) == 1

    def test_commit_with_save_callback(self):
        saved = []
        stream = ContextStream(query="test", on_save=lambda s: saved.append(s.id))
        stream.commit("scout", [])
        assert len(saved) == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_context_stream.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scholaragent.core.context'`

**Step 3: Write the implementation**

```python
# scholaragent/core/context.py
"""Persistent context stream for inter-agent communication.

A ContextStream carries structured pipeline state and conversation traces
between agents during a research run. It supports streaming mid-execution
updates (push) and final snapshots (commit), with optional persistence
via a save callback.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PipelineState:
    """Structured state accumulated across pipeline stages."""

    papers: list[dict] = field(default_factory=list)
    findings: dict[str, dict] = field(default_factory=dict)
    assessments: dict[str, dict] = field(default_factory=dict)
    themes: dict = field(default_factory=dict)
    synthesis: str = ""

    def to_dict(self) -> dict:
        return {
            "papers": list(self.papers),
            "findings": dict(self.findings),
            "assessments": dict(self.assessments),
            "themes": dict(self.themes),
            "synthesis": self.synthesis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PipelineState:
        return cls(
            papers=d.get("papers", []),
            findings=d.get("findings", {}),
            assessments=d.get("assessments", {}),
            themes=d.get("themes", {}),
            synthesis=d.get("synthesis", ""),
        )


@dataclass
class StreamEvent:
    """A timestamped incremental update from an agent."""

    agent: str
    event_type: str
    data: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StreamEvent:
        return cls(
            agent=d["agent"],
            event_type=d["event_type"],
            data=d["data"],
            timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


# Maps (agent_name, event_type) to the state update logic.
_STATE_UPDATERS: dict[str, Callable[[PipelineState, dict], None]] = {}


def _update_papers(state: PipelineState, data: dict) -> None:
    state.papers.extend(data.get("papers", []))

def _update_findings(state: PipelineState, data: dict) -> None:
    paper_ref = data.get("paper_ref", "")
    if paper_ref:
        state.findings[paper_ref] = data.get("finding", {})

def _update_assessments(state: PipelineState, data: dict) -> None:
    paper_ref = data.get("paper_ref", "")
    if paper_ref:
        state.assessments[paper_ref] = data.get("assessment", {})

def _update_themes(state: PipelineState, data: dict) -> None:
    state.themes = data.get("themes", {})

def _update_synthesis(state: PipelineState, data: dict) -> None:
    state.synthesis = data.get("synthesis", "")


_STATE_UPDATERS = {
    "papers_found": _update_papers,
    "finding_extracted": _update_findings,
    "assessment_complete": _update_assessments,
    "themes_identified": _update_themes,
    "synthesis_complete": _update_synthesis,
}


@dataclass
class ContextStream:
    """Persistent context stream for a single pipeline run.

    Carries structured state and conversation traces between agents.
    Supports streaming writes (push) and final snapshots (commit).

    Args:
        query: The research query that spawned this stream.
        on_save: Optional callback invoked after every push/commit for persistence.
    """

    query: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    state: PipelineState = field(default_factory=PipelineState)
    traces: dict[str, list[dict]] = field(default_factory=dict)
    events: list[StreamEvent] = field(default_factory=list)
    on_save: Callable[[ContextStream], None] | None = field(default=None, repr=False)

    def push(self, agent: str, event_type: str, data: dict) -> None:
        """Record an incremental update from an agent mid-execution."""
        event = StreamEvent(agent=agent, event_type=event_type, data=data)
        self.events.append(event)
        self.updated_at = datetime.now(timezone.utc).isoformat()

        updater = _STATE_UPDATERS.get(event_type)
        if updater is not None:
            updater(self.state, data)

        if self.on_save is not None:
            self.on_save(self)

    def commit(self, agent: str, messages: list[dict]) -> None:
        """Save an agent's conversation trace as a final snapshot."""
        self.traces[agent] = messages
        self.updated_at = datetime.now(timezone.utc).isoformat()

        if self.on_save is not None:
            self.on_save(self)

    def read(self, agent: str | None = None) -> dict:
        """Read stream data, optionally filtered to a single agent."""
        if agent is None:
            return {
                "state": self.state.to_dict(),
                "traces": dict(self.traces),
                "events": [e.to_dict() for e in self.events],
            }
        return {
            "state": self.state.to_dict(),
            "traces": {k: v for k, v in self.traces.items() if k == agent},
            "events": [e.to_dict() for e in self.events if e.agent == agent],
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.to_dict(),
            "traces": dict(self.traces),
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ContextStream:
        return cls(
            id=d["id"],
            query=d["query"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            state=PipelineState.from_dict(d["state"]),
            traces=d.get("traces", {}),
            events=[StreamEvent.from_dict(e) for e in d.get("events", [])],
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_context_stream.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scholaragent/core/context.py tests/test_context_stream.py
git commit -m "feat: add ContextStream data model with push/commit/read"
```

---

### Task 2: MemoryStore Persistence for Streams

**Files:**
- Modify: `scholaragent/memory/store.py:25-50` (add table), `:236-252` (add methods)
- Test: `tests/test_context_stream.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_context_stream.py`:

```python
from scholaragent.memory.store import MemoryStore
from tests.helpers import FakeEmbeddings


class TestMemoryStoreStreams:
    """Tests for ContextStream persistence in MemoryStore."""

    @pytest.fixture()
    def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        return MemoryStore(db_path=db_path, embeddings=FakeEmbeddings())

    def test_save_and_load_stream(self, store):
        stream = ContextStream(query="test query")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        store.save_stream(stream)
        loaded = store.load_stream(stream.id)
        assert loaded is not None
        assert loaded.id == stream.id
        assert loaded.query == stream.query
        assert loaded.state.papers == [{"title": "A"}]
        assert loaded.traces["scout"] == [{"role": "user", "content": "task"}]
        assert len(loaded.events) == 1

    def test_load_nonexistent_stream(self, store):
        assert store.load_stream("nonexistent") is None

    def test_save_stream_upsert(self, store):
        stream = ContextStream(query="test")
        store.save_stream(stream)
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        store.save_stream(stream)
        loaded = store.load_stream(stream.id)
        assert len(loaded.state.papers) == 1

    def test_list_streams_empty(self, store):
        assert store.list_streams() == []

    def test_list_streams_returns_compact(self, store):
        stream = ContextStream(query="test query")
        stream.push("scout", "papers_found", {"papers": []})
        stream.commit("scout", [])
        store.save_stream(stream)
        results = store.list_streams()
        assert len(results) == 1
        assert results[0]["id"] == stream.id
        assert results[0]["query"] == "test query"
        assert "scout" in results[0]["agents"]
        assert results[0]["event_count"] == 1

    def test_list_streams_filter_by_query(self, store):
        s1 = ContextStream(query="protein folding")
        s2 = ContextStream(query="quantum computing")
        store.save_stream(s1)
        store.save_stream(s2)
        results = store.list_streams(query="protein")
        assert len(results) == 1
        assert results[0]["query"] == "protein folding"

    def test_list_streams_respects_limit(self, store):
        for i in range(5):
            store.save_stream(ContextStream(query=f"query {i}"))
        results = store.list_streams(limit=3)
        assert len(results) == 3

    def test_list_streams_ordered_by_updated_at(self, store):
        s1 = ContextStream(query="first")
        s2 = ContextStream(query="second")
        store.save_stream(s1)
        store.save_stream(s2)
        results = store.list_streams()
        # Most recent first
        assert results[0]["query"] == "second"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_context_stream.py::TestMemoryStoreStreams -v`
Expected: FAIL — `MemoryStore` has no `save_stream` method

**Step 3: Write the implementation**

In `scholaragent/memory/store.py`:

Add to `_create_tables()` (after the existing `CREATE INDEX` at line 48):

```python
                CREATE TABLE IF NOT EXISTS context_streams (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    state TEXT NOT NULL,
                    traces TEXT NOT NULL,
                    events TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_streams_updated ON context_streams(updated_at);
                CREATE INDEX IF NOT EXISTS idx_streams_query ON context_streams(query);
```

Add three methods before `close()` (before line 236):

```python
    def save_stream(self, stream: ContextStream) -> None:
        """Persist a ContextStream (upsert)."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO context_streams
                   (id, query, state, traces, events, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    stream.id,
                    stream.query,
                    json.dumps(stream.state.to_dict()),
                    json.dumps(stream.traces),
                    json.dumps([e.to_dict() for e in stream.events]),
                    stream.created_at,
                    stream.updated_at,
                ),
            )
            self._conn.commit()

    def load_stream(self, stream_id: str) -> ContextStream | None:
        """Load a ContextStream by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM context_streams WHERE id = ?", (stream_id,)
            ).fetchone()
        if row is None:
            return None
        from scholaragent.core.context import ContextStream
        return ContextStream.from_dict({
            "id": row["id"],
            "query": row["query"],
            "state": json.loads(row["state"]),
            "traces": json.loads(row["traces"]),
            "events": json.loads(row["events"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })

    def list_streams(
        self, query: str | None = None, limit: int = 10
    ) -> list[dict]:
        """List recent context streams as compact metadata dicts."""
        with self._lock:
            if query:
                rows = self._conn.execute(
                    "SELECT id, query, traces, events, created_at, updated_at "
                    "FROM context_streams WHERE query LIKE ? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, query, traces, events, created_at, updated_at "
                    "FROM context_streams ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        results = []
        for row in rows:
            traces = json.loads(row["traces"])
            events = json.loads(row["events"])
            results.append({
                "id": row["id"],
                "query": row["query"],
                "agents": list(traces.keys()),
                "event_count": len(events),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return results
```

Add import at top of `store.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scholaragent.core.context import ContextStream
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_context_stream.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add scholaragent/memory/store.py tests/test_context_stream.py
git commit -m "feat: add ContextStream persistence to MemoryStore"
```

---

### Task 3: Agent Integration — SpecialistAgent

**Files:**
- Modify: `scholaragent/core/agent.py:82-209` (add stream param, REPL injection, auto-commit)
- Modify: `scholaragent/environments/base.py:17-27` (add stream names to RESERVED_NAMES)
- Modify: `scholaragent/environments/local_repl.py:353-373` (restore stream scaffold)
- Test: `tests/test_context_stream.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_context_stream.py`:

```python
from unittest.mock import MagicMock, patch
from scholaragent.core.types import AgentResult, ModelUsageSummary, UsageSummary
from scholaragent.clients.base import BaseLM


class FakeLM(BaseLM):
    """Fake LM that returns a configurable response."""

    def __init__(self, response: str = "FINAL(done)"):
        super().__init__(model_name="fake-model")
        self._response = response

    def completion(self, prompt: str) -> str:
        return self._response

    async def acompletion(self, prompt: str) -> str:
        return self._response

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class MockStreamAgent(SpecialistAgent):
    """Agent for testing stream integration."""

    @property
    def name(self) -> str:
        return "scout"

    @property
    def system_prompt(self) -> str:
        return "You are a scout."


class TestAgentStreamIntegration:
    """Tests for stream injection into agent REPL."""

    @pytest.fixture()
    def handler(self):
        h = LMHandler(client=FakeLM("FINAL(done)"), token_counter=None, verbose=False)
        h.start()
        yield h
        h.stop()

    def test_stream_functions_injected_into_repl(self, handler):
        """When stream is provided, stream_push and stream_read are in REPL."""
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=handler, stream=stream)
        assert result.success

    def test_auto_commit_on_completion(self, handler):
        """Agent auto-commits trace to stream on successful completion."""
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=handler, stream=stream)
        assert result.success
        assert "scout" in stream.traces
        assert len(stream.traces["scout"]) > 0

    def test_auto_commit_on_failure(self, handler):
        """Agent auto-commits trace even on max-iterations failure."""
        lm = FakeLM("no final answer here")
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=h, max_iterations=1, stream=stream)
        assert not result.success
        assert "scout" in stream.traces
        h.stop()

    def test_stream_push_from_repl_code(self, handler):
        """Agent REPL code can call stream_push."""
        code_response = '```repl\nstream_push("papers_found", {"papers": [{"title": "A"}]})\nFINAL("done")\n```'
        lm = FakeLM(code_response)
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()
        stream = ContextStream(query="test")
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=h, stream=stream)
        assert result.success
        assert len(stream.state.papers) == 1
        h.stop()

    def test_stream_read_from_repl_code(self, handler):
        """Agent REPL code can call stream_read."""
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "Prior"}]})
        code_response = '```repl\ndata = stream_read()\nresult = str(len(data["events"]))\nFINAL(result)\n```'
        lm = FakeLM(code_response)
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()

        class ReaderAgent(SpecialistAgent):
            @property
            def name(self): return "reader"
            @property
            def system_prompt(self): return "You are a reader."

        agent = ReaderAgent()
        result = agent.run(task="test", handler=h, stream=stream)
        assert result.success
        assert result.result == "1"
        h.stop()

    def test_no_stream_no_injection(self, handler):
        """When stream=None, no stream functions in REPL."""
        agent = MockStreamAgent()
        result = agent.run(task="test", handler=handler)
        assert result.success
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_context_stream.py::TestAgentStreamIntegration -v`
Expected: FAIL — agent.run() doesn't accept `stream` param

**Step 3: Write the implementation**

In `scholaragent/environments/base.py`, update `RESERVED_NAMES` (line 17-27):

```python
RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "llm_query",
        "call_agent",
        "FINAL",
        "FINAL_VAR",
        "SHOW_VARS",
        "SHOW_PROGRESS",
        "context",
        "stream_push",
        "stream_read",
    }
)
```

In `scholaragent/environments/local_repl.py`, add to `_restore_scaffold()` (after the existing cases around line 365):

```python
            elif name == "stream_push":
                if hasattr(self, "_stream_push"):
                    self.globals["stream_push"] = self._stream_push
            elif name == "stream_read":
                if hasattr(self, "_stream_read"):
                    self.globals["stream_read"] = self._stream_read
```

In `scholaragent/core/agent.py`:

Add import (after line 22):

```python
if TYPE_CHECKING:
    from scholaragent.utils.budget import Budget
    from scholaragent.memory.store import MemoryStore
    from scholaragent.core.context import ContextStream
```

Update `run()` signature (line 82) to add `stream` parameter:

```python
    def run(
        self,
        task: str,
        handler: LMHandler,
        max_iterations: int = 10,
        agent_call_fn: Callable | None = None,
        verbose: bool = False,
        budget: Budget | None = None,
        store: MemoryStore | None = None,
        stream: ContextStream | None = None,
    ) -> AgentResult:
```

After the REPL is created and `call_agent` injected (after line 118), add stream injection:

```python
        # Inject stream functions if a ContextStream is provided
        if stream is not None:
            def _stream_push(event_type: str, data: dict) -> str:
                stream.push(self.name, event_type, data)
                return f"pushed:{event_type}"

            def _stream_read(agent: str | None = None) -> dict:
                return stream.read(agent=agent)

            repl.globals["stream_push"] = _stream_push
            repl.globals["stream_read"] = _stream_read
            repl._stream_push = _stream_push
            repl._stream_read = _stream_read
```

Before every `return AgentResult(...)` (lines 157, 180, 203), add auto-commit:

```python
            if stream is not None:
                stream.commit(self.name, messages)
```

There are three return points — the inline FINAL return (~line 157), the REPL FINAL_VAR return (~line 180), and the max-iterations return (~line 203). Add the commit before each one.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_context_stream.py -v`
Expected: All PASS

**Step 5: Run existing tests to verify no regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests PASS (stream param is optional, defaults to None)

**Step 6: Commit**

```bash
git add scholaragent/core/agent.py scholaragent/environments/base.py scholaragent/environments/local_repl.py tests/test_context_stream.py
git commit -m "feat: inject ContextStream into agent REPL with auto-commit"
```

---

### Task 4: Dispatcher Integration

**Files:**
- Modify: `scholaragent/core/dispatcher.py:55-110` (create stream, pass to children)
- Test: `tests/test_context_stream.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_context_stream.py`:

```python
from scholaragent.core.dispatcher import Dispatcher
from scholaragent.core.registry import AgentRegistry
from scholaragent.core.handler import LMHandler


class SimpleAgent(SpecialistAgent):
    """Minimal agent that immediately returns."""

    def __init__(self, agent_name: str = "scout"):
        self._name = agent_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def system_prompt(self) -> str:
        return f"You are {self._name}."


class TestDispatcherStream:
    """Tests for Dispatcher creating and threading ContextStream."""

    @pytest.fixture()
    def registry(self):
        reg = AgentRegistry()
        for name in ["scout", "reader", "critic", "analyst", "synthesizer"]:
            reg.register(SimpleAgent(name))
        return reg

    @pytest.fixture()
    def handler(self):
        h = LMHandler(client=FakeLM("FINAL(done)"), token_counter=None, verbose=False)
        h.start()
        yield h
        h.stop()

    def test_dispatcher_creates_stream(self, registry, handler):
        """Dispatcher creates a ContextStream on run()."""
        dispatcher = Dispatcher(registry=registry, handler=handler)
        result = dispatcher.run(task="test query")
        assert result.success
        assert dispatcher._stream is not None
        assert dispatcher._stream.query == "test query"

    def test_dispatcher_passes_stream_to_child(self, registry, handler):
        """Dispatched agents receive the stream."""
        # Use a dispatcher that calls scout
        code = '```repl\nresult = call_agent("scout", "find papers")\nFINAL(result)\n```'
        lm = FakeLM(code)
        h = LMHandler(client=lm, token_counter=None, verbose=False)
        h.start()
        dispatcher = Dispatcher(registry=registry, handler=h)
        result = dispatcher.run(task="test")
        assert dispatcher._stream is not None
        # Scout should have committed its trace
        assert "scout" in dispatcher._stream.traces
        h.stop()

    def test_dispatcher_stream_persists_with_store(self, registry, handler, tmp_path):
        """When store is set, stream is persisted on completion."""
        store = MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())
        dispatcher = Dispatcher(registry=registry, handler=handler, store=store)
        result = dispatcher.run(task="persist test")
        assert result.success
        streams = store.list_streams()
        assert len(streams) == 1
        assert streams[0]["query"] == "persist test"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_context_stream.py::TestDispatcherStream -v`
Expected: FAIL — Dispatcher doesn't create `_stream`

**Step 3: Write the implementation**

In `scholaragent/core/dispatcher.py`:

Add import (after line 12):

```python
from scholaragent.core.context import ContextStream
```

In `_dispatch_agent()` (line 74), pass `stream=self._stream`:

```python
        result = agent.run(
            task=task,
            handler=self._handler,
            max_iterations=10,
            budget=sub_budget,
            store=self._store,
            stream=self._stream,
        )
```

In `run()` (starting at line 90), create stream before calling super:

```python
    def run(
        self,
        task: str,
        handler: LMHandler | None = None,
        max_iterations: int = 15,
        agent_call_fn: Callable | None = None,
        verbose: bool = False,
        budget: Budget | None = None,
    ) -> AgentResult:
        """Override run() to create ContextStream and inject _dispatch_agent."""
        if budget is not None:
            self._budget = budget

        # Create a ContextStream for this pipeline run
        self._stream = ContextStream(
            query=task,
            on_save=self._store.save_stream if self._store is not None else None,
        )

        result = super().run(
            task=task,
            handler=self._handler,
            max_iterations=max_iterations,
            agent_call_fn=self._dispatch_agent,
            verbose=verbose,
            budget=self._budget,
            stream=self._stream,
        )

        # Final persist
        if self._store is not None:
            self._store.save_stream(self._stream)

        return result
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_context_stream.py -v`
Expected: All PASS

**Step 5: Run full test suite for regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add scholaragent/core/dispatcher.py tests/test_context_stream.py
git commit -m "feat: Dispatcher creates and threads ContextStream to agents"
```

---

### Task 5: ResearchPipeline Integration

**Files:**
- Modify: `scholaragent/memory/research.py:147-210` (`_run_normal`), `:212-250` (`_run_deep`)
- Test: `tests/test_context_stream.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_context_stream.py`:

```python
from scholaragent.memory.research import ResearchPipeline


class TestResearchPipelineStream:
    """Tests for ContextStream in ResearchPipeline."""

    @pytest.fixture()
    def store(self, tmp_path):
        return MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())

    @pytest.fixture()
    def pipeline_with_agents(self, store):
        handler = LMHandler(client=FakeLM("FINAL(done)"), token_counter=None, verbose=False)
        handler.start()
        registry = AgentRegistry()
        for name in ["scout", "reader", "critic", "analyst", "synthesizer"]:
            registry.register(SimpleAgent(name))
        dispatcher = Dispatcher(registry=registry, handler=handler, store=store)
        pipeline = ResearchPipeline(store=store)
        pipeline.set_agent_infra(handler, registry, dispatcher)
        return pipeline, handler

    def test_deep_run_creates_stream(self, pipeline_with_agents):
        pipeline, handler = pipeline_with_agents
        result = pipeline.run("test deep", depth="deep", force=True)
        assert result["status"] == "completed"
        streams = pipeline.store.list_streams()
        assert len(streams) >= 1
        handler.stop()

    def test_normal_run_creates_stream(self, pipeline_with_agents):
        pipeline, handler = pipeline_with_agents
        with patch("scholaragent.memory.research.search_arxiv", return_value="[]"), \
             patch("scholaragent.memory.research.search_semantic_scholar", return_value="[]"), \
             patch("scholaragent.memory.research.search_github_code", return_value=[]), \
             patch("scholaragent.memory.research.search_docs", return_value=[]):
            result = pipeline.run("test normal", depth="normal", force=True)
        assert result["status"] == "completed"
        handler.stop()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_context_stream.py::TestResearchPipelineStream -v`
Expected: FAIL or unexpected behavior (stream not created in pipeline path)

**Step 3: Write the implementation**

The `_run_deep()` method already delegates to `self.dispatcher.run()`, which now creates its own stream internally (from Task 4). No changes needed for deep.

For `_run_normal()` in `scholaragent/memory/research.py`, create a stream and pass it to agent runs. At the start of `_run_normal()` (after line 148):

```python
        from scholaragent.core.context import ContextStream
        stream = ContextStream(
            query=query,
            on_save=self.store.save_stream,
        )
```

In `_process_papers_normal()`, pass `stream` through. Update `_run_normal` to pass stream to `_process_papers_normal`:

```python
        enriched = self._process_papers_normal(raw_results, focus_hint, stream)
```

Update `_process_papers_normal` signature to accept stream:

```python
    def _process_papers_normal(
        self, raw_results: list[dict], focus_hint: str, stream: ContextStream | None = None
    ) -> list[dict]:
```

In `_process_one` inside `_process_papers_normal`, pass stream to reader/critic runs:

```python
                reader_result = reader.run(
                    task=reader_task,
                    handler=self.handler,
                    max_iterations=6,
                    store=self.store,
                    stream=stream,
                )
```

And similarly for critic:

```python
                critic_result = critic.run(
                    task=critic_task,
                    handler=self.handler,
                    max_iterations=6,
                    store=self.store,
                    stream=stream,
                )
```

At end of `_run_normal`, persist the stream:

```python
        self.store.save_stream(stream)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_context_stream.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add scholaragent/memory/research.py tests/test_context_stream.py
git commit -m "feat: thread ContextStream through ResearchPipeline"
```

---

### Task 6: MCP Tools

**Files:**
- Modify: `scholaragent/mcp_server.py:288-413` (add two tools + handler functions)
- Test: `tests/test_context_stream.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_context_stream.py`:

```python
from scholaragent.mcp_server import _memory_stream_list, _memory_stream_get


class TestMCPStreamTools:
    """Tests for MCP stream tool handler functions."""

    @pytest.fixture()
    def store(self, tmp_path):
        return MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())

    def test_stream_list_empty(self, store):
        result = _memory_stream_list(store)
        assert result["streams"] == []

    def test_stream_list_with_data(self, store):
        stream = ContextStream(query="test")
        stream.commit("scout", [{"role": "user", "content": "hi"}])
        store.save_stream(stream)
        result = _memory_stream_list(store)
        assert len(result["streams"]) == 1
        assert result["streams"][0]["query"] == "test"

    def test_stream_list_filter(self, store):
        store.save_stream(ContextStream(query="protein folding"))
        store.save_stream(ContextStream(query="quantum computing"))
        result = _memory_stream_list(store, query="protein")
        assert len(result["streams"]) == 1

    def test_stream_get_full(self, store):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": [{"title": "A"}]})
        stream.commit("scout", [{"role": "user", "content": "task"}])
        store.save_stream(stream)
        result = _memory_stream_get(store, stream.id)
        assert result["query"] == "test"
        assert "state" in result
        assert "traces" in result
        assert "events" in result

    def test_stream_get_filtered(self, store):
        stream = ContextStream(query="test")
        stream.push("scout", "papers_found", {"papers": []})
        stream.push("reader", "finding_extracted", {"paper_ref": "x", "finding": {}})
        stream.commit("scout", [])
        stream.commit("reader", [])
        store.save_stream(stream)
        result = _memory_stream_get(store, stream.id, agent="scout")
        assert "scout" in result["traces"]
        assert "reader" not in result["traces"]
        assert all(e["agent"] == "scout" for e in result["events"])

    def test_stream_get_not_found(self, store):
        result = _memory_stream_get(store, "nonexistent")
        assert "error" in result
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_context_stream.py::TestMCPStreamTools -v`
Expected: FAIL — `_memory_stream_list` and `_memory_stream_get` don't exist

**Step 3: Write the implementation**

In `scholaragent/mcp_server.py`, add handler functions (after `_memory_status` around line 286):

```python
def _memory_stream_list(
    store: MemoryStore,
    query: str | None = None,
    limit: int = 5,
) -> dict:
    if not 1 <= limit <= 50:
        return {"error": "limit must be between 1 and 50"}
    streams = store.list_streams(query=query, limit=limit)
    return {"streams": streams}


def _memory_stream_get(
    store: MemoryStore,
    stream_id: str,
    agent: str | None = None,
) -> dict:
    stream = store.load_stream(stream_id)
    if stream is None:
        return {"error": f"No stream found with id: {stream_id}"}
    data = stream.read(agent=agent)
    data["id"] = stream.id
    data["query"] = stream.query
    data["created_at"] = stream.created_at
    data["updated_at"] = stream.updated_at
    return data
```

Add MCP tool wrappers (after `memory_model_config` around line 401):

```python
@mcp.tool()
def memory_stream_list(
    query: str | None = None,
    limit: int = 5,
) -> str:
    """List recent research pipeline context streams.

    Shows metadata for past research runs including which agents
    participated and how many events were recorded. Use to find
    a stream_id for memory_stream_get.

    Args:
        query: Filter by research query text (optional)
        limit: Maximum streams to return (default 5)
    """
    result = _memory_stream_list(_get_store(), query, limit)
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_stream_get(
    stream_id: str,
    agent: str | None = None,
) -> str:
    """Get full context from a research pipeline run.

    Returns the structured state (papers, findings, assessments,
    themes, synthesis), conversation traces, and event log from
    a specific research run. Use to understand HOW a conclusion
    was reached, not just WHAT was concluded.

    Args:
        stream_id: Stream ID from memory_stream_list
        agent: Filter to one agent's data (optional: "scout", "reader", "critic", "analyst", "synthesizer")
    """
    result = _memory_stream_get(_get_store(), stream_id, agent)
    return json.dumps(result, indent=2)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_context_stream.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add scholaragent/mcp_server.py tests/test_context_stream.py
git commit -m "feat: add memory_stream_list and memory_stream_get MCP tools"
```

---

### Task 7: Final Integration Test + Cleanup

**Files:**
- Test: `tests/test_context_stream.py` (append end-to-end test)
- Modify: `scholaragent/__init__.py` (export ContextStream)

**Step 1: Write end-to-end integration test**

Append to `tests/test_context_stream.py`:

```python
class TestEndToEnd:
    """End-to-end test: full pipeline with stream persistence."""

    def test_full_pipeline_with_stream(self, tmp_path):
        """Dispatcher → agents → stream → persist → load → read."""
        store = MemoryStore(db_path=str(tmp_path / "test.db"), embeddings=FakeEmbeddings())

        # Agent that calls scout then returns
        dispatch_code = '```repl\nscout_result = call_agent("scout", "find papers")\nFINAL(scout_result)\n```'
        handler = LMHandler(client=FakeLM(dispatch_code), token_counter=None, verbose=False)
        handler.start()

        registry = AgentRegistry()
        registry.register(SimpleAgent("scout"))
        registry.register(SimpleAgent("reader"))
        registry.register(SimpleAgent("critic"))
        registry.register(SimpleAgent("analyst"))
        registry.register(SimpleAgent("synthesizer"))

        dispatcher = Dispatcher(registry=registry, handler=handler, store=store)
        result = dispatcher.run(task="end to end test")

        # Verify stream was created and persisted
        streams = store.list_streams()
        assert len(streams) == 1
        assert streams[0]["query"] == "end to end test"

        # Verify we can load and read it
        loaded = store.load_stream(streams[0]["id"])
        assert loaded is not None
        assert "dispatcher" in loaded.traces
        data = loaded.read()
        assert "state" in data
        assert "traces" in data
        assert "events" in data

        handler.stop()
```

**Step 2: Run the end-to-end test**

Run: `python -m pytest tests/test_context_stream.py::TestEndToEnd -v`
Expected: PASS

**Step 3: Add export to `__init__.py`**

In `scholaragent/__init__.py`, add to `__all__` and imports:

```python
__all__ = [
    "ScholarAgent",
    "ModelConfig",
    "ModelRouter",
    "Dispatcher",
    "AgentResult",
    "ResearchReport",
    "ContextStream",
]
```

```python
from scholaragent.core.context import ContextStream
```

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add scholaragent/__init__.py tests/test_context_stream.py
git commit -m "feat: end-to-end integration test and export ContextStream"
```

---

### Task 8: Final Full Suite Verification

**Step 1: Run complete test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS including all existing tests (no regressions)

**Step 2: Verify MCP server starts cleanly**

Run: `cd /Volumes/WD_4D/RLM/scholaragent && python -c "from scholaragent.mcp_server import mcp; print('MCP server imports OK')"`
Expected: `MCP server imports OK`

**Step 3: Verify public API**

Run: `python -c "from scholaragent import ContextStream; s = ContextStream(query='test'); print(s.id, s.state.papers)"`
Expected: UUID and `[]`
