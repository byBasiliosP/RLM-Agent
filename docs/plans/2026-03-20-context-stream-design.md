# Persistent Context Stream — Design Document

**Date:** 2026-03-20
**Status:** Approved

## Problem

Agents in the ScholarAgent pipeline run in isolation. Each gets a task string and returns a flat `AgentResult.result` string. The Dispatcher stitches results together via ad-hoc string concatenation in its REPL code. There is no shared structured state, no conversation trace continuity, and no persistence of pipeline reasoning across runs.

## Goals

1. **Structured pipeline state** — typed slots (papers, findings, assessments, themes, synthesis) that accumulate as each agent runs
2. **Conversation trace continuity** — each agent can access prior agents' full LLM conversation history
3. **Streaming + snapshot writes** — agents push incremental updates mid-execution and commit a final snapshot on completion
4. **Persistent to disk** — context streams survive across runs, stored in SQLite alongside existing memory DB
5. **MCP-accessible** — coding agents can query pipeline reasoning through new MCP tools

## Approach

**ContextStream as a standalone dataclass + SQLite table** — a new `ContextStream` object created per pipeline run, persisted to a `context_streams` table. Agents receive a reference and interact via `push()` and `commit()`.

### Alternatives considered

- **Extend MemoryStore entries** — reuse `entries` table with `source_type="stream"`. Rejected: mixes concerns, bloats entries, conversation traces don't benefit from embeddings.
- **Event-sourced append-only log** — separate `stream_events` table with replay. Rejected: over-engineered for current needs, requires aggregation logic.

---

## Data Model

### `scholaragent/core/context.py` (new file)

```python
@dataclass
class ContextStream:
    id: str                          # UUID, one per pipeline run
    query: str                       # research query that spawned this
    created_at: str                  # ISO 8601 UTC
    updated_at: str                  # ISO 8601 UTC, bumped on push/commit

    state: PipelineState             # typed slots per pipeline stage
    traces: dict[str, list[dict]]    # {agent_name: [{role, content}, ...]}
    events: list[StreamEvent]        # append-only log of incremental updates

    def push(agent: str, event_type: str, data: dict) -> None
    def commit(agent: str, messages: list[dict]) -> None
    def read(agent: str | None = None) -> dict
    def to_dict() -> dict
    @classmethod
    def from_dict(cls, d: dict) -> ContextStream

@dataclass
class PipelineState:
    papers: list[dict]               # Scout output
    findings: dict[str, dict]        # Reader output keyed by paper ref
    assessments: dict[str, dict]     # Critic output keyed by paper ref
    themes: dict                     # Analyst output
    synthesis: str                   # Synthesizer output

@dataclass
class StreamEvent:
    timestamp: str                   # ISO 8601 UTC
    agent: str                       # agent name
    event_type: str                  # "papers_found", "finding_extracted", etc.
    data: dict                       # event payload
```

### State slot mapping

| Agent | State slot | Event types |
|-------|-----------|-------------|
| scout | `state.papers` | `papers_found` |
| reader | `state.findings[paper_ref]` | `finding_extracted` |
| critic | `state.assessments[paper_ref]` | `assessment_complete` |
| analyst | `state.themes` | `themes_identified` |
| synthesizer | `state.synthesis` | `synthesis_complete` |

### `push()` behavior
- Appends a `StreamEvent` to `events`
- Updates the appropriate `state` slot based on agent name and event type
- Triggers `store.save_stream(self)` for crash safety

### `commit()` behavior
- Saves the agent's full `messages` list into `traces[agent_name]`
- Updates the `state` slot with the agent's final result
- Triggers `store.save_stream(self)` for persistence

---

## Persistence Layer

### SQLite table (in existing `memory.db`)

```sql
CREATE TABLE IF NOT EXISTS context_streams (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    state TEXT NOT NULL,           -- JSON blob of PipelineState
    traces TEXT NOT NULL,          -- JSON blob of {agent: messages}
    events TEXT NOT NULL,          -- JSON blob of [StreamEvent, ...]
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### MemoryStore additions (in `memory/store.py`)

- `save_stream(stream: ContextStream)` — upsert, called on every push/commit
- `load_stream(stream_id: str) -> ContextStream | None` — reconstruct from row
- `list_streams(query: str | None = None, limit: int = 10) -> list[dict]` — recent streams, compact metadata

---

## Agent Integration

### `SpecialistAgent.run()` changes

```python
def run(self, task, handler, ..., stream: ContextStream | None = None) -> AgentResult:
```

- Stream reference injected into REPL namespace as two functions:
  - `stream_push(event_type, data)` — mid-execution, auto-tagged with agent name
  - `stream_read(agent=None)` — read state/events, optionally filtered
- On completion (before returning `AgentResult`), `run()` auto-calls `stream.commit(self.name, messages)`
- Stream is optional — all existing behavior unchanged when `stream=None`

### `Dispatcher` changes

- Creates `ContextStream` at start of `run()`
- Stores as `self._stream`
- Passes to every child agent via `_dispatch_agent()`
- `stream_push` and `stream_read` also available in Dispatcher's own REPL

### `ResearchPipeline` changes

- `_run_deep()`: stream created by Dispatcher (no change needed)
- `_run_normal()`: create stream, pass to Scout/Reader/Critic runs, persist at end

---

## MCP Tools

Two new tools in `mcp_server.py`:

| Tool | Signature | Returns |
|------|-----------|---------|
| `memory_stream_list` | `(query: str \| None, limit: int = 5)` | List of `{id, query, created_at, agents: [...], event_count}` |
| `memory_stream_get` | `(stream_id: str, agent: str \| None)` | Full stream dict, or filtered to one agent's trace + events |

No changes to existing 7 MCP tools.

---

## Files Touched

| File | Change |
|------|--------|
| `scholaragent/core/context.py` | **NEW** — ContextStream, PipelineState, StreamEvent |
| `scholaragent/core/agent.py` | Add `stream` param, REPL injection, auto-commit |
| `scholaragent/core/dispatcher.py` | Create stream, pass to children |
| `scholaragent/memory/store.py` | `context_streams` table, save/load/list methods |
| `scholaragent/mcp_server.py` | `memory_stream_list`, `memory_stream_get` tools |
| `scholaragent/memory/research.py` | Pass stream in `_run_normal` and `_run_deep` |
| `tests/test_context_stream.py` | **NEW** — full test coverage |

## What stays the same

- Agent system prompts (no changes)
- `AgentResult` dataclass (stream is additive)
- Existing 7 MCP tools (untouched)
- Budget system (unaffected)
- Existing memory entries and research log tables
