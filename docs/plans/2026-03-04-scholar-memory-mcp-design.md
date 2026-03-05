# ScholarAgent Memory MCP Server — Design Document

**Date**: 2026-03-04
**Status**: Approved

## Goal

Turn ScholarAgent into a persistent, semantically-indexed knowledge layer that any coding agent can query via MCP. The agent calls `memory_lookup` for fast retrieval of known knowledge, and `memory_research` when it needs to learn something new. Results accumulate over time, forming a recursive memory that grows smarter with use.

## MCP Tool Interface

| Tool | Speed | Signature |
|---|---|---|
| `memory_lookup` | ~100ms | `(query: str, sources?: list[str], max_results?: int) -> Results` |
| `memory_research` | 5s–5min | `(query: str, depth?: "quick"\|"normal"\|"deep", focus?: "implementation"\|"theory"\|"comparison") -> Results` |
| `memory_store` | Fast | `(content: str, source: str, tags: list[str]) -> Confirmation` |
| `memory_forget` | Fast | `(query_or_id: str) -> Confirmation` |
| `memory_status` | Fast | `() -> Stats` |

### Return Format

All tools return JSON. `memory_lookup` and `memory_research` return:

```json
{
  "results": [
    {
      "id": "uuid",
      "content": "The actual finding, code snippet, or insight...",
      "summary": "One-line summary",
      "source_type": "paper|docs|code",
      "source_ref": "arxiv:2401.12345 or URL",
      "relevance_score": 0.87,
      "tags": ["rlhf", "reward-model"]
    }
  ],
  "total_indexed": 342,
  "query_time_ms": 45
}
```

## Storage

### MemoryEntry Schema

```
id: str (uuid)
content: str
summary: str
source_type: "paper" | "docs" | "code"
source_ref: str
tags: list[str]
embedding: list[float]
created_at: datetime
access_count: int
```

### Database

SQLite at `~/.scholaragent/memory.db` with two tables:

- `entries` — all MemoryEntry fields, embedding stored as blob
- `research_log` — tracks past queries with timestamps for deduplication

### Embeddings

Swappable backend behind an `EmbeddingBackend` ABC:

- **Initial**: OpenAI API embeddings (`text-embedding-3-small`)
- **Future**: Local sentence-transformers model

Embedding cache at `~/.scholaragent/embeddings_cache/` to avoid recomputation.

## Sources

### Papers (existing)

Already built: `search_arxiv()`, `search_semantic_scholar()`, `get_citations()`, `get_references()`. Wire directly into the research pipeline.

### Documentation (new)

Fetch and extract text from web pages using `httpx` + HTML-to-text extraction. Used when researching library docs, API references, READMEs.

### Code (new)

GitHub API search (`api.github.com/search/code`). Returns file snippets with repo context. No auth required for public repos (rate-limited), optional `GITHUB_TOKEN` for higher limits.

### Adapter Interface

All sources return `list[dict]` with `content`, `source_ref`, `source_type`. The pipeline normalizes them into MemoryEntries.

## Research Pipeline

### Depth Levels

| Depth | Time | Agents Used | Behavior |
|---|---|---|---|
| `quick` | 5–10s | Scout only | Search all sources, store raw results |
| `normal` | 30–60s | Scout → Reader → Critic | Search, extract findings, score relevance |
| `deep` | 2–5min | Scout → Reader → Critic → Analyst → Synthesizer | Full pipeline with cross-paper synthesis |

### Focus Parameter

Shapes specialist agent system prompts:

- `"implementation"` — prioritize code examples, API usage, how-to
- `"theory"` — prioritize concepts, algorithms, trade-offs
- `"comparison"` — prioritize alternatives, benchmarks, pros/cons

### Deduplication

Before running research, check `research_log` for semantically similar queries within 7 days. If found, return cached results. Force re-research via a flag if needed.

## MCP Server Architecture

### Transport

stdio — standard for local MCP servers.

### Configuration

```json
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
```

### Process Model

Single long-running process. SQLite handles concurrency. The LMHandler TCP server starts on first `memory_research` call and stays alive for the session.

### Data Directory

`~/.scholaragent/` containing:
- `memory.db` — SQLite database
- `embeddings_cache/` — cached embedding vectors

## New Files

Built on top of the existing ScholarAgent codebase at `/Volumes/WD_4D/RLM/scholaragent/`:

- `scholaragent/mcp_server.py` — MCP server entry point + tool definitions
- `scholaragent/memory/store.py` — SQLite storage + semantic search
- `scholaragent/memory/embeddings.py` — swappable embedding backend (ABC + OpenAI impl)
- `scholaragent/memory/types.py` — MemoryEntry, ResearchLog dataclasses
- `scholaragent/sources/docs.py` — documentation fetcher
- `scholaragent/sources/github.py` — GitHub code search adapter
- Tests for all of the above

## Compatibility

Designed to work with any MCP-compatible coding agent: Claude Code, Codex, Gemini, Copilot.

## Key Differences from Base ScholarAgent

| Aspect | Base ScholarAgent | Memory MCP Server |
|---|---|---|
| Interface | Python API | MCP tools over stdio |
| Persistence | None | SQLite + embeddings |
| Sources | Papers only | Papers + docs + code |
| Research depth | Always full pipeline | Configurable: quick/normal/deep |
| Memory | Ephemeral per session | Grows across sessions |
| Consumer | Direct Python caller | Any MCP-compatible agent |
