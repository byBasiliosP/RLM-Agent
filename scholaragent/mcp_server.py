"""ScholarAgent Memory MCP Server.

Run: python -m scholaragent.mcp_server
Or:  uv run mcp run scholaragent/mcp_server.py

Configuration for coding agents:
{
    "mcpServers": {
        "scholar-memory": {
            "command": "scholaragent-server"
        }
    }
}

API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) are read from the user's
shell environment at runtime — never store them in config files.

Model backend configuration via environment variables:
    SCHOLAR_STRONG_BACKEND  - "openai", "anthropic", or "lmstudio" (default: "anthropic")
    SCHOLAR_STRONG_MODEL    - Model name for strong agents (default: "claude-sonnet-4-6")
    SCHOLAR_CHEAP_BACKEND   - "openai", "anthropic", or "lmstudio" (default: "openai")
    SCHOLAR_CHEAP_MODEL     - Model name for cheap agents (default: "gpt-4o-mini")
    SCHOLAR_LMSTUDIO_URL    - LM Studio base URL (default: "http://localhost:1234/v1")
    SCHOLAR_EMBEDDING_BACKEND - "openai" or "lmstudio" (default: "openai")
    SCHOLAR_EMBEDDING_MODEL - Embedding model name (defaults depend on backend)
    SCHOLAR_EMBEDDING_BASE_URL - Optional embedding endpoint override
"""

import atexit
import json
import logging
import threading

logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP

from scholaragent.memory.research import ResearchPipeline
from scholaragent.memory.store import MemoryStore

# --- Validation constants ---

VALID_DEPTHS = frozenset({"quick", "normal", "deep"})
VALID_FOCUSES = frozenset({"implementation", "theory", "comparison"})
VALID_SOURCE_TYPES = frozenset({"paper", "code", "docs", "synthesized_report"})

# Input length limits (characters)
MAX_QUERY_LEN = 2000
MAX_ID_LEN = 256
MAX_TAGS = 20
MAX_TAG_LEN = 64
MAX_CONTENT_LEN = 200_000


def _validate_text(name: str, value: str, max_len: int, allow_empty: bool = False) -> str | None:
    """Return an error string if invalid, else None."""
    if not isinstance(value, str):
        return f"{name} must be a string"
    stripped = value.strip()
    if not allow_empty and not stripped:
        return f"{name} must be non-empty"
    if len(value) > max_len:
        return f"{name} exceeds max length of {max_len} characters"
    return None

# --- Runtime container (replaces former module-level globals) ---

_container = None  # RuntimeContainer, lazy-init
_container_lock = threading.Lock()


def _build_model_config() -> dict:
    """Build strong/cheap model config dicts from environment variables."""
    from scholaragent.config import ScholarConfig

    return ScholarConfig.from_env().model_config_dict()


def _get_container():
    """Lazy-init the runtime container (thread-safe)."""
    global _container
    if _container is not None:
        return _container
    with _container_lock:
        if _container is not None:
            return _container
        from scholaragent.config import ScholarConfig
        from scholaragent.runtime import RuntimeContainer

        cfg = ScholarConfig.from_env()
        _container = RuntimeContainer(
            data_dir=cfg.data_dir,
            db_path=cfg.db_path,
            model_config=cfg.model_config_dict(),
        )
        atexit.register(_container.close)
        return _container


def _get_store() -> MemoryStore:
    return _get_container().get_store()


def _get_pipeline() -> ResearchPipeline:
    return _get_container().get_pipeline()


def _ensure_pipeline_agents(pipeline: ResearchPipeline) -> None:
    """Upgrade a pipeline with agent infrastructure if not already set."""
    if pipeline.has_agent_infra:
        return
    handler, registry, dispatcher = _get_container().get_agent_infra()
    pipeline.set_agent_infra(handler, registry, dispatcher)


# --- Tool handler functions (testable without MCP transport) ---


def _memory_lookup(
    store: MemoryStore,
    query: str,
    sources: list[str] | None = None,
    max_results: int = 5,
    compact: bool = True,
) -> dict:
    if err := _validate_text("query", query, MAX_QUERY_LEN):
        return {"error": err}
    if not 1 <= max_results <= 50:
        return {"error": "max_results must be between 1 and 50"}
    if sources is not None:
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            return {"error": "sources must be a list of strings"}
        invalid = [s for s in sources if s not in VALID_SOURCE_TYPES]
        if invalid:
            return {"error": f"invalid source types: {invalid}. Allowed: {sorted(VALID_SOURCE_TYPES)}"}
    results = store.search(query, max_results=max_results, sources=sources)
    return {
        "results": [
            {
                **(entry.to_compact_dict() if compact else entry.to_dict()),
                "relevance_score": round(score, 3),
            }
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
    if err := _validate_text("query", query, MAX_QUERY_LEN):
        return {"error": err}
    if depth not in VALID_DEPTHS:
        return {"error": f"depth must be one of {sorted(VALID_DEPTHS)}"}
    if focus not in VALID_FOCUSES:
        return {"error": f"focus must be one of {sorted(VALID_FOCUSES)}"}
    # Upgrade pipeline with agent infrastructure for normal/deep depths
    if depth != "quick":
        try:
            _ensure_pipeline_agents(pipeline)
        except Exception as e:
            logger.warning("Failed to init agent infra, falling back to quick: %s", e)
            depth = "quick"
    return pipeline.run(query=query, depth=depth, focus=focus)


def _memory_store(
    store: MemoryStore,
    content: str,
    source: str,
    tags: list[str],
) -> dict:
    if err := _validate_text("content", content, MAX_CONTENT_LEN):
        return {"error": err}
    if err := _validate_text("source", source, MAX_ID_LEN):
        return {"error": err}
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        return {"error": "tags must be a list of strings"}
    if len(tags) > MAX_TAGS:
        return {"error": f"tags exceeds max of {MAX_TAGS} items"}
    if any(len(t) > MAX_TAG_LEN for t in tags):
        return {"error": f"each tag must be at most {MAX_TAG_LEN} characters"}
    from scholaragent.memory.types import MemoryEntry

    # Infer source_type from source string
    source_type = "docs"
    if source.startswith("arxiv:") or source.startswith("s2:"):
        source_type = "paper"
    elif source.startswith("github:") or source.startswith("https://github.com/"):
        source_type = "code"

    entry = MemoryEntry(
        content=content,
        summary=MemoryEntry.smart_summary(content),
        source_type=source_type,
        source_ref=source,
        tags=tags,
    )
    store.add(entry)
    return {"status": "stored", "id": entry.id, "source_type": source_type}


def _memory_get(store: MemoryStore, entry_id: str) -> dict:
    if err := _validate_text("entry_id", entry_id, MAX_ID_LEN):
        return {"error": err}
    entry = store.get(entry_id)
    if entry is None:
        return {"error": f"No entry found with id: {entry_id}"}
    return entry.to_dict()


def _memory_forget(
    store: MemoryStore,
    query_or_id: str,
) -> dict:
    if err := _validate_text("query_or_id", query_or_id, MAX_QUERY_LEN):
        return {"error": err}
    deleted = store.forget(query_or_id)
    return {"deleted": deleted, "query_or_id": query_or_id}


def _memory_status(store: MemoryStore, token_counter=None) -> dict:
    status = store.status()
    # Include token usage and cost info if available
    if token_counter is not None:
        status["token_usage"] = token_counter.summary()
        status["estimated_costs"] = token_counter.cost_summary()
    return status


def _memory_stream_list(
    store: MemoryStore,
    query: str | None = None,
    limit: int = 5,
) -> dict:
    if not 1 <= limit <= 50:
        return {"error": "limit must be between 1 and 50"}
    if query is not None:
        if err := _validate_text("query", query, MAX_QUERY_LEN, allow_empty=True):
            return {"error": err}
        query = query.strip() or None
    streams = store.list_streams(query=query, limit=limit)
    return {"streams": streams}


def _memory_stream_get(
    store: MemoryStore,
    stream_id: str,
    agent: str | None = None,
) -> dict:
    if err := _validate_text("stream_id", stream_id, MAX_ID_LEN):
        return {"error": err}
    if agent is not None:
        if err := _validate_text("agent", agent, MAX_ID_LEN):
            return {"error": err}
    stream = store.load_stream(stream_id)
    if stream is None:
        return {"error": f"No stream found with id: {stream_id}"}
    data = stream.read(agent=agent)
    data["id"] = stream.id
    data["query"] = stream.query
    data["created_at"] = stream.created_at
    data["updated_at"] = stream.updated_at
    return data


# --- MCP Server ---

mcp = FastMCP("scholar-memory", json_response=True)


@mcp.tool()
def memory_lookup(
    query: str,
    sources: list[str] | None = None,
    max_results: int = 5,
    compact: bool = True,
) -> str:
    """Fast semantic search over all indexed knowledge.

    Returns the most relevant findings, code snippets, and insights
    from past research. Call this frequently while coding -- it's fast.

    Args:
        query: What you're looking for (natural language)
        sources: Filter by source type: "paper", "docs", "code" (optional)
        max_results: Maximum results to return (default 5)
        compact: Return summaries only (default True). Set False for full content.
    """
    result = _memory_lookup(_get_store(), query, sources, max_results, compact)
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_get(entry_id: str) -> str:
    """Get the full content of a single memory entry by ID.

    Use after memory_lookup to retrieve full details for a specific
    result. This avoids loading all results into context at once.

    Args:
        entry_id: The entry ID from a memory_lookup result
    """
    result = _memory_get(_get_store(), entry_id)
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
    container = _get_container()
    result = _memory_status(container.get_store(), token_counter=container.get_token_counter())
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_model_config() -> str:
    """Show the current LLM backend configuration.

    Returns which models and backends are configured for strong
    (analytical) and cheap (search) agents.
    """
    config = _build_model_config()
    return json.dumps(config, indent=2)


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


def main():
    config = _build_model_config()
    logger.info("Model config: strong=%s/%s, cheap=%s/%s",
                config["strong"]["backend"], config["strong"]["model_name"],
                config["cheap"]["backend"], config["cheap"]["model_name"])
    mcp.run()


if __name__ == "__main__":
    main()
