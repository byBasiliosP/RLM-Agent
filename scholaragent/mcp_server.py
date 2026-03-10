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
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP

from scholaragent.memory.embeddings import OpenAIEmbeddings
from scholaragent.memory.store import MemoryStore
from scholaragent.memory.research import ResearchPipeline

# --- Validation constants ---

VALID_DEPTHS = frozenset({"quick", "normal", "deep"})
VALID_FOCUSES = frozenset({"implementation", "theory", "comparison"})

# --- Global state ---

_store: MemoryStore | None = None
_pipeline: ResearchPipeline | None = None
_init_lock = threading.Lock()

DATA_DIR = Path(os.environ.get("SCHOLAR_MEMORY_DIR", Path.home() / ".scholaragent"))
DB_PATH = os.environ.get("SCHOLAR_MEMORY_DB", str(DATA_DIR / "memory.db"))


def _build_model_config() -> dict:
    """Build strong/cheap model config dicts from environment variables."""
    lmstudio_url = os.environ.get("SCHOLAR_LMSTUDIO_URL", "http://localhost:1234/v1")

    strong_backend = os.environ.get("SCHOLAR_STRONG_BACKEND", "anthropic")
    strong_model = os.environ.get("SCHOLAR_STRONG_MODEL", "claude-sonnet-4-6")
    cheap_backend = os.environ.get("SCHOLAR_CHEAP_BACKEND", "openai")
    cheap_model = os.environ.get("SCHOLAR_CHEAP_MODEL", "gpt-4o-mini")

    strong = {"backend": strong_backend, "model_name": strong_model}
    cheap = {"backend": cheap_backend, "model_name": cheap_model}

    if strong_backend == "lmstudio":
        strong["base_url"] = lmstudio_url
    if cheap_backend == "lmstudio":
        cheap["base_url"] = lmstudio_url

    return {"strong": strong, "cheap": cheap}


def _cleanup():
    """Close the global memory store on interpreter exit."""
    global _store
    if _store is not None:
        _store.close()
        _store = None


atexit.register(_cleanup)


def _get_store() -> MemoryStore:
    """Lazy-init the global memory store (thread-safe)."""
    global _store
    if _store is not None:
        return _store
    with _init_lock:
        if _store is not None:
            return _store
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        embeddings = OpenAIEmbeddings()
        _store = MemoryStore(db_path=DB_PATH, embeddings=embeddings)
        logger.info("Initialized memory store at %s", DB_PATH)
        return _store


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


# --- Tool handler functions (testable without MCP transport) ---


def _memory_lookup(
    store: MemoryStore,
    query: str,
    sources: list[str] | None = None,
    max_results: int = 5,
    compact: bool = True,
) -> dict:
    if not 1 <= max_results <= 50:
        return {"error": "max_results must be between 1 and 50"}
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
    if depth not in VALID_DEPTHS:
        return {"error": f"depth must be one of {sorted(VALID_DEPTHS)}"}
    if focus not in VALID_FOCUSES:
        return {"error": f"focus must be one of {sorted(VALID_FOCUSES)}"}
    return pipeline.run(query=query, depth=depth, focus=focus)


def _memory_store(
    store: MemoryStore,
    content: str,
    source: str,
    tags: list[str],
) -> dict:
    if not content or not content.strip():
        return {"error": "content must be non-empty"}
    from scholaragent.memory.types import MemoryEntry

    # Infer source_type from source string
    source_type = "docs"
    if source.startswith("arxiv:") or source.startswith("s2:"):
        source_type = "paper"
    elif source.startswith("github:") or source.startswith("https://github.com"):
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
    entry = store.get(entry_id)
    if entry is None:
        return {"error": f"No entry found with id: {entry_id}"}
    return entry.to_dict()


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
    result = _memory_status(_get_store())
    return json.dumps(result, indent=2)


@mcp.tool()
def memory_model_config() -> str:
    """Show the current LLM backend configuration.

    Returns which models and backends are configured for strong
    (analytical) and cheap (search) agents.
    """
    config = _build_model_config()
    return json.dumps(config, indent=2)


def main():
    config = _build_model_config()
    logger.info("Model config: strong=%s/%s, cheap=%s/%s",
                config["strong"]["backend"], config["strong"]["model_name"],
                config["cheap"]["backend"], config["cheap"]["model_name"])
    mcp.run()


if __name__ == "__main__":
    main()
