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
) -> dict:
    if not 1 <= max_results <= 50:
        return {"error": "max_results must be between 1 and 50"}
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
    from past research. Call this frequently while coding -- it's fast.

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
