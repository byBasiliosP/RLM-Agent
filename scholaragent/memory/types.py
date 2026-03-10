"""Data types for the memory layer."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

VALID_SOURCE_TYPES: frozenset[str] = frozenset({"paper", "docs", "code"})


@dataclass
class MemoryEntry:
    content: str
    summary: str
    source_type: str
    source_ref: str
    tags: list[str]
    id: str = field(default_factory=lambda: str(uuid4()))
    embedding: list[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0

    def __post_init__(self):
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}, got '{self.source_type}'"
            )

    @staticmethod
    def smart_summary(content: str, max_length: int = 200) -> str:
        """Truncate content at the last sentence boundary within max_length.

        Produces clean summaries that end at sentence boundaries instead
        of cutting mid-word. Falls back to word boundaries for content
        without sentence-ending punctuation (e.g., code snippets).
        """
        if len(content) <= max_length:
            return content
        truncated = content[:max_length]
        # Find last sentence-ending punctuation
        for sep in ('. ', '.\n', '? ', '!\n'):
            idx = truncated.rfind(sep)
            if idx > max_length // 3:
                return truncated[:idx + 1].strip()
        # Fallback: truncate at last space to avoid mid-word cuts
        idx = truncated.rfind(' ')
        if idx > max_length // 3:
            return truncated[:idx] + '...'
        return truncated + '...'

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

    def to_compact_dict(self) -> dict:
        """Return a compact representation without full content.

        Used by MCP tools to reduce response size and avoid
        filling up the client agent's context window.
        """
        return {
            "id": self.id,
            "summary": self.summary,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "tags": list(self.tags),
        }


@dataclass
class ResearchLogEntry:
    query: str
    depth: str
    focus: str
    result_count: int
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "depth": self.depth,
            "focus": self.focus,
            "result_count": self.result_count,
            "created_at": self.created_at,
        }
