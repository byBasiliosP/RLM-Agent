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
