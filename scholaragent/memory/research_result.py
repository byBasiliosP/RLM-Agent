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
