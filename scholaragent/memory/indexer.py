"""ResultIndexer — persists research results into the MemoryStore.

Extracted from ResearchPipeline._run_quick / _run_normal / _run_deep.
Keeps all MemoryEntry construction and tagging in one place so that
research.py can stay focused on depth orchestration.
"""
from __future__ import annotations

from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry


class ResultIndexer:
    """Writes MemoryEntries into the store with consistent tagging."""

    def __init__(self, store: MemoryStore):
        self._store = store

    def index_raw(self, query: str, raw_results: list[dict]) -> int:
        """Index raw source results. Returns number of entries added."""
        added = 0
        tag = query.lower().replace(" ", "-")
        for raw in raw_results:
            entry = MemoryEntry(
                content=raw["content"],
                summary=MemoryEntry.smart_summary(raw["content"]),
                source_type=raw["source_type"],
                source_ref=raw["source_ref"],
                tags=[tag],
            )
            self._store.add(entry)
            added += 1
        return added

    def index_enriched(self, query: str, enriched_results: list[dict]) -> int:
        """Index agent-enriched results with reader/critic output concatenated."""
        added = 0
        tag = query.lower().replace(" ", "-")
        for item in enriched_results:
            content = item["content"]
            if item.get("reader_findings"):
                content += f"\n\n--- Reader Analysis ---\n{item['reader_findings']}"
            if item.get("critic_assessment"):
                content += f"\n\n--- Critic Assessment ---\n{item['critic_assessment']}"

            tags = [tag]
            if item.get("reader_findings") or item.get("critic_assessment"):
                tags.append("agent-processed")

            entry = MemoryEntry(
                content=content,
                summary=MemoryEntry.smart_summary(content),
                source_type=item["source_type"],
                source_ref=item["source_ref"],
                tags=tags,
            )
            self._store.add(entry)
            added += 1
        return added

    def index_synthesis(self, query: str, synthesis_text: str) -> int:
        """Index a deep-pipeline synthesized report. Returns 1 or 0."""
        if not synthesis_text:
            return 0
        entry = MemoryEntry(
            content=synthesis_text,
            summary=MemoryEntry.smart_summary(synthesis_text),
            source_type="synthesized_report",
            source_ref=f"deep-research:{query[:50]}",
            tags=[query.lower().replace(" ", "-"), "deep-pipeline", "synthesized"],
        )
        self._store.add(entry)
        return 1

    def log_research(
        self, query: str, depth: str, focus: str, result_count: int
    ) -> None:
        self._store.log_research(
            query=query, depth=depth, focus=focus, result_count=result_count
        )
